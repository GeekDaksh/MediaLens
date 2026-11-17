"""Word lists, severity tiers, text normalisation and matching.
No offensive word is written in this file: every word comes from the downloaded lists."""
import hashlib, json, re, unicodedata, urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from rapidfuzz import fuzz, process
from wordfreq import zipf_frequency

LDNOOBW_COMMIT = "5faf2ba42d7b1c0977169ec3611df25a3c08eb13"   # pinned so the lists never change under us
LDNOOBW_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/{commit}/{lang}"
LDNOOBW_SHA256 = {"en": "af851ecef1d5f212caba17339b12ac39cc2fef7d78c74876f67237644fcee8bd",
                  "hi": "83636e9b7a5260451e09e9e471f81dad9f082d107a82da77047e57d7856ed0cc"}
TIER_RANK = {"none": 0, "mild": 1, "strong": 2, "slur": 3}
DEVANAGARI = re.compile("[ऀ-ॿ]")


def download_lists(dest):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for lang in ("en", "hi"):
        p = dest / f"ldnoobw_{lang}.txt"
        if not p.exists():
            urllib.request.urlretrieve(LDNOOBW_URL.format(commit=LDNOOBW_COMMIT, lang=lang), p)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == LDNOOBW_SHA256[lang], f"{lang} list checksum changed: {digest}"
        paths[lang] = p
    return paths


@dataclass
class Entry:
    id: int
    text: str      # normalised, may contain spaces (phrases)
    lang: str      # 'en' or 'hi' (which list it came from)

    @property
    def tokens(self):
        return self.text.split()


def clean_entry(s):
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return re.sub(r"\s+", " ", s)


def load_entries(paths):
    entries, seen = [], set()
    for lang, p in paths.items():
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            t = clean_entry(line)
            if len(re.findall(r"[a-z]", t)) < 2 or t in seen:   # drop emoji / 1-letter entries / duplicates
                continue
            seen.add(t)
            entries.append(Entry(len(entries), t, lang))
    return entries


def mask(text):
    """Printable form of a listed word: first letter + stars. Used for every printout and the report."""
    return " ".join(w[0] + "*" * (len(w) - 1) for w in str(text).split())


def scrub(text, entries, repl="[term]"):
    """Remove every listed word (whole-word, any case) from free text, e.g. an LLM's reason."""
    for e in entries:
        if len(e.text) >= 3:
            text = re.sub(r"\b" + re.escape(e.text) + r"\b", repl, text, flags=re.I)
    return text


def norm_token(t):
    t = unicodedata.normalize("NFKC", str(t)).lower().replace("’", "'")
    t = re.sub(r"^[^a-z0-9*ऀ-ॿ]+|[^a-z0-9*ऀ-ॿ]+$", "", t)
    return t.replace("'", "")


def deva_to_roman(word):
    """Devanagari word -> loose Latin spelling (ITRANS, lower-cased, final schwa dropped)."""
    from indic_transliteration import sanscript
    r = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
    for nasal in (".N", ".n", "~N", "~n", "M"):
        r = r.replace(nasal, "n")
    r = re.sub(r"[^a-z]", "", r.lower())
    last = [c for c in word if DEVANAGARI.match(c)]
    if last and ("क" <= last[-1] <= "ह" or "क़" <= last[-1] <= "य़") and r.endswith("a"):
        r = r[:-1]          # Hindi drops the inherent final 'a' (कुत्ता keeps its matra, साल -> sal)
    return r


def skeleton(w):
    """Spelling-insensitive key for Hinglish: kh/k, ph/f, w/v, z/j, aa/a, ee/i, double letters, ..."""
    w = re.sub(r"[^a-z]", "", w.lower())
    w = w.replace("chh", "ch").replace("ch", "\x00").replace("ck", "k").replace("c", "k").replace("\x00", "c")
    for a, b in (("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"),
                 ("sh", "s"), ("jh", "j"), ("w", "v"), ("z", "j"), ("q", "k"),
                 ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u")):
        w = w.replace(a, b)
    w = re.sub(r"ey$", "e", w)
    w = re.sub(r"y$", "i", w)
    w = re.sub(r"(.)\1+", r"\1", w)
    return re.sub(r"(?<=[^aeiou])h$", "", w)


@dataclass
class Hit:
    i0: int            # first word index (inclusive)
    i1: int            # last word index (exclusive)
    start: float
    end: float
    heard: str         # normalised ASR/OCR text that matched (never printed unmasked)
    entry_id: int
    lang: str
    tier: str
    needs_context: bool
    match: str         # exact | phrase | translit | skeleton | fuzzy | masked
    score: float

    def to_dict(self):
        return asdict(self)


class Matcher:
    def __init__(self, entries, tiers, fuzzy_threshold=88, common_zipf=3.5, min_fuzzy_len=4):
        self.entries, self.tiers = entries, tiers
        self.fuzzy_threshold, self.common_zipf, self.min_fuzzy_len = fuzzy_threshold, common_zipf, min_fuzzy_len
        self.exact, self.phrases, self.skel = {}, {}, {}
        for e in entries:
            self.exact.setdefault(e.text.replace(" ", ""), e)
            if len(e.tokens) > 1:
                self.phrases.setdefault(tuple(e.tokens), e)
            sk = skeleton(e.text)
            if len(sk) >= 3:
                self.skel.setdefault(sk, e)
        self.fuzzy_keys = [k for k in self.exact if len(k) >= min_fuzzy_len]

    def info(self, e):
        t = self.tiers.get(e.id, {"tier": "strong", "needs_context": True})
        return t["tier"], bool(t["needs_context"])

    def match_token(self, tok):
        """Return (entry, match_type, score) or None for one normalised token."""
        if not tok:
            return None
        if "*" in tok:                                   # e.g. an ASR / caption that masks letters itself
            pat = re.compile("^" + re.escape(tok).replace(r"\*", "[a-z]") + "$")
            cands = [e for k, e in self.exact.items() if pat.match(k)]
            if cands and re.search(r"[a-z]", tok):
                e = max(cands, key=lambda e: TIER_RANK[self.info(e)[0]])
                return e, "masked", 100.0
            return None
        translit = bool(DEVANAGARI.search(tok))
        if translit:
            tok = deva_to_roman(tok)
            if not tok:
                return None
        if tok in self.exact:
            return self.exact[tok], ("translit" if translit else "exact"), 100.0
        if not translit and zipf_frequency(tok, "en") >= self.common_zipf:
            return None                                  # common English word: only exact matches allowed
        sk = skeleton(tok)
        if len(sk) >= 3 and sk in self.skel:
            return self.skel[sk], ("translit" if translit else "skeleton"), 100.0
        if self.fuzzy_threshold <= 100 and len(tok) >= self.min_fuzzy_len and self.fuzzy_keys:   # >100 = fuzzy off
            best = process.extractOne(tok, self.fuzzy_keys, scorer=fuzz.ratio, score_cutoff=self.fuzzy_threshold)
            if best:
                return self.exact[best[0]], ("translit" if translit else "fuzzy"), float(best[1])
        return None

    def find(self, words):
        """words: list of dicts with 'word', 'start', 'end' (Whisper output). Returns non-overlapping Hits."""
        toks = [norm_token(w["word"]) for w in words]
        used, hits = set(), []

        def add(i0, i1, e, kind, score):
            tier, ctx = self.info(e)
            hits.append(Hit(i0, i1, float(words[i0]["start"]), float(words[i1 - 1]["end"]),
                            " ".join(toks[i0:i1]), e.id, e.lang, tier, ctx, kind, score))
            used.update(range(i0, i1))

        for n in (4, 3, 2):                              # phrases and words split by the ASR
            for i in range(len(toks) - n + 1):
                span = toks[i:i + n]
                if used & set(range(i, i + n)) or not all(span):
                    continue
                if tuple(span) in self.phrases:
                    add(i, i + n, self.phrases[tuple(span)], "phrase", 100.0)
                elif "".join(span) in self.exact:
                    add(i, i + n, self.exact["".join(span)], "phrase", 100.0)
        for i, tok in enumerate(toks):
            if i not in used:
                m = self.match_token(tok)
                if m:
                    add(i, i + 1, *m)
        return sorted(hits, key=lambda h: h.start)
