"""Content policy -> FAISS index -> top-k rules -> LLM context check for ambiguous hits."""
import re
import numpy as np


def load_policy(path):
    text = open(path, encoding="utf-8").read()
    rules = []
    for m in re.finditer(r"^## (R\d+)\s*[—-]+\s*(.+?)\n(.*?)(?=^## R\d+|\Z)", text, flags=re.M | re.S):
        rules.append({"rule_id": m.group(1), "title": m.group(2).strip(), "text": " ".join(m.group(3).split())})
    return rules


class PolicyIndex:
    def __init__(self, rules, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.rules = rules
        self.model = SentenceTransformer(model_name, device=device)
        emb = self.model.encode([f"{r['title']}. {r['text']}" for r in rules], normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.asarray(emb, dtype="float32"))

    def search(self, queries, k=3):
        q = self.model.encode(list(queries), normalize_embeddings=True)
        scores, idx = self.index.search(np.asarray(q, dtype="float32"), k)
        return [[dict(self.rules[j], score=float(s)) for s, j in zip(sr, ir)] for sr, ir in zip(scores, idx)]


def window_text(words, i0, i1, span=10):
    lo, hi = max(0, i0 - span), min(len(words), i1 + span)
    parts = [w["word"].strip() for w in words[lo:i0]] + ["⟦" + " ".join(w["word"].strip() for w in words[i0:i1]) + "⟧"] \
        + [w["word"].strip() for w in words[i1:hi]]
    return " ".join(parts)


CONTEXT_PROMPT = """You review speech-recognition transcripts from sports highlight clips for a general-audience broadcast.
In each item one span is marked with ⟦ ⟧. It matched the listed term shown (match type tells you how:
exact, phrase, translit = Hindi script converted to Latin, skeleton/fuzzy = approximate spelling, masked = letters hidden).
Decide, using ONLY the policy rules given with that item, whether the marked span is offensive in this context.
Return ONLY a JSON list: [{{"k": <item number>, "offensive": true/false, "severity": "none"|"mild"|"strong"|"slur",
"rule_id": "<one rule id from that item's rules>", "reason": "<max 20 words, do NOT repeat the word itself>"}}, ...]

{items}"""


def context_check(judge, index, items, batch=8, log=print):
    """items: dicts with keys key, window, term, match, tier, lang. Returns {key: decision}."""
    out = {}
    retrieved = index.search([it["window"].replace("⟦", "").replace("⟧", "") for it in items], k=3)
    for b in range(0, len(items), batch):
        chunk, rules = items[b:b + batch], retrieved[b:b + batch]
        blocks = []
        for k, (it, rs) in enumerate(zip(chunk, rules)):
            rtxt = "\n".join(f"   {r['rule_id']} ({r['title']}): {r['text']}" for r in rs)
            blocks.append(f"Item {k}: transcript: \"{it['window']}\"\n   listed term: \"{it['term']}\" | match: {it['match']}"
                          f" | lexicon tier: {it['tier']} | list language: {it['lang']}\n   Rules:\n{rtxt}")
        want = set(range(len(chunk)))

        def ok(o):
            return (isinstance(o, list) and {d.get("k") for d in o if isinstance(d, dict)} >= want
                    and all(isinstance(d.get("offensive"), bool) for d in o if d.get("k") in want))
        obj, src = judge.ask(CONTEXT_PROMPT.format(items="\n\n".join(blocks)), ok)
        for d in obj:
            if d.get("k") not in want:
                continue
            it, rs = chunk[d["k"]], rules[d["k"]]
            sev = d.get("severity") if d.get("severity") in ("none", "mild", "strong", "slur") else it["tier"]
            reason = str(d.get("reason", ""))
            for w in {it["term"], *it.get("heard", "").split()}:
                if len(w) >= 2:
                    reason = re.sub(re.escape(w), "[term]", reason, flags=re.I)
            out[it["key"]] = {"offensive": d["offensive"], "severity": sev if d["offensive"] else "none",
                              "rule_id": str(d.get("rule_id", "")), "reason": reason[:200], "source": src,
                              "retrieved": [r["rule_id"] for r in rs]}
        log(f"  context-checked {min(b + batch, len(items))}/{len(items)} via {src}")
    return out
