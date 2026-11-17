"""One JSON-in/JSON-out LLM judge: Gemini Flash models in order, then a local open model as fallback.
Every answer is cached on disk, so re-running the notebook never spends API quota twice."""
import hashlib, json, re, time
from collections import Counter
from pathlib import Path

GEMINI_CANDIDATES = ["models/gemini-flash-latest", "models/gemini-3.5-flash", "models/gemini-2.5-flash",
                     "models/gemini-flash-lite-latest", "models/gemini-2.5-flash-lite"]
SAFETY_CATEGORIES = ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                     "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]


def extract_json(text):
    if text is None:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for o, c in (("[", "]"), ("{", "}")):
        i, j = text.find(o), text.rfind(c)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    return None


class Judge:
    def __init__(self, cache_file, api_key=None, candidates=GEMINI_CANDIDATES,
                 local_model="Qwen/Qwen2.5-3B-Instruct", local_device="cuda", log=print):
        self.cache_file = Path(cache_file)
        self.cache = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        self.local_model_name, self.local_device, self.log = local_model, local_device, log
        self.usage, self.dead, self._local = Counter(), set(), None
        self.client, self.models = None, []
        if api_key:
            from google import genai
            from google.genai import types
            self.types = types
            self.client = genai.Client(api_key=api_key)
            try:
                listed = [(m.name, list(getattr(m, "supported_actions", None) or [])) for m in self.client.models.list()]
                by_short = {n.split("/")[-1]: n for n, _ in listed}
                self.models = [by_short[c.split("/")[-1]] for c in candidates if c.split("/")[-1] in by_short]
                if not self.models:      # names changed: use any text Flash model this key can call
                    self.models = [n for n, acts in listed if "flash" in n and (not acts or "generateContent" in acts)
                                   and not any(x in n for x in ("tts", "image", "live", "audio", "embed"))]
                self.log(f"Gemini key accepted: {len(listed)} models listed, {len(self.models)} usable Flash models")
            except Exception as e:
                self.log(f"model list failed ({str(e)[:120]}); trying candidates blindly")
                self.models = list(candidates)

    def _gemini_config(self):
        t = self.types
        return t.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json",
            safety_settings=[t.SafetySetting(category=c, threshold="BLOCK_NONE") for c in SAFETY_CATEGORIES])

    def _ask_gemini(self, prompt, validate):
        for m in self.models:
            if m in self.dead:
                continue
            for attempt in range(3):
                try:
                    resp = self.client.models.generate_content(model=m, contents=prompt, config=self._gemini_config())
                    obj = extract_json(resp.text)
                    if obj is not None and validate(obj):
                        return obj, m
                    self.log(f"  {m}: invalid/blocked answer (attempt {attempt + 1})")
                except Exception as e:
                    s = str(e)
                    delay = re.search(r"retry(?:Delay| in)[^0-9]*([0-9.]+)\s*s", s)
                    if ("429" in s or "RESOURCE_EXHAUSTED" in s) and delay and float(delay.group(1)) <= 65 and attempt < 2:
                        self.log(f"  {m}: per-minute limit, waiting {float(delay.group(1)):.0f}s")
                        time.sleep(float(delay.group(1)) + 1)
                        continue
                    if "503" in s or "UNAVAILABLE" in s or "500" in s:
                        self.log(f"  {m}: 503/500, retry in {5 * (attempt + 1)}s")
                        time.sleep(5 * (attempt + 1))
                        continue
                    self.log(f"  {m}: {s[:90]} -> next model")
                    self.dead.add(m)
                    break
        return None, None

    def _ask_local(self, prompt, validate):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self._local is None:
            self.log(f"loading local fallback {self.local_model_name} ...")
            tok = AutoTokenizer.from_pretrained(self.local_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.local_model_name, dtype=torch.float16 if "cuda" in self.local_device else torch.float32
            ).to(self.local_device).eval()
            self._local = (tok, model)
        tok, model = self._local
        msgs = [{"role": "system", "content": "You are a careful broadcast standards editor. Reply with JSON only."},
                {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)   # plain string in every version
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=2048, do_sample=False, pad_token_id=tok.eos_token_id)
        obj = extract_json(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return (obj, "local:" + self.local_model_name) if obj is not None and validate(obj) else (None, None)

    def ask(self, prompt, validate=lambda o: True):
        key = hashlib.sha1(prompt.encode()).hexdigest()
        if key in self.cache:
            self.usage["cache"] += 1
            return self.cache[key]["obj"], self.cache[key]["source"]
        obj, src = self._ask_gemini(prompt, validate) if self.client else (None, None)
        if obj is None and self.local_model_name:
            obj, src = self._ask_local(prompt, validate)
        if obj is None:
            raise RuntimeError("No LLM gave a valid answer for this prompt")
        self.usage[src] += 1
        self.cache[key] = {"obj": obj, "source": src}
        self.cache_file.write_text(json.dumps(self.cache))
        return obj, src


TIER_PROMPT = """You are building a broadcast-standards lexicon for live sports highlights in English and Hinglish
(romanised Hindi). For every numbered entry below decide:
- "tier": "mild" (mild swearing, crude or non-graphic sexual words, clinical sexual vocabulary),
          "strong" (strong swearing, graphic sexual terms, strong personal insults), or
          "slur" (slur or hate term aimed at a protected group: race, ethnicity, nationality, religion, caste,
                  gender, sexuality, disability).
- "needs_context": true if the entry also has a common innocent meaning (a name, place, animal, object,
  sports term, medical or legal word, or an ordinary word in another language); otherwise false.
Return ONLY a JSON list with one object per entry: [{{"i": <number>, "tier": "...", "needs_context": true/false}}, ...]

Entries (list language in brackets):
{items}"""


def tier_entries(judge, entries, batch=60, log=print):
    tiers, sources = {}, Counter()
    for b in range(0, len(entries), batch):
        chunk = entries[b:b + batch]
        items = "\n".join(f"{k}. [{e.lang}] {e.text}" for k, e in enumerate(chunk))
        want = set(range(len(chunk)))

        def ok(o):
            return (isinstance(o, list) and {d.get("i") for d in o if isinstance(d, dict)} >= want
                    and all(d.get("tier") in ("mild", "strong", "slur") for d in o if d.get("i") in want))
        obj, src = judge.ask(TIER_PROMPT.format(items=items), ok)
        sources[src] += 1
        for d in obj:
            if d.get("i") in want:
                tiers[chunk[d["i"]].id] = {"tier": d["tier"], "needs_context": bool(d.get("needs_context"))}
        log(f"  tiered {min(b + batch, len(entries))}/{len(entries)} via {src}")
    return tiers, sources
