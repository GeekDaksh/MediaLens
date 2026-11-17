
import os, re, json, pickle, functools
import numpy as np, pandas as pd, torch, open_clip
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

C = "/kaggle/working/cache"
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MODELS = ["gemini-flash-lite-latest", "gemini-3.5-flash"]

@functools.lru_cache(maxsize=1)
def load():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = {"dev": dev, "meta": pd.read_parquet(f"{C}/meta.parquet")}
    for k in ["text_emb", "clip_emb", "dia_emb", "dia_idx"]:
        R[k] = np.load(f"{C}/{k}.npy")
    R["kw_clip"], R["Kc"], R["kw_dia"], R["Kd"] = pickle.load(open(f"{C}/kw.pkl", "rb"))
    R["cfg"] = json.load(open(f"{C}/best_cfg.json"))
    R["tm"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    R["cm"], _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
    R["cm"].eval()
    R["ctok"] = open_clip.get_tokenizer("ViT-B-32")
    R["client"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    R["empty"] = (R["meta"]["whisper"].fillna("").str.strip() == "").values
    R["speakers"] = sorted(R["meta"]["Speaker"].unique(), key=len, reverse=True)
    return R

def gen(R, prompt, json_mode=False):
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json" if json_mode else None)
    last = None
    for m in MODELS:
        try:
            return R["client"].models.generate_content(model=m, contents=prompt, config=cfg).text, m
        except Exception as e:
            last = e
    raise last

PARSE = """You parse a search query about clips from the TV show Friends. Return ONE JSON object with keys:
"speaker": the speaking character's name exactly as written in the query, or null;
"emotion": one of anger, disgust, fear, joy, neutral, sadness, surprise if implied (mad/furious->anger, crying/sad/in tears->sadness,
happy/excited->joy, shocked->surprise, scared/terrified->fear, grossed out->disgust), else null;
"topic": the remaining subject words, or "" if none.
Query: """

def parse(R, q):
    out, used = {"speaker": None, "emotion": None, "topic": q}, "none (Gemini unavailable)"
    try:
        txt, used = gen(R, PARSE + json.dumps(q), json_mode=True)
        o = json.loads(txt); o = o[0] if isinstance(o, list) else o
        em = (o.get("emotion") or "").lower()
        out = {"speaker": o.get("speaker"), "emotion": em if em in LABELS else None, "topic": (o.get("topic") or "").strip()}
    except Exception:
        pass
    g = (out["speaker"] or "").strip()
    cast = {s.lower(): s for s in R["speakers"]}
    spk = cast.get(g.lower())
    if g and not spk:
        firsts = [s for s in R["speakers"] if s.lower().split()[0] == g.lower().split()[0]]
        spk = firsts[0] if len(firsts) == 1 else None
    if g:
        for s in R["speakers"]:
            if s.startswith(g.split()[0]) and re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", q):
                spk = s
                break
    out["speaker"] = spk
    return out, used

def Z(v):
    return (v - v.mean()) / (v.std() + 1e-8)

@torch.no_grad()
def search(R, q, p, k=5):
    meta, cfg, topic = R["meta"], R["cfg"], p["topic"] or q
    t = R["tm"].encode([topic], normalize_embeddings=True)[0]
    s = cfg["dense"] * np.where(R["empty"], 0, Z(R["text_emb"] @ t))
    s += cfg["kw"] * np.where(R["empty"], 0, Z((R["kw_clip"].transform([topic]) @ R["Kc"].T).toarray()[0]))
    s += cfg["scene"] * Z(R["dia_emb"] @ t)[R["dia_idx"]]
    s += cfg["kw_scene"] * Z((R["kw_dia"].transform([topic]) @ R["Kd"].T).toarray()[0])[R["dia_idx"]]
    ce = R["cm"].encode_text(R["ctok"]([q]).to(R["dev"]))
    s += cfg["clip"] * Z((ce / ce.norm(dim=-1, keepdim=True)).float().cpu().numpy()[0] @ R["clip_emb"].T)
    if p["speaker"]:
        s = np.where(meta["Speaker"].values == p["speaker"], s, -np.inf)
    if p["emotion"]:
        s = s + cfg["emo"] * (meta["pred_emotion"].values == p["emotion"])   # PREDICTED emotion only
    top = [i for i in np.argsort(-s)[:k] if np.isfinite(s[i])]
    return meta.iloc[top].assign(score=s[top].round(3))

def norm(t):
    t = str(t).lower().replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def build_ctx(hits):
    return "\n".join(
        f"[{r.clip_id}] S{r.Season}E{r.Episode} start {str(r.StartTime).split(',')[0]} | {r.Speaker} | "
        f"predicted emotion: {r.pred_emotion} | said: \"{r.whisper or '(no speech detected)'}\""
        for r in hits.itertuples())

ANSWER = """You answer questions about clips from the TV show Friends, using ONLY the clips listed below.
Question: {q}
Clips:
{ctx}

Rules:
- Write 2-4 short sentences.
- After every claim, cite the clip as [clip_id, S<season>E<episode> <start time>], copying the id, season, episode and start time exactly as listed, e.g. [dia12_utt3, S3E19 00:14:38].
- Only quote words that appear in a clip's "said" text.
- If none of the clips really match the question, say so honestly.
- Never mention any clip, person, event or time that is not listed above, even if the question asks you to."""

def answer(R, q, hits):
    try:
        return gen(R, ANSWER.format(q=q, ctx=build_ctx(hits)))
    except Exception as e:
        return f"(Gemini unavailable: {str(e)[:80]})", "none"

def to_sec(t):
    h, m, s = [int(x) for x in str(t).split(",")[0].split(":")]
    return h * 3600 + m * 60 + s

CITE = re.compile(r"\[(dia\d+_utt\d+)(?:,\s*S(\d+)\s*E(\d+)\s*,?\s*([\d:,]+))?[^\]]*\]")

def check_answer(text, hits):
    """Hallucination checks: cited clip ids, season/episode, timestamps and quotes must all come from the results."""
    ids, by, problems = set(hits["clip_id"]), hits.set_index("clip_id"), []
    mentioned = set(re.findall(r"dia\d+_utt\d+", text))
    problems += [f"cites clip not in results: {c}" for c in sorted(mentioned - ids)]
    for cid, S, E, t in CITE.findall(text):
        if cid not in ids:
            continue
        r = by.loc[cid]
        if S and (int(S) != int(r.Season) or int(E) != int(r.Episode)):
            problems.append(f"wrong season/episode for {cid}: S{S}E{E}")
        if t:
            try:
                if min(abs(to_sec(t) - to_sec(r.StartTime)), abs(to_sec(t) - to_sec(r.EndTime))) > 1:
                    problems.append(f"wrong timestamp for {cid}: {t}")
            except Exception:
                problems.append(f"unparseable timestamp for {cid}: {t}")
    corpus = " | ".join(norm(w) for w in hits["whisper"].fillna(""))
    for qt in re.findall(r"[\"\u201c]([^\"\u201d]{8,})[\"\u201d]", text):
        if norm(qt) and norm(qt) not in corpus:
            problems.append(f"quote not in transcripts: \"{qt[:60]}\"")
    return {"cited": sorted(mentioned), "n_cites": len(mentioned), "problems": problems}
