import json, os, sys, tempfile, torch
import pandas as pd, streamlit as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safecut import lexicon as lx, llm, rag, pipeline as pl
PROJ = os.environ.get("SAFECUT_PROJ", "/kaggle/working/safecut_run")

@st.cache_resource
def load_ctx():
    import easyocr
    from faster_whisper import WhisperModel
    cfg = json.load(open(f"{PROJ}/run_cfg.json"))
    entries = lx.load_entries(lx.download_lists(f"{PROJ}/lists"))
    tiers = {int(k): v for k, v in json.load(open(f"{PROJ}/tiers.json")).items()}
    policy = next(p for p in [os.environ.get("SAFECUT_POLICY", ""), f"{PROJ}/content_policy.md"] if p and os.path.exists(p))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dict(cfg=cfg, entries=entries, matcher=lx.Matcher(entries, tiers, cfg["fuzzy_threshold"], cfg["common_zipf"]),
                judge=llm.Judge(f"{PROJ}/cache/llm_cache.json", api_key=os.environ.get("GEMINI_API_KEY"), local_device=dev),
                index=rag.PolicyIndex(rag.load_policy(policy), device=dev),
                whisper=WhisperModel(os.environ.get("SAFECUT_WHISPER", "large-v3"), device=dev, compute_type="float16" if dev == "cuda" else "int8"),
                reader=easyocr.Reader(["en"], gpu=dev == "cuda", verbose=False), rules={r["rule_id"]: r for r in rag.load_policy(policy)})

st.set_page_config(page_title="SafeCut", layout="wide")
st.title("SafeCut — context-aware censoring for sports highlights")
up = st.file_uploader("Upload a short clip (mp4)", type=["mp4", "mov", "mkv"])
if up:
    ctx = load_ctx()
    work = tempfile.mkdtemp(); src = os.path.join(work, "upload.mp4"); open(src, "wb").write(up.read())
    with st.spinner("ASR → detection → context check → OCR blur → render ..."):
        report, out = pl.process_video(src, ctx, work)
    a, b = st.columns(2)
    a.subheader("Before"); a.video(src)
    b.subheader("After"); b.video(str(out))
    ev = pd.DataFrame(report["events"])
    st.subheader("Report (tier and action only — the word is never shown)")
    st.dataframe(ev.drop(columns=["heard_masked"], errors="ignore") if len(ev) else pd.DataFrame([{"result": "nothing censored"}]))
    st.json(report["timings_s"])
    st.subheader("Rules cited")
    for rid in sorted(set(ev["rule_id"])) if len(ev) else []:
        r = ctx["rules"].get(rid)
        if r:
            st.markdown(f"**{rid} — {r['title']}**: {r['text']}")
