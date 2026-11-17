
import streamlit as st, engine
st.set_page_config(page_title="Friends Scene Search", layout="wide")
R = st.cache_resource(engine.load)()

@st.cache_data(show_spinner=False)
def parse(q):
    return engine.parse(R, q)

@st.cache_data(show_spinner=False)
def answer(q, hits):
    return engine.answer(R, q, hits)

st.title("Emotion-aware Friends scene search")
st.caption(f"MELD test split, {len(R['meta']):,} clips | Whisper + MiniLM + CLIP + fine-tuned RoBERTa emotions + Gemini")
q = st.text_input("Describe a scene", "Ross being angry about the couch")
if q:
    with st.spinner("Searching..."):
        p, used = parse(q)
        hits = engine.search(R, q, p)
    st.markdown(f"**Parsed** - speaker: `{p['speaker']}` | emotion: `{p['emotion']}` | topic: `{p['topic']}` (via {used})")
    if hits.empty:
        st.warning("No clips found for that speaker.")
    else:
        for col, r in zip(st.columns(len(hits)), hits.itertuples()):
            with col:
                st.video(r.path)
                st.markdown(f"**{r.clip_id}** - {r.Speaker}  \nS{r.Season}E{r.Episode} {r.StartTime}  \n"
                            f"predicted emotion: *{r.pred_emotion}*  \n\"{r.whisper or '(no speech detected)'}\"")
        with st.spinner("Writing answer..."):
            text, used2 = answer(q, hits)
        st.subheader("Answer")
        st.write(text)
        chk = engine.check_answer(text, hits)
        if chk["problems"]:
            st.error("Hallucination check failed: " + "; ".join(chk["problems"]))
        else:
            st.success(f"Hallucination check passed: {chk['n_cites']} cited clips, all ids/timestamps/quotes match the results (via {used2}).")
