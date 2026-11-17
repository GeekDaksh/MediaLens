import os, tempfile
import streamlit as st
import pipeline as P

st.set_page_config(page_title='Sarcasm Cascade', page_icon='🎭', layout='centered')
st.title('🎭 Cost-aware sarcasm detection')
st.caption('Stage 1: text → Stage 2: text + audio + video → Stage 3: Gemini + retrieved examples. '
           'Each stage runs only if the previous one is unsure.')

if not all(os.path.exists(os.path.join(P.HERE, f)) for f in ('app_bundle.npz', 'app_meta.json')):
    st.error('Missing app_bundle.npz / app_meta.json. Download them from Kaggle '
             '(/kaggle/working/app_export/) into the app/ folder, then refresh this page.')
    st.stop()


@st.cache_resource(show_spinner='Loading models (first run downloads ~2 GB)…')
def get_models():
    return P.Models()


@st.cache_resource
def get_bundle():
    return P.load_bundle()


B, M = get_bundle()

with st.sidebar:
    st.header('Settings')
    mode = st.radio('Thresholds', ['cost-saver', 'best-F1', 'custom'],
                    help='cost-saver and best-F1 use the thresholds chosen in the notebook (median over folds).')
    d = M['thresholds'].get(mode, M['thresholds']['best-F1'])
    t1 = st.slider('Stage 1 → 2 if |p − 0.5| <', 0.0, 0.51, d['t1'], 0.005, disabled=mode != 'custom')
    t2 = st.slider('Stage 2 → 3 if |p − 0.5| <', 0.0, 0.5, d['t2'], 0.005, disabled=mode != 'custom')
    api_key = st.text_input('Gemini API key', value=os.environ.get('GEMINI_API_KEY', ''), type='password')
    gemini_model = st.text_input('Gemini model', 'gemini-3.5-flash')
    always_explain = st.checkbox('Always ask Gemini for an explanation', value=True,
                                 help='Shows an explanation even when Stage 1 or 2 decided (does not change the label).')

utt_file = st.file_uploader('Utterance clip (.mp4)', type=['mp4'])
ctx_file = st.file_uploader('Context clip (.mp4, optional — the lines before the utterance)', type=['mp4'])
local_path = st.text_input('…or path to a local .mp4', '', key='local_path')


def save(f):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.write(f.getbuffer()); tmp.close()
    return tmp.name


utt_path = save(utt_file) if utt_file else (local_path.strip() or None)
ctx_path = save(ctx_file) if ctx_file else None

if utt_path:
    if not os.path.exists(utt_path):
        st.error(f'File not found: {utt_path}'); st.stop()
    st.video(utt_path)
    if st.button('Detect sarcasm', type='primary'):
        with st.spinner('Running the cascade…'):
            R = P.run_cascade(get_models(), B, M, utt_path, ctx_path, t1, t2, api_key, gemini_model, always_explain)

        color = 'red' if R['label'] == 'SARCASTIC' else 'green'
        st.markdown(f"## :{color}[{R['label']}]")
        st.markdown(f"**Decided by Stage {R['decided_by']}** · total {R['total_sec']:.2f} s on this machine")
        if R.get('note'):
            st.warning(R['note'])

        st.subheader('Transcript (Whisper)')
        st.write(R['transcript'] or '_(no speech detected)_')
        if ctx_path:
            st.caption(f"Context: {R['context'] or '(no speech detected)'}")
        else:
            st.caption('No context clip given; the model was trained with context, so accuracy may be lower.')

        st.subheader('Stages')
        st.table([{'Stage': s['stage'], 'Model': s['model'],
                   'P(sarcastic)': '—' if s['p_sarcastic'] is None else f"{s['p_sarcastic']:.3f}",
                   'Time (s)': f"{s['sec']:.2f}"} for s in R['stages']])

        st.subheader('Gemini explanation')
        if R['gemini']:
            g = R['gemini']
            st.write(f"**{g['label']}** (confidence {g.get('confidence')}, main cue: {g.get('main_cue')})")
            st.write(g.get('explanation', ''))
            st.caption('Retrieved examples: ' + ', '.join(g['retrieved']))
        elif R['gemini_error']:
            st.info(f"Gemini not available: {R['gemini_error']}")
        else:
            st.caption('Not requested.')
