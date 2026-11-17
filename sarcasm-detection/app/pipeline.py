"""Cascade inference for one clip. Feature extraction mirrors the Kaggle notebook exactly
(Whisper-small, RoBERTa mean-pool, wav2vec2 mean-pool, CLIP ViT-B/32 quickgelu on 3 frames)."""
import json, os, re, subprocess, time
import numpy as np, torch, av, imageio_ffmpeg, open_clip
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, Wav2Vec2Model

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cpu'


class Models:
    def __init__(self):
        self.whisper = WhisperModel('small', device='cpu', compute_type='int8')
        self.tok = AutoTokenizer.from_pretrained('roberta-base')
        self.rob = AutoModel.from_pretrained('roberta-base').to(DEV).eval()
        self.fe = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base')
        self.w2v = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').to(DEV).eval()
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
        self.clip = self.clip.to(DEV).eval()


def load_bundle(folder=HERE):
    B = dict(np.load(os.path.join(folder, 'app_bundle.npz')))
    M = json.load(open(os.path.join(folder, 'app_meta.json')))
    return B, M


# ---------- features (same settings as the notebook) ----------
def load_audio(path):
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-i', path, '-f', 's16le',
                          '-ac', '1', '-ar', '16000', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def transcribe(m, audio):
    if audio.size == 0:
        return ''
    segs, _ = m.whisper.transcribe(audio, language='en', beam_size=5, condition_on_previous_text=False)
    return ' '.join(s.text.strip() for s in segs).strip()


@torch.no_grad()
def embed_text(m, text):
    enc = m.tok(text or ' ', return_tensors='pt', truncation=True, max_length=256).to(DEV)
    h = m.rob(**enc).last_hidden_state
    mask = enc['attention_mask'].unsqueeze(-1).float()
    return ((h * mask).sum(1) / mask.sum(1)).squeeze(0).numpy()


@torch.no_grad()
def embed_audio(m, audio):
    inp = m.fe(audio, sampling_rate=16000, return_tensors='pt').input_values.to(DEV)
    return m.w2v(inp).last_hidden_state.mean(1).squeeze(0).numpy()


def three_frames(path):
    with av.open(path) as c:
        frames = [f.to_image() for f in c.decode(video=0)]
    if not frames:
        raise ValueError('no video frames found in the clip')
    return [frames[int(len(frames) * f)] for f in (0.25, 0.5, 0.75)]


@torch.no_grad()
def embed_video(m, path):
    batch = torch.stack([m.preprocess(f) for f in three_frames(path)]).to(DEV)
    e = m.clip.encode_image(batch)
    e = e / e.norm(dim=-1, keepdim=True)
    return e.mean(0).numpy()


def lr_prob(B, name, x):
    z = (x - B[f'{name}_mean']) / B[f'{name}_scale'] @ B[f'{name}_coef'] + B[f'{name}_b'][0]
    return float(1 / (1 + np.exp(-z)))


# ---------- Stage 3: Gemini + RAG (same prompt as the notebook) ----------
def short(s, n=300):
    s = str(s)
    return s if len(s) <= n else s[:n] + '…'


def build_prompt(B, M, q_text_emb, utt, ctx, with_video):
    q = q_text_emb / np.linalg.norm(q_text_emb)
    nn = np.argsort(-(B['Xn'] @ q))[:5]
    ex = [f"Example {j}\nContext: {short(M['ctx_asr_text'][r])}\nUtterance: {M['asr_text'][r]}\n"
          f"Label: {'SARCASTIC' if B['y'][r] else 'NOT SARCASTIC'}" for j, r in enumerate(nn, 1)]
    video_line = ("The attached video is the target utterance (use the speaker's tone of voice and facial expression too)."
                  if with_video else "No video is available; judge from the text.")
    return ("You detect sarcasm in TV sitcom dialogue. Transcripts are automatic and may contain errors.\n\n"
            "Here are 5 similar labelled examples:\n\n" + "\n\n".join(ex) +
            f"\n\nNow classify the TARGET clip. {video_line}\n"
            f"Context: {short(ctx, 600)}\nUtterance: {utt}\n\n"
            'Reply with JSON only: {"label": "sarcastic" or "not_sarcastic", "confidence": number 0-1, '
            '"main_cue": "words" or "tone" or "face", "explanation": one short sentence}'), nn


def ask_gemini(api_key, model, prompt, video_path=None):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)))
    parts = [prompt]
    if video_path:
        parts.insert(0, types.Part.from_bytes(data=open(video_path, 'rb').read(), mime_type='video/mp4'))
    r = client.models.generate_content(model=model, contents=parts,
                                       config=types.GenerateContentConfig(temperature=0, response_mime_type='application/json'))
    out = json.loads(re.search(r'\{.*\}', r.text, re.S).group(0))
    out['label'] = str(out.get('label', '')).lower().strip()
    if out['label'] not in ('sarcastic', 'not_sarcastic'):
        raise ValueError(f"unexpected label from Gemini: {out['label']!r}")
    return out


# ---------- the cascade ----------
def run_cascade(m, B, M, utt_path, ctx_path, t1, t2, api_key='', gemini_model='gemini-3.5-flash',
                always_explain=False):
    R = {'stages': [], 'gemini': None, 'gemini_error': None}
    t = time.perf_counter()
    utt = transcribe(m, load_audio(utt_path))
    ctx = transcribe(m, load_audio(ctx_path)) if ctx_path else ''
    x_text = np.concatenate([embed_text(m, utt), embed_text(m, ctx)])
    p1 = lr_prob(B, 'text', x_text)
    R.update(transcript=utt, context=ctx, p_text=p1)
    R['stages'].append({'stage': 1, 'model': 'Text (Whisper + RoBERTa)', 'p_sarcastic': p1,
                        'sec': time.perf_counter() - t})
    label, decided = int(p1 >= .5), 1

    if abs(p1 - .5) < t1:
        t = time.perf_counter()
        audio = load_audio(utt_path)
        x_full = np.concatenate([x_text, embed_audio(m, audio), embed_video(m, utt_path)])
        p2 = lr_prob(B, 'full', x_full)
        R['p_full'] = p2
        R['stages'].append({'stage': 2, 'model': 'Text + Audio + Video fusion', 'p_sarcastic': p2,
                            'sec': time.perf_counter() - t})
        label, decided = int(p2 >= .5), 2
        need_llm = abs(p2 - .5) < t2
    else:
        need_llm = False

    if need_llm or always_explain:
        if not api_key:
            R['gemini_error'] = 'No Gemini API key given.'
        else:
            t = time.perf_counter()
            prompt, nn = build_prompt(B, M, x_text, utt, ctx, with_video=True)
            try:
                try:
                    g = ask_gemini(api_key, gemini_model, prompt, utt_path)
                except Exception:   # video refused / busy -> one text-only try
                    prompt, nn = build_prompt(B, M, x_text, utt, ctx, with_video=False)
                    g = ask_gemini(api_key, gemini_model, prompt, None)
                g['retrieved'] = [M['clip_id'][r] for r in nn]
                R['gemini'] = g
                if need_llm:
                    label, decided = int(g['label'] == 'sarcastic'), 3
                    R['stages'].append({'stage': 3, 'model': f'{gemini_model} + RAG', 'p_sarcastic': None,
                                        'sec': time.perf_counter() - t})
            except Exception as e:
                R['gemini_error'] = f'{type(e).__name__}: {str(e)[:200]}'
    if need_llm and decided != 3:
        R['note'] = 'Stage 3 was needed but Gemini was unavailable, so the Stage 2 label is used.'
    R.update(label='SARCASTIC' if label else 'NOT SARCASTIC', decided_by=decided,
             total_sec=sum(s['sec'] for s in R['stages']))
    return R
