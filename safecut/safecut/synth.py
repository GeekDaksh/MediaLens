"""Synthetic benchmark builder: TTS a listed word, splice it into a clean clip at a pause,
freeze the video for the same time, and (optionally) draw text on frames. Ground truth is exact."""
import hashlib, subprocess
from pathlib import Path
import cv2
import numpy as np
from .audio import SR, EnergyCurve, load_audio

EN_VOICES = ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural",
             "en-IN-PrabhatNeural", "en-IN-NeerjaNeural"]
HINGLISH_VOICES = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural"]   # romanised Hindi said with an Indian accent
HI_VOICES = ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"]           # Devanagari sentences


def tts(text, voice, cache_dir):
    """edge-tts first, gTTS as fallback. Returns (16 kHz mono float32, engine name). Cached on disk."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]
    mp3, tag = cache_dir / f"{key}.mp3", cache_dir / f"{key}.engine"
    if not (mp3.exists() and tag.exists()):
        try:
            import edge_tts
            edge_tts.Communicate(text, voice).save_sync(str(mp3))
            engine = "edge-tts:" + voice
        except Exception:
            from gtts import gTTS
            lang = "hi" if voice.startswith("hi-") else "en"
            tld = {"en-GB": "co.uk", "en-IN": "co.in"}.get(voice[:5], "com")
            gTTS(text, lang=lang, tld=tld).save(str(mp3))
            engine = f"gtts:{lang}-{tld}"
        tag.write_text(engine)
    return load_audio(mp3), tag.read_text()


def trim_silence(y, rel_db=35):
    """Keep the part of a TTS clip that is within rel_db of its loudest 10 ms frame (= ground-truth word)."""
    n = SR // 100
    fr = y[: len(y) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-8)
    on = np.where(db > db.max() - rel_db)[0]
    return y[on[0] * n:(on[-1] + 1) * n].copy()


def room_tone(y, n, rng):
    """n samples of the clip's own quietest background (so pauses are not digital silence)."""
    w = SR // 10
    if len(y) < 2 * w:
        return np.zeros(n, np.float32)
    starts = np.arange(0, len(y) - w, w // 2)
    s = starts[np.argmin([np.sqrt((y[i:i + w] ** 2).mean()) for i in starts])]
    piece = y[s:s + w]
    reps = np.concatenate([np.roll(piece, int(rng.integers(w))) for _ in range(n // w + 1)])
    return reps[:n].astype(np.float32)


def speech_rms(y):
    n = SR // 50
    fr = y[: len(y) // n * n].reshape(-1, n)
    r = np.sqrt((fr ** 2).mean(1))
    return float(r[r >= np.median(r)].mean()) if len(r) else 0.05


def splice(y, word, fps, rng):
    """Insert `word` at a low-energy point between 15% and 85% of the clip, aligned to a video frame.
    Returns new audio, ground-truth (start, end) in seconds, the freeze frame index and number of frames."""
    word = word * (speech_rms(y) / (speech_rms(word) + 1e-8)) * 10 ** (rng.uniform(-3, 3) / 20)
    c = EnergyCurve(y)
    dur = len(y) / SR
    t = c.t[(c.t > 0.15 * dur) & (c.t < 0.85 * dur)]
    d = c.db[(c.t > 0.15 * dur) & (c.t < 0.85 * dur)]
    quiet = t[d < c.floor + 0.25 * (c.peak - c.floor)]
    t_ins = float(rng.choice(quiet)) if len(quiet) else float(t[np.argmin(d)])
    f = int(round(t_ins * fps))
    p = int(round(f / fps * SR))
    g1 = int(rng.uniform(0.04, 0.20) * SR)
    n_freeze = int(np.ceil((g1 + len(word) + 0.04 * SR) / SR * fps))
    total = int(round(n_freeze / fps * SR))
    seg = room_tone(y, total, rng)
    seg[g1:g1 + len(word)] += word
    y2 = np.concatenate([y[:p], seg, y[p:]]).astype(np.float32)
    gt = ((p + g1) / SR, (p + g1 + len(word)) / SR)
    return y2, gt, f, n_freeze


def draw_text(frame, text, x, y, scale, thick, color):
    """Caption-style text on a dark banner. Returns the exact text box (x0, y0, x1, y1)."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    over = frame.copy()
    cv2.rectangle(over, (x - 8, y - h - 8), (x + w + 8, y + base + 8), (20, 20, 20), -1)
    cv2.addWeighted(over, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return (x, y - h, x + w, y + base)


def place_text(text, W, H, rng):
    scale = float(rng.uniform(1.0, 2.0)); thick = int(rng.integers(2, 4))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    x = int(rng.integers(20, max(21, W - w - 20)))
    y = int(rng.integers(h + 20, max(h + 21, H - base - 20)))
    color = [(255, 255, 255), (0, 255, 255), (255, 255, 0)][int(rng.integers(3))]
    return dict(x=x, y=y, scale=scale, thick=thick, color=color)


def build_video(src, out_noaudio, freeze_f, n_freeze, overlays):
    """Copy src frames, repeat frame `freeze_f` n_freeze times, draw overlays (times in the OUTPUT timeline).
    Returns fps, (W, H), n_frames, per-frame list of ground-truth boxes for offensive overlays."""
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {src}")
    H, W = frames[0].shape[:2]
    if freeze_f is not None and n_freeze:
        f = min(freeze_f, len(frames) - 1)
        frames = frames[:f] + [frames[f]] * n_freeze + frames[f:]
    vw = cv2.VideoWriter(str(out_noaudio), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gt_boxes = [[] for _ in frames]
    for k, fr in enumerate(frames):
        fr = fr.copy()
        t = k / fps
        for ov in overlays:
            if ov["t0"] <= t < ov["t1"]:
                box = draw_text(fr, ov["text"], ov["x"], ov["y"], ov["scale"], ov["thick"], ov["color"])
                ov["box"] = box
                if ov["offensive"]:
                    gt_boxes[k].append(box)
        vw.write(fr)
    vw.release()
    return fps, (W, H), len(frames), gt_boxes


def mux(video_noaudio, wav, out_mp4):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video_noaudio), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out_mp4)], check=True)
