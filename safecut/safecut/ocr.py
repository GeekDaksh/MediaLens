"""On-screen text: OCR every `every` seconds, blur boxes whose text matches the lexicon, hold the blur."""
import cv2
import numpy as np
from .lexicon import norm_token


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


def sample_indices(n_frames, fps, every=0.5):
    return sorted({min(n_frames - 1, int(round(k * every * fps))) for k in range(int(n_frames / fps / every) + 1)})


def ocr_hits(reader, matcher, frame, pad=0.1):
    """Boxes (x0,y0,x1,y1) whose OCR text contains a listed word, plus their tiers."""
    H, W = frame.shape[:2]
    out = []
    for pts, text, conf in reader.readtext(frame):
        toks = [norm_token(t) for t in text.split()]
        cands = toks + ["".join(toks)]
        m = next((r for r in (matcher.match_token(t) for t in cands if t) if r), None)
        if not m:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        dx, dy = pad * (max(xs) - min(xs)), pad * (max(ys) - min(ys))
        box = (max(0, min(xs) - dx), max(0, min(ys) - dy), min(W, max(xs) + dx), min(H, max(ys) + dy))
        out.append({"box": tuple(float(v) for v in box), "entry_id": m[0].id, "tier": matcher.info(m[0])[0],
                    "match": m[1], "conf": float(conf)})
    return out


def active_boxes(sample_idx, sample_boxes, k, mode="forward"):
    """Boxes applied to frame k. forward = hold each sample until the next one (live-safe).
    both = also apply to frames before the sample (possible offline, because we see the future)."""
    j = int(np.searchsorted(sample_idx, k, side="right")) - 1
    boxes = list(sample_boxes[j]) if j >= 0 else []
    if mode == "both" and j + 1 < len(sample_idx) and k != sample_idx[j]:
        boxes += sample_boxes[j + 1]
    return boxes


def blur_video(frames, fps, sample_idx, sample_boxes, out_path, mode="forward", ksize=51):
    H, W = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for k, fr in enumerate(frames):
        fr = fr.copy()
        for x0, y0, x1, y1 in active_boxes(sample_idx, sample_boxes, k, mode):
            x0, y0, x1, y1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
            if x1 > x0 and y1 > y0:
                fr[y0:y1, x0:x1] = cv2.GaussianBlur(fr[y0:y1, x0:x1], (ksize, ksize), 0)
        vw.write(fr)
    vw.release()
