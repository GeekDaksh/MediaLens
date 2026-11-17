"""Audio I/O (via ffmpeg, never PyAV), energy curve and the three bleep-window strategies."""
import subprocess
import numpy as np

SR = 16000


def load_audio(path, sr=SR):
    out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "f32le",
                          "-ac", "1", "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).copy()


def save_wav(path, y, sr=SR):
    from scipy.io import wavfile
    wavfile.write(str(path), sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return float("nan")


class EnergyCurve:
    """Short-time energy (dB) of a pre-emphasised signal. Pre-emphasis lifts weak fricatives like f/s/sh,
    which plain energy would treat as silence."""
    def __init__(self, y, sr=SR, win=0.02, hop=0.005, preemph=0.97):
        y = np.append(y[:1], y[1:] - preemph * y[:-1]) if len(y) > 1 else y
        n, h = int(win * sr), int(hop * sr)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        frames = np.lib.stride_tricks.sliding_window_view(y, n)[::h]
        self.db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-8)
        self.t = np.arange(len(self.db)) * hop + win / 2
        self.hop, self.dur = hop, len(y) / sr
        self.floor, self.peak = np.percentile(self.db, 10), np.percentile(self.db, 95)

    def idx(self, t):
        return int(np.clip(round((t - self.t[0]) / self.hop), 0, len(self.t) - 1))


def pad_window(s, e, pad, dur):
    return max(0.0, s - pad), min(dur, e + pad)


def refine_window(c, s, e, search=0.3, alpha=0.3, min_sil=0.03, guard=0.01):
    """Snap each edge to the nearest low-energy point within +-search seconds.
    Edge inside speech -> walk outward to the first quiet run of >= min_sil.
    Edge already in silence (Whisper too generous) -> walk inward to where speech starts."""
    low = c.db < c.floor + alpha * (c.peak - c.floor)
    run = max(1, int(round(min_sil / c.hop)))
    mid = c.idx((s + e) / 2)

    def quiet(j, step):
        k = [j + step * r for r in range(run)]
        return all(0 <= q < len(low) and low[q] for q in k)

    def snap(t, outward):                      # outward = -1 for start edge, +1 for end edge
        i, lim = c.idx(t), int(round(search / c.hop))
        if not low[i]:
            for d in range(lim + 1):
                j = i + outward * d
                if 0 <= j < len(low) and quiet(j, outward):
                    return c.t[j]
            lo, hi = sorted((i, i + outward * lim))
            lo, hi = max(lo, 0), min(hi, len(low) - 1)
            return c.t[lo + int(np.argmin(c.db[lo:hi + 1]))]
        for d in range(lim + 1):
            j = i - outward * d
            if (outward < 0 and j > mid) or (outward > 0 and j < mid) or not (0 <= j < len(low)):
                break
            if not low[j]:
                return c.t[j]
        return t

    return float(max(0.0, snap(s, -1) - guard)), float(min(c.dur, snap(e, +1) + guard))


def leak_overmute(gt, win):
    """leak = fraction of the true word NOT covered; over-mute = clean audio muted (ms)."""
    (gs, ge), (ws, we) = gt, win
    ov = max(0.0, min(ge, we) - max(gs, ws))
    return 1 - ov / (ge - gs), 1000 * ((we - ws) - ov)
