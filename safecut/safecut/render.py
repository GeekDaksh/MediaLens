"""ffmpeg rendering: 1 kHz bleep over mild/strong words, drop slur segments, concat into a reel."""
import subprocess
import numpy as np
from .audio import probe_duration


def _expr(wins):
    return "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in wins) or "0"


def render_clip(video_in, audio_in, out, bleeps=(), drops=(), tone_gain=0.3):
    """video_in: video stream source; audio_in: audio source (wav or same mp4). Times in input seconds."""
    dur = probe_duration(audio_in)
    B, D = _expr(bleeps), _expr(drops)
    fc = (f"[1:a]aresample=48000,aformat=channel_layouts=mono,asetnsamples=n=64:p=0,"
          f"volume=volume='if(gt({B},0),0,1)':eval=frame[sp];"
          f"sine=frequency=1000:sample_rate=48000:duration={dur:.3f},asetnsamples=n=64:p=0,"
          f"volume=volume='if(gt({B},0),{tone_gain},0)':eval=frame[tn];"
          f"[sp][tn]amix=inputs=2:duration=first:normalize=0[mx];"
          f"[0:v]select='not(gt({D},0))',setpts=N/FRAME_RATE/TB[v];"
          f"[mx]aselect='not(gt({D},0))',asetpts=N/SR/TB[a]")
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video_in), "-i", str(audio_in),
           "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return cmd


def concat(clips, out, W=1280, H=720, fps=24):
    ins, fc = [], ""
    for i, c in enumerate(clips):
        ins += ["-i", str(c)]
        fc += (f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
               f"fps={fps},setsar=1[v{i}];[{i}:a]aresample=48000,aformat=channel_layouts=mono[a{i}];")
    fc += "".join(f"[v{i}][a{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=1[v][a]"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *ins, "-filter_complex", fc, "-map", "[v]",
                    "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", str(out)], check=True)


def tone_ratio(y, sr, f=1000.0, bw=30.0):
    """Share of signal power within +-bw Hz of f (1.0 = pure tone)."""
    if len(y) < 32:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    return float(spec[np.abs(freqs - f) <= bw].sum() / (spec.sum() + 1e-12))
