"""Glue used by both the notebook and the Streamlit app: hits -> decisions -> actions -> rendered clip + report."""
import json, time
from pathlib import Path
from .asr import transcribe
from .audio import SR, EnergyCurve, load_audio, refine_window, save_wav
from .lexicon import mask, scrub
from .rag import context_check, window_text
from .render import render_clip

DEFAULT_RULE = {"mild": "R01", "strong": "R02", "slur": "R03"}
ACTION = {"mild": "bleep", "strong": "bleep", "slur": "drop", "none": "keep"}


def needs_check(hit):
    """Ambiguous = the lexicon says the word has innocent senses, or the match was not exact."""
    return hit.needs_context or hit.match not in ("exact", "phrase")


def context_items(clip_id, words, hits, entries):
    return [{"key": f"{clip_id}#{k}", "window": window_text(words, h.i0, h.i1), "term": entries[h.entry_id].text,
             "heard": h.heard, "match": h.match, "tier": h.tier, "lang": h.lang}
            for k, h in enumerate(hits) if needs_check(h)]


def decide(hit, decision, use_context=True):
    if use_context and needs_check(hit) and decision is not None:
        sev = decision["severity"] if decision["offensive"] else "none"
        return {"severity": sev, "rule_id": decision["rule_id"], "reason": decision["reason"], "judge": decision["source"]}
    return {"severity": hit.tier, "rule_id": DEFAULT_RULE[hit.tier], "reason": "unambiguous lexicon match",
            "judge": "lexicon"}


def plan_actions(clip_id, y, words, hits, decisions, cfg, use_context=True, entries=()):
    """Returns report events + bleep and drop windows (seconds)."""
    curve = EnergyCurve(y)
    events, bleeps, drops = [], [], []
    for k, h in enumerate(hits):
        d = decide(h, decisions.get(f"{clip_id}#{k}"), use_context)
        win = refine_window(curve, h.start, h.end, **cfg["refine"])
        act = ACTION[d["severity"]]
        if act == "bleep":
            bleeps.append(win)
        elif act == "drop":
            w = words[h.i0]
            win = (min(win[0], w.get("seg_start", win[0])), max(win[1], w.get("seg_end", win[1])))
            drops.append(win)
        events.append({"clip": clip_id, "modality": "audio", "start": round(win[0], 3), "end": round(win[1], 3),
                       "tier": d["severity"], "lexicon_tier": h.tier, "match": h.match, "action": act,
                       "rule_id": d["rule_id"], "reason": scrub(d["reason"], entries), "judge": d["judge"], "heard_masked": mask(h.heard)})
    return events, bleeps, drops


def process_video(path, ctx, workdir):
    """Full pipeline on one uploaded video. ctx: dict with whisper, matcher, entries, judge, index, reader, cfg."""
    from .ocr import blur_video, ocr_hits, read_frames, sample_indices
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    T, cfg, clip_id = {}, ctx["cfg"], Path(path).stem
    t = time.time(); y = load_audio(path); wav = workdir / f"{clip_id}.wav"; save_wav(wav, y); T["decode"] = time.time() - t
    t = time.time(); words, lang, _ = transcribe(ctx["whisper"], y); T["asr"] = time.time() - t
    t = time.time(); hits = ctx["matcher"].find(words); T["detect"] = time.time() - t
    t = time.time()
    items = context_items(clip_id, words, hits, ctx["entries"])
    decisions = context_check(ctx["judge"], ctx["index"], items, log=lambda *a: None) if items else {}
    T["context"] = time.time() - t
    events, bleeps, drops = plan_actions(clip_id, y, words, hits, decisions, cfg, entries=ctx["entries"])
    t = time.time()
    frames, fps = read_frames(path)
    si = sample_indices(len(frames), fps, cfg["ocr_every"])
    sb, vis = [], []
    for k in si:
        hs = ocr_hits(ctx["reader"], ctx["matcher"], frames[k])
        sb.append([h["box"] for h in hs])
        for h in hs:
            vis.append({"clip": clip_id, "modality": "video", "start": round(k / fps, 3),
                        "end": round(k / fps + cfg["ocr_every"], 3), "tier": h["tier"], "action": "blur",
                        "rule_id": "R13", "reason": "listed word visible on screen", "judge": "lexicon"})
    blurred = workdir / f"{clip_id}_blur.mp4"
    blur_video(frames, fps, si, sb, blurred, mode=cfg["ocr_mode"])
    T["ocr_blur"] = time.time() - t
    t = time.time(); out = workdir / f"{clip_id}_safe.mp4"; render_clip(blurred, wav, out, bleeps, drops); T["render"] = time.time() - t
    report = {"clip": clip_id, "duration_s": round(len(y) / SR, 3), "asr_language": lang,
              "events": events + vis, "timings_s": {k: round(v, 3) for k, v in T.items()}}
    (workdir / f"{clip_id}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report, out
