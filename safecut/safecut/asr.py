def transcribe(model, y, language=None, beam_size=5):
    """faster-whisper on a 16 kHz float array (arrays avoid the PyAV decoding bug seen on Kaggle)."""
    segs, info = model.transcribe(y, language=language, beam_size=beam_size, word_timestamps=True,
                                  condition_on_previous_text=False, vad_filter=False)
    words = [{"word": w.word, "start": float(w.start), "end": float(w.end), "prob": float(w.probability),
              "seg_start": float(s.start), "seg_end": float(s.end)}
             for s in segs for w in (s.words or [])]
    return words, info.language, float(info.language_probability)
