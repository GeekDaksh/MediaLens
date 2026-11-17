# MediaLens: Multimodal AI for Video and TV Content

Three tools that understand video through what is said, how it is said, and what is on screen.
Each tool has its own folder, notebook, app and README with full results.

| Tool | What it does | Headline result |
|---|---|---|
| [Scene search](scene-search/) | Type "Ross being angry about the couch" and get the matching *Friends* clips, plus an answer that cites them | Recall@5 from **2.5% to 76.0%** on 200 held-out queries (MELD, 2,610 clips) |
| [Sarcasm detection](sarcasm-detection/) | Finds sarcasm from text, voice and face. A cheap model runs first, and a clip goes to costlier models only when the cheap one is unsure | F1 from **65.8 (text) to 76.2 (fused)**. The cascade keeps 97.7% of that F1 with 6.7% less compute (MUStARD) |
| [SafeCut](safecut/) | Finds profanity and slurs in clips (English + Hinglish), then bleeps, blurs or cuts them | **100% precision**. A context check cut false censoring of innocent sentences from 93.5% to 19.4% |

## How the results were checked

- Every number comes from a run notebook, and each README says which one.
- Benchmarks use held-out splits. Settings were tuned on a separate dev or practice set.
- Confidence intervals are reported, and McNemar tests where two systems are compared.
- Failures are written up too. For example, SafeCut's boundary-snapping idea leaked more audio than a simple fixed pad.

## Stack

PyTorch, Whisper (faster-whisper), RoBERTa, wav2vec2, CLIP, Qwen2.5, Gemini API, FAISS, EasyOCR, ffmpeg, Streamlit.
Everything was run on free Kaggle T4 GPUs.

## Repo layout

```
scene-search/       engine.py, app.py, scene_search.ipynb, demo/
sarcasm-detection/  app/, sarcasm_cascade.ipynb
safecut/            safecut/ package, app.py, safecut_kaggle.ipynb, results/
```

Datasets: [MELD](https://github.com/declare-lab/MELD) and [MUStARD](https://github.com/soujanyaporia/MUStARD) are public, and you download them separately.
