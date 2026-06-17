# A-JEPA — the first port of the I-JEPA latent-prediction principle to audio spectrograms, with curriculum time-frequency masking

**Authors:** Zhengcong Fei, Mingyuan Fan, Junshi Huang (Kunlun Inc.) **Venue/Year:** arXiv preprint, Nov 2023 (v3 Jan 2024); not a published venue **arXiv:** 2311.15830 **Repo:** "code and models will be publicly available" (no link in paper; no official repo cited)

## TL;DR
A-JEPA is a near-verbatim transplant of I-JEPA (Assran et al., 2023) onto Mel-spectrograms: a ViT context encoder ingests visible spectrogram patches, an EMA target encoder embeds masked target blocks, and a predictor regresses the target representations in latent space under an L2 loss. The two audio-specific contributions are (i) a **curriculum masking schedule** that anneals from random-block to time-frequency-aware masking during pretraining, and (ii) **regularized masking (RM)** during fine-tuning (masked tokens excluded from contributing attention but still attended-to, rather than zeroed/dropped). Pretrained audio-only on AudioSet, it sets SOTA among in-domain self-supervised models, beating AudioMAE by +1.3 mAP on AS-2M.

## Problem & motivation
Masked latent-space prediction (I-JEPA) avoids predicting low-level pixels/tokens and concentrates the model on semantic features, but had not been applied to audio. The authors argue naive I-JEPA random-block masking is "comparably easier than time-frequency aware masking" on spectrograms, because strong local correlations along the time and frequency axes let the model extrapolate missing patches from neighbors (e.g. "formants in vowels and frictional sounds in consonants"). They want a JEPA-style audio recipe that is harder, scalable, and audio-only (no ImageNet transfer).

## Method
- **Input:** raw waveform → mono, 16 kHz → 128 Mel bands (25 ms Hanning window, 10 ms shift). A 10 s AudioSet clip yields a 1×1024×128 spectrogram, patched with non-overlapping 16×16 conv kernels + fixed sinusoidal positional encodings.
- **Architecture:** standard ViT context encoder (default ViT-B, 12 layers) processing only visible patches; structurally-identical EMA target encoder; a 16-layer vanilla Transformer predictor/"decoder" with masked tokens + sinusoidal PEs and a linear head predicting latent features.
- **Curriculum masking:** at each step, choose random-block vs. time-frequency masking via `p ~ Bernoulli(f(s))`, with progressing function `f(s)=min(1, sqrt(s·(1-c0²)/S)+c0²)`, `c0=0.01`; anneals from easy (0) to hard time-frequency (1). Targets: 4 random blocks (scale 0.15–0.2, aspect 0.75–1.5) or 3 time-freq blocks (scale 0.05–0.075); 1 context block (scale 0.85–1.0) with overlapping target regions removed (Algorithm 1).
- **Objective:** averaged L2 distance between predictor outputs and target-encoder outputs, with a multi-mask strategy to amortize target computation.
- **Regularized masking (fine-tuning):** discard predictor, keep encoder + average-pool + linear head; in each layer a random fraction of patch tokens are barred from contributing to attention weights (their attention score is entirely supplied by others), forcing reliance on partial-neighbor info. RM is used only at fine-tuning, attention reverts to vanilla at test time.
- **Training:** AudioSet-2M pretraining, 24 epochs, batch 512, lr 2e-4, ±6 dB magnitude jitter, cyclic 10 s crops; fine-tune with 10% RM ratio (100 fine-tune epochs ≈ 10 full AS-2M passes with class-balanced weighted sampling).

## Key results
- **AS-2M:** 48.6 mAP (ViT-B), vs. AudioMAE 47.3 (+1.3); **AS-20K:** 38.4 mAP vs. AudioMAE 37.0. Also tops ImageNet-supervised models (e.g. HTS-AT/PaSST 47.1) without any out-of-domain data (Table 1).
- **Speech/env tasks (ViT-B):** ESC-50 96.3, SPC-2 98.5, SPC-1 97.7, SID (VoxCeleb) 95.8 — all ≥ AudioMAE.
- **Scaling (Table 2):** ViT-S/B/L = 46.1/48.6/48.8 AS-2M mAP; gains larger on small AS-20K (33.7/38.4/38.8).
- **Ablations:** curriculum masking 48.6 > random 47.8 > inverse 46.3 (Table 3); RM +0.5 mAP (48.6 vs 48.1, Table 4); predictor depth 16>8 (48.6 vs 48.3, Table 5); predictor width 512 optimal (48.6, Table 6); mAP rises monotonically with pretraining data size (Fig 6) and plateaus after epoch 24 (Fig 7).
- RCDM-decoder visualizations (Fig 5) suggest the predictor captures positional uncertainty of high-level audio components.

## Relevance to the EB-JEPA hackathon (audio track)
- This is the **earliest** audio-JEPA and the most direct I-JEPA clone, so it is the cleanest baseline/ancestor to position EB-JEPA against on the audio track. Energy-based / variance-covariance anti-collapse machinery is entirely absent here: A-JEPA relies on the I-JEPA recipe (EMA target encoder + multi-block masking) to avoid collapse, with no explicit regularizer — a natural lever for an EB-JEPA audio variant.
- **Contrast with the already-summarized Audio-JEPA paper (`references/paper/audio-jepa/`):** A-JEPA (Kunlun, Nov 2023) is the original "I-JEPA on Mel-spectrograms" with two bespoke tricks (curriculum time-freq masking + fine-tuning RM) and reports headline AS-2M/AS-20K classification SOTA over AudioMAE. It is a single-paper proof-of-concept, audio-only AudioSet pretraining, fine-tuned classification numbers. Use Audio-JEPA's summary for the complementary/later framing; A-JEPA is the foundational citation establishing that latent masked prediction works on audio at all.
- Concrete transferable design knobs for an audio EB-JEPA: the spectrogram patchification (1×1024×128, 16×16 patches), the time-frequency-aware target sampling, and the Bernoulli curriculum schedule are reusable regardless of the anti-collapse objective.

## Caveats & open threads
- **No public code/repo link** despite the promise; reproducing exact numbers requires re-deriving hyperparameters from Appendix A.
- Terminology is sloppy: the paper calls the latent predictor a "decoder" and an "RCDM decoder" interchangeably, and the Table 5/6 captions say "decoder depth/width" while the body discusses the predictor — read carefully.
- Evaluation is **classification-only** (mAP/accuracy on AudioSet/ESC/SPC/SID); no generation, retrieval, or transfer-to-other-modality numbers. Linear-probe results for A-JEPA itself are not reported (only fine-tuning), so the raw representation quality vs. AudioMAE under frozen features is untested here.
- Gains over AudioMAE are modest (+1.3 mAP on AS-2M) and the headline relies on supervised fine-tuning; the "self-supervised representation" advantage is conflated with the RM fine-tuning trick.
- Not peer-reviewed (arXiv only); no error bars / multiple seeds reported.
