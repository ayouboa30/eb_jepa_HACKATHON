# Audio-JEPA — A from-scratch, I-JEPA-style adaptation of masked latent prediction to log-mel spectrograms, competitive with wav2vec 2.0 / data2vec on <1/5 of their pretraining data.

**Authors:** Ludovic Tuncay (IRIT-SAMoVA, Université de Toulouse), Etienne Labbé (IRIT-SAMoVA), Emmanouil Benetos (Queen Mary University of London), Thomas Pellegrini (IRIT-SAMoVA) **Venue/Year:** ICME 2025 (Audio Encoder Capability Challenge) **arXiv:** 2507.02915 **Repo:** https://github.com/LudovicTuncay/Audio-JEPA (stated as "code and pretrained checkpoints will be released on GitHub")

## TL;DR
Audio-JEPA is a direct port of image I-JEPA to audio: it treats a log-mel spectrogram as a single-channel (non-square) image, masks 40-60% of its 16×16 patches, and trains a ViT context encoder + lightweight predictor to regress the latent embeddings of the masked patches produced by an EMA target encoder (with stop-gradient). Pretrained on ~1.92M AudioSet clips (5,338 h) on 4 V100s for only 14 h, it reaches performance "comparable to wav2vec 2.0 and data2vec" on the 21-dataset X-ARES suite while using less than one-fifth of their training data and with no hyper-parameter tuning. It is strongest under kNN evaluation (1st on ESC-50, FMA-small, GTZAN), weakest on fine-grained speech tasks (speaker verification, keyword spotting).

## Problem & motivation
SSL has driven audio representation learning (wav2vec 2.0, HuBERT, WavLM, data2vec/data2vec 2.0, M2D), but most prior audio SSL either does contrastive prediction over quantized latents (wav2vec 2.0), predicts offline cluster labels (HuBERT), or reconstructs the masked signal. The JEPA paradigm (LeCun's vision; I-JEPA for images, V-JEPA for video) instead predicts the *high-level latent representation* of masked regions rather than low-level signal, forcing the model to capture abstract semantics and ignore minute pixel-level (here, time-frequency-bin-level) detail. The authors note that an audio JEPA was sketched by Fei et al. (A-JEPA) but with no public implementation or checkpoints, and that Stem-JEPA applies JEPA to whole instrument stems rather than patches. The gap the paper fills: a clean, open-source, from-scratch I-JEPA-style audio model with purely random masking (no spec-augment annealing as in A-JEPA, no data augmentation as in M2D), benchmarked on the ICME 2025 challenge's X-ARES suite.

## Method

### Input representation
Each waveform is an AudioSet clip resampled to **32 kHz, 10 s** duration. It is converted to a **128-band mel-spectrogram with 256 time bins** (frame size = 2.5× the hop). The spectrogram is treated as a single-channel, possibly non-square "image" and partitioned into non-overlapping time-frequency patches of **16×16**. (128 mel bins / 16 = 8 frequency patches; 256 time bins / 16 = 16 time patches → 128 patches per clip.)

### Masking strategy
Purely **random patch masking** throughout training: per example, a fraction of patch indices in **40%–60%** is masked, with the exact ratio uniformly sampled per batch. The authors explicitly tried the I-JEPA **block masking** strategy and found it gave *lower* performance than random masking. No spec-augment-style structured masking (contrast with A-JEPA, which anneals toward it) and no data augmentation (contrast with M2D).

### Encoder / predictor / target-encoder architecture
A three-module I-JEPA design:
- **Context encoder** `f_ctx`: ViT over the *visible* patches. Patch size 16×16, embedding dim **768**, depth **12**, **12** heads, MLP ratio **4.0**. ~85.4M params.
- **Target encoder** `f_tgt`: architecturally identical ViT, but updated by **EMA** of the context encoder's weights (not by gradient descent), encoding the *full* spectrogram (context + masked) so targets see rich context. ~85.4M params. EMA decay τ scheduled "the same way as in BYOL."
- **Predictor** `g_pred`: a lighter ViT, embedding dim **384**, depth **6**, **12** heads, MLP ratio 4.0, with a re-projection back to **768** so its outputs match the target encoder's dim. ~11.3M params.

Total **96.7M** trainable params at training time; **85.4M** at inference (the predictor is discarded; only the frozen target encoder is used for downstream features).

### Prediction objective (latent regression; loss)
Standard JEPA latent regression. With `c_i = f_ctx(x_\M)` (context embeddings of visible patches), `ĉ_j = g_pred(c)` (predicted embeddings for masked positions), and `t_j = f_tgt(x)_j` (target embeddings of masked patches), the loss is the **average squared L2 (Euclidean) distance over masked patches**:

L = (1/|M|) Σ_{j∈M} ‖ĉ_j − t_j‖₂².

The context encoder and predictor are trained by backpropagation; the target encoder is updated as θ_tgt ← τ·θ_tgt + (1−τ)·θ_ctx.

### Anti-collapse mechanism
No explicit variance/covariance or contrastive regularizer. Collapse is prevented purely by the **asymmetry of EMA target + stop-gradient** (the standard BYOL/I-JEPA recipe): the stop-gradient sits between the predictor and the target encoder, and the EMA-stabilized target "stabilizes target representation and prevents collapse." The authors note (citing V-JEPA) that this objective favors embedding *cohesion*, so the embedding space is not guaranteed to be linearly separable.

## Key results

**Pretraining data.** 1,921,982 AudioSet clips, 10 s @ 32 kHz, totaling **5,338 hours**. Trained on **4 NVIDIA V100 GPUs**, batch size 256 clips (≈42.7 min of audio per batch), for **100,000 steps (~13 epochs), 14 hours** total. AdamW (β1=0.9, β2=0.95, weight decay 0.05), LR 3e-4 with warmup-cosine (1e-6 → 3e-4 over 1,000 steps, then anneal to 0). For comparison, wav2vec 2.0 and data2vec each trained for 400k steps with larger batches (1.6 h and 63 min of audio per batch respectively) — Audio-JEPA uses far fewer steps and "less than one-fifth" of the data.

**Benchmark (X-ARES).** The eXtensive Audio Representation and Evaluation Suite: **21 public datasets** spanning speech, music, and environmental-sound tasks. Two frozen-feature protocols on the frozen *target encoder*:
- **Linear probe (MLP):** single linear/MLP head on frozen features, fixed hyperparameters. Reported on 20 of 21 datasets (all except LibriSpeech-100h).
- **kNN:** non-parametric classifier on frozen embeddings, no training. Reported on the 16 X-ARES tasks compatible with the probe.

**Headline comparisons (vs wav2vec 2.0 and data2vec).**
- *kNN (the favorable, "true representational power" protocol):* Audio-JEPA is **1st on 3 datasets — ESC-50 (0.140 vs 0.081 / 0.040), FMA-small (0.449 vs 0.251 / 0.106), GTZAN (0.452 vs 0.303 / 0.108)** — and 2nd on 7 more, outperforming both baselines on music and environmental-sound tasks despite far less pretraining. It ranks last on roughly a third of tasks (the fine-grained speech ones).
- *Linear probe:* "first or second place on several benchmarks but falls to last on roughly half." It does well on, e.g., FMA-small (0.553 vs 0.469 / 0.334) and DESED (0.306 vs 0.313 / 0.136), but collapses on Fluent Speech Commands (0.025 vs 0.468 / **0.978**), Speech Commands V1 (0.152 vs 0.714 / 0.927), VoxCeleb1 (0.041 vs 0.340 / 0.105), and VoxLingua33 (0.093 vs 0.553 / 0.620). The authors attribute this to the JEPA objective not guaranteeing linear separability (a known V-JEPA caveat), not to weak features per se.

**Data-efficiency claim.** Competitive-to-better embeddings (especially under kNN on music/environmental sound) with <1/5 the data and a fraction of the compute (14 h on 4 V100s vs 400k-step baselines).

**Ablation.** The only explicit ablation: random masking beats I-JEPA block masking ("preliminary experiments showed the block masking strategy from I-JEPA yielded lower performance than random masking"). No sweep over mask ratio, EMA decay, or optimizer settings was performed (explicitly listed as future work).

## Relevance to the EB-JEPA hackathon
This paper is the canonical reference for the **audio modality track (Track 2)**: it shows exactly how to lift the image-JEPA recipe onto spectrograms, and its failure modes are instructive.

**Which eb_jepa setting it resembles.** Audio-JEPA maps cleanly onto **`examples/image_jepa`**, NOT `examples/video_jepa`. Despite being audio, it is a *masked-patch latent-prediction over a 2D (time×freq) "image"* task — i.e., spatial masked modeling — not a temporal next-frame predictor. The eb_jepa `video_jepa` example predicts the *future* `K+1`-th frame representation autoregressively (temporal prediction); Audio-JEPA predicts *spatially masked* patches within one clip. So a team should think "I-JEPA on a 128×256 single-channel image" and reuse the image_jepa scaffolding (ViT backbone, patchify, frozen-feature linear-probe eval), swapping CIFAR images for log-mel spectrogram tensors.

**How it prevents collapse — and the key contrast with eb_jepa.** This is the most important mapping to flag. Audio-JEPA uses the **I-JEPA / BYOL recipe: EMA target encoder + stop-gradient, no explicit regularizer.** The eb_jepa examples take the *opposite* anti-collapse route: there is **no EMA target encoder and no stop-grad** in `image_jepa` (it is a JEA with no predictor) or `video_jepa`; collapse is prevented by an explicit **VICReg variance-covariance loss or SIGReg/LeJEPA** term plus a projector. So a hackathon team replicating Audio-JEPA inside eb_jepa has two clean options:
1. **Faithful replication:** add an EMA target encoder + predictor + stop-grad path (closer to the paper, but not the native eb_jepa idiom).
2. **eb_jepa-native variant ("EB-Audio-JEPA"):** keep masked-patch latent prediction but drop the EMA/stop-grad and instead regularize the embeddings with VICReg or SIGReg (a single `λ` in the SIGReg case). This is arguably the more interesting hackathon contribution — it directly tests whether eb_jepa's energy-based regularizers can replace the BYOL-style asymmetry on audio, mirroring the image_jepa SIGReg-vs-EMA story.

**What a team can realistically reuse/replicate in 24 h.**
- Reuse `examples/image_jepa` (ViT path via `cfgs/transformers.yaml`, SIGReg via `cfgs/sigreg.yaml`) as the trainer; write only a log-mel dataset that emits `[1, 128, 256]` tensors (torchaudio MelSpectrogram, 32 kHz, 128 mels, 256 frames, frame = 2.5× hop).
- Use a *small subset* of AudioSet (or a quicker proxy like ESC-50 / FMA-small / GTZAN, where Audio-JEPA itself wins) so pretraining fits the window — full reproduction (5,338 h, 100k steps, 14 h on 4 V100s) is out of scope for 24 h; aim for a scaled-down proof-of-concept.
- Evaluate frozen features with **kNN first** (the protocol where JEPA-style features shine and which needs no probe training) and a light linear/MLP probe second, matching X-ARES.
- The single most reproducible ablation: **random vs block masking** at a 40-60% ratio.

**What to watch out for.**
- Expect **weak linear-probe separability** (the objective favors cohesion, not linear separability) — judge primarily by kNN, and consider an attentive-pooling head if doing linear probes (the paper's own recommended remedy, from V-JEPA).
- Expect **poor performance on fine-grained speech** (speaker ID/verification, keyword spotting) and strong performance on **music / environmental sound** — pick demo tasks accordingly (ESC-50, GTZAN, FMA-small are the wins).
- Mind the **non-square patch grid** (8 freq × 16 time) and 2D positional encodings; the paper flags positional encoding (rotary / conditional sine-cosine) as a likely improvement.

## Caveats / open threads
- **No hyper-parameter tuning at all** — mask ratio, EMA decay τ, and optimizer settings were taken off-the-shelf; the authors explicitly list a systematic sweep as future work, so reported numbers are a floor, not a tuned ceiling.
- **Linear-probe results are confounded** by the known JEPA non-separability issue; the paper itself argues kNN better reflects feature quality, so cross-method linear-probe comparisons should be read with that caveat (attentive pooling untested here).
- **Self-described as "a straightforward translation"** — no architectural novelty beyond porting I-JEPA; future directions named are attention-pooling probe heads, modern audio backbones (ConvFormer, CAFormer), better positional encodings, and HP tuning.
- **Grounding status of this summary:** all numbers, architecture sizes, dataset stats, and both results tables were extracted directly from the local PDF text (via PyMuPDF), not from the arXiv HTML (which is not rendered for this paper). The X-ARES per-dataset numbers and the 96.7M / 85.4M / 11.3M param split, 1,921,982 clips / 5,338 h, and 100k steps / ~13 epochs / 14 h figures are quoted verbatim from the paper. Not independently verified: the exact patch-grid count (128 patches) is inferred arithmetically from 128 mels and 256 frames at 16×16 (the paper does not state the patch count), and the τ schedule is only described as "the same way as in BYOL" without a numeric value.
