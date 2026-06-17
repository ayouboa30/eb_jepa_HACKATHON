# Stem-JEPA — A JEPA that predicts the embedding of a *compatible missing instrument stem* from a context mix, conditioned on the target instrument label, and is the first work to use the JEPA *predictor* at inference time.

**Authors:** Alain Riou (LTCI, Télécom-Paris, IP Paris & Sony CSL Paris), Stefan Lattner (Sony CSL Paris), Gaëtan Hadjeres (Sony AI), Michael Anslow (Sony CSL Paris), Geoffroy Peeters (LTCI, Télécom-Paris, IP Paris) **Venue/Year:** ISMIR 2024 (25th Int. Society for Music Information Retrieval Conf., San Francisco) **arXiv:** 2408.02514 (v1, 5 Aug 2024) **Repo:** https://github.com/SonyCSLParis/Stem-JEPA (demo: https://sonycslparis.github.io/Stem-JEPA)

## TL;DR
Stem-JEPA reframes JEPA for *musical stem compatibility*: given a multi-track song, it crops 8 s, picks one active stem as the **target** and a mix of 1–3 of the remaining stems as the **context**, converts both to log-mel spectrograms, and trains a ViT-Base context encoder + a 6-layer MLP predictor (conditioned on the missing stem's instrument label) to predict the EMA-target encoder's patch-wise embeddings of the held-out stem, under a patch-wise normalized-L2 (MSE on L2-normalized vectors) loss. Crucially the context/target split comes from **omitting stems during mixing**, not from masking the input. Trained on a proprietary 20k-track, 1350 h dataset (4 instrument classes: Bass/Drums/Vocals/Other) for 300k steps (~4 days, single 40 GB A100). On a MUSDB18 stem-retrieval task it hits **R@1 = 33%, median Normalized Rank = 0.5%**; a listening study rates retrieved stems roughly *double* random and close to ground truth; embeddings encode local temporal alignment and meaningful key/chord structure; downstream MARBLE probes are on par with SOTA on tagging/instrument but lag on key/genre given ~100× less data. It is, to the authors' knowledge, the **first use of a JEPA predictor at inference time**.

## Problem & motivation
**Musical stem compatibility** = how well a single-instrument stem fits a given musical context (another stem or a mix) when played together. Automatic estimation enables stem retrieval, automatic arrangement/mashup creation, and conditioning for stem generation. Compatibility depends on global factors (tonality, tempo, genre, timbre, playing/singing style) and on local features (chords, pitches) that govern temporal alignment. Prior compatibility work was either music-theory-driven (beat tracking + chord estimation, e.g. AutoMashupper) or learned end-to-end, but learning extends "compatibility" beyond theory toward sound/expressive characteristics. The authors also motivate it for **generation**: with Stem-JEPA, a stem representation can be *predicted* from context at inference time, so a stem generator can be trained on stem representations alone, "eliminating the need for context/target pairs." The gap: most SSL is ported from vision and is not music-specific, whereas musical audio is naturally composed of stems with rich compositional structure; only a few prior SSL works leverage separated stems, and JEPAs had not been applied to stem compatibility.

## Method

### Training pipeline (stem omission, not masking)
Given a track of `S` stems `x_1,…,x_S`, crop an **8-second** chunk. Randomly pick one active stem as **target** `x̄ = x_t`; build the **context** as a *sum* (mix) of a random subset `C ⊂ {1,…,S}\{t}` of the *other* active stems: `x = Σ_{c∈C} x_c`. This is the paper's defining departure from I-JEPA/M2D: the context/target pair is created by **omitting stems within the mixing process** (and conditioning the predictor on the missing stem's *label*), *not* by masking patches in the input space.

### Input representation
Both `x` and `x̄` → **log-mel spectrograms**: **80 mel bins**, window 25 ms, hop 10 ms. Divided into a regular time×frequency grid of **16×16 patches**, giving sequences of (80/16)×(800/16) = **250 tokens** during training (one embedding per 160 ms of audio).

### Encoder / target-encoder / predictor
- **Context encoder** `f_θ`: standard **ViT-Base** producing patch-wise embeddings `z = (z_1,…,z_K)`.
- **Target encoder** `f_θ̄`: same architecture, weights updated by **EMA** of the context encoder: `θ̄_i = τ_i θ̄_{i-1} + (1−τ_i) θ_i`, with EMA rate `τ_i` linearly interpolated from `τ_0` to `τ_T` over `T` steps. Encodes the target stem to `z̄ = (z̄_1,…,z̄_K)`.
- **Predictor** `g_φ`: a **6-layer MLP**, ReLU, 1024-dim hidden layers. **Conditioned** on the missing stem's instrument label `l` by concatenating a learnable embedding `emb(l)` to each `z_k`: `z̃_k = g_φ(concat(z_k, emb(l)))`. (An ablation swaps in a Transformer predictor identical to M2D's.)

### Prediction objective (loss)
Patch-wise MSE between L2-**normalized** predicted and target embeddings, with **stop-gradient** on the target branch (the EMA encoder):

L(z̃, z̄) = (1/K) Σ_{k=1}^{K} ‖ z̃_k/‖z̃_k‖ − z̄_k/‖z̄_k‖ ‖².

Only `(θ, φ)` are updated by gradient descent; `θ̄` follows the EMA.

### Anti-collapse mechanism
Pure **BYOL/I-JEPA recipe**: **stop-gradient on the non-predictor (target) branch** + **EMA-tied target encoder** + an asymmetric **predictor**. The paper explicitly states blocking gradients in the non-predictor branch is crucial to prevent collapse, and that tied-but-different encoders stabilize training. **No** VICReg/SIGReg-style variance-covariance regularizer is used.

### Sampling (avoiding silence)
Let `A` be the indices of *active* (non-silent) stems. Pick target `t ∈ A`; pick context subset `C ⊂ A\{t}` with `|C|` uniform between 1 and `|A|−1`. Usually `|C|>1` (more context than target, easier prediction); occasionally `|C|=1` so the model also learns single-stem representations (needed because stems are also used as targets). If `|A| < 2`, resample another chunk from the same track.

### Architecture & training details
ViT-Base encoder; 6-layer 1024-d MLP predictor. **300k steps**, **AdamW**, **batch size 256**, base LR **3e-4**, cosine annealing after **20k** linear-warmup steps; other hyperparameters follow M2D [9]. ~**4 days on a single 40 GB A100**. Training data: proprietary **20k multi-track recordings, 1350 hours**, diverse genres (pop/rock, R&B, rap, country), four instrument categories **Bass, Drums, Vocals, Other**.

## Key results

### Stem retrieval on MUSDB18
MUSDB18: `N = 150` tracks × `S = 4` stems = 600 runs. For stem `x_s^{(n)}`, encode the leave-one-out mix `x_{¬s}^{(n)}`, push through the predictor conditioned on `s`, average over time+freq → query `q`; test whether the true stem embedding (averaged over time) is among `q`'s nearest neighbors in the 600-item reference set. Metrics: **Recall@K** (K∈{1,5,10}) and **Normalized Rank** (rank of ground truth / 600).

| Predictor | R@1 | R@5 | R@10 | mean NR | median NR |
|---|---|---|---|---|---|
| **MLP w/ cond. (main)** | **33.0** | **63.2** | **76.2** | **2.0** | **0.5** |
| MLP w/o cond. | 28.2 | 58.0 | 69.2 | 3.3 | 0.7 |
| Transformer | 5.2 | 17.5 | 25.7 | 12.1 | 6.0 |
| AutoMashupper [1] | 1.0 | 8.8 | 15.5 | 29.1 | 19.5 |

(All in %.) The main model retrieves the correct stem in the **top 0.5%** of neighbors in half the cases; **conditioning is essential** and, notably, the **MLP predictor strongly beats the Transformer** for retrieval (the authors argue an MLP forces the *encoder* to capture global info, yielding more informative embeddings). Per-instrument: "other" is easiest (R@1 ≈ 45%), "drums" hardest (R@1 ≈ 25%, plausibly because many drum patterns fit a given mix). "Both wrong" is rare; for bass/drums the model often returns the right song but wrong instrument, with "other" (a broad, ill-defined class) the main confound.

### User study
23 participants (20 musically experienced, 11 with ≥10 years), Go Listen platform, 60 trials (12 per user), 16 s chunks randomly cropped to 10 s during the test to prevent reliance on temporal alignment. Retrieved stems are rated **slightly below ground truth but ≈ double random**, confirming retrieval of compatible non-original stems. Drums show the closest match to ground truth and highest variance (drums are broadly compatible).

### Temporal alignment analysis
Using time-resolved (one embedding / 160 ms) embeddings, cosine similarity `s(z,q,j)` between embeddings and predictions is measured across temporal shifts `j`. A **sharp peak at j=0** indicates locally aligned detail; similarity stays high overall (global info present; random baseline ≈ **0.17**); **periodic patterns** reveal beat/bar structure; small peaks every 8 s expose absolute-position leakage from the encoder's absolute positional encodings.

### Musical plausibility
k-means (`k=32`) on patch embeddings from 174 Beatles songs (Isophonics key/chord labels): within-cluster key/chord co-occurrence shows neighbors share tonal relationships (tonic, subdominant, dominant, e.g. C/F, C/G), so embeddings capture meaningful harmony.

### Downstream MARBLE probes
Frozen encoder, patch outputs concatenated+averaged → **3840-d** global embedding → MLP (512 hidden) + softmax. Datasets: Giantsteps (key), GTZAN (genre), MagnaTagATune (tagging), NSynth (instrument).

| Model | GS Acc_refined | GTZAN Acc | MTT ROC | MTT AP | NSynth Acc |
|---|---|---|---|---|---|
| MLP w/ cond. | 40.2 | 68.6 | 89.9 | 42.8 | 73.5 |
| MLP w/o cond. | 36.8 | 72.5 | 90.1 | 42.9 | 75.0 |
| Transformer | 46.0 | 68.1 | 90.0 | 42.7 | 73.3 |
| MULE [12] | 64.9 | 75.5 | 91.2 | 40.1 | 74.6 |
| Jukebox [42] | 63.8 | 77.9 | 91.4 | 40.6 | 70.4 |

Predictor choice barely affects downstream (except key detection, where the Transformer predictor clearly wins). Strong NSynth shows the encoder keeps **stem-specific** info (not just shared-track info); surprisingly, **dropping conditioning improves most downstream tasks**. On MTT and NSynth Stem-JEPA is **on par with SOTA**; on Giantsteps and GTZAN it lags substantially, attributed to **~100× less training data** than MULE (117k h) / Jukebox (1.2M songs).

## Relevance to the EB-JEPA hackathon
This is a high-value reference for the **audio / music track**, and the most *conceptually distinctive* of the audio JEPA papers in `references/paper/` because its "view-generation" mechanism is **stem omission + mixing**, not patch masking — a genuinely different pretext task from Audio-JEPA (spatial masked modeling) and S-JEPA (channel masking).

**Which eb_jepa setting it resembles.** Like Audio-JEPA, it maps onto an **image-style ViT-over-spectrogram** trainer (`examples/image_jepa`-shaped: ViT backbone, 16×16 patchify of an 80×800 log-mel "image", frozen-feature probe eval), **not** the temporal next-frame `video_jepa`. The twist a team must implement is **dataset/collate-side, not model-side**: build context as a *sum of a random subset of stems* and target as a held-out stem, plus an **instrument-label conditioning embedding** concatenated to the predictor input. This is the cheapest place a hackathon team can add real novelty without touching the backbone.

**Anti-collapse — the key eb_jepa contrast.** Stem-JEPA uses the **EMA target + stop-gradient + predictor** (BYOL/I-JEPA) recipe, with **no** variance-covariance regularizer. The eb_jepa examples instead prevent collapse with explicit **VICReg / SIGReg / LeJEPA** energy terms (and no EMA/stop-grad in `image_jepa`). So the natural hackathon contribution mirrors the Audio-JEPA story: an **"EB-Stem-JEPA"** that keeps the stem-omission pretext + label conditioning but **drops EMA/stop-grad and regularizes with SIGReg/VICReg**, testing whether eb_jepa's regularizers can replace the BYOL asymmetry on the *compatibility* (not similarity) task.

**What's reusable in a hackathon window.**
- The retrieval eval is **clean and self-contained on public MUSDB18** (150 tracks, 4 stems, 600 runs; R@K + Normalized Rank) and needs **no probe training** — an excellent demo metric.
- A compatibility-prediction model is small (ViT-B + 6-layer MLP) and the *eval* protocol (leave-one-stem-out, predict, nearest-neighbor) is reproducible even with a much smaller / shorter-trained encoder.
- The **MLP-beats-Transformer-for-retrieval** and **conditioning-is-essential-for-retrieval-but-hurts-downstream** findings are concrete, cheap ablations to re-run.
- Strongest single contribution angle: **using the predictor at inference** (compatibility prediction / retrieval / could feed a generator) — unique among the JEPA references here.

**What to watch out for.**
- Main results use a **proprietary 20k-track / 1350 h dataset** — *not* reproducible; a team must substitute MUSDB18 (train) or another stemmed corpus (MoisesDB, Slakh) and expect lower numbers.
- Only **4 instrument classes** (Bass/Drums/Vocals/Other) — generalization is limited, "other" is a noisy catch-all that drives many confusions.
- **Absolute positional encodings leak global position** (8 s periodic peaks); consider relative/rotary PE.
- Downstream key/genre are **data-hungry** — don't expect MULE/Jukebox-level numbers from a 24 h run.

## Caveats & open threads
- **Reproducibility gap:** training data is proprietary (20k tracks, 1350 h); only the *evaluation* (MUSDB18) and code are public. All headline retrieval/downstream numbers were obtained with that private corpus.
- **Limitations stated by the authors:** SSL wants very large stem corpora but separated-stem datasets are scarce (source-separation advances may help); restriction to **four instruments** limits generalizability; extending the predictor to **arbitrary instrument conditioning** is named as the key future direction to remove the section-4.1.3 failure modes.
- **Novelty framing:** "first use of the predictor component of JEPAs during inference" and modeling **compatibility instead of similarity** via conditioning — flagged as potentially generalizable beyond music.
- **Grounding status:** all quoted numbers (R@1 33%, median NR 0.5%; the two result tables; 80 mel bins / 16×16 patches / 250 tokens / one embedding per 160 ms; 300k steps, batch 256, LR 3e-4, 20k warmup; ViT-Base + 6-layer 1024-d MLP; 20k tracks / 1350 h; ~4 days on one 40 GB A100; random-baseline cosine 0.17; k=32, 174 Beatles songs; 23 participants) were extracted directly from the local PDF text via PyMuPDF (9 pages, header b'%PDF', non-truncated). Not independently verified: exact `τ_0`/`τ_T` EMA values (paper says only "linearly interpolated", no numbers), the precise proprietary-dataset composition, and the 3840-d figure's derivation (paper states it follows M2D [9]).
