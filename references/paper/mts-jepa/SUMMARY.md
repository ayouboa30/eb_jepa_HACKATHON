# MTS-JEPA — A multi-resolution JEPA with a soft-codebook bottleneck for proactive multivariate time-series *anomaly prediction* (early warning), where the discrete bottleneck doubles as the anti-collapse regularizer

**Authors:** Yanan He, Yunshi Wen, Xin Wang, Tengfei Ma (Purdue / RPI / Stony Brook; correspondence Tengfei Ma) **Venue/Year:** Preprint, dated "February 5, 2026" (no venue stated; ICML-style template) **arXiv:** 2602.04643 (v1, "4 Feb 2026") **Repo:** none stated

## TL;DR
MTS-JEPA reframes time-series **anomaly prediction** (anticipate whether the *next* non-overlapping window is anomalous, i.e. early warning) as latent-space predictive world modeling à la JEPA. It adds two ingredients on top of a plain TS-JEPA: (1) a **multi-resolution predictive objective** — the online branch sees only a fine (patched) view of the current window but is supervised to match both a fine and a coarse (time-averaged/downsampled) target produced by an EMA encoder, decoupling transient shocks from slow drift; and (2) a **soft codebook bottleneck** (differentiable VQ over K=128 prototypes) that maps continuous encoder features to a probability distribution on the simplex, so self-distillation becomes **KL matching between code distributions** rather than unconstrained MSE in R^D. The paper's central empirical+theoretical claim is that this discrete bottleneck is the load-bearing **anti-collapse / stability** mechanism (not just a semantic prior): removing the codebook module collapses performance to near-random (std≈0 across seeds). On four standard MTS anomaly benchmarks (MSL, SMAP, SWaT, PSM), under their early-warning protocol MTS-JEPA reports **best F1 and best AUC on all four datasets**, including beating the single-scale **TS-JEPA** baseline.

## Problem & motivation
The paper distinguishes **anomaly prediction** (anticipate future abnormality from the current window, capture *precursors*) from both **detection** (reactive, scores the current window after a fault occurred) and **forecasting** (regresses numerical values, "prone to high-frequency noise"). It argues point-wise reconstruction/regression overfits stochastic volatility, motivating representation-space prediction (JEPA) that "captures high-level state transitions while ignoring unpredictable details." Two stated gaps when porting JEPA to continuous time series: (i) **representation collapse** — without negatives, latent self-distillation admits the constant-vector degenerate solution (covariance rank → 0); (ii) **multi-scale precursors** — standard JEPA operates at a single fixed temporal granularity, but precursors appear across "distinct frequency bands or temporal resolutions," so a single-resolution model misses dynamics outside its scale.

Contributions as listed: **(i)** formulate anomaly prediction as JEPA-style representation-space predictive modeling; **(ii)** the **first multi-resolution architecture within the JEPA framework for time series**, capturing scale-variant precursors (shocks → drifts); **(iii)** a **soft codebook bottleneck** (differentiable quantization) that imposes a discrete inductive bias and, as a "vital secondary benefit," stabilizes optimization / prevents collapse without explicit negative sampling.

## Method

### Problem setup
Multivariate series x ∈ R^{T×V}, partitioned into non-overlapping windows {W_t}. Goal: from current window W_t, predict the anomaly status of the next window W_{t+1} (proactive intervention). Channel-independent formulation throughout.

### Input formulation — dual-scale views
Each window is first passed through **RevIN** (reversible instance norm; affine stats cached for inverting normalization at reconstruction). Then two views:
- **Fine view (patching):** X^fine_t = Patch(Ŵ_t) ∈ R^{P×L×V}, i.e. P patches of length L (T_w = P·L).
- **Coarse view (downsampling):** X^coarse_t = DownAvg_P(Ŵ_t) ∈ R^{1×L×V} — averages every P consecutive time points (stride P) to capture low-frequency trends, then treated as a *single* patch to match the patch format.
- **Asymmetry / information gap:** the **online branch processes ONLY the fine view**, while the **EMA target branch encodes BOTH resolutions at t+1**. The online encoder must internalize global context from local evidence — a structural inductive bias for scale-aware reps.

### Architecture
A shared encoder + soft codebook + dual-resolution predictor + auxiliary decoder:
- **Encoder E:** channel-independent residual-CNN tokenizer + Transformer backbone; supports variable-length inputs so the P fine patches and the 1 coarse token share parameters/embedding space. Online encoder E_θ (sees fine only) and EMA encoder E_ξ (momentum copy, sees both → produces fine + coarse targets h^fine_{t+1}, h^coarse_{t+1}). EMA decay ρ=0.996, stop-gradient on targets.
- **Soft codebook Q:** maps each patch feature h_{t,i} ∈ R^D to a soft code distribution p_{t,i} ∈ Δ^{K−1} via **temperature-scaled cosine similarity** between ℓ2-normalized features and K learnable prototypes {c_k} (τ=0.1). Stacked → code sequence Π_t ∈ R^{P×K} (predictor input). Expected embedding z_{t,i} = Σ_k p_{t,i,k} c_k is a convex combination of prototypes (bounded convex hull) — used by the decoder. The codebook itself is EMA-maintained (an "EMA codebook" for the target branch).
- **Dual predictor:**
  - **Fine predictor (micro-dynamics):** Transformer maps Π_t → fine prediction Π̂^fine_{t+1} ∈ R^{P×K}, preserving patch resolution (transient disturbances not smoothed).
  - **Coarse predictor (macro-dynamics):** a learnable query token q aggregates the whole history via cross-attention into a single global prediction Π̂^coarse_{t+1} ∈ R^{1×K}.
- **Auxiliary decoder D:** reconstructs the input fine patches from the online soft-quantized embeddings z_t (after inverting RevIN with cached stats) — anchors signal-level semantics, prevents over-abstract codes.

### Objective
`L = L_pred + L_code + λ_r L_rec` (Eq. 5). Three coupled groups:
- **(A) Predictive (L_pred = λ_f(L^fine_KL + γ L^fine_MSE) + λ_c L^coarse_KL):** KL divergence between EMA-target and predicted code distributions at fine (per-patch) and coarse (window) resolution, plus a fine **MSE** term on the soft-quantized latents. KL is justified because targets are probability vectors on the simplex (principled distributional alignment vs. treating codes as Euclidean regressands); slowly-varying EMA targets stabilize the KL matching.
- **(B) Codebook (L_code):** **alignment** = bidirectional VQ-style commitment, L_emb = Σ‖sg(z)−h‖² and L_com = Σ‖z−sg(h)‖² (stop-grad both ways) to keep prototypes synced with encoder features; plus **dual-entropy calibration** — *minimize* per-sample entropy L^sample_ent = E[H(p)] (sharp, decisive assignments) and *maximize* batch-marginal entropy L^batch_ent = H(E[p]) (diverse code usage, prevents index collapse).
- **(C) Reconstruction (L_rec):** squared error between input patches and their denormalized reconstructions; called a "non-negotiable anchor" against collapse.

### Theory (Appendix A.3)
Two guarantees tied to the codebook radius M = max_k ‖c_k‖:
- **Stability upper bound (Thm A.3):** representation drift ‖ẑ_{t+1}−ẑ_t‖₂ ≤ M(√(2ε_{t+1}) + δ_t + √(2ε_t)), via an ℓ1→ℓ2 Lipschitz bridge (Lemma A.2) + Pinsker's inequality. So drift is *strictly controlled by M*; unbounded continuous models correspond to M→∞. Filters out false positives from trivial numerical perturbations.
- **Non-collapse lower bound (Thm A.10):** under multi-code usage (batch entropy ≥ η), sharp assignments (controlled by τ), and prototype separation Δ_c, the batch covariance trace Tr(Cov({z_i})) ≥ α²(Δ_c − 2Mε)² > 0 — a non-collapse certificate. The three assumptions are exactly what L^batch_ent, L^sample_ent, and L_com/EMA enforce.

### Downstream protocol (App B.4)
Pre-train SSL on the official train set (unlabeled), 9:1 train/val for model selection + early stopping. Then **freeze encoder + codebook** (discard projection heads), aggregate soft-code probs p_t ∈ R^{V×P×K} via **variable-wise max-pooling** → R^{P×K}, flatten, feed an MLP classifier trained with BCE on a chronological **6:2:2** split of the official test set; threshold δ* chosen on val to maximize window-level F1. Window-level label y_{t+1}=1 iff any timestep in the next window is anomalous.

## Key results
Four MTS anomaly benchmarks: **MSL, SMAP** (NASA spacecraft telemetry), **SWaT** (industrial water-treatment ICS), **PSM** (eBay server metrics). Metrics in % (window-level Prec/Rec/F1/AUC), mean over **5 seeds**. Setup: context T_c=100, target T_t=100, stride 100, P=5 patches of L=20; D=256, K=128, τ=0.1, 6-layer Transformer encoder (8 heads), 2-layer Transformer predictors; Adam lr 5e-4, wd 1e-5, batch 128, grad-clip 0.5, ≤100 epochs; single RTX 4090.

**Main table (Table 1, F1 / AUC):**
- **MSL:** MTS-JEPA **33.58 / 66.08** vs. best baseline (TimesNet F1 28.44; TS2Vec AUC 64.86); TS-JEPA 25.49 / 60.33.
- **SMAP:** **33.64 / 65.41** vs. TS2Vec 32.81 / iTransformer 60.91; TS-JEPA 26.57 / 57.38.
- **SWaT:** **72.89 / 84.95** vs. TS-JEPA 71.95 (F1) / TS2Vec 83.76 (AUC).
- **PSM:** **61.61 / 77.85** vs. PAD 57.82 / 73.84.
- MTS-JEPA holds **top AUC on all four and top F1 on all four**. Authors note TS-JEPA's inferior numbers show "the limitations of a pure continuous predictive model; lacking both a discrete codebook and multi-resolution modeling, it suffers from optimization instability."

**Ablations (Table 3 / Table 8, F1 / AUC, "Full Model" = 33.58/66.08, 33.64/65.41, 72.89/84.95, 61.61/77.85):**
- **w/o Codebook Module** is by far the most damaging — collapses to near-random with std≈0 (e.g. MSL **21.82±0.00 / 43.02**, SWaT **11.51±0.00 / 50.00±0.00**, SMAP 21.69±0.00 / 51.00); the paper explicitly reads std≈0-with-low-mean as model collapse.
- **w/o Reconstruction Decoder** also triggers collapse on SWaT (**14.77 / 53.30**) — "non-negotiable anchor."
- **w/o KL** (KL→MSE-only) and **w/o Predictive Objective** each cause large drops (e.g. MSL ~29/53.9 and ~28.9/53.2).
- **w/o Codebook *Loss*** (keep the bottleneck, zero the auxiliary regularizers) is only a mild drop (MSL 31.62/58.93) — so the *structure* of the bottleneck, not its auxiliary losses, is what stabilizes training; the auxiliary losses are an "outperformance mechanism."
- **w/o Downsampling** (drop the coarse view → single-resolution) costs a few F1/AUC points across datasets (MSL 28.77/63.16), validating multi-resolution.

**Generality (Table 2/7):** cross-domain (pre-train on the union of the *other three* datasets, target fully excluded; channels treated univariately) vs. in-domain. MTS-JEPA transfers best of the three studied (vs. PatchTST, TS2Vec): AUC even *improves* cross-domain on MSL (+1.05), SMAP (+3.47); largest drop on hard PSM (AUC −11.05) but still the strongest there.

**Qualitative / efficiency:** a small subset of codes shows large anomaly-vs-normal activation gaps (Fig 4/6) and distinct codes map to distinct physical patch morphologies (Fig 3/7) — the bottleneck learns interpretable regime prototypes. App D.3: MTS-JEPA is **slower than PatchTST** at inference (richer architecture), but the per-sample latency gap shrinks at larger batch sizes — explicitly flagged as a limitation, not a contribution.

## Relevance to the EB-JEPA hackathon
This is a **multivariate time-series (MTS)** track paper, and a near-direct A/B target against the repo's existing time-series entries:
- **vs. TS-JEPA (Ennadir et al. 2025, "Joint embeddings go temporal"):** MTS-JEPA is explicitly positioned as TS-JEPA + (multi-resolution targets) + (soft-codebook bottleneck), with TS-JEPA used as a baseline it beats on all four datasets. This is the cleanest "what does adding a regularized discrete bottleneck buy you over a vanilla continuous TS-JEPA" experiment in the references set.
- **vs. LaT-PFN (`paper/lat-pfn/`):** both are latent-space, energy-/JEPA-style time-series models, but orthogonal — LaT-PFN does *zero-shot univariate forecasting* via PFN in-context meta-learning; MTS-JEPA does *multivariate anomaly prediction* via per-window self-distillation with an EMA target. Different task (forecast value vs. early-warn next window), different conditioning (in-context vs. EMA self-distillation), different output (continuous trajectory vs. simplex code distribution).
- **Anti-collapse axis (the eb_jepa BCS/regularizer slot):** unlike the LeJEPA/SIGReg or VICReg variance-covariance camp, MTS-JEPA's anti-collapse mechanism is the **discrete soft-codebook bottleneck + dual-entropy calibration**, backed by an explicit variance lower-bound certificate (Tr(Cov(z))>0). That makes it a third, distinct anti-collapse strategy to A/B against SIGReg and VICReg in eb_jepa — "constrain the latent to a bounded convex hull of prototypes" vs. "push the embedding distribution toward isotropic Gaussian." A natural hackathon experiment: swap the codebook bottleneck for SIGReg/VICReg on the same TS-JEPA backbone and compare collapse behavior + anomaly AUC.
- **Reusable JEPA mechanics:** EMA target encoder + EMA codebook, stop-gradient, multi-resolution targets from one shared variable-length encoder, KL-on-the-simplex prediction loss, RevIN normalization with cached affine stats — all map onto eb_jepa's encoder/predictor/EMA-target/loss slots.
- **24h replicable slice:** the benchmarks are small (single RTX 4090, ≤100 epochs, batch 128) and standard (MSL/SMAP/SWaT/PSM). A team could reproduce the **w/o Codebook Module → collapse** result as a one-axis ablation, or reproduce the multi-resolution gain (w/o Downsampling) on a single dataset, then plug in SIGReg as a fourth anti-collapse arm.

## Caveats & open threads
- **No repo, no released code** found in the paper; reproduction is from scratch against the described hyperparameters (which are fully specified in App B.2).
- **Anomaly-prediction protocol is non-standard:** results are *not* comparable to the usual point-adjusted anomaly *detection* numbers on these datasets. The early-warning, window-level, frozen-encoder + supervised-MLP-on-6:2:2-test-split setup is the paper's own protocol; baselines were re-adapted to it ("when applicable, baselines are pre-trained on the same data and aligned to the anomaly prediction setting"), so the comparison hinges on the authors' re-implementation of nine baselines.
- **Downstream uses labels on the test split** (6:2:2 chronological split of the *official test set* to train the classifier + pick the threshold); the SSL pre-training is unlabeled, but the headline numbers reflect a supervised probe, not pure zero-shot.
- **Absolute F1 numbers are low** (e.g. MSL/SMAP ~33%), reflecting the hard precursor-prediction task; AUC is the more comparable threshold-free metric and is where the model's lead is most consistent.
- **High variance on some cells** (e.g. MSL Rec 40.80±15.68, PSM Rec 72.00±12.96) — wide 5-seed spreads on Precision/Recall even when F1/AUC lead holds.
- **Theory is conditional:** the non-collapse bound is vacuous if batch entropy is low or sharpness/separation is weak (2Mε ≥ Δ_c); the paper argues App D.2 shows these conditions hold empirically, but the certificate is not unconditional.
- **Efficiency is a stated limitation:** richer multi-branch architecture is slower than a single-backbone baseline (PatchTST); the gap narrows only at larger batch sizes.
- **No head-to-head vs. SIGReg/VICReg-regularized JEPA** — exactly the comparison the eb_jepa hackathon could add.
