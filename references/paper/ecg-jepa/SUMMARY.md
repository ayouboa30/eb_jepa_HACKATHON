# ECG-JEPA — A 12-lead ECG JEPA that predicts masked patches in latent space with a clinically-motivated Cross-Pattern Attention, beating contrastive/MAE SSL on diagnosis, feature regression, and segmentation

**Authors:** Sehun Kim (Samsung Medical Center) **Venue/Year:** arXiv preprint, v5 dated 10 Apr 2026 (first posted Oct 2024) **arXiv:** 2410.08559 **Repo:** https://github.com/sehunfromdaegu/ECG_JEPA

## TL;DR
ECG-JEPA ports the Joint-Embedding Predictive Architecture to 12-lead ECG: it splits a multi-lead recording into per-lead temporal patches, masks all leads at the same temporal positions (synchronized temporal masking), and trains a student transformer + small per-lead predictor to regress the EMA-teacher's latent embeddings of the masked patches via an **L1 loss**. Two ECG-specific design elements carry the paper: (1) **Cross-Pattern Attention (CroPA)**, a masked self-attention that only lets a patch attend within its own lead or its own temporal column, encoding the clinical "compare leads at the same instant" inductive bias; and (2) latent prediction instead of raw-signal reconstruction, which avoids forcing the model to reproduce ECG noise. Pretrained on ~180k 12-lead ECGs (Chapman + Ningbo + CODE-15), it reaches "state-of-the-art performance in various downstream tasks including diagnostic classification, feature extraction, and segmentation," and is notably strong in low-shot and reduced-lead regimes. Uses only 8 leads (I, II, V1-V6) since the other 4 are linear combinations.

## Problem & motivation
Supervised ECG models "often face significant performance degradation when applied to data distributions different from those on which they were trained," and medical labels are scarce/costly, so SSL is attractive. But SSL transfers poorly to ECG out of the box: standard CV augmentations (rotation, scaling, flipping) "can distort the physiological meaning of ECG signals," and autoencoder/MAE-style reconstruction "may cause autoencoder-based SSL models to struggle with reconstructing raw signals" because ECG "recordings often contain artifacts and noise," and may "miss visually subtle but diagnostically critical features, such as P-waves and T-waves." JEPA sidesteps both: it predicts in latent space (so it never has to reconstruct unpredictable noise) and needs no augmentations. The stated contributions: (1) an ECG-specific JEPA with synchronized temporal masking, a lead-wise predictor, and CroPA; (2) broad empirical evidence of transferable representations across linear eval, fine-tuning, reduced-lead, low-shot, and noisy settings; (3) showing the representations support ECG **feature regression** and **segmentation**, not just classification — claimed as "the first work to jointly evaluate ECG feature prediction and ECG segmentation."

## Method
### Input & patching
- 10-second multi-lead ECG resampled to 250 Hz → T = 2500 time points. Interval [0, T) split into N = 50 non-overlapping subintervals of length t = 50. Each (lead, subinterval) is a patch x_{c,i} ∈ R^t, giving C × N patches; a linear layer projects each to a D = 768 token plus positional embeddings. **2-D sinusoidal** positions for student/teacher, **1-D sinusoidal** for the predictor.
- Only **8 leads** (I, II, V1-V6) are used; III, aVR, aVL, aVF are reconstructed from Einthoven's law (e.g. III = II - I), giving comparable accuracy to the full 12-lead model (Table 12) at lower cost.

### Masking (synchronized across leads)
Because same-time patches across different leads are highly correlated, masking is applied to **all leads at the same temporal positions** (otherwise the task is trivially easy). Two strategies:
- **Random masking (rb):** randomly select a fraction of subintervals; ratio sampled in (0.6, 0.7).
- **Multi-block masking (mb):** select several (possibly overlapping) consecutive subinterval blocks; ratio (0.175, 0.225) at frequency 4, forcing prediction of longer contiguous spans.
Masked indices I_msk, visible indices I_vis partition [N]. Visible patches feed the student; masked patches are the latent targets.

### Teacher, student, predictor
Standard I-JEPA / data2vec-style asymmetric pair:
- **Teacher** (12 layers, 16 heads, D = 768): sees all C × N patches → fully contextualized targets z_{c,i}. Updated by **EMA** of student weights — the anti-collapse mechanism (no VICReg/SIGReg variance-covariance term).
- **Student** (same 12L/16H/768): sees only visible patches; its outputs are concatenated with learnable mask tokens z_msk to restore C × N positions.
- **Predictor** (smaller: 6 layers, 12 heads, D = 384): operates **per lead** (single-channel sequences) to produce predictions ẑ_{c,i}, but via self-attention each lead's tokens still encode information from all leads.
- **Loss:** L = Σ_c (1/|I_msk|) Σ_{i∈I_msk} ‖ẑ_{c,i} − z_{c,i}‖_1 — L1 distance on masked positions only.
- **EMA schedule:** teacher = β·teacher + (1−β)·student with β ramped linearly from ema0 = 0.996 to ema1 = 1.0 over training.
- **Inference:** only the student encoder is used; its patch outputs are **average-pooled** into a single D = 768 ECG representation for downstream tasks.

### Cross-Pattern Attention (CroPA)
A masked self-attention that restricts a patch x_{c,i} to attend to x_{c',i'} **iff** same lead (c = c') **or** same temporal column (i = i'). This mirrors how clinicians read ECGs (e.g., reciprocal changes across leads at the same instant), injecting a structured inductive bias vs. dense self-attention. It is the paper's headline architectural novelty alongside synchronized masking.

### Training setup
100 pretraining epochs, **no augmentation and no noise removal** (trained on raw noisy signals by design), AdamW, cosine LR with 5 warmup epochs, drop-path 0.1. ECG-JEPA_rb: LR 2.5e-5, batch 128; ECG-JEPA_mb: LR 5e-5, batch 64 (higher memory). Trained on a **single NVIDIA RTX 3090**. PyTorch 2.3 / CUDA 11.8 / Python 3.10.

## Key results
Pretrain: Chapman + Ningbo (43,240 after cleaning) + CODE-15 (130,900 10-s recordings) ≈ 180k ECGs. Downstream: **PTB-XL**, **CPSC2018**, **G12EC**. Baselines: ST-MEM, SimCLR, ECG-FM, KED (run directly), plus MoCo v3 / MTAE / MLAE (scores cited from Na et al.). Metrics: AUC (multi-label avg of binary; multi-class one-vs-rest) and macro-F1.

- **Linear evaluation (Table 1, multi-label / multi-class AUC):** ECG-JEPA_mb tops most cells — PTB-XL **0.912 / 0.903**, CPSC2018 **0.966 / 0.973**, G12EC **0.895 / 0.908**, consistently above ST-MEM, SimCLR, ECG-FM, KED.
- **Reduced-lead (Table 2):** with only Lead II (1-lead) ECG-JEPA_mb hits **0.849 AUC** (vs ST-MEM 0.831); with II+V1 (2-lead) **0.879 AUC / 0.641 F1** — strongest, relevant to mobile/wearable monitoring.
- **Low-shot (Table 3, PTB-XL multi-label macro AUC):** at **1%** data ECG-JEPA_rb **0.839 ± 0.002** (best; ST-MEM 0.817, ECG-FM 0.729); at **10%** ECG-JEPA_mb **0.893 ± 0.001**. The advantage is largest in the scarce-label regime.
- **Fine-tuning (Table 4):** ECG-JEPA competitive-to-best — PTB-XL multi-label **0.931** (rb), multi-class **0.934** (mb); CPSC2018 multi-class **0.980** (mb). Supervised-from-scratch baseline lags (e.g. PTB-XL ML 0.878).
- **ECG feature regression (Table 5, MAE):** heart-rate MAE **0.40 ± 0.67 BPM** (mb) — far below ST-MEM 0.68 and ECG-FM 2.67 (test mean HR 69.67 ± 12.92 BPM); QRS-duration MAE ~1.9 ms (rb best among JEPA, ST-MEM slightly lower at 1.42).
- **Segmentation (Table 6, P/QRS/T/none IoU on PTB-XL):** fine-tuned ECG-JEPA_rb best **mIoU 0.954**; frozen-linear already 0.888.
- **Noise robustness (Fig. 7):** across noise levels 0→2, both JEPA variants stay most stable; the inter-method gap widens with noise. KED slightly beats JEPA at level 0 but "drops more sharply under stronger corruption."
- **CroPA ablation (Table 7):** CroPA consistently improves multi-class AUC (e.g. G12EC multi-block lin 0.832→0.908); bootstrapped 95% CIs for ΔAUC are strictly positive on all three datasets (App. A.4) — modest but statistically significant.
- **Nearest-neighbor classifier (Table 10):** training-free NCC, ECG-JEPA still wins (e.g. CPSC2018 acc 0.707 rb).
- **Masking-ratio ablation (Table 11):** high mask ratios help; mb (0.175,0.225)@freq4 → 0.912 AUC; random (0.7,0.8) marginally best (0.909) but (0.6,0.7) chosen for consistency.
- **Scaling to MIMIC-IV-ECG (App. A.1, Tables 8-9):** adding MIMIC-IV-ECG (~780k usable) to reach ~960k pretrain samples gave a **slight drop in linear eval** and roughly unchanged fine-tuning, attributed to MIMIC's acute/ICU distribution bias and no model-size increase — i.e. no clean scaling win observed here.

## Relevance to the EB-JEPA hackathon
This is a strong fit for the **wearable / biosignal track (Track 3)** — a 1-D **multi-lead (multi-channel) time-series JEPA**, complementary to the EEG papers (S-JEPA, Brain-JEPA, Laya) but on cardiac signals. It maps cleanly onto the eb_jepa recipe:
- **Encoder/predictor:** linear-patch tokenizer + 12-layer transformer student/teacher + small 6-layer per-lead predictor — the standard encoder + predictor slots, with average-pooling readout.
- **Anti-collapse:** firmly in the **EMA-target + stop-gradient (I-JEPA/data2vec) camp** with an **L1 latent loss** (β ramp 0.996→1.0), *no* VICReg/SIGReg term. This is the obvious A/B axis for the hackathon — drop the EMA target and swap in eb_jepa's BCS/SIGReg variance-covariance regularizer on a two-view or single-view ECG setup, and compare anti-collapse mechanisms directly.
- **Transferable tricks:** (a) **synchronized temporal masking** across channels (mask the same time columns in every lead) — the multi-channel-sensor analogue of patch masking that avoids a trivially-easy task; (b) **CroPA**, a same-lead-or-same-time attention mask — a cheap, domain-grounded structured-attention inductive bias that any multi-channel JEPA could try.
- **24h-replicable slice:** code is released (github.com/sehunfromdaegu/ECG_JEPA) and the model trains on a **single RTX 3090** at ~180k samples. The cleanest reproduction is *pretrain (Chapman/Ningbo/CODE-15) → frozen linear probe on PTB-XL multi-label*, which already shows the gap over ST-MEM/SimCLR. The reduced-lead (II / II+V1) and 1%/10% low-shot results are the most hackathon-friendly headlines (wearable single-lead, label scarcity). Sweep masking strategy (rb vs mb) and toggle CroPA as the two primary ablation knobs.

## Caveats & open threads
- **Single author / single-GPU scale:** D = 768, 12-layer encoder, ~180k samples — small by foundation-model standards; the MIMIC-IV scale-up (~960k) did **not** improve linear eval and was run "all at once" with no model-size increase, so the paper has **no positive scaling result** for ECG JEPA yet (flagged as preliminary).
- **No VICReg/SIGReg baseline:** collapse handled purely by EMA + stop-grad; the regularizer-based alternative central to eb_jepa is untested — exactly the gap a team could fill.
- **rb vs mb is task-dependent:** multi-block better for linear eval, random better for fine-tuning and NCC; no single masking strategy dominates, so any reproduction must pick per-task.
- **Segmentation/feature ground truth is pseudo-labeled:** HR/QRS targets and segmentation masks come from a separate publicly-available segmentation model [43], not gold clinical annotations; Fig. 6 even notes the fine-tuned head appears to *correct* some pseudo-label errors, so the regression/segmentation numbers measure agreement with a model, not clinicians.
- **8-lead reduction is an assumption:** derivable-lead argument (Einthoven) holds in theory but the derived leads can differ on noisy real data; the paper shows 8-lead ≈ 12-lead only on PTB-XL multi-label linear eval.
- **Versioning:** the bare arXiv PDF served here is **v5, dated 10 Apr 2026** (the work is from Oct 2024); numbers quoted are from this latest revision. Related work and baseline tables (ECG-FM, KED, etc.) reflect that updated version.
