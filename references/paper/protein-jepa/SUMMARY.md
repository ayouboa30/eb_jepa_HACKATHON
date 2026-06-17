# ProteinJEPA — latent prediction *complements* (not replaces) MLM for protein language models

**Authors:** Dan Ofer, Michal Linial, Dafna Shahaf (Hebrew University of Jerusalem) **Venue/Year:** Preprint (arXiv), 2026 **arXiv:** 2605.07554 **Repo:** https://anonymous.4open.science/r/protJepa-FF24 (anonymized for review)

## TL;DR
Protein sequence encoders are trained almost exclusively with masked language modeling (MLM = predict the masked amino-acid token). This paper asks whether a JEPA-style **latent-space** prediction loss can *improve* on token-level MLM under a **matched wall-clock budget**. The winning recipe, dubbed **masked-position MLM+JEPA**, is *not* the literal vision/world-model JEPA port. It keeps the MLM cross-entropy AND adds a cosine latent-prediction loss applied **only at the masked positions** (the same positions MLM scores), with detached (stop-gradient, no-EMA) targets + SIGReg. On a 16-task downstream suite under an 8 h budget it beats MLM-only continuation 10/3/3 (W/L/T) on ESM2-35M and 11/2/3 on ESM2-150M. Two negative controls clarify the result: all-position MLM+JEPA only reaches macro parity, and JEPA-only (no MLM) "collapses in nearly every experiment."

## Problem & motivation
- MLM is the default objective for sequence-only protein language models (PLMs: ESM2, ProtTrans, ProteinBERT, AMPLIFY, ProGen2). It is cheap and strong, so it is a hard baseline to beat.
- JEPA (I-JEPA, V-JEPA, LLM-JEPA, JEPA-DNA) replaces input reconstruction with latent-target prediction and has beaten reconstruction in vision/video. Recent LeJEPA/LeWorldModel work simplified collapse-prevention to **detached targets + SIGReg, no EMA teacher**.
- Open question: does latent prediction help for **proteins**, where inputs are discrete, the vocabulary is tiny (20 amino acids), and statistics differ from images/NL? The bar is "improve downstream **without** excessive added compute," explicitly ruling out PSSM/MSA-style heavy inputs.
- "To our knowledge, this is the first controlled study of JEPA-style latent prediction for protein language models under matched MLM baselines."

## Method
Four self-supervised objectives share architecture, optimizer, corpus, and matched wall-clock budget:
- **MLM-only (reference).** Standard bidirectional masked-token cross-entropy: 20% positions masked with 80/10/10 mask/random/keep; matched continuation defines the baseline.
- **Masked-position MLM+JEPA (primary recipe).** `L = L_MLM + λ·L_JEPA^masked + α·L_reg` (Eq. 1).
  - `L_JEPA^masked`: **cosine-similarity** loss between the student's hidden states **at the masked positions only** and **detached** target representations from the *same* backbone applied to the **unmasked** input (stop-gradient; **no separate EMA teacher**).
  - Student hidden states pass through a **two-layer SwiGLU predictor** (expansion 8/3, no bias, LayerNorm on both predictions and targets) before the JEPA loss.
  - `L_reg` = **SIGReg** (Maes et al. 2026) with **256 random projections**, regularizing the predictor output toward a standard Gaussian (anti-collapse).
  - `λ = 0.45`, `α = 1.0`, inherited from the all-position recipe sweep (Appendix A.3) and *not* re-tuned after switching to masked positions.
  - Design rationale: restricting the latent loss to masked positions preserves the identity-recovery training that makes MLM effective, swapping identity recovery for *latent* recovery as the auxiliary signal. "Retaining the MLM term was crucial to performance."
- **All-position MLM+JEPA (control).** Same combined loss but latent loss at **all** non-padding positions with **MSE** instead of cosine. (Headline contrast thus varies both target-set and loss-form, an acknowledged confound.)
- **JEPA-only (control).** All-position latent prediction + SIGReg, **no MLM cross-entropy**. An EMA-teacher classic MLM+JEPA variant was also tested.

**Tokenization.** All backbones use a single-character amino-acid tokenizer (no structure tokens; sequence-only).

**Backbones (5 families, the relevant pretraining regimes).** (i) ESM2-35M pretrained (`esm2_t12_35M_UR50D`); (ii) ESM2-150M (`Synthyra/ESM2-150M`); (iii) AMPLIFY-120M (modern PLM, different corpus); (iv) ESM2-35M random-init; (v) **ProteinBERT2-35M** random-init, a custom 12-layer/hidden-512/8-head encoder with RoPE, RMSNorm, SwiGLU FFN, a 3-layer depthwise-separable conv stem, and alternating local/global attention (window 256), inspired by ProteinBERT + ModernBERT.

**Training.** UniRef50 (same data as ESM2), BF16, single A100-80GB, FlashAttention-2, seq len 512. AdamW, lr `3e-4` (from-scratch) / `3e-5` (continued), 1000 warmup steps, weight decay 0.01, effective batch 128 (192 ProteinBERT2, 208 AMPLIFY). Checkpoints at {1, 4, 8} h. They **match wall-clock, not steps** — the JEPA branch costs ~1.8× per step, so in 8 h on 35M models MLM-only does ~160K steps vs ~90K for MLM+JEPA (a setup harsh toward MLM+JEPA).

**Evaluation.** 16-task suite = 15 frozen **linear probes** on mean-pooled embeddings (TAPE + ProteinBERT-style + public splits, spanning function, structure, interaction, localization, physicochemical) plus **zero-shot SCOPe-40 fold retrieval** (cosine Recall@k on L2-normalized mean-pooled embeddings). Within-family masked-pos vs MLM-only uses one-sided binomial sign tests (α=0.05); all-pos vs MLM-only uses paired Wilcoxon with Holm–Bonferroni over the 5×3 cells.

## Key results
- **Masked-position MLM+JEPA beats matched MLM-only (8 h):** ESM2-35M **10/3/3** (sign-test `p=0.046`), ESM2-150M **11/2/3** (`p=0.011`); both reject H0. From-scratch is architecture-sensitive: ProteinBERT2-35M **11/4/1** (`p=0.059`, macro Δ `+0.127`) but random-init ESM2-35M only **6/8/2** (`p=0.79`, though macro Δ `+0.148` over its random init). AMPLIFY-120M near-neutral **7/6/3**.
- **Gains concentrate on regression/fitness + structural retrieval.** Across the 6-task regression slice, pooling the two pretrained-ESM checkpoints gives **9 wins / 2 losses / 1 tie**. ProteinBERT2-35M: Stability **+12.0** and β-lactamase **+6.8** absolute Spearman points vs MLM-only.
- **SCOPe-40 Recall@1 lift (masked-pos vs MLM-only):** **+5.3 pp** (ESM2-35M), **+8.1 pp** (ESM2-150M), **+8.7 pp** (ProteinBERT2-35M), **+1.7 pp** (AMPLIFY-120M); roughly flat (**−1.2 pp**) on random-init ESM2.
- **Per-task winners (n=5 backbones):** β-lactamase 4/0/1 (median Δ +0.059), Solubility 4/0/1, SCOPe-40 4/1/0 (+0.053), CheZoD disorder 4/1/0 (+0.029), Variant Effect 4/1/0 (+0.024), Stability 4/1/0 (+0.020). **Consistent losers:** Fluorescence (TAPE) **0/5/0** (median −0.006), Peptide-HLA Binding 2/3/0.
- **All-position control = macro parity only.** `|ΔΔ| ≤ 0.04` on every pretrained backbone vs MLM-only; per-cell wins exist (e.g. Stability +11.6, β-lactamase +12.1, SCOPe +4.5 pp) but do **not** aggregate. Pooled at 8 h, masked-pos beats all-pos on **60/10/10** of 80 (backbone, task) cells (one-sided binomial `p < 1e-6`, median Δ `+0.030`).
- **JEPA-only collapses.** Family-mean Δ vs off-the-shelf at 8 h: **−0.250** (ESM2-35M), **−0.235** (ESM2-150M), **−0.198** (AMPLIFY-120M), −0.131 (ProteinBERT2); worst on identity-dependent tasks (Stability −0.53, CheZoD −0.53, EC −0.48, Remote Homology −0.41). Interpretation: without MLM, the model "fails to learn good representations... trapping it in a local minima of trying to learn random representations."
- **Step-matched diagnostic:** at ~91K steps the all-pos MLM+JEPA checkpoint reaches paired Δ `+0.012` vs MLM-only (vs ΔΔ `−0.042` at 8 h wall-clock), i.e. step-matching over-credits MLM+JEPA — they deliberately report the harsher wall-clock view.

## Relevance to the EB-JEPA hackathon
- **New modality: biomolecular sequences (proteins).** Extends the EB-JEPA / energy-based JEPA story beyond vision/video/audio/DNA to amino-acid sequences, a discrete, small-vocabulary regime where masked-token prediction is already a strong baseline.
- **Directly reuses the EB-JEPA-adjacent collapse-prevention stack:** **SIGReg + detached targets + no EMA teacher** (LeJEPA / LeWorldModel lineage). It is a concrete data point that this no-teacher, SIGReg-regularized JEPA recipe transfers to a new modality, and that **anti-collapse regularization (SIGReg, 256 projections) is load-bearing** — JEPA-only collapses without the auxiliary MLM identity signal.
- **Headline design lesson for JEPA practitioners:** the literal "predict latents everywhere" port is *not* optimal for proteins; restricting the latent loss to **masked positions** and **keeping a token-level (MLM) objective** is what unlocks the gains. A cautionary tale that JEPA-only is not a drop-in replacement and that hybrid token+latent objectives can dominate.
- Cheap, reproducible setup (single A100, 8 h, public UniRef50 + public benchmarks) — a plausible hackathon-scale track for a protein/biomolecular JEPA experiment.

## Caveats & open threads
- **Loss-form vs target-set confound (authors flag this):** the headline masked-pos recipe uses **cosine**, the all-pos control uses **MSE**, so masked-pos gains cannot be attributed to target selection alone. The clean ablations (all-pos cosine, masked-pos MSE) were **not run**.
- **Evaluation is linear probes on static mean-pooled embeddings only** — no fine-tuning, no per-residue tasks. SCOPe-40 retrieval is where the geometric benefit is clearest.
- **From-scratch is unreliable at this scale** (works for ProteinBERT2, mixed/negative for vanilla ESM2). Continued pretraining is the clear use case; improved from-scratch protocols (delayed JEPA schedule, separate JEPA lr) left to future work.
- **Limited statistical scope:** not all cells have repeated pretraining seeds; reported p-values are **task-level, not seed-level**. Pretrained baselines are heavily overtrained (~53 passes over UR50).
- **Scope:** sequence-only PLMs at **35–150M** params. Multimodal PLMs (SaProt, ESM-3, PTM-Mamba) and asymmetric representation-learning variants are out of scope.
