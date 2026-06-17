# TS-JEPA — the first systematic adaptation of JEPA to time series, balancing classification and forecasting from a single self-supervised encoder

**Authors:** Sofiane Ennadir (KTH; internship at Flatiron Institute), Siavash Golkar (NYU), Leopoldo Sarra (Flatiron Institute). **Venue/Year:** NeurIPS 2024 Workshop on Time Series in the Age of Large Models (paper title "Joint Embeddings Go Temporal"; arXiv v1 posted 29 Sep 2025). **arXiv:** 2509.25449 **Repo:** https://github.com/Sennadir/TS_JEPA

## TL;DR
TS-JEPA ports the image/video JEPA recipe (encoder + predictor + EMA target encoder, latent-space masked prediction) to univariate time series. It patchifies a series with a 1D-CNN tokenizer + sin-cos positional encoding, masks 70% of patches, and predicts the EMA-encoder's embeddings of masked patches from the encoded visible patches under an L1 loss. On a frozen-encoder linear-probe protocol it "outperforms both contrastive and autoregressive approaches in the majority of classification tasks" and is "comparable" to MAE, while staying competitive on forecasting, where a pure autoregressive model normally dominates. The pitch: one architecture that does not trade one task off against the other.

## Problem & motivation
Masked/autoregressive (input-space reconstruction) SSL for time series is "susceptible to the presence of noise and other non-predictable confounding factors," because good reconstruction requires modeling the entire input including noise, "rather than the underlying patterns." JEPA's latent-space prediction is argued to be robust to such confounders (per LeCun's energy-based vision [12], I-JEPA [1], V-JEPA [2]). Time series are "often inherently noisy," so the authors hypothesize JEPA transfers well. They position this as "the first systematic study of the JEPA architecture to time series," whereas prior JEPA-on-time-series work was either a narrow application to encoded frames [8] or combined with other techniques for in-context prediction (LaT-PFN [19]).

## Method
Four components (Fig. 1), univariate but "easily adaptable to multivariate":
1. **Tokenizer:** splits x into non-overlapping patches, embeds each with a 1D-CNN to capture local temporal patterns; adds absolute sin-cos positional embeddings. Patches are split into masked P_M and non-masked P_N via a **uniform masking strategy**.
2. **Encoder E_theta:** standard self-attention transformer; encodes only the visible patches, z_N = E_theta(P_N).
3. **Predictor P_beta:** transformer that maps encoded visible tokens to predictions of the encoded masked tokens, z'_M = P_beta(E_theta(P_N)).
4. **EMA-Encoder Ebar_theta:** weight-EMA copy of the encoder, encodes the masked patches to produce targets t_M = Ebar_theta(P_M); prevents collapse (BYOL [10]-style slow-moving target).

Loss (L1 in latent space over masked indices): `L = (1/|M|) sum_{i in M} || P_beta(E_theta(P_N))_i - Ebar_theta(p_i) ||_1`.

**Config (Appendix):** encoder and predictor share architecture = transformer with **2 attention heads, embedding dim 128**; **10 patches** per series; **batch size 32**; **masking ratio 70%** for TS-JEPA (75% for the MAE baseline); EMA momentum **m = 0.998**; AdamW, LR swept over {1e-3, 1e-4, 1e-5, 1e-6}; single **NVIDIA V100**. Evaluation = frozen encoder + small trained classification/regression head (linear probe). Long-term forecasting is done by autoregressively rolling out next-patch predictions over a horizon.

## Key results
**Classification (accuracy +/- std over 10 runs, Table 1; TS-JEPA vs. baselines vs. supervised upper bounds):**
- FordA: TS-JEPA **91.5 +/- 0.1**, beating TS2Vec 86.4, MAE 85.1, Auto-regressive 69.6, and matching/exceeding the fully-supervised Transformer (91.8) and CNN (86.8).
- FordB (transfer, pretrain FordA -> eval FordB): TS-JEPA **73.8 +/- 0.3**, best among SSL (TS2Vec 72.4, MAE 59.6, AR 61.9) and near supervised Transformer 74.8.
- FaultDetectionA: TS-JEPA 85.8 vs MAE **90.4**, TS2Vec 83.9, AR 81.6 (MAE wins here; supervised CNN 98.4).
- FaultDetectionB (transfer): all methods weak; TS-JEPA 50.6, TS2Vec 53.9 = MAE 54.3 (tie), AR 51.4.
- ECG5000: TS-JEPA 89.5 vs MAE **91.6**, AR 87.5, TS2Vec 86.9.
- A frozen **randomly-initialized** encoder ("Classification Head" only) is far worse (e.g. FordA 46.3, FaultDetectionA 54.3), confirming the pretraining actually learns useful structure.

**Label efficiency (Fig. 2):** with only 5-20% of labels used downstream (rest used for pretraining), TS-JEPA beats a fully-supervised Transformer; the gap shrinks as labels increase.

**Short-term forecasting (next-patch, MSE/MAE, Table 2):** autoregressive wins on ETT-Small (AR 0.009 MSE vs JEPA 0.017) and Electricity (AR 0.010 vs JEPA 0.014); JEPA slightly better on Weather (JEPA 0.015 vs AR 0.022 MSE). "As anticipated, the autoregressive approach outperforms TS-JEPA in short-term forecasting," consistent with AR's training paradigm.

**Long-term forecasting (cumulative MSE over ~100 rollout steps, Fig. 3):** both methods show roll-out error amplification, but "TS-JEPA demonstrates superior performance to autoregressive strategies in two out of three datasets (i.e. ETT and Electricity), suggesting an enhanced stability."

**Overall framing:** TS-JEPA "does not consistently outperform all baselines on every dataset and task," but "by maintaining competitive forecasting capabilities and outperforming on classification, it offers a compelling trade-off between the two tasks for using a single architecture."

## Relevance to the EB-JEPA hackathon
Directly on the **time-series track**. This is the canonical "vanilla JEPA, but for 1D temporal sequences" baseline: it is the simplest faithful transposition of the I-JEPA/V-JEPA template (1D-CNN patch tokenizer, transformer encoder/predictor, EMA target, masked latent prediction) with an **L1** rather than L2 prediction loss and collapse prevented purely by the EMA target (no explicit variance/covariance/energy term). For an energy-based EB-JEPA variant on time series, TS-JEPA is the natural starting scaffold and ablation reference: swap its EMA-only anti-collapse for an explicit energy/regularizer, or its L1 latent loss for an EB objective.

**Contrast with LaT-PFN** (Verdenius et al., 2024, arXiv:2405.10093, their ref [19]): LaT-PFN is a JEPA *combined with prior-fitted-network in-context forecasting* -- a more specialized, forecasting-centric system. TS-JEPA is deliberately the opposite: a minimal, general-purpose JEPA studied *systematically* across both classification and forecasting, explicitly claiming the first such study and emphasizing the cross-task balance rather than peak forecasting accuracy. So in a hackathon lineup, LaT-PFN ~= the heavyweight forecasting-specialized JEPA, TS-JEPA ~= the clean general-representation JEPA baseline to build/compare against.

## Caveats & open threads
- **Workshop-scale, small models.** Fixed tiny backbone (2 heads, dim 128, 10 patches, batch 32, single V100); authors explicitly ran "no ablation study on this configuration" and defer architecture choices and **scaling** to future work. The "foundation model" framing is aspirational, not demonstrated at scale.
- **Univariate only** in experiments; multivariate is claimed easy but untested.
- **Mixed wins:** beaten by MAE on FaultDetectionA and ECG5000, by autoregressive on short-term forecasting; the claim is balance, not dominance.
- **Anti-collapse rests entirely on the EMA target** (m=0.998) with no explicit collapse-prevention term and only an informal note that it "empirically" avoids collapse -- no collapse diagnostics reported.
- **Masking is uniform/random** over 70% of 10 patches; no exploration of block/temporal-causal masking strategies that might matter more for sequential structure.
- Long-term forecasting uses autoregressive rollout of the patch predictor, so it inherits roll-out error amplification.
- Loss is **L1** (note: differs from the L2 typical of I-JEPA/V-JEPA) -- an under-discussed design choice worth probing.
- Dataset table lists ECG5000 as "ECG500" in places (typo); FaultDetectionA/B share identical train/test sizes (10912/2728) suggesting a shared split.
