# T-JEPA — augmentation-free JEPA for tabular data: predict the latent of one feature-subset from another within the same row

**Authors:** Hugo Thimonier, José Lucas De Melo Costa (equal), Fabrice Popineau, Arpad Rimmel, Bich-Liên Doan (Université Paris-Saclay / CentraleSupélec, LISN; Emobot). **Venue/Year:** ICLR 2025 (v3, 3 May 2025). **arXiv:** 2410.05016 **Repo:** https://github.com/jose-melo/t-jepa

## TL;DR
T-JEPA adapts I-JEPA to structured tabular data: it predicts, in latent space, the representation of one subset of a row's features from the representation of a different, non-overlapping subset of the *same* row. This sidesteps the central pain point of SSL on tables — that meaningful data augmentations are "non-trivial" to construct and easily "generate samples outside the data manifold." Used as a pretraining step, it lifts five deep classifiers (MLP, DCNv2, ResNet, AutoInt, FT-Transformer) so that, augmented by T-JEPA, "most methods consistently outperform GBDT or match their performance" (XGBoost/CatBoost). Headline ranking on 7 datasets: ResNet+T-JEPA has the best avg rank (2.6), MLP+T-JEPA second (3.9), beating PTaRL, SwitchTab, VIME, BinRecon, SubTab. A novel **regularization token [REG]** is found to be "critical to escape collapsed training regimes."

## Problem & motivation
SSL normally needs two views of a sample, which requires augmentations. For images/text these are easy (crop, rotate, token-mask); for tabular data they are domain-specific and risk leaving the data manifold. Tabular data is also heterogeneous (mixed numerical/categorical), and NNs notoriously trail GBDTs there. T-JEPA's pitch: do mask-reconstruction *in latent space* (à la I-JEPA), with no augmentation, extending the "mask reconstruction is a relevant pretext task for tabular data" line (SubTab, VIME) from data space to embedding space.

## Method
Three transformer modules: context encoder `fθ`, target encoder `f̄θ`, predictor `gϕ`.
- **Embedding/masking.** Numerical features normalized to 0 mean/unit var; categoricals one-hot. Each feature `j` gets its own learned `Linear(eⱼ, h)` into hidden dim `h`, plus learned index- and feature-type embeddings. A mask vector `m ∈ {0,1}ᵈ` *drops* masked features (sequence shrinks): masked context input is `zₓᵐ ∈ ℝ^{lₘ×h}` where `lₘ = d − ‖m‖₁`.
- **Asymmetric masking.** Several context masks `M_context` and target masks `M_target` per row. The context encoder sees a *masked* row; the target encoder sees the *full* unmasked row `zₓ^{0_d}`, then its output is sliced by each target mask (Eq. 4). Intra-set overlaps allowed, but a context mask may NOT inter-overlap a target mask (non-overlapping subsets). Min/max masking shares sampled per row (ctx ~0.07–0.9, trgt ~0.05–0.9).
- **Prediction & loss.** `gϕ` (its own transformer, downsized hidden dim) takes the context output plus a target mask (learnable mask tokens + positional embedding) and predicts each target's latent; run `|M_target|` times per context output. Loss is mean ℓ2: `L = (1/|M_target|)(1/|M_context|) Σ Σ ‖gϕ(h_context^m, mₖ) − h_target^{mₖ}‖₂²` (Eq. 5).
- **Anti-collapse.** Target encoder is an EMA of the context encoder (decay 0.996→1) with stop-gradient — but T-JEPA found this *insufficient* on tables. The **[REG] token** (inspired by ViT register tokens, Darcet et al. 2024) is appended to both context and target representations (never masked), discarded before the predictor and at downstream time. It is what lets training escape the initial collapsed equilibrium.
- **Downstream use.** Discard target encoder + [REG]; use the trained context encoder's `ℝ^{d×h}` representations, adapted to each model via a jointly-trained projection (linear-flatten, linear-per-feature, conv, max/mean pool).

## Key results
- **vs raw data (Table 1).** T-JEPA improves every model on nearly every dataset. E.g. MLP on Adult 0.827→0.866, Jannis 0.672→0.728, ALOI 0.916→0.961; ResNet on California (RMSE↓) 0.534→0.441. Higgs is the weak spot (most models drop; only ResNet improves there).
- **vs SSL baselines (Table 2, avg rank over 7 datasets, lower better).** ResNet+T-JEPA **2.6** (best), MLP+T-JEPA **3.9**, MLP+SwitchTab 4.3, MLP+PTaRL 5.1; SubTab 6.3–8.1, VIME 7.4–7.6, BinRecon 6.9–11.7. Raw MLP/ResNet 9.4/10.4.
- **vs GBDT.** Raw deep nets are "significantly outperformed by both GBDT methods," but T-JEPA-augmented deep nets "regularly obtain the best performance," with ResNet+T-JEPA beating XGBoost/CatBoost on most datasets.
- **Representation quality.** Over training on Jannis, uniformity 3.12→11.38, pairwise KL 9.3e−4→9.38e−2, pairwise ℓ2 5.83→70.0 — i.e. it escapes collapse and spreads out. Unsupervised feature relevance: T-JEPA embedding-variance ranking correlates with XGBoost importance (Kendall τ = 0.44, p = 1.73e−6) with no labels.
- **[REG] ablation.** Without any [REG] token, training on Jannis stays stuck in the initial collapse; one or more [REG] tokens let it escape. Main experiments use a single [REG] token.
- **Cost.** Pretraining 0.34–4.80 GPU-hours per dataset on one A100-40GB; batch size 512; AdamW + cosine annealing; 4 prediction masks.

## Relevance to the EB-JEPA hackathon
- **New modality = tabular.** A clean template for porting JEPA to a non-spatial, non-temporal, heterogeneous modality: there is no patch grid, so "masking" = dropping a random non-overlapping subset of *columns*, and the asymmetric context-vs-target masking is over feature subsets, not image blocks. The per-feature learned linear embedders + index/type embeddings are the modality-specific front-end worth reusing.
- **Anti-collapse choice (central to EB framing).** T-JEPA's notable empirical finding: EMA + stop-gradient alone, which suffices for I-JEPA/V-JEPA, **does not prevent collapse on tables** — the loss crashes to ~0. Their fix is architectural (a register-style [REG] token) rather than a loss term (no VICReg-style variance/covariance penalty). This is a useful contrast for any energy-based / anti-collapse analysis: an "implicit, plumbing" anti-collapse mechanism vs an explicit regularizer, plus quantitative collapse diagnostics (uniformity, pairwise KL/ℓ2) that EB-JEPA could borrow.

## Caveats & open threads
- The collapse claim leans on a single ImageNet-1K I-JEPA re-run at a *drastically reduced batch size of 16* (rest of hyperparameters unchanged) "for computational purposes"; whether the "JEPA initially collapses" narrative generalizes is not stressed at scale.
- *Why* [REG] works is empirical only — no theory; authors explicitly call for "better theoretical insight into why non-contrastive SSL works."
- Representations are tailored to transformer-style consumption (`ℝ^{d×h}` + projection layer); the paper flags adapting to other architectures as future work.
- Only 7 datasets, max ~108k rows; no large-scale or cross-table foundation-model setting. Higgs regresses under T-JEPA for most models.
- Index embeddings only *approximately* preserve a feature↔representation correspondence (self-attention mixes tokens), so the embedding-variance feature-importance story is suggestive, not exact.
