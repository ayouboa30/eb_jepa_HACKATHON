# JETS — a JEPA adapted to long-horizon, irregularly-sampled multivariate wearable time series, learning robust health representations by predicting masked patches in latent space

**Authors:** Erik Xie (MIT EECS), Wyatt Chang (Empirical Health), Raquel Rodriguez Martinez, MD (Empirical Health), Brandon Ballinger (Empirical Health) **Venue/Year:** NeurIPS 2025 Workshop on Time Series for Health (TS4H), 2025 (poster) **arXiv:** none (non-archival workshop + company blog) **Repo/Source:** OpenReview https://openreview.net/forum?id=QqQDjLgHab (PDF https://openreview.net/pdf?id=QqQDjLgHab); blog https://www.empirical.health/blog/wearable-foundation-model-jets/

> Provenance note: this is a NON-archival TS4H workshop poster plus a company blog post, with no arXiv id — the weakest provenance of the reference set. A 17-page OpenReview PDF was retrieved and this summary is grounded in it.

## TL;DR
JETS (Joint Embedding for Time Series) is, per the authors, the first application of a JEPA-style joint-embedding architecture to long-horizon, irregularly-sampled multivariate time series (IMTS) of high-level *behavioral* wearable data (heart rate, sleep, activity, etc.). It masks 70% of token patches and trains a predictor to regress the EMA target encoder's latent representations of the masked patches from the visible context. Pretrained on ~3M person-days from 16,522 individuals (63 daily-resolution metrics), its frozen representations beat MAE, contrastive (PrimeNet) and mean-pooling baselines under linear probing on self-reported disease diagnosis and biomarker prediction.

## Problem & motivation
Wearables produce long-term behavioral time series rich in health signal, but the data is Irregular Multivariate Time Series (IMTS): high-dimensional, sparse, irregularly sampled (intermittent device use, sensor failures, variable participation). This breaks classic time-series models that need dense, regular, fixed-length inputs, and clinical labels are scarce/expensive, ruling out large-scale supervised learning. Prior SSL/foundation work targeted dense, low-level, short-window physiological signals (LSM-2, Apple HMS, DeepHeart); JEPAs for time series (e.g. TS-JEPA) handled only continuous univariate data. JETS targets high-level, low-resolution, mixed-source (sensor + self-report) behavioral IMTS.

## Method
Four components: learnable tokenizer, patch masking, dual encoder, predictor.
- **Tokenizer:** each observation is a triplet (day t_i, value v_i, metric type m_i); for the Mamba backbone, time differences Δt_i = t_i − t_{i-1} are used (state-space nature). Each of the three fields is embedded to hidden dim D and combined into a token sequence T ∈ R^{L×D}.
- **Masking:** T is split into patches; 70% are randomly removed to form the visible context T_ctx (MAE-style).
- **Encoders:** a bidirectional **Mamba** context encoder E_θ encodes T_ctx; an identically-structured **target encoder** E_φ encodes the full sequence T. E_φ is an EMA of E_θ (φ ← τφ + (1−τ)θ), as in JEPA/BYOL, providing the asymmetry that prevents representation collapse.
- **Predictor:** a small transformer-decoder network P takes E_θ(T_ctx) plus positional (time + variable) embeddings of the masked patches and predicts their target representations.
- **Objective:** latent-space MSE over masked positions M: L = (1/|M|) Σ_{j∈M} ‖ P(E_θ(T_ctx), pos_j) − E_φ(T) ‖_2^2.
- **Ablation variant JETS-Former** replaces Mamba blocks with bidirectional transformer blocks.

## Key results
Evaluation = **frozen encoder + single linear probe** (D=256 for all models) on a held-out 15% with self-reported medical history. Note: the blog says "fine-tuned"; the paper itself uses frozen + linear probing.
- **Diagnosis (AUROC, JETS-Mamba):** Hypertension 0.868, Sick Sinus Syndrome 0.868, Substance abuse 0.915, ME/CFS 0.810, Atrial flutter 0.705, POTS 0.731, Osteoporosis 0.758. JETS is best or near-best on most of the 14 conditions vs Mean-pooling, MAE, JETS-Former, PrimeNet; PrimeNet wins a few (e.g. Autism, Osteoporosis-AUPRC, Anxiety-ties).
  (Blog rounds these to "87% hypertension, 70% atrial flutter, 81% ME/CFS, 87% sick sinus syndrome.")
- **Biomarker prediction (Mean Relative Error ↓):** JETS had the best MRE across models (e.g. A1C 3.167 vs MAE 3.262, PrimeNet 5.721; Glucose 0.081), though absolute accuracies are low, attributed to small training/eval sample sizes.
- Takeaway: the joint-embedding (latent-prediction) objective + Mamba backbone outperforms raw-signal MAE reconstruction and a contrastive IMTS baseline on behavioral wearable data.

## Relevance to the EB-JEPA hackathon
Directly relevant to the **wearable-health biosignal track (track 3)**. JETS is a near-textbook EB-JEPA instance on biosignals: EMA target encoder + stop-gradient asymmetry as the anti-collapse mechanism, latent-space prediction loss, and high-ratio patch masking, all transplanted onto irregularly-sampled multivariate physiological time series. It is a concrete demonstration that the EB-JEPA recipe (context encoder, EMA target, predictor, latent MSE) transfers to health biosignals and beats reconstructive (MAE) and contrastive baselines — a useful blueprint and baseline framing for a biosignal JEPA hackathon entry. The IMTS tokenizer (triplet day/value/metric + Δt) and 63-channel daily-resolution behavioral framing are reusable design ideas for wearable inputs.

## Caveats & open threads
- **Weak / non-archival provenance:** TS4H 2025 workshop poster + company (Empirical Health) blog, no arXiv, no peer-reviewed archival version. Treat numbers as preliminary.
- **Private, non-reproducible data:** ~3M person-days from 16,522 individuals is a proprietary Empirical Health cohort; no public dataset or (located) code release, so results are not independently reproducible.
- **Self-reported labels:** disease "diagnoses" are self-reported, not clinically confirmed ground truth; AUROC interpretation is limited and some positive classes are very rare (low AUPRC).
- **Small eval sets:** authors flag limited sample sizes (esp. biomarker regression) as a cap on accuracy.
- **Method details:** EMA decay τ, exact patch/token sizes, predictor depth, and full architecture sizes live in the Appendix (not all transcribed here); worth a full read before reuse.
- **Blog vs paper discrepancy:** blog describes "fine-tuning"; paper text specifies frozen-encoder linear probing. Trust the paper.
