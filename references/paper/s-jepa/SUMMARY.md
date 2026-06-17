# S-JEPA — A Signal-JEPA for EEG that masks *channels* (not time) and learns spatially-transferable representations for cross-dataset BCI

**Authors:** Pierre Guetschel, Thomas Moreau, Michael Tangermann **Venue/Year:** 9th Graz Brain-Computer Interface Conference (Graz BCI 2024), pp. 11–16; 2024 **arXiv:** 2403.11772 **Repo:** none stated

## TL;DR
S-JEPA (Signal-JEPA) adapts the Joint-Embedding Predictive Architecture to EEG by introducing a *spatial* block-masking strategy: instead of masking patches in time, it masks all channels within a radius of a randomly chosen electrode, and a JEPA predictor reconstructs the masked channel tokens in embedding space. Pre-trained self-supervised on the Lee2019 dataset (54 subjects, 62 channels), it is evaluated on three BCI paradigms (motor imagery, ERP, SSVEP) via three lightweight downstream heads. The headline finding is that an explicit *spatial filtering* step (the "pre-local" head that linearly combines channels before the local encoder) is critical for accuracy, and longer pre-training windows (16 s) consistently help; mask size has no clear effect.

## Problem & motivation
BCI deployment is bottlenecked by the need for per-subject calibration data, which is "time-consuming and demanding for participants." Self-supervised learning (SSL) could amortize this with cross-dataset/cross-subject pretraining, but EEG poses two problems: (i) datasets differ in channel montage/geometry, so models must adapt to varying channel configurations, and (ii) direct reconstruction (MAE-style) is expensive and ill-posed in high-dimensional signal space because "the reconstruction's difficulty can vary significantly across different areas of the input data" and may "necessitate domain-specific constraints to produce valid signals." JEPA sidesteps this by predicting *latent* embeddings rather than raw signal, which is "computationally efficient" and makes the choice of reconstruction metric far less critical. The paper's specific gap: channel-based (spatial) block masking for EEG SSL "remains unexplored," whereas BCI application of SSL "remains largely untapped" (prior SSL EEG work targeted sleep staging). The work is framed as an exploratory study around research questions, not a numbered contributions list; the two stated novelties are a "domain-specific spatial block masking strategy" and "three novel architectures for downstream classification."

## Method

### Architecture
- **Local encoder:** a per-channel CNN, 5 conv layers each with GELU. It encodes *each channel independently*: a 1.19 s window (1.0 s stride) per channel becomes a 64-dim token. First conv kernel spans 0.25 s; subsequent layers pair features with kernel = stride = 2. This per-channel tokenization is what makes spatial masking and montage-agnostic transfer possible.
- **Contextual encoder:** an 8-layer transformer over the token sequence, with temporal position encoding (cosine, first 34 dims) and *spatial* position encoding (trainable channel embeddings initialized from cosine encodings of the electrodes' 3D coordinates). The spatial embeddings are what let the model reason about electrode geometry.
- **Predictor:** a 4-layer transformer decoder that predicts the masked channel tokens in embedding space.
- **Target encoder:** a non-trainable duplicate ("contextual target encoder") updated by EMA of the contextual encoder's weights.

### Objective
Standard I-JEPA / data2vec-style latent prediction with a stop-gradient EMA target: only the unmasked (context) tokens go through the (online) contextual encoder + predictor; the EMA target encoder sees the *full* token sequence and produces the targets. The loss is an **L1 loss** between predicted and EMA-target embeddings of the masked tokens. Collapse is prevented by the asymmetric online/EMA-target design (stop-gradient), not by a VICReg/SIGReg variance-covariance term. Pretraining stops on 10-epoch validation-loss plateau (early stopping).

### Spatial block masking
The core novelty: pick a random central electrode, then mask **all channels within a radius** of it. Three radii are tested, with diameters ≈ **40%, 60%, 80% of head size**. Because electrode density varies, this "inherently introduces variability in the number of masked tokens." Context = unmasked channels; targets = the masked channels' embeddings.

### Downstream heads (three variants)
Each head adds two layers: a **spatial aggregation** (a conv that forms weighted channel combinations into `V << C` "virtual" channels; exact V not specified) plus a fully-connected classifier.
- **Contextual:** both layers placed after the contextual encoder (uses the full pretrained transformer).
- **Post-local:** discards the contextual encoder; head sits on top of the local encoder.
- **Pre-local:** discards the contextual encoder and places spatial aggregation *before* the local encoder — i.e., a learned spatial-filtering step "as commonly present in BCI architectures."

Two fine-tuning regimes: **new-** (freeze pretrained weights, train only new layers) and **full-** (train the whole network with a 10-epoch warmup).

## Key results
- **Dataset:** Lee2019 (MOABB), 54 subjects, 62 channels. Split: 40 subjects pretraining / 7 validation / 7 downstream test. Pretraining slices continuous recordings into 16.9 s windows across all paradigms/subjects combined: 36,576 train / 6,528 val examples; no artifact rejection.
- **Pretraining sweep:** signal durations {1, 4, 16 s} (log scale) x mask {40%, 60%, 80%}.
- **Downstream protocol:** 5-fold within-subject stratified CV; example length 4.19 s for MI/SSVEP, 1.19 s for ERP.
- **Headline numbers (vs. Riemannian-geometry SOTA):**
  - **ERP:** ~97% AUC (best pipeline `16s-40%-full-pre-local`), near SOTA 98.41%.
  - **SSVEP (4-class):** ~94% accuracy (best `16s-60%-new-pre-local`), *exceeding* SOTA 89.44%.
  - **MI (left vs. right hand):** ~65% accuracy (best `16s-40%-new-pre-local`), well below SOTA 84.74%.
- **Ablations:**
  - *Pre-training window length:* 16 s pipelines consistently best across all paradigms; "long pre-training windows favor the local features encoder." 1 s/4 s do poorly with contextual heads but acceptably with pre-local.
  - *Mask size:* no clear trend — "mask radius' impact on downstream performance uncertain"; 40/60/80% roughly equivalent.
  - *Spatial filtering / head choice:* pre-local (spatial filtering before features) is clearly best; the pure-contextual heads "frequently result at chance level on average."

## Relevance to the EB-JEPA hackathon
This is the **EEG / signal-modality** track informer — a 1-D multi-channel time-series JEPA. It maps cleanly onto the eb_jepa recipe:
- **Encoder:** per-channel CNN local encoder + transformer contextual encoder. The per-token-per-channel design is the reusable idea — it lets one model ingest arbitrary montages, the EEG analogue of patch tokenization.
- **Predictor:** a small (4-layer) transformer-decoder predicting masked-token embeddings — the standard JEPA predictor slot.
- **Regularizer / collapse prevention:** this paper is firmly in the **masking + EMA-target + stop-gradient (I-JEPA/data2vec) camp**, not the VICReg/SIGReg two-view camp. The anti-collapse mechanism is the asymmetric online/EMA-target pair with an L1 latent loss — directly comparable to eb_jepa's EMA/stop-grad path, and a natural A/B against swapping in a SIGReg/VICReg variance-covariance term to drop the EMA target.
- **Masking-vs-two-view:** single-view masking (context = unmasked channels), but the masking axis is **spatial (channels)** rather than spatio-temporal patches — a transferable trick for any multi-channel sensor JEPA.
- **24h replicable slice:** the cleanest reproduction is the *pretrain-then-pre-local-head* pipeline on a single MOABB dataset (Lee2019), since the SSVEP/ERP results are strong and the head is lightweight. A team could (1) implement per-channel tokenization + radius-based channel masking, (2) reuse eb_jepa's existing EMA-target JEPA loop with L1 loss, (3) plug a spatial-aggregation linear head, and (4) sweep pretrain window length (the dominant factor) rather than mask size (negligible).

## Caveats / open threads
- **Single dataset, tiny test set:** evaluated only on Lee2019 with 7 downstream subjects; authors flag this as exploratory and note 6 of 7 MI test subjects are "hard to classify," likely deflating the weak MI result. Cross-*dataset* transfer (the title's motivation) is argued architecturally (montage-agnostic tokenization) but not actually benchmarked across datasets here.
- **Contextual encoder underperforms:** pure transformer heads often hit chance; authors attribute this to "the need for large datasets ... when training transformers," i.e., the transformer is undertrained — so the strong numbers come mostly from the local encoder + spatial filter, not the JEPA-pretrained contextual representation. This weakens the claim that the SSL pretraining itself is doing the heavy lifting.
- **Spatial vs temporal masking not compared:** the paper introduces spatial masking but does not run it head-to-head against temporal/patch masking; an obvious follow-up.
- **Unspecified details:** number of virtual channels `V`, full hyperparameters, and (notably) no released code/repo found in the HTML. Numbers here are read from the arXiv HTML rendering, rounded as reported (~97% / ~94% / ~65%).
- **No VICReg/SIGReg baseline:** collapse is handled purely by EMA+stop-grad; the paper does not test a regularizer-based alternative, which is exactly the axis the eb_jepa hackathon could add.
