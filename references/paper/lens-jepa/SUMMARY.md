# Lens-JEPA — A physics-informed I-JEPA that bakes the gravitational-lens equation into the encoder for strong-lensing foundation models

**Authors:** J Rishi, Pranath Reddy Kumbam, Michael W. Toomey, Sergei Gleyzer **Venue/Year:** NeurIPS 2025 ML4PS Workshop **arXiv:** none (ML4PS 2025 paper #340) **Repo:** none stated

## TL;DR
Lens-JEPA extends I-JEPA to strong gravitational-lensing imagery by replacing the standard ViT context/target encoders with a **physics-informed encoder** that explicitly applies the gravitational lens equation before attention. The encoder predicts a per-pixel learnable correction `k(x,y)` to an analytic Singular-Isothermal-Sphere (SIS) potential ansatz, "un-lenses" the image to reconstruct the source-plane galaxy, then tokenizes that reconstruction (Shifted Patch Tokenization) and feeds it to transformer blocks with Locality Self-Attention. Pretrained self-supervised on 10k simulated lenses (instrument-agnostic "Model A"), then fine-tuned/evaluated on Euclid-mock "Model B" for a 3-class dark-matter-substructure classification (axion / CDM / no-substructure). It reaches **0.9120 accuracy**, beating plain I-JEPA (0.9017) and all supervised baselines (ViT, ViTSD, CaiT, ResNet18, Lensformer), with ROC-AUC 0.97 / 0.96 / 1.00 across the three classes.

## Problem & motivation
Strong gravitational lensing is a sensitive probe of dark-matter substructure on subgalactic scales, but ML for lensing is fragmented: separate task-specific models for classification, regression, segmentation, super-resolution, each needing large labeled sets and bespoke pipelines. The authors argue for a **foundation model** for lensing learned self-supervised on abundant (simulated) unlabeled images and adaptable across downstream tasks (lens finding, mass modeling, substructure analysis), reducing retraining and label cost. I-JEPA is the chosen SSL backbone because predicting *latent* representations of masked blocks (vs. pixel reconstruction) yields semantically rich, scalable embeddings well-suited to scientific imaging where high-level physical features matter more than pixel-perfect output. The gap they target: vanilla I-JEPA is domain-agnostic and encodes no astrophysical priors, so they inject the lens equation directly into the architecture to enforce physical consistency.

## Method

### Setup / preliminaries
Standard I-JEPA structure is retained: a context encoder + an EMA target encoder + a predictor that predicts masked-block representations from visible context, with a latent (not pixel) objective. The contribution is entirely in the **encoder**, which is swapped for a physics-informed module.

### Physics-informed encoder (the core idea)
- **Lens equation.** Works from the dimensionless lens equation `S = I − ∇Ψ(I)`, where `I` is the observed image-plane position, `S` the source-plane position, and `Ψ` the dimensionless gravitational potential. Only `I` is observed, so the lens potential must be assumed.
- **Ansatz.** Adopts an analytic SIS potential `Ψ_SIS(x,y) = sqrt(x^2 + y^2)` as a first-order approximation of the halo, scaled by a **learnable per-pixel function** `k(x,y)`: `Ψ(x,y) = k(x,y)·Ψ_SIS(x,y)`. The `k_ij` field is predicted from the image by a ViT-for-Small-Datasets (ViTSD), trained to capture subtle gradient variations induced by dark-matter substructure.
- **Inverse-lens layer.** With the estimated `Ψ`, the encoder solves/inverts the lens equation to reconstruct the **source-plane galaxy image** (an "un-lensing" step). The pipeline shown in Fig. 1 includes log-transform, instance norm, tanh/abs saturation operations feeding the inverse-lens layer.
- **Tokenization + attention.** The reconstructed source galaxy is tokenized via **Shifted Patch Tokenization (SPT)** and processed by transformer blocks using **Locality Self-Attention / Locally Multi-Head Attention (LMA/LSA)** with add-&-norm, feedforward, dropout — the small-dataset ViT recipe (ViTSD). Both context and target encoders use this physics encoder. The design is explicitly inspired by **Lensformer** (Velôso, Toomey, Gleyzer 2023), a physics-informed ViT for lensing.

## Key results
- **Datasets.** Two simulated galaxy-galaxy strong-lensing datasets from `lenstronomy`, 150×150 single-channel, SNR ≈ 25, Sérsic source profile, SIS lens halo. Three substructure classes: (1) CDM with truncated NFW subhalos, (2) axion dark matter (m ~ 10⁻²³ eV) with vortex defects, (3) no-substructure baseline. **Model A** = generic instrument, Gaussian PSF 0.05″ (used for pretraining, instrument-agnostic). **Model B** = Euclid-mock (used for fine-tuning/eval). Pretraining: 10,000 Model-A sims. Downstream: 3,000 sims per class from Model B, 80:20 train/test.
- **Protocol.** Pretrain Lens-JEPA on Model A → fine-tune on Model B → compare to supervised baselines (ViT, ViTSD, CvT, CaiT, ResNet18, Lensformer) and original I-JEPA, all similar parameter count, AdamW, lr 1e-5, CrossEntropy, 50 epochs.
- **Headline (Table 1, accuracy / ROC-AUC axion-CDM-noSubs):**
  - **Lens-JEPA: 0.9120** / 0.97 / 0.96 / 1.00 (best on every metric).
  - I-JEPA: 0.9017 / 0.96 / 0.95 / 0.98.
  - Lensformer: 0.8969 / 0.94 / 0.92 / 0.98.
  - ViT: 0.8931; ViTSD: 0.8838; ResNet18: 0.8207; CaiT: 0.8065.
- **Takeaway.** Embedding the lens equation as an inductive bias gives consistent gains over plain I-JEPA and over a previous physics-informed ViT (Lensformer), and the gains hold across all three dark-matter scenarios. The authors frame classification as a proof of concept toward a broader lensing foundation model.

## Relevance to the EB-JEPA hackathon
This is the clearest **physical-sciences / scientific-imaging JEPA** informer in the memory: a worked example of injecting a known **PDE / physical equation as an inductive bias inside the JEPA encoder** rather than treating images as generic patches. It maps onto the eb_jepa recipe as:
- **Encoder:** the swappable slot here is the whole encoder. The reusable trick is a physics pre-processing front-end (analytic ansatz + learnable per-pixel correction + an "invert-the-forward-model" layer) feeding ViTSD-style SPT + locality attention. For any track with a known generative/forward model (lensing, optics, diffusion, fields in The Well), one can prepend an analytic-inversion layer before tokenization.
- **Predictor / objective:** unchanged from I-JEPA — predict masked-block *latent* representations against an EMA target. The paper sits firmly in the **masking + EMA-target (I-JEPA) camp**, not the VICReg/SIGReg two-view camp; collapse handling is the standard asymmetric context/EMA-target pair. So an obvious eb_jepa A/B is to drop the EMA target and substitute a SIGReg/VICReg variance-covariance regularizer while keeping the physics encoder.
- **Pretrain → transfer story:** pretrain on a generic/instrument-agnostic simulator, fine-tune on a realistic-instrument distribution — directly the "foundation-model-for-science" pattern, and a tidy 24h-replicable slice (10k sims pretrain, 3×3k classification fine-tune).
- **Best-fit track:** the **physical-fields / The Well track** (scientific imaging with known physics), with a secondary fit to any "does physics-informed inductive bias help a JEPA?" ablation track.

## Caveats / open threads
- **Simulation-only.** All data are `lenstronomy` simulations; no real observations. Authors explicitly flag reliance on simulated datasets and that validation on real-world data is future work.
- **Single downstream task.** Only 3-class substructure classification is shown; the foundation-model claim (lens finding, mass modeling, super-resolution, regression) is aspirational, not demonstrated. Embeddings are not evaluated by linear probing or transfer breadth.
- **No ablation of the physics components.** The +1.0% over I-JEPA is from one table; there is no breakdown isolating the SIS ansatz vs. the learnable `k` field vs. SPT/LSA, no error bars / seeds, and no compute or parameter-count details beyond "similar number of parameters."
- **Ansatz rigidity.** The SIS potential is a first-order halo approximation; whether the learnable `k(x,y)` can absorb deviations for non-SIS / elliptical lenses is untested. Generalization to other lens profiles or PSFs beyond Model A→B is open.
- **Reproducibility.** No code released; several pipeline details (e.g., exact inverse-lens solver, how SPT/LSA hyperparameters are set, predictor depth) are only sketched in Fig. 1. Workshop-length (7 pages incl. references), so depth is limited by design.
