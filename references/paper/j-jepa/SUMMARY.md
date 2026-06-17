# J-JEPA — A jet-based Joint-Embedding Predictive Architecture that learns augmentation-free, symmetry-independent particle-jet representations by predicting masked *subjet* embeddings from context subjets

**Authors:** Subash Katel*, Haoyang Li*, Zihan Zhao* (equal), Farouk Mokhtar, Javier Duarte (UC San Diego); Raghav Kansal (Caltech, also Fermilab) **Venue/Year:** Machine Learning and the Physical Sciences Workshop, NeurIPS 2024 **arXiv:** 2412.05333 (v1, 5 Dec 2024, hep-ph) **Repo:** https://github.com/ucsd-hep-ex/J-JEPA (Zenodo v0.1.0, doi:10.5281/zenodo.14251372)

## TL;DR
J-JEPA ports I-JEPA to high-energy-physics jets. A large-radius jet is reclustered into *subjets*; some subjets are designated "target" and the rest "context." A context encoder and an EMA target encoder both see the *full* jet, then masks are applied to the *encoder outputs* (not the inputs) to split context vs. target representations. A predictor takes the context subjet representations, conditioned on the spatial positions (eta, phi) of the target subjets as "hints," and predicts the target subjet embeddings; training minimizes an L2 loss against the EMA-target encoder. Because it predicts in latent space and uses no hand-crafted augmentations, J-JEPA avoids baking in symmetry assumptions that could bias downstream tasks, positioning it as a step toward a cross-task particle-physics foundation model. Pretrained on 1 M JetClass jets and finetuned for top-vs-QCD jet tagging, J-JEPA beats from-scratch training, with the largest gains in the low-label regime.

## Problem & motivation
LHC physics analyses (triggering, tracking, calorimetry, particle-flow, jet tagging, mass regression) are dominated by supervised ML trained on *labeled simulations*, then applied to real data. The core drawback: simulation mismodeling means performance "may not translate to real data." The proposed fix is the pretrain-then-finetune paradigm: SSL-pretrain on large unlabeled data, then adapt with limited labels. Existing SSL-for-jets approaches (contrastive, masked autoencoders, masked particle modeling, generative pretraining) often rely on data augmentations that assume a symmetry. But "different tasks generally require invariance under different augmentations," so augmentation-baked invariances can *harm* downstream transfer. J-JEPA's stated thesis: an **augmentation-free** latent-predictive method removes per-task augmentation engineering and the biases it introduces, "offering a pathway toward a cross-task foundation model." Related work cited: MAE [9], contrastive SSL for jets [10], resimulation-based SSL [11], masked particle modeling [12,13], generative pretraining (OmniJet-alpha) [14], and dataset scaling [15].

## Method

### Tokenization: jets -> subjets (the "patches")
A large-radius jet (anti-kT, R = 0.8) is **reclustered into smaller subjets** with the Cambridge-Aachen algorithm (R = 0.2) via FastJet and its Python bindings. Subjets are the J-JEPA analogue of I-JEPA image patches. Jets with fewer than 20 subjets are **padded with empty subjets** to a fixed dimension for batched processing.

### Masking (the key I-JEPA-inspired design choice)
For each jet, a fixed fraction of subjets is randomly chosen as targets (the paper states **30% targets / 70% context**, a multi-block masking strategy inspired by I-JEPA). Crucially, **the full jet is passed through both encoders, and the mask is applied to the encoder *outputs*, not the inputs** — so both encoders have access to the complete semantic content of the jet before the context/target split. Masking the output rather than the input is called out as a key distinction from MAE (along with predicting in representation space).

### Encoders, predictor, EMA target
- **Subjet Transformers (SjTs):** transformer encoders adapted from ViT serve as context encoder, target encoder, and predictor. The SjT differs from a ViT by using **nonlinear embedding layers designed to disregard padded particles** within a subjet.
- **Two subjet-embedding variants:** (1) an MLP on the flattened array of the subjet's particles' four-vectors (GELU + residual connections) -> model "SjT-T"; (2) an attention-based embedding ("AE-SjT-T") that MLP-embeds each particle, runs transformer encoder blocks, then aggregates particles into one subjet embedding via class-attention blocks — argued to better ignore padding.
- **Predictor:** like I-JEPA, a smaller SjT with a linear dimension-expanding layer to create an information bottleneck that distills the most valuable context features.
- **Target encoder via EMA:** context-encoder + predictor params are learned by gradient descent; the **target encoder is an EMA of the context encoder**. EMA is cited (per DINO [22]) as crucial to prevent informational collapse, and the authors confirm this holds for J-JEPA.

### Physical positional encoding ("hints")
The predictor is conditioned on the **positions of the target subjets** to make the latent space highly semantic. Positions are the subjets' momentum direction relative to the jet — pseudorapidity eta and azimuthal angle phi — turned into spatial embeddings added to a learnable token. phi is processed as **sin(phi/2)** so embeddings reflect Cartesian rather than angular distance. They also describe a *novel* four-vector phase-space embedding (pT, eta, phi, E) as joint information to the predictor, but in this paper report **spatial (eta, phi) embeddings only**.

### Objective
Minimize the **L2 (MSE) loss** between the predictor's predicted target subjet representations and the EMA-target encoder's actual target subjet representations. Collapse is prevented by the EMA target (stop-gradient asymmetry), not by a VICReg/SIGReg variance-covariance term.

## Key results
- **Pretraining data:** a small fraction of JetClass [26] — 500 k top jets + 500 k QCD jets = **1 M jets**.
- **Finetuning data:** Top Tagging dataset [27]; two regimes — full (kept jets with >10 valid subjets: **785,767 of 1.2 M**) and **10%** (120 K train / 12 K val) to probe the low-label setting.
- **Setup:** single NVIDIA A100; AdamW, lr 1e-3, weight decay 1e-2, cosine schedule with 10% warmup, 80 epochs, batch size 64.
- **Eval metrics:** classification accuracy and **background rejection at 50% signal efficiency, 1/eps_B(eps_S = 0.5)**. Std devs from 5 random-init trials.
- **Headline finding:** finetuning a J-JEPA-pretrained model beats training the same architecture from scratch, "especially for smaller dataset sizes."
- **Selected numbers (Table 1; Baseline = from scratch, Finetuned = pretrained):**
  - **Accuracy:** best Full = AE-SjT-T/Flatten **90.03 +/- 0.07%** finetuned (vs 90.01 +/- 0.08 baseline); at 10%, AE-SjT-T/Flatten **88.94 +/- 0.13%** finetuned vs 88.92 +/- 0.15 baseline.
  - **Background rejection 1/eps_B:** SjT-T/Flatten 10% jumps from **40.50 +/- 1.26** (scratch) to **53.67 +/- 9.97** (finetuned); SjT-T/Cls-Attn 10% from 52.56 -> **61.32**; SjT-T/Flatten Full from 70.70 -> **90.06**. The attention-embedding AE-SjT-T already has strong baselines (e.g. Cls-Attn Full baseline 99.38 +/- 2.80) so its pretraining gains in rejection are smaller/mixed.
- **Ablations (Fig. 2):** (i) J-JEPA pretraining helps most when labeled samples are limited; (ii) class-attention aggregation of subjet representations > simple flattening for SjT-T; (iii) the custom **attention-based subjet embedding significantly outperforms the MLP-based embedding** downstream.

## Relevance to the EB-JEPA hackathon
This is the **particle-physics / jet-modality** track informer — a JEPA over *sets of subjets* (variable-cardinality point clouds with physical coordinates), a sibling of the other physical-sciences entries (e.g. Lens-JEPA and a "The Well" track). How it maps onto the eb_jepa recipe:
- **Tokenization is the transferable idea:** jet -> subjets via jet-clustering is a domain-specific "patchifier." Any modality where the natural unit is a variable-length cluster of constituents (point clouds, sets) can reuse this reclustering-into-tokens + padding-aware embedding pattern.
- **Encoder/predictor slots:** ViT-style "Subjet Transformer" context + EMA target encoders, plus a smaller bottlenecked predictor — a direct structural match to eb_jepa's encoder/predictor split.
- **Anti-collapse camp:** firmly **masking + EMA-target + stop-gradient (I-JEPA/DINO style)** with an **L2 latent loss**, *no* VICReg/SIGReg variance-covariance regularizer. This is exactly the axis an eb_jepa team could A/B — swap the EMA target for a regularizer-based (SIGReg/VICReg) two-view objective and test whether augmentation-free jet SSL survives without EMA.
- **Mask-the-output trick:** masking encoder *outputs* (not inputs) so both encoders see the full jet is an unusual, reusable design worth probing in other set/point-cloud JEPAs.
- **Conditioning on physical positions:** the predictor "hints" (eta, phi -> sin(phi/2) Cartesian-aware spatial embeddings, plus a sketched four-vector phase-space embedding) are the JEPA analogue of positional/action conditioning — a clean example of injecting physics priors via the predictor rather than via augmentations.
- **24h-replicable slice:** pretrain on the 1 M-jet JetClass subset, finetune a single-linear-layer head on Top Tagging at **10%** labels, and compare 1/eps_B(eps_S=0.5) finetuned vs scratch — a single A100, 80-epoch recipe, and the low-label rejection gain (e.g. 40.5 -> 53.7) is the most legible win to reproduce.

## Caveats & open threads
- **Workshop-scale, not full-scale:** only **1 M of JetClass's >100 M jets** used for pretraining; the authors explicitly flag scaling to full JetClass as future work, so the foundation-model claim is aspirational here.
- **Gains concentrate in the weaker config / low-label regime:** the biggest pretraining lifts are for the MLP-embedding SjT-T and at 10% labels; the strong AE-SjT-T baselines leave little headroom (some Full-data rejection numbers go *down* after pretraining, e.g. AE-SjT-T/Cls-Attn 99.38 -> 95.47), and standard deviations on rejection are large (e.g. +/-9.97).
- **Augmentation-free / cross-task claims under-tested:** the headline motivation (symmetry-independence enabling cross-task transfer) is argued by design but only demonstrated on a *single* downstream task (top-vs-QCD tagging); no cross-task or cross-symmetry benchmark is run.
- **Four-vector embedding not evaluated:** the novel phase-space (pT, eta, phi, E) predictor conditioning is described but the paper reports spatial-only embeddings; physics-informed backbones (e.g. Particle Transformer) and event-level clustering are listed as future work.
- **Reproducibility gaps:** target fraction stated as 30% in text but the architecture figure and "fixed number of target subjets" wording leave the exact masking schedule somewhat under-specified; full hyperparameters beyond the optimizer block are not tabulated. Code is released (ucsd-hep-ex/J-JEPA, Zenodo v0.1.0).
