# Polymer-JEPA — JEPA self-supervised pretraining on stochastic polymer molecular graphs

**Authors:** Francesco Piccoli*, Gabriel Vogel*, Jana M. Weber (TU Delft, Dept. of Intelligent Systems; *equal contribution)
**Venue/Year:** arXiv preprint, June 2025 (cs.LG)
**arXiv:** 2506.18194
**Repo:** https://github.com/Intelligent-molecular-systems/Polymer-JEPA

## TL;DR
Applies the JEPA idea (predict in embedding space, not input space) to **stochastic polymer molecular graphs**. The graph is partitioned into subgraph "patches"; a context encoder embeds a large context subgraph and a predictor reconstructs the embeddings of smaller target subgraphs (conditioned on positional encodings). Pretraining a wD-MPNN GNN this way "enhances downstream performance, particularly when labeled data is very scarce, achieving improvements across all tested datasets." Improvement in R2 ranges from "39.8% in the smallest labeled data scenario to 0.4% in the scenario with 8% labelled data."

## Problem & motivation
Polymer ML is "hampered by the scarcity of high-quality labeled datasets." Prior polymer SSL is mostly text-based (pSMILES); graph-based SSL is underexplored, and existing graph SSL (node/edge masking [Gao et al. 2024]) reconstructs in the noisy input space. Polymer graphs are *stochastic* (weighted/dashed edges encode monomer ensembles, chain topology, stoichiometry; rep. from Aldeghi & Coley 2022), distinguishing them from small-molecule graphs. The authors transpose I-JEPA / Graph-JEPA to this modality to learn from a large unlabeled corpus, then fine-tune on label-scarce tasks.

## Method
- **Representation:** stochastic polymer graph (monomer graphs linked by weighted stochastic edges encoding connection probabilities & chain architecture).
- **Subgraphing ("patches"):** partition graph G into subgraphs, then build one large **context** subgraph x and one or more smaller **target** subgraphs y. Three algorithms compared: **random-walk** (stochastic, varies per epoch — best), **motif-based** (BRICS, chemically meaningful but deterministic/few subgraphs), **METIS** (clustering, min edge-cut). Constraints in Appendix A (context spans both monomers; full node/edge coverage; context larger than targets; minimal context-target overlap; subgraphs re-drawn each loop; directed-edge symmetry for wD-MPNN).
- **Encoders:** two GNNs (context & target), a variant of the **weighted directed message-passing NN (wD-MPNN)** from Aldeghi & Coley, switched to **node-centred** message passing (Appendix B shows parity with edge-centred). Context encoder takes the context subgraph; target encoder takes the *entire* graph (global contextualization), then pools node embeddings → sx and sy. Polymer graphs are small (20-30 nodes), so no self-attention needed.
- **Predictor:** MLP h_phi. For target i, predicts ŝy(i) = h_phi(sx + p̃_i T̃), conditioned on a target subgraph **positional token** (Eq. 1). Positional encoding via **RWSE** at both node level and subgraph/patch level. Loss = average L2 (MSE) between m predicted and true target embeddings.
- **Optional pseudolabel objective:** *jointly* (not sequentially, unlike Gao et al.) predict **polymer molecular weight Mw** (stoichiometry-weighted sum of monomer weights) from the target-encoder fingerprint via an MLP.
- **Train/finetune:** pretrain on 40% (17,186) of the conjugated-copolymer dataset; only the **target encoder** is kept for finetuning; an MLP head is added and finetuned **end-to-end** (encoder weights updated). Split 40% pretrain / 40% finetune / 20% test.

## Key results
- **Datasets:** (1) conjugated copolymer photocatalysts for H2 production (Aldeghi & Coley): 42,966 polymers, 9 A-monomers + 862 B-monomers, 3 architectures (alternating/random/block) × 3 stoichiometries (1:1, 1:3, 3:1); labels EA (electron affinity) & IP. (2) Arora et al. diblock copolymers: phase-behavior classification (lamellae, cylinders, gyroid, …), 49/50 diblocks, 4,780 samples.
- **Same-space (EA, regression):** pretraining helps most in low-label regimes (tested 0.4% / 192 → 24% / 10,311); "performance improvements in scenarios up to a data size of 4% (1728 polymers)", then "the benefits of pretraining plateau."
- **Transfer (diblock phase, classification):** pretraining on conjugated copolymers "consistently improves the classification performance (between around 0.02 and 0.1 in AUPRC), even in higher labeled data scenarios" (tested 4% / 191 → 80% / 3,824) — across a *different* chemical space ⇒ "learning general chemical knowledge about polymers."
- **vs input-space SSL (Gao et al. node/edge masking):** "comparable"; JEPA "slightly better in the very low data scenarios", Gao et al. better with more labels. Pseudolabel objective helps both but less for JEPA ("potentially already captures relevant information related to the polymer molecular weight").
- **vs random forest (ECFP, 2048-bit, radius 2, 32-oligomer ensemble avg):** RF wins at 0.4%/0.8% on EA and across all scenarios on the diblock task; JEPA's edge is only around 4% data on EA. RF's strength leans on handcrafted descriptors / strong mole-fraction correlation; JEPA learns from the graph directly without expert fingerprinting.
- **Subgraphing ablation (pretrain 40%, finetune 0.4% EA; no-pretrain baseline R2 = 0.46 ± 0.15):**
  - Context size: best **60%** (R2 0.65 ± 0.03); 50-75% broadly effective.
  - Target size: best **10%** (R2 0.66 ± 0.02); recommend 10-20%.
  - Number of targets: best **1** (R2 0.67 ± 0.01), not very sensitive.
  - Algorithm: RW 0.67 ± 0.01 ≈ METIS 0.67 ± 0.04 > Motif 0.63 ± 0.05. "Variations in subgraphs and subgraph sizes play a more crucial role … than the chemical meaningfulness of the subgraphs."

## Relevance to the EB-JEPA hackathon
A clean **molecular-graph** instantiation of JEPA and the natural **sibling of Graph-JEPA** (it explicitly builds on and extends the Graph-JEPA work [ref 32]). It demonstrates the full EB-JEPA recipe on a non-image, non-video modality: context/target *subgraph* views (analogous to image patches / video blocks), a single (target) GNN encoder kept for downstream, a predictor conditioned on **positional encodings** (RWSE) of the target, and an L2 embedding-space loss. Notable for the hackathon: (i) no momentum/EMA target encoder or stop-gradient anti-collapse machinery is highlighted here — collapse is implicitly avoided via the predictor + positional conditioning + per-epoch subgraph resampling, a contrast worth probing; (ii) it shows a JEPA winning specifically in the **label-scarce / transfer** regime, the regime EB-JEPA cares about; (iii) the joint **pseudolabel (Mw) auxiliary** is an example of adding a graph-level objective alongside the JEPA loss. Most relevant track: **graph/molecular modality extension of the JEPA energy-based framework** (sibling-of-Graph-JEPA track).

## Caveats & open threads
- Small-scale, domain-specific study: two datasets, both copolymer; graphs are tiny (20-30 nodes), so conclusions on encoder choice (no attention) may not transfer to larger graphs.
- A **random forest on handcrafted ECFP fingerprints beats** the method in the most data-scarce EA scenarios and across the whole diblock task — JEPA's practical edge is narrow and regime-dependent.
- Gains plateau quickly (≈4-8% labeled data); benefit is concentrated in extreme low-data regimes.
- Pseudolabel (Mw) auxiliary gives smaller gains for JEPA than for input-space SSL — unclear how much the auxiliary vs the JEPA objective drives results.
- No explicit collapse analysis / anti-collapse regularizer discussion; relies on architectural/sampling choices.
- Authors hypothesize (untested) that larger, more diverse pretraining corpora and multimodal data would lift performance further.
