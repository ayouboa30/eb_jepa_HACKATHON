# ST-JEMA — A JEPA-style latent-reconstruction objective for self-supervised learning of *dynamic* functional connectivity from fMRI brain graphs

**Authors:** Jungwon Choi, Hyungi Lee, Byung-Hoon Kim, Juho Lee (KAIST AI; Yonsei Univ. College of Medicine; AITRICS) **Venue/Year:** arXiv preprint v2, 5 May 2025 (v1 Mar 2024) **arXiv:** 2403.06432 **Repo:** none stated in paper (no GitHub URL in the PDF or its HTML)

## TL;DR
ST-JEMA (Spatio-Temporal Joint Embedding Masked Autoencoder) ports the I-JEPA idea — predict masked *latent* representations rather than raw inputs — onto **dynamic graphs** built from resting-state fMRI. Each time window of BOLD signal becomes a brain graph (nodes = Schaefer-atlas ROIs, edges = thresholded Pearson functional connectivity). A context GIN encoder sees a block-masked graph; an EMA target GIN encoder sees the full graph; MLP-Mixer node/edge decoders reconstruct the masked **node embeddings (MSE)** and **adjacency submatrix (BCE)**. The novelty over the authors' prior ST-MAE is a **dual reconstruction objective**: a *spatial* term (reconstruct masked nodes/edges within a timestep) plus a *temporal* term (reconstruct a timestep's nodes/edges from two non-overlapping neighboring timesteps ta, tb). Pretrained on UK Biobank (40,913 rs-fMRI subjects), it tops 8 downstream benchmarks (gender classification, age regression, psychiatric diagnosis) and is the rank-1 method on every benchmark in all three tables.

## Problem & motivation
GNNs on fMRI functional connectivity (FC) are strong for phenotype/disorder prediction, but clinical labeled fMRI is scarce and expensive, capping supervised scalability. SSL on abundant unlabeled fMRI is the answer. Among SSL paradigms, generative masked-autoencoding beats contrastive in many domains, and prior graph MAEs (GraphMAE) and the authors' dynamic-graph ST-MAE work on fMRI. But existing generative SSL "tend to focus on reconstructing lower-level features," which "hinders the ability of the model to capture generalizable representations." The paper's gap: bring JEPA's *latent*-target reconstruction (which "avoid[s] the potential pitfall of generating representations overly fixated on low-level details") to **dynamic** brain graphs, and make it explicitly spatio-temporal so the encoder captures the temporal dynamics of dynamic FC.

## Method
**Dynamic graph construction.** ROI BOLD time series P (Schaefer atlas, 400 ROIs) is sliced by a temporal window (length Γ=50, stride S=16 on UKB) into TG graphs G(t). Node feature xi(t) = W[ei ‖ η(t)] concatenates a one-hot spatial ROI embedding ei with a GRU temporal encoding η(t). Edges A(t) ∈ {0,1}^{N×N} = top-30%-thresholded Pearson FC within the window.

**Joint-embedding architecture (collapse prevention).** Two structurally-identical GIN encoders: context fcxt(·;θ) trained by SGD, target ftar(·;θ̄) **frozen + EMA-updated** (θ̄ ← βθ̄ + (1−β)θ). Separate context/target encoders + stop-gradient on the target are what "prevent representation collapse during training" — the same I-JEPA/BYOL mechanism, no VICReg/SIGReg variance term. Decoders gnode and gedge are **MLP-Mixers** (token-mixing lets context nodes interact to reconstruct targets), chosen over plain MLP/GIN.

**Block masking.** K=4 binary masks per timestep. Nodes are masked in **sequential-index blocks** (the Kim et al. dynamic-FC construction aligns adjacent node indices across time, so contiguous-index masking is semantically meaningful, the graph analogue of I-JEPA's large contiguous target blocks). Edges are masked as a random **square submatrix**. Mask ratio α ~ Uniform(αmin, αmax) = 10–30%; a single *global* context (logical-AND of the K masks) keeps memory linear, and a learnable mask vector m conditions which of the K blocks to reconstruct.

**Spatial objective (Lspat).** Reconstruct only the masked target node embeddings via gnode against the EMA target encoder's embeddings (LMSE), and the masked adjacency submatrix via gedge (sigmoid(HHᵀ), LBCE). Only the *target* nodes are reconstructed (not the whole graph), forcing focus on masked semantics. `Lspat = Σ_t (λnode·L_node-spat + λadj·L_adj-spat)`.

**Temporal objective (Ltemp).** The distinctive part. For target time t, sample two timesteps ta, tb from **non-overlapping** windows (ta ≤ t−dmin, tb ≥ t+dmin, dmin = ½(Γ/S+1)) so the task isn't trivial. Fill the masked target slots not with the learnable vector m but with a projection of the **concatenated context embeddings at ta and tb** (Z̃a,b = WT[Z̃(ta)‖Z̃(tb)]); decode with the same gnode (LMSE). Adjacency is reconstructed cross-temporally: Ā(t) = ½(sig(H(ta)H(tb)ᵀ) + sig(H(tb)H(ta)ᵀ)), LBCE. `Ltemp = Σ_t (λnode·L_node-temp + λadj·L_adj-temp)`.

**Total.** `L_ST-JEMA = γ·Lspat + (1−γ)·Ltemp`, γ=0.5. Pretrain: 10,000 steps, batch 16, cosine LR, GIN 4-layer encoder + 1-layer MLP-Mixer decoder. Fine-tune: encoder + SERO attention readout + task head, 30 epochs, batch 32, one-cycle, orthogonal reg. Discard decoders downstream.

## Key results
Pretrained on UKB (40,913 subjects); evaluated on 8 datasets (ABCD, HCP-YA/A/D/EP, ABIDE, ADHD200, COBRE). ST-JEMA is **rank 1.00** (mean rank) in all three main tables.
- **Gender classification (AUROC↑, Table 1):** best on every benchmark; e.g. **84.16** ABCD, **93.99** HCP-YA, **85.65** HCP-A, and a large clinical jump on **COBRE 80.61** (next best ST-DGI/dynamic-Random-Init ≈ 71.8 / 72.5; static GraphMAE 68.89).
- **Age regression (MAE↓, Table 2):** best or tied-best on most; e.g. **7.94** HCP-A, **2.73** HCP-YA, **8.60** COBRE.
- **Psychiatric diagnosis (AUROC↑, Table 3):** **79.22** HCP-EP, **71.49** ABIDE, **58.89** ADHD200, **70.04** COBRE — beats ST-MAE (78.66 / 69.73 / 56.77 / 62.19) and contrastive ST-DGI everywhere; COBRE again the standout.
- **Low-label robustness (Fig. 2, ABIDE):** beats all baselines even at **20%** labels; ST-DGI (contrastive) fails to beat Random-Init below 50%.
- **Loss ablation (Table 4, ABIDE diagnosis):** full = **71.49**; dropping temporal loss collapses to **65.19** (worse than Random-Init 67.16) — temporal term is the single most important component; dropping node 68.24, edge 70.33, spatial 69.79.
- **Decoder ablation (Table 9):** MLP-Mixer **71.49** >> GIN 66.35 > MLP 64.76.
- **Block-mask ratio (Fig. 6):** 10–15% optimal across clinical datasets.
- **Pretraining-size scaling (Fig. 5):** AUROC rises monotonically 1K→40K UKB subjects.
- **Temporal-missing robustness (Fig. 4):** ST-JEMA degrades least as 30–90% of the time axis is masked, confirming the value of the temporal objective.
- **Linear probing (Table 7) & multi-task (Table 8):** ST-JEMA leads generative ST-MAE and contrastive ST-DGI.

## Relevance to the EB-JEPA hackathon
This is a **neuro / brain-signal track** paper, a sibling of Brain-JEPA and a graph-domain cousin of the S-JEPA EEG entry — useful as the *fMRI-on-dynamic-graphs* point in the design space. Mapping onto the eb_jepa recipe:
- **Encoder:** GIN over brain graphs instead of a ViT over patches — shows JEPA transfers cleanly to a GNN backbone on non-Euclidean data.
- **Predictor / decoder:** MLP-Mixer node + edge decoders that reconstruct only masked target embeddings; the ablation (MLP-Mixer >> MLP) echoes the general JEPA lesson that the predictor must mix tokens, not act per-token.
- **Collapse prevention:** firmly in the **EMA-target + stop-gradient (I-JEPA/BYOL) camp** — no variance/covariance regularizer. A clean A/B for the hackathon: swap the EMA target for a SIGReg/VICReg two-view objective and drop the target encoder.
- **Masking design:** single-view masking with **two masking axes** — *spatial* (sequential-index node blocks + square adjacency submatrix) and a genuinely novel *temporal* objective that predicts a timestep from two non-overlapping neighbors. The temporal term being the ablation's most load-bearing component is the transferable insight for any temporal/sequence JEPA.
- **24h-replicable slice:** pretrain on a single mid-size dataset and linear-probe ABIDE diagnosis; the loss ablation (Table 4) and decoder ablation (Table 9) are each a one-axis sweep that reproduces the paper's headline lessons cheaply.

## Caveats & open threads
- **No released code found** (no GitHub link in the PDF/HTML); reproduction needs reimplementing dynamic-graph construction, block masking, and dual decoders from the equations (full algorithm is in App. A, hyperparameters in Table 6).
- **No VICReg/SIGReg baseline:** collapse handled only by EMA+stop-grad; the regularizer-vs-EMA axis the eb_jepa hackathon targets is untested here.
- **Heavy domain-specific preprocessing:** Schaefer-400 ROI extraction, MNI registration, window/stride choices, and per-dataset dynamic lengths (App. C/E) are nontrivial and gate any from-scratch replication.
- **EMA decay β and λadj range loosely specified** (λnode=1.0, λadj swept over 10⁻³…10⁻⁶; β not given numerically); some hyperparameters require guessing.
- **Encoder is shallow (4-layer GIN):** unclear how the latent-reconstruction benefit scales to larger graph backbones; the data-size ablation scales data, not model.
- **Temporal sampling assumes well-aligned sequential node indices** across time (from the Kim et al. construction); the sequential-block masking trick may not transfer to graph domains lacking that index alignment.
