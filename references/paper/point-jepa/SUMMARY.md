# Point-JEPA — A canonical JEPA for point clouds that learns latent representations without input-space reconstruction, enabled by a greedy "sequencer" that orders unordered patch embeddings so I-JEPA-style spatially-contiguous context/target blocks become cheap to sample.

**Authors:** Ayumu Saito, Prachi Kudeshia, Jiju Poovvancheri (Graphics and Spatial Computing Lab, Saint Mary's University, Halifax, Canada). **Venue/Year:** WACV 2025 (arXiv v6, 9 Feb 2025). **arXiv:** 2404.16432 **Repo:** https://github.com/Ayumu-J-S/Point-JEPA

## TL;DR
Point-JEPA ports I-JEPA to 3D point clouds. The core obstacle is that point clouds are permutation-invariant (unordered), so I-JEPA's trick of selecting spatially-contiguous blocks by index does not apply. The fix is a **greedy sequencer** that reorders the per-patch (center-point) embeddings into a 1-D sequence where index-adjacency approximates spatial adjacency; context/target blocks are then sampled by contiguous index ranges, exactly as in I-JEPA. The model predicts target-encoder embeddings (not input points) with a predictor, uses Smooth L1 loss, and an EMA target encoder. It reaches **93.7±0.2%** linear-SVM accuracy on ModelNet40 (SOTA among SSL point-cloud models) and SOTA across all four few-shot settings, while pretraining in **7.5 h** on an RTX A5500 (~half of Point-M2AE, ~60% of Point2Vec).

## Problem & motivation
- Prior point-cloud SSL has three recurring drawbacks: long pretraining time, the need for reconstruction in the input space (generative models like Point-BERT, Point-MAE, PointGPT), or the need for additional modalities.
- Generative/reconstruction methods are computationally inefficient (they reconstruct in input space); self-distillation methods (e.g. Point2Vec) avoid input-space reconstruction but still train slowly; contrastive methods depend heavily on careful positive/negative selection and augmentation.
- JEPA (LeCun's energy-based framework; I-JEPA in images, V-JEPA in video) predicts in **representation space**, eliminating input-space reconstruction and converging faster. The paper's goal: bring this to point clouds.
- **Key challenge:** unlike image patches, point-cloud patches have no natural index order, so I-JEPA's "select a contiguous block of patches" cannot be done by indexing. Naively computing spatial proximity between all patch pairs at every context/target draw is inefficient.

## Method
- **Patch embedding (tokenization):** Given a point cloud P ⊂ R^3 with n points, sample `c` center points via Farthest Point Sampling (FPS), then take the `k` nearest neighbors (KNN) of each center to form `c` point patches. Patches are normalized by subtracting their center coordinate (separates local structure from position). A **mini-PointNet** (two rounds of shared-MLP → max-pool → concat) embeds each patch into a permutation-invariant patch embedding `T`. Pretraining uses c=64 centers, group size k=32, 1024 input points per object.
- **Greedy sequencer (the contribution; Algorithm 1):** orders the `c` patch embeddings into a 1-D sequence by their center points. Start from the center with the **minimum coordinate sum** (lands on the object's outer edge); repeatedly append the unvisited center closest (Euclidean) to the previously chosen one until all are visited. Result `T' = {t'_1,...,t'_r}` has index-adjacency ≈ spatial-adjacency "in most cases" (not guaranteed; gaps can occur). Analogous in spirit to z-ordering in PointGPT but data-driven. Crucially it is **batch-parallelizable** on GPU (pairwise distances in one pass, next-closest selected simultaneously across the batch), and lets context and target selection **share** the proximity computation.
- **Context/Target (operate on embeddings, not patches):**
  - Targets: pass all patch embeddings through the **target encoder f_θ** to get encoded embeddings y = {y_1,...,y_n}, then sample **M (=4) possibly-overlapping contiguous target blocks** y(i) = {y_j}_{j∈B_i}. Masking for targets is applied *after* the encoder, ensuring high-semantic-level targets.
  - Context: masking is applied to the **patch embeddings** (before encoding); select a contiguous subset T̂ ⊆ T', encode with the **context encoder f_θ** to get context block x. Context indices are forced to differ from target indices to prevent trivial solutions; because target patches are removed, the context usually spans several contiguous runs.
  - Ratios: target ratio range (0.15, 0.2) sampled 4×; context ratio range (0.4, 0.75). I-JEPA-style ranged masking.
- **Predictor p_φ:** takes context x plus, per target block, **mask tokens** (shared learnable params) and **positional encoding from the target center points**, and predicts the target embeddings: ŷ(i) = p_φ(x, {m_j}_{j∈B_i}). Predictor is narrower (dim 192, depth 6, 6 heads), following I-JEPA.
- **Loss:** Smooth L1 (β=2, like Point2Vec) between predicted and target embeddings, averaged over the M blocks: (1/M) Σ_i Σ_{j∈B_i} L(ŷ_j, y_j).
- **Optimization / EMA:** AdamW + cosine LR (warmup 1e-5→1e-3 over 30 epochs, decay to 1e-6), batch size 512. Context encoder updated by backprop; target encoder is the **EMA** of the context encoder, θ ← τθ + (1−τ)θ, with τ ramped **0.995 → 1.0** over pretraining. Encoders: standard Transformer, depth 12, width 384, 6 heads. Pretrained on ShapeNet (41,952 instances, 55 categories).

## Key results
- **ModelNet40 linear (SVM) evaluation:** 93.7±0.2% — SOTA, +0.8% over the best prior (CluRender 93.2, Point-M2AE 92.9). Uses max+mean pooling over the frozen context encoder, 1024 points.
- **Few-shot (ModelNet40), SOTA in all 4 settings:** 5-way 10-shot 97.4±2.2, 5-way 20-shot 99.2±0.8, 10-way 10-shot 95.0±3.6 (+1.1% over prior, hardest setting), 10-way 20-shot 96.4±2.7.
- **End-to-end fine-tuning (ModelNet40):** 94.1±0.1 (+voting) / 93.8±0.2 (−voting), 1k points — competitive but below Point2Vec (94.8/94.7). **ScanObjectNN:** OBJ-BG 92.9±0.4 (+1% over most SSL, just under PointDiff 93.2), OBJ-ONLY 90.1±0.2, OBJ-T50-RS 86.6±0.3.
- **Part segmentation (ShapeNetPart):** 83.9±0.1 mIoU_C / 85.8±0.1 mIoU_I — slightly *below* SOTA (authors note it lags on segmentation).
- **Efficiency:** 7.5 h pretraining on RTX A5500, < half of Point-M2AE and ~60% of Point2Vec.
- **Ablations:**
  - Sequencer (500 ep, RTX 5500): greedy(min-coordinate) 93.7 @ 7.47 h > z-ordering 93.4 @ 8.30 h > greedy(min-index) 92.7 @ 7.47 h > Hilbert 91.8 @ 10.78 h. Starting at the edge (min-coordinate-sum) beats starting at min-index; greedy is also the fastest (parallelizable).
  - Masking: **multi-block** (4 small contiguous targets + ranged context) 93.7 beats single-block contiguous 92.3 and single-block random 92.5 — many small targets + ample context is best.
  - Number of target blocks: rises to a peak at 4 (93.7) then falls (5→93.4, 6→93.2) — needs enough patches left for context.
  - Target ratio: best at (0.15,0.2)=93.3; degrades sharply for large targets (0.35–0.4 → 84.6). Context ratio: wide ranges help, (0.4,0.75)=93.7 best.
  - Predictor depth: monotonically better, 2→92.5 up to 6→93.7.
- **Limitations (authors' own):** weaker on segmentation / local features (the representation emphasizes **global** over local features); effectiveness on **larger** point clouds (with spatial redundancy) is unverified.

## Relevance to the EB-JEPA hackathon
Directly maps onto **Track 6 (point clouds)** as the canonical "JEPA on point clouds" baseline. Concrete takeaways:
- **How it tokenizes points:** FPS center sampling → KNN grouping → per-patch coordinate normalization → mini-PointNet (shared-MLP + max-pool, twice) → one embedding per patch. A patch = a small local neighborhood; a token = its PointNet embedding. Standard, reusable recipe (same as Point-BERT/MAE/PointGPT/Point2Vec).
- **How it orders points (the crux for any sequence/JEPA model on unordered sets):** a **greedy nearest-neighbor sequencer over center points**, seeded at the min-coordinate-sum (edge) point, turning an unordered set into a 1-D sequence where index-contiguity ≈ spatial-contiguity. This is what makes I-JEPA-style contiguous-block masking and EB-JEPA-style energy/predictive objectives tractable on point clouds without all-pairs proximity at every draw. It is batch-parallel on GPU and shares proximity computation across context+target selection. The ablation shows the *choice of ordering matters* (greedy-edge > z-order > Hilbert) — relevant if EB-JEPA wants to revisit ordering/masking design for 3D.
- **Architecture is vanilla JEPA:** shared point encoder, Transformer context/target encoders, EMA target, narrow predictor with mask tokens + center-point positional encoding, Smooth L1 in latent space. Easy to swap the loss for an energy-based / variance-covariance regularized objective in an EB-JEPA experiment.
- It is non-contrastive, reconstruction-free, single-modality, and fast (7.5 h) — a good cheap baseline to reproduce and then perturb.

## Caveats & open threads
- The sequencer only **approximates** spatial contiguity — adjacent indices "do not always guarantee spatial proximity; there might be a gap." It works empirically but is a heuristic, not a true space-filling guarantee.
- Quoted numbers are from pretraining on **ShapeNet** and evaluating on ModelNet40 / ScanObjectNN / ShapeNetPart only; **object-level** (synthetic) point clouds, ~1k–2k points. No scene-level or large/outdoor LiDAR results; large-cloud behavior explicitly flagged as untested.
- Segmentation underperforms SOTA — the representation is global-feature-biased, a caution if Track 6 cares about dense/local prediction.
- No temporal / action-conditioned / predictive-of-future-pointcloud results; the paper only gestures at motion prediction and dynamic scenes as future work. Energy-based regularizers (variance/covariance, anti-collapse) are not studied; collapse is avoided only via the EMA target + Smooth L1 setup.
- EMA decay τ ramps 0.995→1.0; predictor benefits from depth — both are tuning knobs an EB-JEPA reproduction should mirror.
