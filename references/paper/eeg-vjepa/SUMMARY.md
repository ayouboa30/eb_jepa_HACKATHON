# EEG-VJEPA — adapting V-JEPA's spatiotemporal masked latent prediction to multi-channel EEG, treating EEG as "video", reaching SOTA on TUAB abnormal-EEG classification

**Authors:** Amirabbas Hojjati, Lu Li, Ibrahim Hameed, Anis Yazidi, Pedro G. Lind, Rabindra Khadka (OsloMet / Sun Yat-sen / NMBU / Univ. Oslo / Simula). **Venue/Year:** Preprint submitted to *Pattern Recognition*; arXiv v5, 12 Mar 2026 (orig. Jul 2025). **arXiv:** 2507.03633 **Repo:** none provided in the paper.

## TL;DR
First work to port **V-JEPA** (the video JEPA of Bardes et al., the [22] ref) to EEG: multi-channel EEG is reshaped into a video-like 3D tensor, patched into spatiotemporal tubelets, and trained with V-JEPA's multi-block masking + EMA target encoder + L1 latent-prediction loss. Pretrained label-free on a TUH+NMT combo ("EEGComb2", 4438 subjects), it hits **SOTA on the TUAB abnormal-EEG benchmark**, beating the JEPA-based EEG2Rep, the foundation model LaBraM, and a contrastive baseline, and matching the fully-supervised ChronoNet. This is the cleanest published template for the hackathon's EEG/TUAB flagship track.

## Problem & motivation
EEG = high temporal, low spatial resolution; analysis is bottlenecked by scarce labels, high dimensionality, and the lack of models that capture spatial AND temporal structure jointly. Prior SSL for EEG handles spatial or temporal features in isolation (contrastive needs hand-designed augmentations/inductive biases; generative/masked-reconstruction methods can learn semantically weak features). JEPA-style latent prediction sidesteps both; V-JEPA's tubelet masking is a natural fit if EEG is cast as video.

## Method (how V-JEPA is applied to EEG)
- **EEG-as-video tensor.** Crop recordings to 5 min, keep a common **19 channels** (10–20 montage), downsample to **100 Hz**, bandpass **1–40 Hz**, channel-wise z-normalize. A sliding window makes overlapping **5 s windows of shape (19 × 500)**, stacked into a **3D tensor 118 × 19 × 500** (frames × channels × time). Channels play the "height" axis, time the "width" axis.
- **Patch / tubelet embedding.** 3D conv + linear projection turns non-overlapping spatiotemporal tubelets into tokens. Tubelet notation `h × w × t`; best configs `4 × 30 × 4` (ViT-M) and `4 × 30 × 2` (ViT-B). h≠w because input is 19×500.
- **Architecture = standard V-JEPA.** X-encoder (context, sees masked sequence) + Y-encoder (target, sees full unmasked sequence) + narrow predictor (embed dim 384). Backbones: ViT-S (5M), ViT-M (21M), ViT-B (85M), all L=12. Y-encoder = **EMA of X-encoder** (momentum 0.998→1.0).
- **Masking.** V-JEPA multi-block masking: large spatially-contiguous blocks spanning the full temporal axis, random aspect ratio, to force long-range spatiotemporal dependency learning.
- **Objective.** Latent L1 loss (MAE, "as recommended in [22] for stability"): `min_{θ,φ} ‖ P_φ(E_θ(x), Δy) − sg(E_θ(y)) ‖_1`, with stop-gradient + EMA preventing collapse. (Eq. 1.)
- **Training.** AdamW, WarmupCosine LR (warmup ~40 ep to 2e-4, decay to 6.25e-4 ref then 1e-6), cosine WD 0.04→0.4, grad-clip 10.0. Pretrain up to **400 epochs on 5× V100-32GB**; fine-tune/eval up to 500 ep on 1 V100.
- **Eval protocol.** Frozen (attention-pooling linear probe: cross-attention over token reps + residual + linear head) and full fine-tuning, both on the TUAB train/val subset.

**Does it use TUAB/TUH? Yes.** Pretraining set "EEGComb2" = **TUH [18] + NMT [32]** unlabeled (4438 subjects). Downstream benchmark = **TUAB** (TUH Abnormal Corpus: 2,993 recordings, 2,329 subjects, normal/abnormal labels). Also a second small clinical set (General Hospital of Thessaloniki, 88 subjects, AD/FTD/CN dementia) for generalization.

## Key results
TUAB eval subset: **276 normal / 270 abnormal train; 150 normal / 126 abnormal val.** Mean ± std over 5 runs. Best model = **ViT-M/4×30×4**.

- **Frozen (linear/attention probe), ViT-M/4×30×4:** Accuracy **83.30% ± 0.3**, F1 **82.4% ± 0.2**, AUROC **87.7% ± 0.2**.
- **Fine-tuned, ViT-M/4×30×4:** Accuracy **85.80% ± 0.5**, F1 **85.6% ± 0.4**, AUROC **88.5% ± 0.2**.
- **vs baselines (Tables 3–4):** EEG2Rep frozen 76.6% / FT 80.5% acc (AUROC 83.2 / 88.4); LaBraM FT 82.58% acc, AUROC 92.04%; CL model 80.55–81.9% acc; BSVT 82.67%; supervised ChronoNet 86.57% acc. Abstract headline: EEG-VJEPA beats EEG2Rep / LaBraM / CL by **6.4% / 4% / 2.45%**.
- **Note:** LaBraM still has the highest AUROC (92.04%); EEG-VJEPA wins on accuracy/F1 and ties supervised ChronoNet on accuracy.
- **Interpretability.** UMAP of 384-dim embeddings shows age gradient + pathology + (weak) gender structure. Attention-rollout maps localize diagnostic channels/time; PSD analysis: beta-band power drops 27.81% (normal) → 11.48% (abnormal), matching clinical literature.
- **Thessaloniki dementia (binary, FT, ViT-M/4×30×4):** Acc **83.34% ± 0.2**, F1 **83.48% ± 0.2** — strong but below the handcrafted-feature SVM (93.5%); frozen probe much weaker (67.0%).

## Relevance to the EB-JEPA hackathon (closest analogue to the flagship EEG/TUAB track)
This is the most direct, replicable blueprint for an EEG/TUAB flagship submission:
- **Exact data recipe to copy:** 5-min crop → 19 channels (10–20) → 100 Hz → 1–40 Hz bandpass → channel-wise z-norm → 5 s windows (19×500) → stack to 118×19×500. Reuses the same TUH/TUAB corpus the track targets.
- **Minimal port of an existing JEPA stack:** it is V-JEPA with (i) a 3D-conv tubelet embed sized for 19×500 inputs, (ii) `h × w × t` tubelet split where height=channels, width=time, and (iii) multi-block masking unchanged. A team with an eb_jepa / V-JEPA codebase mainly swaps the patch embedder + data loader; encoder/predictor/EMA/L1-loss are stock.
- **Concrete targets to beat:** frozen 83.3% / FT 85.8% accuracy on the stated TUAB subset; AUROC ~88%. LaBraM's 92% AUROC is the harder bar if AUROC is the track metric.
- **Best-known hyperparameters handed over:** ViT-M/4×30×4 sweet spot; full pretrain HPs in Tables 1–2 (batch 4–40, predictor depth 4/6/12, sampling rate 1–3, frames 16–64, clips 1–6, tubelet temporal 2–8, rrar (0.75,1.35) / rrs (0.3,1.0), spatial+noise+flip augs). Compute is modest (5× V100 / 400 ep) — feasible in a hackathon.
- **Eval scaffolding to reuse:** attention-pooling frozen probe + optional full FT, 5-seed mean±std, plus interpretability (UMAP, attention rollout, PSD band power) as differentiators beyond raw accuracy.

## Caveats & open threads
- **No code released** in the paper (no GitHub/HF link found in full text) — the port must be re-implemented from the description.
- **TUAB eval uses a small subset** (276/270 train, 150/126 val), not the full corpus split — be careful comparing to other TUAB numbers in the literature (different splits → not apples-to-apples).
- **AUROC trails LaBraM** (87.7–88.5% vs 92.04%); the "SOTA" claim is on accuracy/F1, so pick the metric the track scores on.
- **Hyperparameter tables are noisy** (typos: `(0.75,1-35)`, "rrar/rrs", inconsistent tubelet `(h,w,t)` vs `h×w×t` orderings) — the `4×30×4` / `4×30×2` configs and Table 2 rows are the load-bearing ones.
- **Lowest pretrain loss ≠ best val** (they flag pretext overfitting); smaller patches overfit; for tiny downstream sets, restrict FT to the last layer.
- Generalization set (Thessaloniki) frozen probe is weak (67%); gains there are FT-driven and still under a handcrafted SVM — pretraining diversity is the bottleneck.
