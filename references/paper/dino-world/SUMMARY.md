# DINO-world — a generalist video world model trained to predict future frames in the frozen latent space of DINOv2

**Authors:** Federico Baldassarre, Marc Szafraniec, Basile Terver, Vasil Khalidov, Francisco Massa, Yann LeCun, Patrick Labatut, Maximilian Seitzer, Piotr Bojanowski (Meta FAIR)
**Venue/Year:** arXiv preprint, July 2025
**arXiv:** 2507.19468 (https://arxiv.org/abs/2507.19468)
**Repo:** — (none released; encoder is public DINOv2, training data is a private web-video pool)

## TL;DR
DINO-world freezes a DINOv2 image encoder and trains a large transformer **predictor** to forecast next-frame patch features on ~60M uncurated web videos. Decoupling representation learning (frozen DINOv2) from dynamics modeling makes training cheap (<1B params vs. COSMOS's 4–12B / 22M GPU-hours) while beating prior world models on dense feature forecasting (segmentation/depth) and matching them on intuitive-physics probes. The unconditional predictor can then be cheaply **fine-tuned with action blocks** on observation-action trajectories, enabling planning by rolling out candidate action sequences in latent space.

## Problem & motivation
Pixel-space generative world models (SORA, COSMOS, Wan2.1) are enormously resource-hungry (e.g. 22M GPU-hours, up to 12B params for COSMOS) and spend capacity modeling irrelevant pixel detail. Action-annotated video is scarce and task-specific, limiting most successful world models to narrow domains (driving, games). The paper argues for: (1) separating large-scale unconditional video pre-training from later action-conditioned fine-tuning to reduce the demand for action-labeled data; (2) predicting in a high-level semantic latent space instead of pixels; (3) reusing a frozen SSL foundation encoder (DINOv2) to bootstrap semantic/geometric understanding and avoid the difficulty of jointly training encoder+predictor (the V-JEPA path). It also pushes a *comprehensive* evaluation suite spanning dense forecasting, intuitive physics, and planning, since prior world-model evals are narrow.

## Method
**Frozen encoder.** Frames are encoded with a frozen **DINOv2 ViT-B/14 with registers** (last layer only, D=768, same encoder as DINO-Foresight). Each frame v_t -> patch-token tensor x_t in R^{H×W×D}; the patch token x_{t,i,j} is the unit of state the model predicts. The encoder is never updated (no EMA target needed — the prediction targets are just the frozen encoder's outputs).

**Predictor architecture.** A stack of **N residual pre-norm cross-attention blocks** framed as a decoding problem (à la NMT / masked image reconstruction). To predict a future token at coordinates (τ_{t'}, i', j'), a learnable **query token** q cross-attends to key-value pairs from all *past* patch tokens, then passes through an MLP: q <- q + CrossAttn(LN(q), {x_{t,i,j} | τ_t < τ_{t'}}); q <- q + MLP(LN(q)). A final linear projection maps q to the predicted patch token x̂ in R^D. Default predictor: **N=40 blocks, dim D'=1536, 24 heads (~1.1B params, "giant"/ViT-g scale but cross-attention)**.

**Positional encoding.** A **3-axial RoPE**: the head dimension is split in three to encode temporal, horizontal, and vertical coordinates separately. Spatial coords use *relative* positions on a [-1,+1]^2 grid (resolution-invariant); the temporal coord uses *absolute timestamps in seconds* so the model handles variable FPS and extrapolates to longer videos. RoPE periods in [10^-2, 10^2].

**Training objective.** Next-frame prediction (t' = t+1) with **teacher forcing** and a **smooth L1 loss** on predicted vs. frozen-encoder features. Given T frames, all (T-1)·H·W queries are stacked and trained *in parallel* with a **block-triangular (block-causal) attention mask** so a query for frame t+1 attends only to tokens up to frame t. Unlike V-JEPA / DINO-Foresight masked-reconstruction (loss on a few mask tokens only), DINO-world computes a loss on *every* token.

**Variable FPS.** Instead of contiguous frames (which skews Δτ toward short intervals), per video they sample T-1 time deltas uniformly from [Δτ_min, Δτ_max], cumulatively sum them (random start), and decode the nearest real frame + its true timestamp — giving a uniform distribution over prediction horizons.

**Action-conditioned fine-tuning.** Starting from the pre-trained predictor, after each block they insert a zero-/identity-initialized **action block** that updates the query as q + MLP(LN([q, a_t])). These can be trained with a small action dataset; optionally the whole video model stays **frozen** and only action blocks train (less overfitting, one base model reused across tasks). The paper explicitly contrasts this with **DINO-WM**'s interleaving of action tokens into the patch sequence, which complicates batching/masking, adds capacity needs, and forces full fine-tuning that can destroy learned video understanding.

**Training setup.** AdamW, **300k iterations**, batch **1024 clips**, **T=8**, **224×224**; then **50k more iterations at 448×448**. Constant LR 1e-4 after warmup. Data: a private pool of **~66M (≈60M) uncurated web videos**, 5–60s, varied frame rates.

## Key results
**Dense forecasting (Table 1)** — train a "present-time" linear head on DINOv2 features, apply to predictions at ~200ms (short) / ~0.5s (mid). DINO-world uses a ViT-B encoder:
- **VSPW mIoU:** present 52.8, short **51.6**, mid **47.0** (best). COSMOS-12B: 46.6 / 40.7; V-JEPA ViT-H: 4.9 / 4.6; DINO-Foresight: 44.7 / 37.7. Intro highlights **+6.3% mIoU over the second-best at 0.5s**.
- **Cityscapes mIoU:** present 68.6, short **64.7** (best), mid 55.1 (DINO-Foresight edges mid at 57.2 due to driving-domain training).
- **KITTI depth RMSE (↓):** short **3.214** (best), mid 4.268 (DINO-Foresight best on KITTI: 3.562 / 3.740, again domain-specific driving).
- DINO-world clearly beats V-JEPA (which collapses on forecasting: VSPW mid 4.6–7.7) and the much larger pixel-space COSMOS.

**Intuitive physics (Table 2)** — surprise = MAE between predicted and encoded features (perplexity for COSMOS). Mean relative accuracy:
- **IntPhys:** DINO-world **91.3** (COSMOS-4B 99.5 near-perfect on the simplest; V-JEPA ViT-H 89.4; DINO-Foresight 87.8).
- **GRASP:** DINO-world **76.0** (best; V-JEPA ViT-H 73.0, COSMOS-4B 60.1).
- **InfLevel:** DINO-world **63.7** (best; DINO-Foresight 62.8, V-JEPA ViT-H 59.9, COSMOS-4B 44.8).
- DINO-world (ViT-B encoder, 1.1B predictor) matches/beats V-JEPA ViT-H despite a smaller encoder. Authors treat these as a noisy sanity check, not a hard benchmark.

**Ablations (Table 3)** — metrics: IntPhys / Cityscapes-mid / VSPW-mid:
- *Predictor size:* Base (86M) 84.9/47.7/45.4 → Large (304M) 89.1/51.9/46.4 → **Giant (1.1B) 90.6/53.2/46.8**. Clear scaling; dynamics need more capacity than static spatial modeling.
- *Training data:* Cityscapes-only 66.7/45.6/23.1; SSv2 79.3/44.9/45.2; **Ours 66M web 90.6/53.2/46.8**. Scale + diversity is crucial.
- *Encoder:* SD3.5 VAE –/13.0/1.5 (fails); SigLIP2 SO400M 80.7/50.5/41.0; **DINOv2 90.6/53.2/46.8** (best). VAE features are not for understanding; SigLIP2 features are noisier from vision-language pre-training.

**Direct vs. autoregressive (Fig. 3):** direct prediction wins at short Δτ; autoregressive rollout holds better at longer horizons; both degrade as the interval nears ~1s (long-horizon forecasting remains a limitation).

**Planning (Table 4)** — success rate over 512 episodes/env, setup of Zhou et al. (DINO-WM); action blocks trained 25 epochs, T=4 clips at 224px from the "base" model:
| Model | PushT | Wall | PointMaze |
|---|---|---|---|
| Scratch | 46.9 | 87.1 | 59.4 |
| Action-only (base frozen) | 49.4 | 91.1 | 61.6 |
| **Fine-tuned** | **59.4** | **93.8** | **68.7** |
Large-scale pre-training + fine-tuning beats training the predictor from scratch on every environment; the benefit is expected to grow for environments closer to the pre-training distribution.

## Relevance to the EB-JEPA hackathon
DINO-world is squarely in the **frozen-encoder latent video world-model** family that the repo's `ac_video_jepa` example targets: predict-in-latent + action-conditioned planning. It sits between two reference points already summarized here:
- **vs. DINO-WM** (also frozen DINOv2 + latent predictor, plus MPC planning): DINO-world scales the *predictor* to ~1.1B and trains on ~60M uncurated web videos rather than per-environment data, and conditions on actions via **appended zero-init action blocks** instead of DINO-WM's **interleaved action tokens** — explicitly argued to keep batching/masking simple and avoid destroying pretrained video knowledge during full fine-tuning. This is a directly transferable design choice for the repo's action-conditioning code.
- **vs. V-JEPA 2** (encoder *and* predictor trained jointly via masked latent prediction): DINO-world keeps the encoder **frozen** (no EMA target — targets are the frozen encoder's features) and shows joint-trained V-JEPA features are strong for summarization but weak for *forecasting/planning* (V-JEPA collapses to ~5 mIoU on VSPW forecasting). Useful evidence in the frozen-vs-jointly-trained-encoder debate central to EB-JEPA.

Concrete takeaways for the hackathon: the **block-causal cross-attention decoder with 3-axial RoPE and absolute-second temporal coords**, **dense per-token smooth-L1 loss** (vs. mask-only losses), **uniform-Δτ FPS sampling**, and the **two-stage "pretrain unconditional → add action blocks → plan by latent rollout"** recipe are all reusable patterns. The planning protocol (PushT / Wall / PointMaze, 512 eps, DINO-WM setup) overlaps with the repo's two_rooms / Push-T planning eval.

## Caveats / open threads
- Predictions live in DINOv2 latent space and **cannot be rendered to pixels** (only PCA-visualized); evaluation relies on dense linear-head proxies, not pixel fidelity.
- **Long-horizon forecasting degrades by ~1s** for all models; teacher-forced next-frame training does not address the multimodality of the future (authors suggest sampling one of many possible futures as future work).
- Trained on a **private ~66M-video pool** (not released) and an unreleased model — reproducibility is limited to the public DINOv2 encoder + architecture description.
- Intuitive-physics scores are **noisy** (distribution shift, long context needed); authors frame them as a sanity check, not a benchmark.
- Planning is validated only on **three simulated RL environments**; no real-world/robot validation. Language conditioning is left to future work.
