# NWM (Navigation World Models) — a controllable diffusion video world model for visual navigation that plans by simulating and scoring trajectories

**Authors:** Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun (FAIR at Meta; NYU; Berkeley AI Research) **Venue/Year:** CVPR 2025 (arXiv v1 Dec 2024, v2 Apr 2025) **arXiv:** 2412.03572 **Repo/Project:** https://amirbar.net/nwm

## TL;DR
NWM is a 1B-parameter controllable video generation world model that predicts future visual observations from past observation latents + navigation actions. Its core architectural contribution is the **Conditional Diffusion Transformer (CDiT)**: cross-attention from the denoised target frame to past-frame tokens (no self-attention over context), making compute **linear** in context length instead of quadratic — ~4x fewer FLOPs than a standard DiT at equal params, with better future prediction. After training on diverse robot + human egocentric navigation video, NWM plans navigation via MPC + Cross-Entropy Method (CEM), either standalone or by re-ranking trajectories sampled from an external policy (NoMaD). Standalone planning reaches **ATE 1.13 / RPE 0.35 on RECON**, beating NoMaD (1.93/0.52) and GNM (1.87/0.73).

## Problem & motivation
Supervised navigation policies (GNM, NoMaD) are "hard-coded" after training: new constraints (e.g. "no left turns") cannot be injected, and they cannot allocate more compute to hard problems. NWM instead learns an environment simulator that can be queried at inference: imagine candidate trajectories, score whether they reach a goal, and impose arbitrary action/state constraints during planning.

## Method
- **Formulation.** Latent world model F: given past m observation latents s_tau (encoded by a frozen Stable Diffusion VAE) and action a_tau, sample next latent s_{tau+1}. Action a = (u, phi, k): translation u in R^2, yaw phi, plus a **time-shift k in [-16s, +16s]** so the model also learns temporal dynamics. Actions over a horizon are summed (Eq. 2). Multiple goals per state during training create natural counterfactuals to disentangle action vs. time.
- **CDiT block.** First attention restricted to target-frame tokens being denoised; a cross-attention layer lets each target query attend to past-frame keys/values, contextualized via a skip connection. Action, time-shift, and diffusion-timestep scalars -> sine-cosine -> 2-layer MLP, summed into xi (Eq. 3), fed to AdaLN to modulate LayerNorm + attention outputs. Unlabeled data handled by omitting the action term in xi. Complexity O(m n^2 d) (linear in frames m) vs DiT's O(m^2 n^2 d).
- **Training.** Standard DiT diffusion: predict clean target latent under MSE (L_simple) + predict noise covariance with the VLB loss. DiT noise schedule/hyperparams. Default: CDiT-XL, 1B params, context 4 frames, 4 goals, batch 1024 (x4 goals = 4096 effective), AdamW lr 8e-5, 8x8 H100.
- **Planning.** Energy E = -perceptual-similarity(s_T, s*) + large indicator penalties for invalid actions / unsafe states (Eq. 4). Similarity = LPIPS/DreamSim on VAE-decoded pixels. Minimize over action sequence (Eq. 5) via MPC + CEM (gradient-free). Two modes: **standalone planning** (optimize the trajectory) and **ranking** (sample n in {16,32} trajectories from NoMaD, roll out each through NWM, pick lowest energy).

## Key results
- **Standalone planning (RECON, 2s trajectories):** NWM ATE **1.13 +/- 0.02**, RPE **0.35 +/- 0.01** — best of all; NoMaD 1.93/0.52, GNM 1.87/0.73. Ranking NoMaD with NWM (x32) gives 1.78/0.48.
- **CDiT vs DiT:** CDiT matches/beats DiT with <2x FLOPs at up to 1B params; CDiT-L beats DiT-XL while being ~4x faster (Fig. 5).
- **Ablations (4s prediction on RECON):** 4 goals > 1 (LPIPS 0.296 vs 0.312); 4 context frames > 1; action+time (LPIPS 0.295) > action-only (0.318) >> time-only (0.760). Both action and time conditioning matter.
- **Video quality (16s @ 4 FPS, RECON):** FVD **200.97** vs DIAMOND **762.73**. NWM far more accurate than DIAMOND over time; 4 FPS variant overtakes 1 FPS after ~8s as error accumulates.
- **Constraints:** all three tested action constraints (forward-first, left-right-first, straight-then-forward) satisfied with only minor cost vs unconstrained (Table 3).
- **Unknown environments:** adding unlabeled Ego4D (time-shift action only) improves Go Stanford (unknown) predictions (LPIPS 0.658->0.652, DreamSim 0.478->0.464) but degrades the known RECON env — imagination generalization at a cost.

## Relevance to the EB-JEPA hackathon
Directly on the **world-model / planning track** and the **video + action-conditioned** modality, which is the same regime as the repo's `ac_video_jepa` (DROID, Two Rooms, Push-T) action-conditioned video JEPAs. Maps onto the encoder/predictor/regularizer recipe as a contrast point:
- **Encoder:** frozen SD-VAE producing pixel-decodable latents (vs JEPA's learned/EMA target encoder operating in representation space). NWM keeps latents pixel-decodable specifically so planning energy can be measured via LPIPS/DreamSim in pixel space.
- **Predictor:** the CDiT itself is an action+time-conditioned predictor; the **linear-in-context cross-attention-to-past-frames design is a transferable predictor architecture** for AC-Video-JEPA's predictor (cheaper long context than full DiT self-attention). The time-shift k action is a clean trick for variable-horizon prediction.
- **Regularizer / objective:** this is a **generative diffusion** world model, NOT energy-/JEPA-style in latent space — no collapse-avoidance regularizer; the "energy" here is an inference-time planning cost (perceptual similarity + constraint indicators), not a training loss. Useful as the generative baseline to contrast against EB-JEPA's latent energy. Note: by LeCun, and the Discussion explicitly gestures at self-supervised perceive-and-plan systems; the closest in-family cite is DINO-WM [77] (Zhou & LeCun), JEPA-adjacent latent world model + zero-shot planning.
- **Planning recipe:** MPC + CEM over an energy with hard constraint indicators is a directly reusable planning loop for the repo's plan-eval pipelines.

## Caveats & open threads
- **Mode collapse / context loss out-of-distribution:** on OOD data the model drifts back toward training-data-like frames (named explicitly as mode collapse); struggles to simulate dynamic agents (e.g. pedestrian motion).
- **Only 3-DoF actions** (translation in plane + yaw); 6-DoF / arm control left to future work.
- **Generative, pixel-space scoring** is expensive (decode + LPIPS, sampled 3-5x for averaging) — contrast with latent energy scoring in JEPA which avoids decoding.
- Unlabeled-data gain in unknown envs comes with degradation in known envs (Table 4 trade-off), not a free lunch.
- Numbers above are RECON-centric; reported with mean +/- std over 5 samples.
