# Lifting World Models — train a lightweight policy to "lift" a frozen low-level world model to a low-dimensional, searchable high-level action space (2D waypoints) for tractable planning on a human-like embodiment

**Authors:** Alex N. Wang, Trevor Darrell, Pavel Izmailov, Yutong Bai‡, Amir Bar‡ (NYU; BAIR, UC Berkeley; ‡ equally advised) **Venue/Year:** arXiv preprint, 2026 (v1, 28 Apr 2026; cs.CV) **arXiv:** 2604.26182 **Repo:** none listed in paper

## TL;DR
Action spaces for complex (human-like) embodiments are high-dimensional (48-d per-joint actions), making world models hard to control and very expensive to plan with (CEM scales exponentially in action dim). The authors train a lightweight goal-conditioned **policy** π that maps a single **high-level action** to a sequence of T low-level joint actions, then compose it with a **frozen** low-level world model (PEVA) to form a **Lifted World Model (LWM)** that maps one high-level action → a sequence of future observations. The high-level action space is a set of 4 **2D waypoints** (pelvis, head, left/right hand) drawn on the current egocentric frame. CEM search over waypoints (8-d/step) beats CEM in raw joint space (48-d/step) by **3.8× lower mean joint error** to the goal, at lower compute, and generalizes to held-out environments.

## Problem & motivation
- World models predict o_{t+1}=f_φ(o_t, a_t). For human-like embodiments, a_t is per-joint (48 dims here), high-dim and hard to specify (e.g. reaching = coordinated shoulder/elbow/wrist).
- Search-based planning (CEM) scales poorly with action dimensionality × sequence length, so direct low-level planning is expensive and ineffective.
- Goal images o_g are a poor goal-conditioning signal for egocentric agents: the agent's own body is mostly out of frame, so o_g carries little info about the target pose.

## Method
- **High-level action space (waypoints).** a^HL = {w_pelvis, w_head, w_left_hand, w_right_hand}: 2D projections into the current frame o_t of near-term 3D goal positions for the 4 leaf joints. Annotated onto o_t → o_t^ann. Low-dim (8-d), visually interpretable, manually specifiable, searchable. Training labels obtained by forward-kinematics on goal pose p_g then 3D→2D camera projection (P).
- **Policy π (diffusion policy).** Predicts a_{t:t+T-1}^LL ~ π_θ(o_t, p_t, a^HL_t). Context: K_π=3 obs + poses. Encodes o_t and o_t^ann with a **DINOv3-S** encoder (tokens not pooled, to keep spatial info); pose context linearly projected and added; 3D positional embeddings; ViT; later spatial pooling → context vector c_t conditioning a denoising UNet (UNet dims raised from NoMaD's [64,128,256] to [256,384,512]). T=8, sampling 4 Hz.
- **Waypoint masking.** Half the time no masking; half the time each waypoint independently masked w.p. 0.5 → handles sparse/out-of-frame waypoints and infers unconditioned joints. Masking all = unconditioned distribution.
- **Lifted World Model.** o_{t+T}=f^HL(o_t, p_t, a^HL_t): π emits a^LL_{t:t+T-1}, the frozen PEVA world model autoregressively rolls them into observations. No change to the base world model.
- **Planning.** CEM in the high-level (waypoint) space: 8 vs 48 dims/step plus shorter sequence. Base WM: PEVA upper-body checkpoint, 15 joints, 48 action dims, context K=8, 64 denoising iters. CEM: up to 6 iters, 64 samples/iter; action prior N(0,σ²I), σ=0.05 for a^LL, σ=0.3 for a^HL; high-level planning horizon L=1. Trained on Nymeria (egocentric Project Aria video + XSens mocap, 50 environments). Metric: **mean joint error (MJE)** in meters (leaf / intermediate / all joints).

## Key results
- **Policy ablations (Table 1, val MJE):** Initial distance leaf MJE 0.445. Goal-conditioned all-MJE: Base Policy 0.392 → +arch 0.367 → +pose context 0.323 → +waypoint conditioning 0.243 → +waypoint masking 0.226 (best 2D) → +3D conditioning + masking 0.208. Waypoint conditioning improves goal-conditioned MJE by **8.8 cm** over unconditioned. Goal image o_g barely helps (NoMaD reduces MJE only 2.1 cm vs initial; adding o_g only another 1.3 cm).
- **Visibility (Table 2):** for non-visible joints, observation-conditioned policies degrade by **26.5 cm** vs only **12.7 cm** for waypoint conditioning; masking shrinks the non-visible drop to **8.8 cm**.
- **Decomposition (Table 3):** motion generation given goal pose (all MJE 0.105) is far easier than predicting goal pose from goal observation (0.279) → egocentric obs insufficient to identify goal pose.
- **Planning (Table 4, 128 tasks, 6 CEM iters × 64 samples):** Initial 0.704 all-MJE. PEVA CEM 0.616 (−8.8 cm only). **Lifted CEM (2D, ours) 0.374** (−33 cm); Lifted CEM (3D) 0.420; image-cond policy 0.585; uncond policy 0.650. The 3.8× headline = MJE reduction-to-goal ratio of Lifted vs PEVA CEM.
- **Efficiency (Fig 9):** Lifted CEM beats PEVA CEM at every sample budget (n=8/16/64) and every CEM iter count; PEVA CEM even *worsens* MJE after 1 step. 3D waypoints (12-d) lag 2D (8-d) with more iters.
- **Generalization (Table 5, unseen Nymeria locations 6/19/34):** Lifted CEM held-out policy all-MJE 0.362 vs base policy 0.333 vs PEVA CEM 0.553 — generalizes well to unseen environments.

## Relevance to the EB-JEPA hackathon
- **Track/modality:** action-conditioned video world models + planning; egocentric human video (Nymeria), high-dimensional whole-body action spaces. Directly in the **AC-Video-JEPA / planning** lane (CEM/MPC over learned world models), the same family as the repo's `ac_video_jepa` and `h_ac_video_jepa` planning-eval work.
- **Map to recipe — hierarchical action abstraction.** This is essentially a *hierarchy in action space* layered on a frozen flat world model: a learned low→high "lifting" policy turns a 48-d/step joint search into an 8-d/step waypoint search. Mirrors the repo's H-AC-Video-JEPA motivation (level-2 abstract actions, plan-eval with reduced search dimensionality). The key transferable idea: rather than a hierarchical *encoder*, keep the base model frozen and add a lightweight policy that exposes a low-dim, searchable, interpretable high-level action interface for CEM/MPPI planning.
- **Concrete cross-pollination:** (i) 2D image-space waypoints as a searchable high-level action space; (ii) waypoint masking to support sparse/out-of-frame goals; (iii) the empirical finding that *goal images are weak conditioning for egocentric agents* (relevant to any goal-image-conditioned JEPA planning setup); (iv) CEM-in-abstract-space is both better and cheaper — a direct argument for the hierarchical planning-eval story.

## Caveats & open threads
- Base world model is **frozen PEVA** (diffusion/pixel-space egocentric predictor), not a JEPA; the lifting idea is model-agnostic but results are pixel-space, not latent-energy.
- No explicit head-orientation control → agent can look around, perturbing world-model-generated observations.
- Waypoints cannot specify goals fully outside the current frame (world models hallucinate unseen space); high-level planning horizon fixed L=1, policy length fixed T=8.
- 3D (depth-augmented) waypoints use *privileged* depth and only help ≤2 cm in policy MJE, and lag 2D in search — so the cheap 2D interface is the recommended one.
- Both Lifted and PEVA CEM degrade on very-close tasks (initial MJE ≤0.3 m, Fig 11).
- No public code/repo referenced; Nymeria + PEVA upper-body checkpoint required to reproduce.
