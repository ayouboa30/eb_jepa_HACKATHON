# V-JEPA 2 — Internet-scale action-free video JEPA + a frozen-encoder action-conditioned world model that plans zero-shot on real Franka arms

**Authors:** M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. R. Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, N. Ballas (FAIR at Meta; Mila / Polytechnique Montréal). **Venue/Year:** arXiv preprint, 2025 (v1 11 Jun 2025; 48 pages, 19 figures). **arXiv:** 2506.09985 **Repo:** https://github.com/facebookresearch/vjepa2  (blog: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks)

## TL;DR
V-JEPA 2 is a self-supervised video JEPA (action-free, mask-denoising latent prediction) pretrained on >1M hours / 22M samples of internet video at ViT-g/1B scale, reaching SOTA on motion understanding (**77.3** top-1 SSv2), human action anticipation (**39.7** recall@5 on Epic-Kitchens-100, +44% relative), and, after aligning with an 8B LLM, SOTA video-QA (**84.0** PerceptionTest, **76.9** TempCompass). Post-training a *frozen* encoder with a 300M action-conditioned predictor (**V-JEPA 2-AC**) on **<62 hours** of unlabeled Droid robot video gives a latent world model that plans **zero-shot** via energy minimization (CEM-MPC) on real Franka arms in two new labs (reach / grasp / pick-and-place from image goals), with no in-lab data, no task-specific training, and no reward.

## Problem & motivation
Goal (LeCun 2022): learn to understand and act in the physical world largely by observation. Prior world models either train on scarce interaction data alone (poor scaling) or train action-conditioned video *generation* models that look plausible but rarely control real robots and are expensive to plan with (generating pixels). JEPA predicts in learned representation space, modeling the predictable abstract structure (object trajectories) while ignoring unpredictable pixel detail, enabling cheap latent planning and use of action-free internet video.

## Method
**V-JEPA 2 pretraining.** Mask-denoising feature prediction: encoder E_theta, predictor P_phi, loss `min ||P_phi(Delta_y, E_theta(x)) - sg(EMA_E_theta(y))||_1` (L1 on masked patches only, EMA teacher + stop-gradient anti-collapse), V-JEPA multiblock masking, 2x16x16 tubelets. **3D-RoPE** (feature dim split into T/H/W rotations) replaces sincos PE and stabilizes the largest models. Data = **VideoMix22M** (22M samples, >1M hours: SSv2, Kinetics-400/600/700, HowTo100M, retrieval-curated YT-Temporal-1B "YT1B" 19M/1.6M h, ImageNet as 16-frame static clips); cluster-retrieval curation gives **+1.4 pts**. Four cumulative scaling ingredients (84.2 -> 88.2 avg over ViT-L baseline): data scaling 2M->22M (+1.0), model scaling 300M ViT-L -> 1B ViT-g/16 (+1.5), longer training 90K->252K iters (+0.8), higher/progressive resolution (256->384, 16->64 frames at cooldown). Model family: ViT-L 300M, ViT-H 600M, **ViT-g 1B** (width 1408, depth 40, 22 heads); predictor fixed = ViT-s ~22M. **Progressive-resolution training** (warmup-constant-cooldown) gives **8.4x** GPU-time reduction vs full 64x384x384 from scratch (~60 GPU-years).

**V-JEPA 2-AC (action-conditioned world model).** Freeze the ViT-g encoder (used as a per-frame image encoder -> z_k in R^{16x16x1408}); train a new **~300M-param predictor** (24 layers, 16 heads, hidden 1024). Data: raw **Droid** (7-DoF Franka + gripper), **<62 hours / 23k trajectories** (successes AND failures), 256x256, 4 fps -> 16-frame clips. State s_k = 7D (3 pos + 3 Euler orientation + 1 gripper); action a_k = 7D delta between adjacent frames. **Block-causal attention** over temporally-interleaved (a_k, s_k, z_k) tokens; 3D-RoPE on patch tokens, temporal-only RoPE on action/pose tokens. Objective `L = L_teacher-forcing + L_rollout` (both L1; rollout = 2 autoregressive steps to curb error accumulation).

**Planning / control.** Energy = L1 distance to the goal embedding in latent space: `E(a_{1:T}) = ||P(a_{1:T}; s_k, z_k) - z_g||_1` (goal image x_g -> z_g). Optimized with the **Cross-Entropy Method** (gradient-free) under receding-horizon **MPC** (execute first action, re-observe, re-plan). Actions bounded to an L1-ball radius 0.075 (~13 cm/step); pick-and-place uses 3 image sub-goals (grasp / near-goal / placed). Energy landscape is empirically smooth and locally convex (Fig 9).

**LLM alignment for VideoQA.** LLaVA-style non-tokenized early fusion: project patch embeddings into the LLM via a learnable MLP, 3-stage instruction tuning. SOTA setup uses 88.5M image/video-text pairs + Llama 3.1 8B. Notably the first MLLM built on a video encoder trained with *no* language supervision.

## Key results
- **Understanding (frozen attentive probe):** ViT-g384 avg **88.2**; **SSv2 77.3**, Diving-48 90.2, Jester 97.8, K400 87.3, COIN 91.1, IN1K 85.1. SSv2 75.3 (ViT-g 256) vs InternVideo2 69.7, PEcoreG 55.4.
- **Prediction (EK100 anticipation, mean-class recall@5):** ViT-L 32.7 -> ViT-H 36.5 -> ViT-g 38.0 -> **ViT-g384 39.7** (verb 63.6, noun 57.1), beating prior SOTA PlausiVL-8B (27.6) by +12.1 = **+44% relative**.
- **VideoQA (SOTA, Llama 3.1 8B):** ViT-g384 avg **59.5**; **PerceptionTest 84.0**, MVP 44.5, **TempCompass 76.9**, TemporalBench 36.7, TOMATO 40.3, TVBench 60.6, MVBench 73.5. Beats PerceptionLM-8B on 5 of 7 benchmarks.
- **Robot planning (zero-shot, 2 labs, 10 trials each, success rate):** V-JEPA 2-AC -- **Reach 100%**, Grasp cup 65% / box 25%, Reach-w/-obj 75% / 75%, **Pick-and-Place cup 80% / box 65%**. Octo baseline far lower (P&P cup 15% / box 10%).
- **Efficiency vs Cosmos-7B (Lab 2):** V-JEPA 2-AC plans at **16 sec/action** (vs Cosmos **4 min/action**, full P&P >1 hour) on a single RTX 4090, with much higher success; single-goal reaching gets within **<4 cm** of the goal.

## Relevance to the EB-JEPA hackathon
This is the reference blueprint for the **action-conditioned video-JEPA / robotics-planning** track (most relevant track: AC-Video-JEPA on Droid; modality: video + robot actions).
- **Energy-based planning is the core.** The control objective is an explicit energy `E = ||P(a; z, s) - z_g||_1` minimized over actions -- exactly the EB-JEPA framing; the landscape is empirically smooth/convex (a target for gradient-based planners in EB variants).
- **Frozen-encoder + lightweight AC-predictor recipe** (300M, block-causal attention, 3D-RoPE on patches, temporal RoPE on action/pose tokens, interleaved a/s/z tokens, teacher-forcing + 2-step rollout L1 loss) is a concrete architecture to mirror.
- **Droid preprocessing matches the repo:** 4 fps, 16-frame, 256^2 clips, 7D end-effector state, 7D delta actions, <62 h / 23k trajectories (successes + failures, unlike BC baselines).
- **Latent (not pixel) planning** is the efficiency argument (16 s vs 4 min/action) motivating JEPA latent world models for closed-loop MPC.
- **Open threads to attack:** hierarchical/multi-scale predictors for long horizons, language-goal conditioning, gradient-based planning exploiting the convex energy, learned proposals to warm-start CEM.

## Caveats & open threads
- **Camera sensitivity:** no calibration; the action coordinate axis is inferred from monocular RGB and breaks if the robot base is not visible (camera hand-tuned).
- **Long horizon:** autoregressive latent rollouts accumulate error; action search grows exponentially with horizon, hence reliance on image sub-goals (~16 s prediction limit).
- **Image-goal dependence:** no language-goal conditioning yet.
- **EK100 not solved** (closed vocab, kitchen-only, degrades at longer horizons); VideoQA loses to PerceptionLM on TVBench/MVBench and is sensitive to alignment-data scale.
- **Scale:** only to 1B params; cross-encoder comparisons are system-level (different training data per baseline, not apples-to-apples).
