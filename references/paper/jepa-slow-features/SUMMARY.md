# JEPAs Focus on Slow Features — JEPA world models trained with VICReg/SimCLR latch onto the slowest-changing features and collapse to fixed background noise, ignoring the agent

**Authors:** Vlad Sobal, Jyothir S V, Siddhartha Jalagam, Nicolas Carion, Kyunghyun Cho, Yann LeCun (NYU / Meta AI / Prescient Design–Genentech / CIFAR). **Venue/Year:** NeurIPS 2022 (Self-Supervised Learning workshop / SSL-Theory-Practice track), 2022. **arXiv:** 2211.10831 (v1, 20 Nov 2022). **Repo:** https://github.com/vladisai/JEPA_SSL_NeurIPS_2022

## TL;DR
The paper studies offline, reward-free world-model learning on a toy "moving dot" environment with background distractors, comparing JEPA (VICReg- and SimCLR-based), pixel reconstruction, inverse dynamics modeling (IDM), supervised, and random baselines, probing the learned representations linearly for the dot's location. JEPA matches or beats reconstruction when distractor noise **changes every frame**, but **fails catastrophically when the noise is fixed** (static across the sequence but resampled per sequence). The cause is identified both theoretically and empirically: JEPA's prediction + variance/covariance (or InfoNCE) objective is trivially minimized by copying the slowest-varying input feature, so the encoder latches onto temporally-constant background noise and ignores the moving dot.

## Problem & motivation
World models are usually trained with pixel reconstruction, which suffers from "object-vanishing" (the objective does not prioritize task-relevant objects). JEPA is a reconstruction-free alternative, but the paper asks whether JEPA's joint-embedding objective introduces its own shortcut: focusing on easily-predictable "slow features" (Wiskott & Sejnowski 2002, ref [37]) rather than on the controllable agent. Setting: offline MDP M=(O,A,P,R), reward unknown at pretraining; learn encoder g_φ(o_t)=s_t (D-dim) and forward model f_θ(s_t,a_t)=s̃_{t+1}, unrolled auto-regressively, then probe with a single linear layer for dot location.

## Method
- **VICReg-JEPA** (Fig 1a): per-timestep variance + covariance losses plus an L2 prediction loss between forward-model output s̃_t and encoder output s_t. Total L = α·L_pred + β·L_var + L_cov.
- **SimCLR-JEPA** (Fig 1b): InfoNCE loss with (encoder output, forward-model output) at the same step as positive pairs, symmetric InfoNCE(S_t, S̃_t)+InfoNCE(S̃_t, S_t).
- **Baselines:** Reconstruction (adds decoder d_ξ, pixel L2); **IDM** (linear layer predicts a_t from g_φ(o_t),g_φ(o_{t+1}) — makes encoder attend to action-affected parts); Supervised (end-to-end, error lower bound); Random (frozen random weights); Center (always predict center).
- **Architecture:** 3-conv encoder (ReLU+BatchNorm) → linear to 512-dim; predictor is a single-layer **GRU** (hidden 512, action input). 28×28 images, episode length 17 (16 actions).
- **Dataset:** single dot in a unit square, action = coordinate delta (‖a‖≤D=0.14, Von Mises directions). Distractors: **uniform** (random pixels) or **structured** (overlaid CIFAR-10), each either **changing** (resampled per frame) or **fixed** (constant within a sequence, resampled per sequence), with brightness coefficient α∈[0,3]. 1M pretrain / 300k prober / 10k eval sequences; hyperparameters tuned per method/noise.

**Theoretical argument (the core result).** For fixed noise, a trivial solution exists: g_φ(o_t)=s∼N(0,σ²I) directly copies the static noise; since noise is persistent, s_t=s_{t+1}, and the forward model converges to identity f_θ(s,a)=s. Then L_prediction=0 (Eq 1), L_variance=0 for large enough σ (Eq 3), L_covariance=0 because noise dims are independent across episodes (Eq 4) — **all VICReg terms vanish**. For SimCLR, by Wang & Isola (2022, ref [36]) Theorem 1, InfoNCE is minimized when positive pairs are perfectly aligned and outputs are uniform on the sphere, both satisfied by this trivial solution. So both objectives are provably susceptible to fixed distractor noise.

## Key results
- **No distractors:** all methods perform well.
- **Changing noise (uniform & structured):** JEPA performs **on par or better than reconstruction**, even up to high noise (α up to 3). JEPA also needs **no per-noise hyperparameter retuning**, whereas reconstruction degrades badly with untuned hyperparameters (Fig 5 / App A.5).
- **Fixed noise (uniform & structured):** JEPA (VICReg & SimCLR) **fails**; reconstruction stays good for α≤1.5 (Fig 3). RMSE is plotted as mean over 17 timesteps, 3 seeds.
- **3-dots dataset** (Table 1, RMSE over 17 steps, 3 seeds): action / random / stationary dots.
  - VICReg: Avg 0.229±0.031, Action 0.277±0.041, Random 0.273±0.044, **Stationary 0.066±0.026** → captures only the slow (stationary) dot.
  - SimCLR: Avg 0.158±0.001, Action 0.193, Random 0.193, **Stationary 0.025±0.001** → same pathology.
  - **IDM:** **Action 0.035±0.000** (excellent) but Stationary 0.272 (ignores the static dot — IDM only captures action-relevant content).
  - Reconstruction: Action 0.021, Stationary 0.026 (captures both); Supervised Action 0.010, Stationary 0.005 (lower bound).
- **Interpretation:** JEPA collapses onto whichever feature changes slowest (fixed background, or the stationary dot); IDM collapses onto only the action-controllable feature; reconstruction is the only objective tested that captures both fast and slow task-relevant features.

## Relevance to the EB-JEPA hackathon
This is the canonical analysis of **why JEPA world models collapse to spurious / slow features in randomized environments**, and it is the direct conceptual motivation for **eb_jepa's IDM loss**: the paper shows (i) plain VICReg/SimCLR JEPA latches onto temporally-constant nuisance signal and ignores the agent, and (ii) **inverse dynamics modeling is the complementary fix** — IDM forces the encoder to represent action-affected content (Action RMSE 0.035 vs VICReg 0.277), exactly the failure mode the prediction+variance loss alone does not fix. The flip side the paper flags (IDM ignores task-relevant *static* content, e.g. the stationary dot) is the trade-off any IDM-regularized recipe must watch. Most relevant to the **AC-video / world-model tracks** and to the **regularizer / loss-design story**: it supplies the toy diagnostic (changing vs fixed distractors, single-dot probing, 3-dots channel separation), the proof that VICReg + L2-pred is trivially minimized by static noise, and the proposed remedies — **architectural hierarchy (HJEPA)** or **an objective term that prevents representations from being constant across time** — both of which map onto eb_jepa design choices.

## Caveats & open threads
- **Toy-only.** All evidence is a 28×28 moving-dot environment with a tiny 3-conv/GRU model; the authors explicitly note (App A.8) it is "unclear that the same will hold for more complicated video datasets and bigger models." Modern image/video JEPAs (I-JEPA, V-JEPA) on real data are not tested here.
- **Limited objective coverage.** Only VICReg and SimCLR JEPA variants are tested; Barlow Twins, BYOL, etc. (refs [10],[38]) are not, and EMA/stop-gradient target encoders are absent.
- **Proposed remedies are conjectural** — HJEPA and a non-constancy constraint are suggested in the conclusion but **not implemented or evaluated** in this paper.
- **Suggested input fixes have caveats:** the authors note image differences / optical flow could help but "will ignore potentially useful background and may still contain fixed noise."
- **Compute:** per noise type/level, 100 random hyperparameter trials, best run for 3 seeds; each run <1 GPU-hr (AMD MI50 / Nvidia RTX 8000).
