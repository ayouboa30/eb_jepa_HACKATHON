# PEVA — Whole-body pose-conditioned egocentric video prediction as a world model

**Authors:** Yutong Bai*, Danny Tran*, Amir Bar* (equal), Yann LeCun†, Trevor Darrell†, Jitendra Malik† (equal advising) — UC Berkeley (BAIR), FAIR Meta, NYU
**Venue/Year:** arXiv preprint, 2025 (v1, 26 Jun 2025; cs.CV)
**arXiv:** 2506.21552
**Repo / project page:** https://dannytran123.github.io/PEVA

## TL;DR
PEVA ("Predict Ego-centric Video from human Actions") is an autoregressive **conditional diffusion transformer** (CDiT) that predicts the next egocentric video frame given past frames and an action defined as a **relative 3D whole-body pose change** (root translation + 15 upper-body joints' relative Euler rotations, dact = 3 + 15×3 = 48). Trained on the real-world Nymeria ego-video + mocap dataset, it beats CDiT/Diffusion-Forcing baselines on single-step LPIPS/DreamSim/FID, follows atomic actions (hand up/down/left/right, body forward/rotate), rolls out coherently to 16 s, and supports preliminary CEM-based planning by simulating action candidates and scoring against a goal image. This is a **pixel-generative world model, not a JEPA** (no joint-embedding latent prediction, no anti-collapse loss).

## Problem & motivation
Embodied agents must understand how whole-body motion shapes first-person perception (you move your body to reveal new info). Prior video world models (e.g. NWM, Bar et al. 2025) condition on low-dimensional controls (velocity/heading) and ignore the agent's own body dynamics. PEVA conditions on full structured kinematic pose to get physically grounded ego-video prediction.

## Method
- **Structured action representation.** Mocap synchronized to frames, transformed into a pelvis-centered local frame (invariant to initial pose/orientation). Global pos → local coords; quaternions → relative Euler angles structured by the kinematic tree. Normalized: translation to [−1,1], rotations to [−π,π]. Each action = delta between consecutive frames.
- **Latent diffusion.** Frames encoded with a frozen Stable Diffusion VAE; transitions modeled as DDPM (Lsimple + λLvlb, learned covariances à la Nichol & Dhariwal). Markov factorization on last k states + one past action (Eqs. 1–4).
- **Architecture (CDiT extension).** Within-image self-attention on the noisy current frame + cross-attention to clean past-frame tokens; action conditioning injected via **AdaLN** (simple concatenation of all joint actions into a 1D tensor beats an MLP action-embedding). Trained autoregressively with teacher forcing; causal masking allows training on every sequence prefix in one forward/backward pass.
- **Random timeskips:** sample 16 frames from a 32 s window, with the timeskip fed as part of the action — handles delayed visual consequences and long-horizon dynamics efficiently. Sequence-level training over 16-frame sequences.
- **Planning.** Energy-minimization via **Cross-Entropy Method (CEM)**, same setup as NWM: simulate candidate action rollouts, score final frame by LPIPS to the goal image. Demonstrated only for left/right arm (12-dim action: shoulder/upper-arm/forearm/hand × 3 Euler), T=8, k=0.25 s.

## Key results
- **Single-step (2 s ahead) baselines (Table 1):** PEVA LPIPS **0.303**, DreamSim **0.193**, FID **62.293**, vs CDiT 0.313 / 0.202 / 63.714 and DF* 0.352 / 0.244 / 73.052. PEVA wins on all three.
- **FID over time (Fig. 3):** PEVA maintains lower FID than baselines as horizon grows to 16 s.
- **Atomic actions (Table 2):** PEVA-XXL best across all 11 action types (e.g. Navigation-Forward 0.325 vs CDiT 0.348).
- **Ablations (Table 3):** context 3→15 frames helps (FID 63.97→62.29); scale helps monotonically S→XXL (LPIPS 0.370→**0.298**, DreamSim 0.327→**0.186**, FID 101.38→**61.10**); action **concatenation > MLP embedding** (0.303 vs 0.317 LPIPS).
- **Long horizon:** DreamSim degrades gracefully 0.178 (1 s) → 0.390 (16 s).
- **Planning (Fig. 6):** can rule out wrong action sequences and pick goal-reaching ones (LPIPS-scored), but only arm-level, preliminary.
- **Setup:** Nymeria, XSens skeleton, 4 FPS, 224×224 center-crop, 80/20 split; CDiT-S→XXL up to 32 layers, 2×2 patches, AdamW lr=8e−5, batch 512, grad clip 10.0, metrics averaged over 5 samples.

## Relevance to the EB-JEPA hackathon
- **Track / modality:** action-conditioned **video world modeling** for embodied agents — directly the AC-Video-JEPA territory, but with a **generative-diffusion counterpoint instead of a JEPA**. Same LeCun-lineage framing (world model for planning via energy minimization), shares NWM heritage with the EB-JEPA planning recipe.
- **Map to the recipe:** (1) The **structured, kinematic-tree action representation** (pelvis-local relative joint rotations, normalized) is a transferable idea for any whole-body / high-DoF action conditioning in AC-JEPA. (2) **Random timeskips as an action token** is a clean trick for long-horizon, variable-dt training — relevant to the multi-step / val_nsteps invariants in the HJEPA eval workflow. (3) **CEM/energy-minimization planning scored against a goal embedding** mirrors the EB-JEPA planning eval; PEVA scores in pixel-LPIPS, whereas EB-JEPA would score in latent energy — a natural ablation/contrast. (4) Serves as the **diffusion-decoder baseline** to argue why JEPA (latent, anti-collapse, no pixel reconstruction) is cheaper/more plannable than autoregressive pixel diffusion.
- Useful as a **related-work citation** for "egocentric / whole-body action-conditioned world models" and as a baseline-philosophy contrast (predict pixels vs predict embeddings).

## Caveats & open threads
- **Not a JEPA**: predicts pixels via VAE-latent diffusion; no joint-embedding / target encoder / collapse concerns — so methodological transfer is at the action-representation + planning-protocol level, not the loss.
- Planning is **preliminary**: only single-arm (left OR right), straight-line constant action over T=8, no closed-loop control, no full-trajectory optimization, no semantic/task-goal conditioning (uses image similarity as proxy objective).
- Conditioning is **upper-body only** (above pelvis); legs/locomotion folded into root translation.
- 4 FPS, 224×224, Nymeria-only; no real-world deployment or robot transfer.
- v1 preprint, no peer-reviewed venue yet; numbers cited above are from v1 tables.
