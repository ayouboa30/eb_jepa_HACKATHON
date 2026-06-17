# Causal-JEPA (C-JEPA) — A decoder-free object-centric world model that extends masked joint-embedding prediction from image patches to *object-level latents*, masking whole object slots so the predictor must infer each masked object from the others, injecting a "causal" interaction inductive bias

**Authors:** Heejeong Nam (Brown, GalilAI), Quentin Le Lidec\* (NYU), Lucas Maes\* (Mila / U. Montréal), Yann LeCun (NYU), Randall Balestriero (Brown) [\*equal contribution] **Venue/Year:** ICML 2026 (PMLR 306); arXiv v2, 28 May 2026 (25 pp.) **arXiv:** 2602.11389 **Repo:** https://github.com/galilai-group/cjepa

## TL;DR
C-JEPA is an object-centric world model that lifts masked joint-embedding prediction (JEPA) from image patches up to **object-level latents**: on top of a frozen object-centric encoder (VideoSAUR / SAVi), it masks entire object *slots* across the history window (keeping only an earliest-timestep identity anchor) and trains a bidirectional ViT predictor to reconstruct each masked object's latent trajectory from the *other* objects' evolving states, purely with an L2 latent loss (no decoder, no pixel reconstruction). This "latent intervention on observability" creates counterfactual-like queries that make interaction reasoning functionally necessary to minimize the loss. It yields **+21.13% absolute** on CLEVRER counterfactual VQA over the same architecture without object masking, and on Push-T control matches patch-based DINO-WM (88.67% vs 91.33% success) while using only **1.02%** of the latent token budget and planning **>8×** faster.

## Problem & motivation
- World models need relational / interaction understanding, not pixel correlations. Object-centric representations are a useful abstraction but **insufficient alone**: models fall back on object *self-dynamics* or exploit incidental correlations when nothing forces interaction learning.
- Existing fixes enforce interactions *externally* — architectural factorization separating self-dynamics from interactions (OCVP-Seq, SOLD), sparse-attention regularization (SPARTAN), relational graphs (C-SWM), or task-specific RL. None makes interaction structure **functionally necessary through the learning objective itself**; that is the gap C-JEPA targets.
- **Patch-based masked prediction (MAE / VideoMAE / I-JEPA / V-JEPA) optimizes local patch correlations** and does not enforce object interaction reasoning; its predictor also scales quadratically in a large patch-token count (expensive for planning).
- Object masking is designed to **block shortcut solutions** (trivial temporal interpolation / self-dynamics: recovering an object's masked state from its own nearby frames), forcing the predictor to consult other objects.

## Method
**Setup / notation.** Frozen object-centric encoder `g` maps each frame `X_t ∈ R^{H×W×C}` to `N` permutation-equivariant slots `S_t = {s¹_t,…,s^N_t}`, `s^i_t ∈ R^d` (d=128). History length `T_h`, horizon `T_p`. Auxiliary observables `U_t = {a_t, p_t}` (actions, proprioception) are treated as **separate entity tokens**, not concatenated into slots (Fig. 3 shows separate-node beats concatenation). Entity tokens `Z_t = {S_t, U_t}`.

**Architecture (frozen-encoder JEPA, decoder-free).**
- **Encoder `g` (frozen):** VideoSAUR (primary) or SAVi — slot attention over frozen **DINOv2 ViT-S/14** features (196 patch tokens × dim 384, projected to 128-dim slots). Targets are the frozen encoder's slot latents; **no EMA target encoder, no pixel decoder.**
- **Predictor `f`:** a **bidirectional masked ViT** (not autoregressive) — 6 layers, 16 heads, head dim 64, MLP hidden 2048, slot dim 128. Bidirectional because object states aren't first-order Markov, and AR would bias toward self-dynamics. Built on `stable-pretraining` + `stable-worldmodel`.

**What "object-level latent masking" means.** Future tokens are always masked (forward prediction). *Additionally*, across the history the model masks **entire object slots** (all timesteps of objects in index set `M ⊂ {1,…,N}`), keeping **only the earliest timestep `t₀` as an identity anchor.** Masked token: `z̃^i_τ = φ(z^i_{t₀}) + e_τ` — `φ` linear projection, `z^i_{t₀}` the anchor, `e_τ` learnable embedding + temporal position. The anchor is needed because slots carry no entity-axis position, so the predictor must be told *which* masked object to predict. Framed as a **latent intervention on predictor observability** (changes available info, not the data-generating mechanism), creating counterfactual-like queries.

**Training objective (masked latent L2).**
`L_mask = E[ Σ_{τ∈𝒯} Σ_i 1[z̄^i_τ ≠ z^i_τ] · ‖ẑ^i_τ − z^i_τ‖₂² ]`, `Ẑ_𝒯 = f(Z̄_𝒯)`. Decomposes as `L_history` (reconstruct masked history slots under partial observability → suppresses trivial self-dynamics) + `L_future` (forward world-model alignment). **At inference** history is fully observed and masking is applied only to future tokens → standard latent rollout for planning / reasoning.

**Vs I-JEPA / patch JEPA.** I-JEPA/V-JEPA mask spatial blocks / spatiotemporal tubes of *patch* tokens (local correlations); C-JEPA masks *whole object slots across time* in a tiny `T×N` latent space, with an identity anchor + auxiliary-variable nodes, making object interaction the only way to minimize loss — and far cheaper attention.

**Formal analysis (Sec. 6, App. L/M).** Assumes temporally-directed predictive dependencies, a shared/stationary transition, object-aligned latents, finite-history sufficiency; does **not** assume causal sufficiency, first-order Markov, or global sparsity.
- **Def. 1 — Influence Neighborhood `N_t(i)`:** minimal sufficient subset s.t. `p(z^i_t | Z^{(−i)}_T) = p(z^i_t | N_t(i))`.
- **Thm. 1 — Interaction Necessity:** the MSE-optimal masked predictor is the conditional mean `E[z^i_t | N_t(i)]`; any predictor ignoring `N_t(i)` incurs strictly higher loss.
- **Cor. 1:** repeated diverse masking induces state-dependent attention aligned with `N_t(i)` (soft local relational structure; related to ICP / IRM, viewed as latent interventions on observability).
- Careful framing: "causal" = temporally-directed predictive dependencies stable under masking, **not** causal identifiability; influence neighborhoods are predictive-sufficiency sets, not verified causal parents.

## Key results
**Visual reasoning — CLEVRER VQA** (reasoning head ALOE; OC-JEPA = `|M|=0` ablation, same arch, future-only masking). Reported as Avg / Counterfactual-per-option / **Counterfactual-per-question**:
- VideoSAUR encoder: `|M|=0` 82.79 / 79.53 / **47.68** → `|M|=4` **89.40 (↑6.61) / 88.67 (↑9.14) / 68.81 (↑21.13)** — the headline ~20% absolute counterfactual gain (47.68→68.81). `|M|=3` already +15.92 on CF-per-que.
- SAVi encoder: best at `|M|=2`: 83.88 / 85.16 / **60.19 (↑19.09)**; but `|M|=4` **regresses below baseline** (73.28 / 73.55 / 34.06) → an **optimal masking regime** exists; the weaker encoder caps the ceiling.
- Baseline comparison (SAVi, Table 2/A3, CF-per-que): SlotFormer 47.29 but **collapses to 11.10 without reconstruction**; OCVP-Seq 56.06; OC-JEPA 41.10; **C-JEPA 60.19** — best, fully decoder-free. (C-JEPA(V) Descriptive-per-que reaches 91.02%.)

**Predictive control — Push-T** (latent MPC via CEM; `|M|=1`). Success rate vs token budget `#Token×d`:
- DINO-WM (patch, 196×384): **91.33%**; DINO-WM-Reg. 88.00%; OC-DINO-WM (6×128, ref.) 60.67%; OC-JEPA 76.00% (↑+15.33); **C-JEPA 88.67% (↑+28.00)** — comparable to DINO-WM at the object-centric budget.
- **Efficiency:** object latent space is **1.02%** of the patch budget (6×128 vs 196×384). 50-trajectory eval on one L40S: **C-JEPA 673 s vs DINO-WM 5,763 s** (3 seeds) → **>8× faster MPC**.

**Masking-strategy ablation (App. K).** Object- vs token- vs tube-level masking at matched budgets. On Push-T (Table A5) object-level masking is markedly more robust: at 50% masking Object 82.67 / Token 84.00 / **Tube 5.33** (collapse); at 25% Object 88.67 / Token 84.67 / Tube 55.33. On CLEVRER's tiny space differences are smaller but token/tube are less consistent.

**PHYRE** (qualitative only — no ground-truth causal graphs): more physically plausible rollouts, sharper cross-slot attention on interaction-relevant slots vs OC-JEPA.

**Datasets / baselines.** CLEVRER (10k/5k/5k, 7 slots), Push-T (18,410 train, 4 slots, action dim 2, proprio dim 4), PHYRE (qual.). Baselines: SlotFormer / OCVP-Seq (±recon.) + OC-JEPA for VQA; DINO-WM / DINO-WM-Reg. / OC-DINO-WM / OC-JEPA for control. Encoders VideoSAUR / SAVi on frozen DINOv2 ViT-S/14.

## Relevance to the EB-JEPA hackathon
- **Track / modality: planning & control + visual reasoning world models (video / object-centric).** Most directly relevant to a **DINO-WM-style world-modeling-and-planning track** — it is benchmarked head-to-head against DINO-WM on Push-T and reuses the same CEM-MPC-in-latent-space planning recipe, so it slots into the existing AC-Video-JEPA / planning-eval machinery (latent rollout + MPC cost = MSE-to-goal).
- **Map to the recipe.** Keep the JEPA backbone (frozen encoder → predictor → L2 latent loss, no decoder, no anti-collapse regularizer needed since targets come from a frozen encoder — same trick as DINO-WM). The single new ingredient is **object-level latent masking with an identity anchor**: swap patch tokens for slot tokens, mask whole slots across history, treat actions/proprio as separate entity tokens. This is a small, well-scoped extension to add as an ablation axis (`|M|` = number of masked objects) on top of the action-conditioned video-JEPA pipeline.
- **Why it's attractive for a 24h project:** the latent token count drops to ~1% (6 slots vs 196 patches), giving >8× faster planning eval — fast iteration loops. The headline +21% counterfactual VQA gain is a clean, reproducible knob (just `|M|`).

## Caveats & open threads
- Performance is **bounded by object-centric encoder quality**; a weak encoder (SAVi) caps the ceiling and makes over-masking actively hurt (SAVi `|M|=4` regresses below baseline).
- **Optimal masking regime is encoder/task-dependent** — too much masking removes informative dependencies; `|M|` must be tuned.
- Influence neighborhoods are **not validated on data with explicit temporal causal graphs** (PHYRE has none); they are predictive-sufficiency sets, not verified causal parents. The "causal" framing is operational, not identifiability.
- PHYRE results are **qualitative only**; CLEVRER VQA uses the **validation set** as held-out test (eval server unavailable).
- Future work: jointly refine object-centric encoders with strong backbones without representational collapse; richer interaction environments.
