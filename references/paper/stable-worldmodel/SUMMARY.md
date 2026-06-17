# stable-worldmodel — a unified, reproducible platform for the full world-model pipeline (data → baselines → solvers → factor-of-variation evaluation)

**Authors:** Lucas Maes*, Quentin Le Lidec*, Luiz Facury, Nassim Massaudi, Ayush Chaurasia, Francesco Capuano, Richard Gao, Taj Gillin, Dan Haramati, Damien Scieur, Yann LeCun, Randall Balestriero (* equal). Mila/UdeM, NYU, UFMG, LanceDB, Oxford, Brown.
**Venue/Year:** 2026 (preprint, arXiv:2605.21800v1, 20 May 2026)
**arXiv:** 2605.21800
**Repo:** open-source ("Code available here" in the paper; package `stable_worldmodel`, import alias `swm`; built on PyTorch + Gymnasium; `$STABLEWM_HOME` for datasets/checkpoints)

## TL;DR
World-model research is fragmented: every lab re-rolls its own data pipeline, baselines, solvers, and eval protocol, so reported gains can't be told apart from implementation differences. `stable-worldmodel` (swm) is a single, non-invasive platform that standardizes the *infrastructure* around the model (data layer, planning solvers, environments, and especially a controllable factor-of-variation evaluation suite) while imposing **no** constraint on the user's model architecture or training loop. Its built-in case study shows that today's world models (DINO-WM, PLDM, LeWM, TD-MPC2, GCBC) plan well in-distribution but degrade sharply under even mild visual/geometric/physical shifts, and that prediction MSE does **not** correlate with planning success.

## Problem & motivation
Three concrete bottlenecks in current practice:
- **Implementation fragmentation.** CEM alone has been independently re-implemented (with varying fidelity) in at least five recent papers (TD-MPC, PLDM, DINO-WM, LeWM, V-JEPA2); duplicated baselines/envs/solvers inject subtle inconsistencies and kill fair comparison.
- **Data-loading bottleneck.** WMs need temporally contiguous multimodal blocks (frames + actions + proprioception). Per-frame storage = fast random access but huge I/O/storage; compressed MP4 = small but terrible random access (must decode preceding frames). Neither scales; the GPU starves.
- **Weak generalization eval.** Standard Gym benchmarks test near the training distribution, so you can't tell whether a model learned reusable dynamics or exploitable correlations. A WM is a learned representation of dynamics, not a policy, and predictive accuracy in-distribution does not imply temporally stable / intervention-robust / counterfactual-useful latent dynamics.

## Method (the platform)
Three minimal abstractions (Fig. 1):
- **World** — unified Gymnasium-compatible wrapper over a vectorized `EnvPool`; handles collection, policy execution, rendering, evaluation, and *controllable interventions* (the FoV). `World.reset()/step()` don't return obs/reward; they update an in-place `world.infos` dict.
- **Policy** — `get_actions(info)` interface: random, expert (e.g. SAC), RL-learned, or `MPCPolicy` (wraps any WM + solver, encodes obs → latent, delegates to the solver each step).
- **Solver** — single-shooting MPC planners; a WM only needs a `get_cost` method. Sampling-based: Predictive Sampling, CEM, iCEM, MPPI. Gradient-based: GD, Projected GD, GRASP. All tested for numerical stability; pseudocode in App. F.

**Data layer.** Primary format is **Lance** (columnar, ML-optimized: fast random access, high compression, zero-copy, native versioning, S3 streaming). Native support + one-click `swm.data.convert()` for MP4, HDF5, and **LeRobot** (real-robot trajectories via adapter → auto-convert to Lance). Supports offline + online (TD-MPC2-style alternation) collection with FoV active on-the-fly.

**Baselines (six, two paradigms).** Latent WM + test-time planning: **DINO-WM** (frozen DINOv2 + ViT predictor over patch features), **PLDM** (JEPA-style joint repr+dynamics with stabilizing regularization), **LeWM/LeWorldModel** (JEPA-based, simplified objective, fastest baseline), **TD-MPC2** (decoder-free, reward/value-guided). GCRL: **GCBC** (goal-conditioned BC), **GCIQL / GCIVL** (implicit Q/V-learning + AWR; in swm all GCRL methods encode obs/goals into DINOv2 patch embeddings first). Shared Hydra training entry point, shared offline dataset → fair comparison.

**Environments + Factors of Variation (the core contribution).** ~150 environments across game/control/robotics: DeepMind Control Suite, OGBench (3D manip), Classic Control, Fetch-Suite, **Craftax** (open-world 2D survival), **ALE/Atari (100+ games)**, plus `swm/PushT-v1` and `swm/TwoRoom-v1`. Two parallel intervention mechanisms feeding one eval pipeline:
  - **Native FoV** (`swm/*` envs): hierarchical `variation_space` (e.g. `agent.color`, `block.scale`, `physics.floor.friction`), set via Gymnasium `options`; sampled or pinned, held constant for the episode (so a failure is attributable to a persistent change, not per-frame noise). `variation:"all"` samples every native factor.
  - **Boundary-level visual wrappers** (for black-box envs like Atari ROMs / Craftax): 11 universal wrappers — ChromaKey, Noise, Blur, ColorJitter, Grayscale, RandomShift, Cutout, Occlusion, MovingPatch, RandomConv, Resolution — that rewrite the rendered RGB frame. Composable with native FoV on one `World`.

## Key results (Push-T / OGBench case study, success rate %)
- **In-distribution (Tab. 1, same expert dataset, identical planning config):** Push-T — LeWM **94**, DINO-WM **92**, PLDM **78**, GCBC **75**, TD-MPC2 **12**. OGB-Cube — DINO-WM **86**, GCBC **84**, LeWM **72**, PLDM **62**, TD-MPC2 **4**. Recovered values are consistent with the originally reported baselines; TD-MPC2 fails offline (conjectured OOD-action drift fooling the predictor; verified online on DMC vs SAC in Tab. 5).
- **Prediction MSE ≠ planning success.** On LeWM (Fig. 4), across expert-train / expert-valid / random / random+all-variations, success was **88% / 84% / 51% / 30%**; success and failure MSE distributions overlap heavily even under strong shift — OOD *inputs*, not error magnitude, drive failures (same holds for PLDM).
- **Brittle under FoV (Fig. 5b, Push-T SR%):** baseline None = LeWM 50.8 / PLDM 50.8 / DINO-WM 20.0. Color-Canvas drops to **6.0 / 6.0 / 10.0**; Color-Agent to **12.0 / 8.0 / 18.0**; Size-Agent to **22.0 / 18.0 / 4.0**. Visual distractor squares (Fig. 5a) cause roughly **quadratic** SR decay; pattern holds across baselines.
- Takeaway: current WMs have limited zero-shot generalization; robust generalization likely needs both architectural advances and systematic scaling — which swm is built to study reproducibly.

## Relevance to the EB-JEPA hackathon
swm is the **scale-up environment platform** the hackathon guide explicitly points teams to. In `hackathon_guide/main.tex`, `eb_jepa` is positioned as the prototyping rung of a ladder; once an idea works, larger codebases pick it up — `jepa-wms` for planning with frozen encoders, and **`stable-worldmodel` for "a broad, controllable suite of world-model environments"** (cited as `\citep{stableworldmodel}`, line 191). "Prototype small here; scale and validate there."

It directly powers **Track 9 — "Stress-test the recipe under factors of variation"** (`main.tex` line 1123, `\diffhard`). The track asks whether `eb_jepa`'s *minimal* AC-video-JEPA recipe (VC + sim + IDM regularizers, distance-cost MPPI) survives realistic perturbations and which term breaks first — and tells teams to evaluate planning "as you dial up controllable visual, geometric, and physical factors of variation, using the environment suites in `stable-worldmodel` ... rather than hand-building a new environment." Deliverable: a success-vs-perturbation curve against the tiny Two Rooms baseline. swm's native FoV + visual-wrapper machinery is exactly the controllable-perturbation generator this needs.

**How a team plugs eb_jepa into swm.** swm's design is non-invasive: it standardizes data/eval/control and does **not** dictate the model. A team (1) wraps an `eb_jepa` encoder + `JEPA.unroll` predictor as a swm world model exposing `get_cost` (latent distance-to-goal), (2) wraps it in `MPCPolicy` with a solver (CEM/MPPI matching the eb_jepa distance-cost MPPI planner), (3) collects/loads a Lance dataset on a chosen swm env (`swm/TwoRoom-v1` is the natural analogue to eb_jepa's Two Rooms; Push-T/OGBench scale up), and (4) calls `world.evaluate(..., options={"variation": [...]})` to sweep FoV and read `metrics["success_rate"]` — yielding the Track-9 success-vs-perturbation curve with near-zero boilerplate (cf. Algorithm 1).

## Caveats & open threads
- **It is infrastructure, not a new model/algorithm.** The contribution is the standardized platform + the diagnostic finding (WMs brittle OOD), not a method that fixes brittleness.
- **Case study is narrow:** the quantitative robustness numbers come almost entirely from Push-T (+ some OGBench); the ~150-env suite is presented as available, not all benchmarked.
- **Explicitly contrasts with EB-JEPA.** In Related Work (p.9, ref [50] = Terver et al., 2026, arXiv:2602.03604, the *this repo's* library), swm says EB-JEPA is "restricted to JEPA-style architectures, ... mainly developed for educational purposes, and lack[s] scalable research components, such as efficient data-loading, training-agnostic evaluation, and modeling baselines." That is the intended division of labor the hackathon guide endorses (prototype in eb_jepa, scale/validate in swm), but worth flagging as the authors' framing.
- **Adapter work required for Track 9:** eb_jepa models must be wrapped to swm's `get_cost`/`MPCPolicy` interface and the eb_jepa Two Rooms vs `swm/TwoRoom-v1` correspondence verified; not a drop-in.
- **Preprint (v1):** future work (sim-to-real, async real-time interaction, online training) is unimplemented; numbers/repo may move.
