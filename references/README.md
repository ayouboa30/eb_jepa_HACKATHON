# References — paper memory

Curated summaries of papers relevant to this repo, for humans and agents iterating on
`eb_jepa`. One folder per paper: `paper/<slug>/<slug>.pdf` + `paper/<slug>/SUMMARY.md`.
PDFs are git-LFS-tracked automatically (`paper/.gitattributes`); extracted `*.txt`
sidecars are git-ignored.

## Add a paper

1. Fresh clone? Run `git lfs install` once.
2. `mkdir references/paper/<slug>` — short, lowercase, hyphenated.
3. Download the **full latest** arXiv version and verify it is not truncated:
   ```bash
   curl -sL -o references/paper/<slug>/<slug>.pdf https://arxiv.org/pdf/<arxiv-id>
   python3 -c "import fitz; d=fitz.open('references/paper/<slug>/<slug>.pdf'); print('pages:', d.page_count); print(repr(d[d.page_count-1].get_text()[-200:]))"
   ```
   The last page should end on the references/conclusion. **Do not trust `file`'s page
   count** (it undercounts nested page trees); the bare `/pdf/<id>` already serves the
   latest version. If the paper is short with no appendix that may be correct (ICME, Graz
   BCI, etc. are ~6 pages); if it looks cut off, the full version may be OpenReview-only.
4. Write `references/paper/<slug>/SUMMARY.md` using the skeleton below (no need to commit
   a `.txt`; extract on demand with `python3 -c "import fitz; ..."`).
5. Add a row to the appropriate tier table below, then `git add references/paper/<slug>`
   and commit. The PDF is stored in LFS automatically.

## SUMMARY.md skeleton

```
# <Name> — <one-line thesis>
**Authors:** … **Venue/Year:** … **arXiv:** … **Repo:** …
## TL;DR
## Problem & motivation
## Method
## Key results
## Relevance to the EB-JEPA hackathon
## Caveats / open threads
```

## Papers

Grouped into a **Core** set (the `eb_jepa` codebase plus the vision/robotics
world-model line it builds on) and three maturity tiers for the JEPA-to-new-modality
literature. Each row carries a modality/track tag.

### Core — EB-JEPA, JEPA world models & vision foundations

| Directory | Paper |
|---|---|
| `paper/eb-jepa/` | **EB-JEPA** (Terver et al., 2026, [2602.03604](https://arxiv.org/abs/2602.03604)): the hackathon's core library — unifies image SSL, video prediction, and action-conditioned world-model planning under one energy-based JEPA recipe (encoder + predictor + VICReg/SIGReg + K-step rollouts + MPPI/CEM); 91% CIFAR-10, 97% Two Rooms. *Core codebase.* |
| `paper/lejepa/` | **LeJEPA** (Balestriero & LeCun, 2025, [2511.08544](https://arxiv.org/abs/2511.08544)): provable heuristics-free JEPA — isotropic-Gaussian embeddings via SIGReg (Epps-Pulley along random projections), single λ, no stop-grad/teacher; scales to 1.8B params (79% IN-1K). *Regularizer (eb_jepa's BCS/SIGReg).* |
| `paper/jepa-wms/` | **JEPA-WMs** (Terver et al., 2026, [2512.24497](https://arxiv.org/abs/2512.24497)): systematic ablation of what makes latent-space planning with frozen-encoder JEPA world models work (planner, rollout, proprio, encoder, scaling); beats DINO-WM and V-JEPA-2-AC. *Robotics/world-model.* |
| `paper/dino-wm/` | **DINO-WM** (Zhou et al., 2024, [2411.04983](https://arxiv.org/abs/2411.04983)): task-agnostic latent world model on *frozen* DINOv2 features; causal ViT predictor + L2 latent loss, zero-shot CEM-MPC planning to a goal embedding. *Robotics/world-model.* |
| `paper/dino-world/` | **DINO-world** (Baldassarre, Szafraniec, Terver et al.; Meta/FAIR 2025, [2507.19468](https://arxiv.org/abs/2507.19468)): trains a ~1.1B cross-attention predictor to forecast next-frame DINOv2 patch features on ~60M web videos (frozen encoder, smooth-L1, block-causal mask); fine-tunes action blocks for latent-rollout planning. *Video/world-model.* |
| `paper/leworldmodel/` | **LeWorldModel / LeWM** (Maes & Le Lidec et al., 2026, [2603.19312](https://arxiv.org/abs/2603.19312)): first JEPA trained stably end-to-end from pixels with a two-term loss (next-embedding MSE + SIGReg), cutting six loss hyperparameters to one; plans up to 48× faster than DINO-WM. *Robotics (ac_video_jepa).* |
| `paper/v-jepa-2.1/` | **V-JEPA 2.1** (Mur-Labadia et al., 2026, [2603.14482](https://arxiv.org/abs/2603.14482)): applies the JEPA L1 latent loss to *visible context tokens* (distance-weighted) + deep multi-level supervision, unlocking SOTA dense video features (0.307 NYUv2 depth). *Video/world-model.* |
| `paper/reconstruction-or-semantics/` | **Reconstruction or Semantics?** (Nilaksh, Jha, Zholus, Chandar; Mila, 2026, [2605.06388](https://arxiv.org/abs/2605.06388)): controlled study of latent spaces for action-conditioned video-diffusion robotic world models on BridgeV2; semantic encoders (V-JEPA 2.1 best on policy, Web-DINO, SigLIP 2) beat reconstruction VAE latents on planning/policy despite comparable pixel fidelity. *World-model latent-space study.* |
| `paper/vl-jepa/` | **VL-JEPA** (Chen, Shukor, Moutakanni et al., 2026, [2512.10942](https://arxiv.org/abs/2512.10942)): non-generative vision-language model predicting continuous target-text embeddings (frozen V-JEPA2 + query-conditioned Llama predictor, InfoNCE); names VICReg/SIGReg as drop-in anti-collapse swaps. *Multimodal.* |
| `paper/intuitive-physics/` | **Intuitive Physics from Self-Supervised Video** (Garrido et al., 2025, [2502.11831](https://arxiv.org/abs/2502.11831)): V-JEPA latent prediction acquires object permanence/continuity/shape constancy zero-shot via violation-of-expectation (98% IntPhys), while pixel-prediction and LLMs stay near chance. *Track 10 (video_jepa).* |
| `paper/v-jepa/` | **V-JEPA** (Bardes et al., FAIR/Meta 2024, [2404.08471](https://arxiv.org/abs/2404.08471)): the canonical video JEPA — predicts masked spatio-temporal features in latent space (L1 loss, EMA target + stop-grad + ~90% tube masking); frozen ViT-H hits 81.9% K400 / 77.9% IN1K, beating pixel-prediction video methods. *Video.* |
| `paper/vjepa2/` | **V-JEPA 2** (Assran et al., Meta FAIR 2025, [2506.09985](https://arxiv.org/abs/2506.09985)): internet-scale action-free video JEPA (ViT-g/1B, >1M h) plus a frozen-encoder action-conditioned predictor (V-JEPA 2-AC) trained on <62 h of Droid that plans zero-shot on real Franka arms via CEM-MPC. *Video + robot actions.* |
| `paper/iwm-image-world-models/` | **IWM — Image World Models** (Garrido et al., CVPR 2024, [2403.00504](https://arxiv.org/abs/2403.00504)): generalizes the I-JEPA task to photometric augmentations as "actions", conditioning the predictor on the transform to learn a reusable equivariant world model (84.4% IN-1k finetuned); interpolates invariant↔equivariant representations via predictor capacity. *Image/world-model.* |
| `paper/jepa-slow-features/` | **JEPAs Focus on Slow Features** (Sobal et al., NeurIPS 2022, [2211.10831](https://arxiv.org/abs/2211.10831)): shows VICReg/SimCLR JEPA world models collapse onto the slowest-changing feature (fixed background) and ignore the agent — the direct motivation for eb_jepa's inverse-dynamics (IDM) loss. *Theory / world-model collapse.* |
| `paper/stable-worldmodel/` | **stable-worldmodel** (Maes et al., 2026, [2605.21800](https://arxiv.org/abs/2605.21800)): a unified platform for the full world-model pipeline — tested WM baselines (DINO-WM, PLDM, LeWM, TD-MPC2), MPC solvers (CEM/MPPI/GD), and ~150 environments with controllable visual/geometric/physical factors of variation; the env suite the guide's Track 9 runs on. *Robotics/world-model platform.* |
| `paper/causal-jepa/` | **Causal-JEPA** (Nam et al., ICML 2026, [2602.11389](https://arxiv.org/abs/2602.11389)): object-centric world model extending masked JEPA from patches to object-level latents (mask whole object slots so the predictor infers each from the others); +21% counterfactual VQA at ~1% of DINO-WM's token budget, >8× faster planning. *Object-centric world models.* |
| `paper/temporal-straightening/` | **Temporal Straightening** (Wang et al., ICML 2026, [2603.12231](https://arxiv.org/abs/2603.12231)): adds a curvature regularizer (1−cos of consecutive latent velocities) so JEPA world-model latent trajectories straighten, making gradient-based latent planning well-conditioned (+20–60% open-loop goal-reaching). *Video world-model / planning.* |

### Related world models (non-JEPA)

Pixel-space / diffusion world models included for comparison: they share the
action-conditioned planning setup (CEM/MPC over imagined rollouts) but predict in
pixel space rather than in a learned JEPA latent, so they sit as generative
counterpoints to the latent JEPA world models above.

| Directory | Paper |
|---|---|
| `paper/navigation-world-models/` | **NWM — Navigation World Models** (Bar et al., CVPR 2025, [2412.03572](https://arxiv.org/abs/2412.03572)): a 1B-param controllable diffusion video world model (linear-in-context CDiT) that plans navigation via MPC+CEM by simulating and scoring imagined trajectories (ATE 1.13 on RECON); the generative counterpoint to latent JEPA world models. *Action-conditioned video / navigation.* |
| `paper/peva-egocentric-video-prediction/` | **PEVA — Whole-Body Conditioned Egocentric Video Prediction** (Bai et al., 2025, [2506.21552](https://arxiv.org/abs/2506.21552)): autoregressive conditional diffusion transformer predicting egocentric video from whole-body 3D-pose actions (Nymeria), CEM-planned over 16 s rollouts; a pixel-space AC world-model counterpoint to AC-Video-JEPA. *Action-conditioned video world model.* |
| `paper/lifting-world-models/` | **Lifting World Models** (Wang et al., 2026, [2604.26182](https://arxiv.org/abs/2604.26182)): "lifts" a frozen PEVA world model with a small policy mapping low-dim 2D-waypoint high-level actions to joint sequences, so CEM planning runs in an 8-d space (3.8× lower joint error vs 48-d joint search). *Egocentric AC world models / hierarchical planning.* |

### Tier 1 — Established / foundational modality JEPAs (>1 yr, strong venue)

| Directory | Paper |
|---|---|
| `paper/graph-jepa/` | **Graph-JEPA** (Skenderi et al., TMLR 2025, [2309.16014](https://arxiv.org/abs/2309.16014)): the first JEPA for graphs — METIS subgraph patches, a hyperbolic-angle 2D target, EMA+stop-grad; SOTA frozen backbone on 5/8 TUD sets. *Graphs.* |
| `paper/a-jepa/` | **A-JEPA** (Fei et al., 2023, [2311.15830](https://arxiv.org/abs/2311.15830)): the early audio JEPA — first port of I-JEPA latent masked-prediction to Mel-spectrograms with curriculum time-frequency masking; SOTA over AudioMAE on AudioSet. *Audio.* |
| `paper/brain-jepa/` | **Brain-JEPA** (Dong & Li et al., NeurIPS 2024 Spotlight, [2409.19407](https://arxiv.org/abs/2409.19407)): fMRI brain-dynamics foundation model porting I-JEPA to noisy neuro time series with a connectivity-gradient positional code and structured Cross-ROI/Cross-Time masking. *fMRI/neuro (Track 1 sibling).* |
| `paper/point-jepa/` | **Point-JEPA** (Saito et al., WACV 2025, [2404.16432](https://arxiv.org/abs/2404.16432)): the canonical point-cloud JEPA — a greedy nearest-neighbor sequencer orders FPS+KNN patches for cheap I-JEPA blocks; 93.7% ModelNet40 linear SVM. *Point clouds (Track 6).* |

### Tier 2 — Solid, mostly peer-reviewed, ~1 year old

| Directory | Paper |
|---|---|
| `paper/audio-jepa/` | **Audio-JEPA** (Tuncay et al., ICME 2025, [2507.02915](https://arxiv.org/abs/2507.02915)): I-JEPA ported to audio, masks log-mel patches and regresses their latent embeddings against an EMA target; matches wav2vec 2.0 / data2vec on X-ARES with under 1/5 the data. *Audio.* |
| `paper/s-jepa/` | **S-JEPA / Signal-JEPA** (Guetschel et al., Graz BCI 2024, [2403.11772](https://arxiv.org/abs/2403.11772)): JEPA for EEG with a spatial (channel-radius) block-masking scheme + EMA-target L1 latent loss for cross-dataset BCI transfer. *EEG.* |
| `paper/st-jema/` | **ST-JEMA** (Choi et al., 2024, [2403.06432](https://arxiv.org/abs/2403.06432)): JEPA-style latent-reconstruction SSL on dynamic fMRI brain graphs (GIN encoders, EMA target) with a dual spatial+temporal objective; rank-1 on 8 rs-fMRI benchmarks. *fMRI/neuro.* |
| `paper/lat-pfn/` | **LaT-PFN** (Verdenius et al., 2024, [2405.10093](https://arxiv.org/abs/2405.10093)): PFN in-context meta-learning + energy-based JEPA for zero-shot univariate time-series forecasting in latent space; beats ARIMA/Prophet/ForecastPFN zero-shot. *Time-series.* |
| `paper/stem-jepa/` | **Stem-JEPA** (Riou et al., ISMIR 2024, [2408.02514](https://arxiv.org/abs/2408.02514)): JEPA predicting the embedding of a compatible missing instrument stem (stem omission, not masking), label-conditioned predictor; first to use the JEPA predictor at inference. *Audio/music.* |
| `paper/t-jepa/` | **T-JEPA** (Thimonier et al., ICLR 2025, [2410.05016](https://arxiv.org/abs/2410.05016)): augmentation-free JEPA for tabular data — predicts the latent of one feature-subset from another; a register `[REG]` token (not VICReg) prevents collapse since EMA+stop-grad alone fails. *Tabular.* |
| `paper/ecg-jepa/` | **ECG-JEPA** (Kim, 2024–26, [2410.08559](https://arxiv.org/abs/2410.08559)): I-JEPA on 12-lead ECG — synchronized cross-lead temporal masking + EMA-target L1 + Cross-Pattern Attention; SOTA on diagnosis/feature/segmentation, strong low-shot/reduced-lead. *ECG/biosignal (Track 3).* |
| `paper/ecg-ptbxl-jepa/` | **ECG-PTBXL-JEPA** (Weimann & Conrad, CBM 2024, [2410.13867](https://arxiv.org/abs/2410.13867)): faithful I-JEPA port to 12-lead ECG (patch masking + EMA-target L1 + ViT) pretrained on ~1M records; beats CPC and ST-MEM on PTB-XL (0.945 macro AUC). *ECG/biosignal (Track 3).* |
| `paper/sar-jepa/` | **SAR-JEPA** (Li et al., ISPRS 2024, [2311.15153](https://arxiv.org/abs/2311.15153)): a JEPA for SAR target recognition that predicts a frozen, speckle-robust multi-scale gradient-by-ratio feature (not pixels or an EMA target), avoiding collapse on noisy radar imagery. *Radar/SAR.* |
| `paper/anysat/` | **AnySat** (Astruc et al., 2024, [2412.14123](https://arxiv.org/abs/2412.14123)): a single multimodal JEPA + scale-adaptive patch encoder trained on 11 Earth-observation sensors at 0.2–250 m; SOTA across land-cover/crop/flood/burn-scar. *Earth observation.* |
| `paper/j-jepa/` | **J-JEPA** (Katel et al., NeurIPS 2024 ML4PS, [2412.05333](https://arxiv.org/abs/2412.05333)): augmentation-free I-JEPA for particle jets — reclusters a jet into subjets, masks encoder outputs, predicts target-subjet embeddings; helps most in low-label top-tagging. *Particle/jet physics.* |
| `paper/locate-3d/` | **Locate 3D / 3D-JEPA** (Arnaud, McVay, … Ballas, Assran, Rajeswaran, Meier; Meta FAIR 2025, [2504.14151](https://arxiv.org/abs/2504.14151)): JEPA-style masked latent prediction on scene point clouds whose per-point features are *lifted from frozen CLIP+DINO+SAM*, contextualizing 2D-foundation features; fine-tuned with a language decoder for SoTA 3D referring-expression localization. *Point clouds (Track 6).* |

### Tier 3 — New / emerging (2025 H2 – 2026), few citations yet

| Directory | Paper |
|---|---|
| `paper/eeg-vjepa/` | **EEG-VJEPA** (Hojjati et al., Pattern Recognition 2026, [2507.03633](https://arxiv.org/abs/2507.03633)): ports V-JEPA to EEG-as-video (19×500 tubelets, multi-block masking, EMA target, L1 latent loss); pretrained label-free on TUH+NMT, SOTA on **TUAB** abnormal-EEG (frozen 83.3% / FT 85.8%). *EEG (flagship analogue).* |
| `paper/laya/` | **Laya** (Panchavati et al., 2026, [2603.16281](https://arxiv.org/abs/2603.16281)): LeJEPA/SIGReg EEG foundation model — masked temporal latent prediction beats reconstruction on noisy clinical EEG under linear probing; ablations show SIGReg is the load-bearing anti-collapse term. *EEG.* |
| `paper/echo-jepa/` | **EchoJEPA** (Munim, Fallahpour, Szasz et al., 2026, [2602.02603](https://arxiv.org/abs/2602.02603)): foundation-scale V-JEPA2 for cardiac ultrasound (18M videos); latent prediction (EMA target, L1, masking) beats compute-matched VideoMAE on LVEF/RVSP, sample efficiency, and acoustic robustness. *Video/medical signal.* |
| `paper/lens-jepa/` | **Lens-JEPA** (Rishi et al., NeurIPS ML4PS 2025): physics-informed I-JEPA for strong gravitational lensing that bakes the lens equation into a ViT encoder, beating plain I-JEPA on dark-matter-substructure classification (0.912 acc). *Physical fields / The Well.* |
| `paper/polymer-jepa/` | **Polymer-JEPA** (Piccoli, Vogel & Weber, 2025, [2506.18194](https://arxiv.org/abs/2506.18194)): JEPA pretraining on stochastic polymer molecular graphs — context/target subgraph views, wD-MPNN encoders, RWSE-conditioned predictor, L2 loss; helps most in label-scarce/transfer regimes. *Molecular graphs.* |
| `paper/ts-jepa/` | **TS-JEPA** (Ennadir et al., NeurIPS 2024 TS Workshop, [2509.25449](https://arxiv.org/abs/2509.25449)): a systematic port of JEPA to (univariate) time series — 1D-CNN patch tokenizer, transformer encoder/predictor, EMA target, L1 latent masked prediction; one frozen encoder balances classification and forecasting. *Time-series.* |
| `paper/mts-jepa/` | **MTS-JEPA** (He et al., 2026, [2602.04643](https://arxiv.org/abs/2602.04643)): multi-resolution JEPA + soft-codebook bottleneck for multivariate time-series *anomaly prediction*; the discrete bottleneck is the load-bearing anti-collapse term (with a covariance-trace certificate). *Multivariate time-series.* |
| `paper/jepa-dna/` | **JEPA-DNA** (Larey et al., 2026, [2602.17162](https://arxiv.org/abs/2602.17162)): a model-agnostic JEPA continual-training branch that grounds pretrained genomic FMs by predicting masked-span global latents (EMA target + cosine loss + VICReg); new SOTA on 17 genomic tasks. *Genomics/DNA.* |
| `paper/var-jepa/` | **Var-JEPA** (Gögl & Yau, 2026, [2603.20111](https://arxiv.org/abs/2603.20111)): recasts JEPA as a deterministic special case of a coupled latent-variable VAE; a single ELBO (predictor = learned conditional prior) avoids ad-hoc anti-collapse losses and adds latent uncertainty (Var-T-JEPA). *Theory / tabular.* |
| `paper/koopman-jepa/` | **Koopman-JEPA** (Ruiz-Morales et al., AAAI 2026, [2511.09783](https://arxiv.org/abs/2511.09783)): proves the idealized JEPA loss is minimized when the encoder spans the eigenvalue-1 invariant subspace of the data's Koopman operator (regime indicators), explaining JEPAs' emergent regime clustering. *Theory.* |
| `paper/protein-jepa/` | **ProteinJEPA** (Ofer, Linial & Shahaf, 2026, [2605.07554](https://arxiv.org/abs/2605.07554)): masked-position MLM+JEPA — a cosine latent-prediction loss at masked positions with detached targets + SIGReg (no EMA teacher) added to MLM; beats MLM-only on ESM2; JEPA-only collapses. *Proteins.* |
| `paper/mini-jepa-hydrology/` | **Mini-JEPA** (Rahman, 2026, [2605.14120](https://arxiv.org/abs/2605.14120)): a fleet of five 22M-param sensor-specialized I-JEPA+VICReg models (ViT-S, 64-d) over satellite products, composed by an LLM router; matches planetary-scale AlphaEarth on physics-matched hydrologic tasks at workstation compute. *Hydrology.* |
| `paper/jets/` | **JETS** (Xie et al., NeurIPS TS4H 2025, non-archival; [OpenReview](https://openreview.net/forum?id=QqQDjLgHab)): first JEPA on long-horizon irregular multivariate wearable behavioral time series; EMA target + 70% patch masking + latent-MSE on ~3M person-days; beats MAE/contrastive on disease & biomarker linear probes. *Wearable health.* |
