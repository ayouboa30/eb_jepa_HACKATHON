# EEG — self-supervised representation learning on TUH EEG (abnormality detection)

**Question.** Can *two-view invariance learning* on **unlabeled** EEG learn features
that linearly separate **normal vs abnormal** recordings, and generalize to
**held-out (patient-disjoint) subjects**?

## Data
TUH Abnormal EEG corpus: raw `.edf` files audited for the common **19-channel**
10-20 montage, source sampling frequency, duration, and patient identity.
The provided EDFs are already bandpass-filtered at 0.1-75 Hz, notch-filtered at
60 Hz, and resampled to 200 Hz; the loader audits these properties and does not
apply them a second time. It windows the data and applies per-channel z-scoring.
Patient-disjoint `train` / `eval`
splits each contain `normal` / `abnormal` recordings. The default location is
`/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED`.
Training is blocked until the audit manifest exists; unreadable recordings are
rejected rather than replaced with zero tensors.

## Mandatory EDF audit

```bash
python -m eb_jepa.datasets.eeg.audit \
  --data-root /lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED \
  --output-dir /lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED/audit
```

This writes `edf_inventory.csv`, `channel_inventory.csv`,
`rejected_files.csv`, and `audit_summary.json`. Any patient found in both
`train` and `eval` is rejected and makes the command exit unsuccessfully.

## Layout
```
eb_jepa/datasets/eeg/   dataset.py (provided EDF loader) + data_config.yaml
examples/eeg/
  main.py     SSL pretraining — TODO: build_encoder() + build_ssl()
  eval.py     patient-disjoint probe — TODO: probe() + metric
  cfgs/    train.yaml, eval.yaml
```

## What you implement (the `# TODO`s)
1. `main.py:build_encoder` — a 1D encoder over `[B, 19, T]` (`represent() -> [B, D]`,
   and `frames()` if you go predictive).
2. `main.py:build_ssl` — the SSL objective: two-view VICReg (`Projector` +
   `VICRegLoss`, the natural choice — the dataset already returns two views)
   **or** predictive JEPA (eb_jepa `RNNPredictor` + EMA target + `VCLoss`).
3. `eval.py:probe` — the **patient-disjoint** frozen-feature probe + metric
   (`LogisticRegression`; accuracy / balanced-acc / AUROC), compared to a
   random-encoder floor and a supervised end-to-end baseline.

Everything else (EDF loading, two-view training loop, recording-level feature
extraction) is provided.

## Run
```bash
python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml
python -m examples.eeg.eval --ckpt <.../latest.pth.tar>
```

## Extension — TUEV (the "hard" one)
TUAB is recording-level binary (normal vs abnormal). The harder variant is **TUEV**
(TUH EEG Events): **6-class**, **second-level** event labels
(`SPSW, GPED, PLED, EYEM, ARTF, BCKG`), a tiny + massively imbalanced corpus. The
same per-frame encoder feeds a temporal model; the probe becomes a patient-disjoint
**6-class** classifier (macro-F1 / macro-AUROC, fighting the background imbalance).
A natural follow-up once the binary TUAB probe works.
