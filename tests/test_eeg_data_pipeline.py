import csv
from pathlib import Path

import numpy as np
import pytest

from eb_jepa.datasets.eeg import audit
from eb_jepa.datasets.eeg.audit import CANONICAL_CHANNELS
from eb_jepa.datasets.eeg.dataset import EEGConfig, _load_manifest, _preprocess_signal


def test_channel_normalization_and_ordered_montage():
    labels = [f"EEG {name}-REF" for name in reversed(CANONICAL_CHANNELS)]
    labels[labels.index("EEG T3-REF")] = "EEG T7-REF"
    labels[labels.index("EEG T4-REF")] = "EEG T8-REF"
    labels[labels.index("EEG T5-REF")] = "EEG P7-REF"
    labels[labels.index("EEG T6-REF")] = "EEG P8-REF"

    indices, errors = audit.resolve_montage(labels)

    assert errors == []
    assert [audit.normalize_channel_name(labels[index]) for index in indices] == list(
        CANONICAL_CHANNELS
    )


def test_extracts_alphanumeric_tuab_patient_id():
    assert audit.extract_patient_id("aaaaajtg_s001_t000.edf") == "aaaaajtg"


def test_bipolar_channel_is_not_mistaken_for_referential_channel():
    labels = [f"EEG {name}-REF" for name in CANONICAL_CHANNELS]
    labels[0] = "EEG FP1-F7"

    _, errors = audit.resolve_montage(labels)

    assert "missing_channel:FP1" in errors


def test_preprocessing_resamples_to_target_frequency_and_zscore_is_finite():
    cfg = EEGConfig(sfreq=200, bandpass_low_hz=0.1, bandpass_high_hz=75.0)
    source_sfreq = 256.0
    seconds = 30
    time = np.arange(int(source_sfreq * seconds)) / source_sfreq
    base = np.sin(2 * np.pi * 10 * time) + 0.2 * np.sin(2 * np.pi * 60 * time)
    x = np.stack([base + channel * 0.01 for channel in range(19)])

    processed = _preprocess_signal(x, source_sfreq, cfg)

    assert processed.shape == (19, seconds * cfg.sfreq)
    assert processed.dtype == np.float32
    assert np.isfinite(processed).all()


def test_manifest_loader_requires_exact_validated_channel_mapping(tmp_path):
    manifest = tmp_path / "edf_inventory.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "split",
                "label",
                "patient_id",
                "duration_sec",
                "source_sfreq",
                "channel_indices",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": "record.edf",
                "split": "train",
                "label": 0,
                "patient_id": "00000001",
                "duration_sec": 60,
                "source_sfreq": 200,
                "channel_indices": "0;1",
            }
        )

    with pytest.raises(ValueError, match="Invalid channel mapping"):
        _load_manifest(manifest, "train")


class _FakeEdfReader:
    def __init__(self, path):
        self.path = path

    def getSignalLabels(self):
        return [f"EEG {name}-REF" for name in CANONICAL_CHANNELS]

    def getSampleFrequencies(self):
        return np.full(len(CANONICAL_CHANNELS), 256.0)

    def getNSamples(self):
        return np.full(len(CANONICAL_CHANNELS), 256 * 20)

    def close(self):
        return None


def test_audit_rejects_patient_overlap_between_train_and_eval(tmp_path, monkeypatch):
    root = tmp_path / "tuab"
    train = root / "train" / "normal" / "00000001_s001_t000.edf"
    evaluation = root / "eval" / "abnormal" / "00000001_s002_t000.edf"
    accepted = root / "train" / "abnormal" / "00000002_s001_t000.edf"
    for path in (train, evaluation, accepted):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(audit, "pyedflib", type("FakeModule", (), {"EdfReader": _FakeEdfReader}))

    output = tmp_path / "audit"
    summary = audit.audit_dataset(root, output)

    assert summary["patient_overlap_train_eval"] == ["00000001"]
    assert summary["files_accepted"] == 1
    assert summary["files_rejected"] == 2
    rejected = (output / "rejected_files.csv").read_text(encoding="utf-8")
    assert "patient_overlap_train_eval" in rejected
    assert (output / "edf_inventory.csv").exists()
    assert (output / "channel_inventory.csv").exists()
    assert (output / "audit_summary.json").exists()
