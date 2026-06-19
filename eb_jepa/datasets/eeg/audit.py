"""Audit TUAB EDF metadata and build a strict, patient-disjoint manifest.

Run before training::

    python -m eb_jepa.datasets.eeg.audit \
        --data-root /path/to/TUAB_PREPROCESSED \
        --output-dir /path/to/TUAB_PREPROCESSED/audit

No signal samples are transformed by this command. It only reads EDF headers and
writes inventories describing which recordings are safe for preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import pyedflib
except ImportError:  # pragma: no cover - exercised by the CLI error path
    pyedflib = None


# TUAB commonly uses the legacy temporal names T3/T4/T5/T6. Modern aliases are
# accepted but represented canonically with the TUAB names.
CANONICAL_CHANNELS = (
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "FZ",
    "CZ",
    "PZ",
)

CHANNEL_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
PATIENT_PATTERN = re.compile(r"([a-z0-9]{8})(?=_s\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class RecordingRecord:
    path: str
    split: str
    label: int
    class_name: str
    patient_id: str
    duration_sec: float
    source_sfreq: float
    n_source_channels: int
    channel_indices: str
    canonical_channels: str
    accepted: bool
    rejection_reason: str


def normalize_channel_name(label: str) -> str:
    """Normalize referential TUH labels without accepting bipolar derivations."""
    name = label.strip().upper().replace(" ", "")
    name = re.sub(r"^EEG", "", name)
    name = re.sub(r"-(REF|LE|AVG|AR)$", "", name)
    name = name.removesuffix("-0")
    return CHANNEL_ALIASES.get(name, name)


def extract_patient_id(path: str | Path) -> str | None:
    """Extract the eight-digit TUH subject identifier from a path."""
    matches = PATIENT_PATTERN.findall(str(path))
    return matches[-1] if matches else None


def infer_split_and_label(path: Path, root: Path) -> tuple[str, str, int]:
    parts = [part.lower() for part in path.relative_to(root).parts]
    split = next((value for value in ("train", "eval") if value in parts), "")
    class_name = next(
        (value for value in ("normal", "abnormal") if value in parts), ""
    )
    label = {"normal": 0, "abnormal": 1}.get(class_name, -1)
    return split, class_name, label


def resolve_montage(labels: Sequence[str]) -> tuple[list[int], list[str]]:
    """Return source indices in canonical order and validation errors."""
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_name[normalize_channel_name(label)].append(index)

    errors = []
    indices = []
    for channel in CANONICAL_CHANNELS:
        candidates = by_name.get(channel, [])
        if not candidates:
            errors.append(f"missing_channel:{channel}")
        elif len(candidates) > 1:
            errors.append(f"duplicate_channel:{channel}")
        else:
            indices.append(candidates[0])
    return indices, errors


def _safe_close(reader) -> None:
    try:
        reader.close()
    except AttributeError:
        reader._close()


def inspect_edf(path: Path, root: Path, min_duration_sec: float) -> tuple[RecordingRecord, list[dict]]:
    split, class_name, label = infer_split_and_label(path, root)
    patient_id = extract_patient_id(path) or ""
    errors = []
    if split not in {"train", "eval"}:
        errors.append("unknown_split")
    if label < 0:
        errors.append("unknown_class")
    if not patient_id:
        errors.append("missing_patient_id")

    reader = None
    channel_rows: list[dict] = []
    duration = 0.0
    source_sfreq = 0.0
    labels: list[str] = []
    indices: list[int] = []
    try:
        reader = pyedflib.EdfReader(str(path))
        labels = list(reader.getSignalLabels())
        sample_freqs = [float(value) for value in reader.getSampleFrequencies()]
        n_samples = [int(value) for value in reader.getNSamples()]
        for index, (raw_name, sfreq, count) in enumerate(
            zip(labels, sample_freqs, n_samples)
        ):
            channel_rows.append(
                {
                    "path": str(path.resolve()),
                    "source_index": index,
                    "raw_name": raw_name,
                    "normalized_name": normalize_channel_name(raw_name),
                    "sfreq": sfreq,
                    "n_samples": count,
                    "duration_sec": count / sfreq if sfreq > 0 else 0.0,
                }
            )
        indices, montage_errors = resolve_montage(labels)
        errors.extend(montage_errors)
        if len(indices) == len(CANONICAL_CHANNELS):
            selected_freqs = [sample_freqs[index] for index in indices]
            selected_durations = [
                n_samples[index] / sample_freqs[index]
                for index in indices
                if sample_freqs[index] > 0
            ]
            if len(selected_durations) != len(CANONICAL_CHANNELS):
                errors.append("invalid_sampling_frequency")
            elif max(selected_freqs) - min(selected_freqs) > 1e-6:
                errors.append("mixed_sampling_frequencies")
            else:
                source_sfreq = selected_freqs[0]
                duration = min(selected_durations)
                if duration < min_duration_sec:
                    errors.append(f"duration_below_{min_duration_sec:g}s")
    except Exception as exc:
        errors.append(f"unreadable_edf:{type(exc).__name__}:{exc}")
    finally:
        if reader is not None:
            _safe_close(reader)

    record = RecordingRecord(
        path=str(path.resolve()),
        split=split,
        label=label,
        class_name=class_name,
        patient_id=patient_id,
        duration_sec=round(duration, 6),
        source_sfreq=round(source_sfreq, 6),
        n_source_channels=len(labels),
        channel_indices=";".join(map(str, indices)),
        canonical_channels=";".join(CANONICAL_CHANNELS),
        accepted=not errors,
        rejection_reason="|".join(errors),
    )
    return record, channel_rows


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(data_root: str | Path, output_dir: str | Path, min_duration_sec: float = 10.0) -> dict:
    if pyedflib is None:
        raise ImportError("pyedflib is required: install the project EEG dependencies")
    root = Path(data_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    files = sorted(root.glob("**/*.edf"))
    if not files:
        raise FileNotFoundError(f"No EDF files found under {root}")
    output.mkdir(parents=True, exist_ok=True)

    records: list[RecordingRecord] = []
    channels: list[dict] = []
    for path in files:
        record, channel_rows = inspect_edf(path, root, min_duration_sec)
        records.append(record)
        channels.extend(channel_rows)

    split_patients: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.accepted:
            split_patients[record.split].add(record.patient_id)
    leaked_patients = split_patients["train"] & split_patients["eval"]
    if leaked_patients:
        records = [
            RecordingRecord(
                **{
                    **asdict(record),
                    "accepted": False,
                    "rejection_reason": (
                        record.rejection_reason + "|" if record.rejection_reason else ""
                    )
                    + "patient_overlap_train_eval",
                }
            )
            if record.patient_id in leaked_patients
            else record
            for record in records
        ]

    accepted = [asdict(record) for record in records if record.accepted]
    rejected = [
        {"path": record.path, "rejection_reason": record.rejection_reason}
        for record in records
        if not record.accepted
    ]
    _write_csv(
        output / "edf_inventory.csv",
        accepted,
        list(RecordingRecord.__dataclass_fields__),
    )
    _write_csv(
        output / "channel_inventory.csv",
        channels,
        [
            "path",
            "source_index",
            "raw_name",
            "normalized_name",
            "sfreq",
            "n_samples",
            "duration_sec",
        ],
    )
    _write_csv(output / "rejected_files.csv", rejected, ["path", "rejection_reason"])

    reason_counts = Counter(
        reason
        for record in records
        if not record.accepted
        for reason in record.rejection_reason.split("|")
    )
    summary = {
        "data_root": str(root),
        "files_found": len(records),
        "files_accepted": len(accepted),
        "files_rejected": len(rejected),
        "accepted_by_split_class": dict(
            Counter(f"{row['split']}/{row['class_name']}" for row in accepted)
        ),
        "patients_by_split": {
            split: len({row["patient_id"] for row in accepted if row["split"] == split})
            for split in ("train", "eval")
        },
        "patient_overlap_train_eval": sorted(leaked_patients),
        "source_sampling_frequencies": dict(
            Counter(str(row["source_sfreq"]) for row in accepted)
        ),
        "duration_sec": {
            "min": min((row["duration_sec"] for row in accepted), default=None),
            "max": max((row["duration_sec"] for row in accepted), default=None),
            "total_hours": round(
                sum(row["duration_sec"] for row in accepted) / 3600.0, 3
            ),
        },
        "rejection_reasons": dict(reason_counts),
    }
    with (output / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-duration-sec", type=float, default=10.0)
    args = parser.parse_args()
    summary = audit_dataset(args.data_root, args.output_dir, args.min_duration_sec)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["patient_overlap_train_eval"]:
        raise SystemExit("Patient leakage detected; overlapping recordings were rejected")


if __name__ == "__main__":
    main()
