"""Strict TUAB EDF loader backed by the output of :mod:`.audit`.

The loader never assumes that the first 19 EDF signals form the desired montage
and never replaces unreadable data with zeros. Channel selection, patient split,
and source sampling frequency come from a validated audit manifest.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from eb_jepa.datasets.eeg.audit import CANONICAL_CHANNELS

try:
    import pyedflib
except ImportError:  # pragma: no cover - clear runtime error below
    pyedflib = None

try:
    from scipy import signal
except ImportError:  # pragma: no cover - clear runtime error below
    signal = None


@dataclass
class EEGConfig:
    data_root: str = (
        "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/"
        "TUAB_PREPROCESSED"
    )
    manifest_path: Optional[str] = None
    split: str = "train"
    mode: str = "ssl"
    n_channels: int = len(CANONICAL_CHANNELS)
    sfreq: int = 200
    window_sec: float = 10.0
    apply_signal_preprocessing: bool = False
    filter_pad_sec: float = 10.0
    bandpass_low_hz: float = 0.1
    bandpass_high_hz: float = 75.0
    notch_hz: float = 60.0
    epoch_size: int = 20000
    n_windows: int = 16
    batch_size: int = 128
    num_workers: int = 8
    aug_noise_std: float = 0.1
    aug_scale_jitter: float = 0.2
    aug_chan_drop_p: float = 0.2
    aug_time_mask_frac: float = 0.2


@dataclass(frozen=True)
class ManifestItem:
    path: str
    split: str
    label: int
    patient_id: str
    duration_sec: float
    source_sfreq: float
    channel_indices: tuple[int, ...]


def _zscore(x: np.ndarray, axis: int) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32, copy=False)


def _load_manifest(path: str | Path, split: str) -> list[ManifestItem]:
    manifest = Path(path).expanduser()
    if not manifest.exists():
        raise FileNotFoundError(
            f"Validated EEG manifest not found: {manifest}. Run "
            "`python -m eb_jepa.datasets.eeg.audit --data-root ... "
            "--output-dir ...` first."
        )
    items = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            indices = tuple(int(value) for value in row["channel_indices"].split(";"))
            if len(indices) != len(CANONICAL_CHANNELS):
                raise ValueError(f"Invalid channel mapping in manifest for {row['path']}")
            items.append(
                ManifestItem(
                    path=row["path"],
                    split=row["split"],
                    label=int(row["label"]),
                    patient_id=row["patient_id"],
                    duration_sec=float(row["duration_sec"]),
                    source_sfreq=float(row["source_sfreq"]),
                    channel_indices=indices,
                )
            )
    if not items:
        raise ValueError(f"No accepted recordings for split={split!r} in {manifest}")
    return items


def _preprocess_signal(x: np.ndarray, source_sfreq: float, cfg: EEGConfig) -> np.ndarray:
    """Bandpass, notch, resample, then z-score an ordered multichannel segment."""
    if signal is None:
        raise ImportError("scipy is required for EEG filtering and resampling")
    nyquist = source_sfreq / 2.0
    if not 0 < cfg.bandpass_low_hz < cfg.bandpass_high_hz < nyquist:
        raise ValueError(
            f"Invalid {cfg.bandpass_low_hz}-{cfg.bandpass_high_hz} Hz bandpass "
            f"for source sfreq={source_sfreq} Hz"
        )
    sos = signal.butter(
        4,
        [cfg.bandpass_low_hz, cfg.bandpass_high_hz],
        btype="bandpass",
        fs=source_sfreq,
        output="sos",
    )
    x = signal.sosfiltfilt(sos, x, axis=-1)
    if 0 < cfg.notch_hz < nyquist:
        b_notch, a_notch = signal.iirnotch(cfg.notch_hz, Q=30.0, fs=source_sfreq)
        x = signal.filtfilt(b_notch, a_notch, x, axis=-1)
    ratio = Fraction(cfg.sfreq / source_sfreq).limit_denominator(1000)
    if ratio.numerator != ratio.denominator:
        x = signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)
    return x.astype(np.float32, copy=False)


def _close_reader(reader) -> None:
    try:
        reader.close()
    except AttributeError:
        reader._close()


class EEGDataset(torch.utils.data.Dataset):
    """SSL random windows or recording-level probe windows from audited EDFs."""

    def __init__(self, cfg: EEGConfig):
        if pyedflib is None:
            raise ImportError("pyedflib is required to read EDF files")
        if cfg.n_channels != len(CANONICAL_CHANNELS):
            raise ValueError(
                f"n_channels must be {len(CANONICAL_CHANNELS)} for the audited montage"
            )
        if cfg.split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'")
        if cfg.mode not in {"ssl", "probe", "supervised"}:
            raise ValueError("mode must be 'ssl', 'probe', or 'supervised'")
        self.cfg = cfg
        manifest_path = cfg.manifest_path or str(Path(cfg.data_root) / "audit" / "edf_inventory.csv")
        self.items = _load_manifest(manifest_path, cfg.split)
        self.window = int(round(cfg.window_sec * cfg.sfreq))
        self._rng = np.random.default_rng()

    def __len__(self):
        return self.cfg.epoch_size if self.cfg.mode == "ssl" else len(self.items)

    def _read_window(self, item: ManifestItem, start_sec: float) -> np.ndarray:
        cfg = self.cfg
        pad = cfg.filter_pad_sec if cfg.apply_signal_preprocessing else 0.0
        read_start_sec = max(0.0, start_sec - pad)
        read_end_sec = min(item.duration_sec, start_sec + cfg.window_sec + pad)
        start_sample = int(np.floor(read_start_sec * item.source_sfreq))
        n_samples = int(np.ceil((read_end_sec - read_start_sec) * item.source_sfreq))
        reader = None
        try:
            reader = pyedflib.EdfReader(item.path)
            x = np.empty((cfg.n_channels, n_samples), dtype=np.float64)
            for out_index, source_index in enumerate(item.channel_indices):
                values = reader.readSignal(source_index, start_sample, n_samples)
                if len(values) != n_samples:
                    raise IOError(
                        f"Short EDF read in {item.path}: expected {n_samples}, got {len(values)}"
                    )
                x[out_index] = values
        except Exception as exc:
            raise RuntimeError(f"Failed to read audited EDF {item.path}: {exc}") from exc
        finally:
            if reader is not None:
                _close_reader(reader)

        if cfg.apply_signal_preprocessing:
            x = _preprocess_signal(x, item.source_sfreq, cfg)
        elif abs(item.source_sfreq - cfg.sfreq) > 1e-6:
            raise RuntimeError(
                f"Audited source sfreq={item.source_sfreq} differs from target "
                f"sfreq={cfg.sfreq}; enable apply_signal_preprocessing"
            )
        else:
            x = x.astype(np.float32, copy=False)
        crop_start = int(round((start_sec - read_start_sec) * cfg.sfreq))
        x = x[:, crop_start : crop_start + self.window]
        if x.shape != (cfg.n_channels, self.window):
            raise RuntimeError(
                f"Preprocessed window has shape {x.shape}, expected "
                f"({cfg.n_channels}, {self.window}) for {item.path}"
            )
        return _zscore(x, axis=1)

    def _read_random_window(self) -> np.ndarray:
        errors = []
        for _ in range(8):
            item = self.items[int(self._rng.integers(len(self.items)))]
            max_start = item.duration_sec - self.cfg.window_sec
            start = float(self._rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
            try:
                return self._read_window(item, start)
            except RuntimeError as exc:
                errors.append(str(exc))
        raise RuntimeError("Unable to read an audited EEG window after 8 attempts: " + " || ".join(errors))

    def _read_recording_windows(self, item: ManifestItem) -> np.ndarray:
        max_start = item.duration_sec - self.cfg.window_sec
        starts = np.linspace(0.0, max(0.0, max_start), self.cfg.n_windows)
        return np.stack([self._read_window(item, float(start)) for start in starts])

    def _augment(self, x: np.ndarray) -> np.ndarray:
        cfg, rng = self.cfg, self._rng
        x = x.copy()
        if cfg.aug_scale_jitter > 0:
            scale = 1.0 + rng.uniform(
                -cfg.aug_scale_jitter,
                cfg.aug_scale_jitter,
                size=(cfg.n_channels, 1),
            ).astype(np.float32)
            x *= scale
        if cfg.aug_noise_std > 0:
            x += rng.normal(0, cfg.aug_noise_std, size=x.shape).astype(np.float32)
        if cfg.aug_chan_drop_p > 0:
            mask = (rng.random(cfg.n_channels) > cfg.aug_chan_drop_p).astype(np.float32)
            x *= mask[:, None]
        if cfg.aug_time_mask_frac > 0:
            mask_len = int(rng.uniform(0, cfg.aug_time_mask_frac) * self.window)
            if mask_len > 0:
                start = int(rng.integers(0, self.window - mask_len + 1))
                x[:, start : start + mask_len] = 0.0
        return x

    def __getitem__(self, index):
        if self.cfg.mode == "ssl":
            x = self._read_random_window()
            return torch.from_numpy(self._augment(x)), torch.from_numpy(self._augment(x))
        item = self.items[index]
        windows = self._read_recording_windows(item)
        return torch.from_numpy(windows), item.label, True


def make_loader(cfg: EEGConfig, shuffle=None):
    dataset = EEGDataset(cfg)
    is_train = cfg.mode == "ssl" and cfg.split == "train"
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=cfg.num_workers > 0,
    )
