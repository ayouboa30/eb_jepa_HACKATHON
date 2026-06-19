"""EEG augmentation benchmark with reconstruction-based doubling.

Evaluates normal vs abnormal classification as train fraction increases:
5%, 25%, 75%, 100%.
For each fraction, trains a linear probe on:
  - original data
  - original + reconstructed data
and reports accuracy / recall on the patient-disjoint eval split.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from eb_jepa.models.spectral_state_vae import SpectralStateVAE, spectral_features


def load_recordings(split: str, cfg: EEGConfig, max_items: int | None = None):
    ds = EEGDataset(EEGConfig(**{**cfg.__dict__, "split": split, "mode": "probe"}))
    idxs = range(len(ds)) if max_items is None else range(min(max_items, len(ds)))
    X, y = [], []
    for i in idxs:
        wins, label, ok = ds[i]
        if ok:
            X.append(wins.float())
            y.append(int(label))
    return X, np.asarray(y)


def window_features(window: torch.Tensor) -> np.ndarray:
    mu = window.mean(dim=-1)
    sd = window.std(dim=-1)
    _, log_amp, _, _ = spectral_features(window.unsqueeze(0))
    spec = log_amp.squeeze(0)
    bands = [spec[..., 1:5].mean(), spec[..., 4:8].mean(), spec[..., 8:13].mean(), spec[..., 13:30].mean(), spec[..., 30:75].mean()]
    return torch.cat([mu, sd, torch.stack(bands).repeat(window.shape[0])]).cpu().numpy()


@torch.no_grad()
def encode_windows(windows, device):
    feats = []
    for x in windows:
        feats.append(window_features(x.to(device)))
    return np.stack(feats)


@torch.no_grad()
def reconstruct_windows(model, windows, device):
    outs = []
    for x in windows:
        x = x.to(device)
        _, log_amp, cos, sin = spectral_features(x.unsqueeze(0))
        out = model(x.unsqueeze(0), log_amp, cos, sin, torch.zeros(1, 1, device=device), sample=False)
        outs.append(out["F_hat"].squeeze(0).cpu())
    return outs


def fit_probe(Xtr, ytr, Xev, yev):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xev)
    return {
        "accuracy": float(accuracy_score(yev, pred)),
        "recall": float(recall_score(yev, pred, pos_label=1, zero_division=0)),
    }


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device)
    dcfg = EEGConfig(
        data_root=args.data_root,
        manifest_path=args.manifest,
        split="train",
        mode="probe",
        n_channels=19,
        sfreq=200,
        window_sec=10.0,
        batch_size=1,
        num_workers=0,
    )
    train_wins, ytr = load_recordings("train", dcfg, args.max_train)
    eval_wins, yev = load_recordings("eval", dcfg, args.max_eval)
    model = SpectralStateVAE(channels=19, time_samples=train_wins[0].shape[-1], latent_dim=32, hidden_dim=128).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    fractions = [0.05, 0.25, 0.75, 1.0]
    n_train = len(train_wins)
    for frac in fractions:
        k = max(1, int(round(frac * n_train)))
        subset_wins = train_wins[:k]
        subset_y = ytr[:k]
        aug_wins = subset_wins + reconstruct_windows(model, subset_wins, device)
        aug_y = np.concatenate([subset_y, subset_y], axis=0)

        Xev = encode_windows(eval_wins, device)
        Xbase = encode_windows(subset_wins, device)
        Xaug = encode_windows(aug_wins, device)
        base = fit_probe(Xbase, subset_y, Xev, yev)
        aug = fit_probe(Xaug, aug_y, Xev, yev)
        rows.append((frac, base, aug))

    print("| data % | acc base | recall base | acc +x2 | recall +x2 |")
    print("|---:|---:|---:|---:|---:|")
    for frac, base, aug in rows:
        print(
            f"| {int(frac*100)}% | {base['accuracy']:.4f} | {base['recall']:.4f} | "
            f"{aug['accuracy']:.4f} | {aug['recall']:.4f} |"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-train", type=int, default=200)
    p.add_argument("--max-eval", type=int, default=100)
    run(p.parse_args())
