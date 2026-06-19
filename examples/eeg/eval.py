"""EEG — downstream evaluation (the patient-disjoint abnormality probe).

The feature-extraction harness is provided: per recording, encode N evenly-spaced
10 s windows with the FROZEN encoder and mean-pool them into ONE embedding. What
you implement (`# TODO`) is the probe + metric.

GOLDEN RULE — patient-disjoint split: fit the probe on `train` patients, score on
`eval` patients (no subject overlap). A probe that scores well *within* a subject
but collapses across subjects is measuring identity, not pathology — so the held-
out-patient number is the only one that answers the transferability question.

Run:  python -m examples.eeg.eval --ckpt <.../latest.pth.tar>
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, device, data_cfg):
    """Provided: frozen encoder -> [N_rec, D] recording-level features + labels.

    One embedding per recording: encode its N windows and mean-pool them.
    """
    values = OmegaConf.to_container(data_cfg, resolve=True)
    values.update({"split": split, "mode": "probe"})
    ds = EEGDataset(EEGConfig(**values))
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=16,
                                         pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:          # wins: [B, N, C, T]
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        z = encoder.represent(flat).reshape(B, N, -1).mean(dim=1)  # [B, D]
        z = z.cpu().numpy()
        for k in range(B):
            if bool(ok[k]):                  # drop unreadable recordings
                X.append(z[k]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


# --------------------------------------------------------------------------- #
# PROBE + METRIC  — # TODO
# --------------------------------------------------------------------------- #
def probe(Xtr, ytr, Xev, yev):
    """TODO: fit a PATIENT-DISJOINT linear probe on the FROZEN train features and
    score on the held-out-patient eval features. Return a metrics dict.

    No leakage: standardize features on TRAIN stats only (sklearn StandardScaler
    fit on Xtr), then fit a LogisticRegression (class_weight='balanced' helps the
    normal/abnormal imbalance) and score on the eval embeddings. Report:
        accuracy / balanced-accuracy / AUROC   (normal=0 vs abnormal=1)

    To make the number meaningful, also run this same probe on (a) a RANDOM
    untrained encoder (floor) and (b) a supervised end-to-end baseline, and
    compare. The eval metrics are on held-out patients — stress that."""
    raise NotImplementedError("TODO: implement the patient-disjoint probe + metric (see docstring)")


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    print("[eeg-eval] extracting TRAIN embeddings (fit set)...", flush=True)
    Xtr, ytr = extract_features(encoder, "train", device, cfg.data)
    print("[eeg-eval] extracting EVAL embeddings (held-out patients)...", flush=True)
    Xev, yev = extract_features(encoder, "eval", device, cfg.data)
    print("[eeg-eval]", probe(Xtr, ytr, Xev, yev))


if __name__ == "__main__":
    main()
