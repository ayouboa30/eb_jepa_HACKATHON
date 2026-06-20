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
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import wandb
except ImportError:  # optional dependency
    wandb = None

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


def _arg_value(flag: str, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _load_checkpoint_with_optional_artifact(eval_cfg, wb_run, device):
    wandb_cfg = eval_cfg.get("wandb", {}) if eval_cfg is not None else {}
    eval_section = eval_cfg.get("eval", {}) if eval_cfg is not None else {}
    use_artifact = bool(wandb_cfg.get("use_artifact", False))

    if use_artifact:
        if wb_run is None:
            raise ValueError("wandb.use_artifact=true requires an active wandb run")
        artifact_name = wandb_cfg.get("artifact_name", None)
        if not artifact_name:
            raise ValueError("wandb.use_artifact=true but wandb.artifact_name is missing")
        artifact = wb_run.use_artifact(artifact_name, type="model")
        artifact_dir = Path(artifact.download())
        ckpt_path = artifact_dir / "best.pth.tar"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Artifact {artifact_name} downloaded but best.pth.tar not found")
        return str(ckpt_path), f"artifact:{artifact_name}"

    ckpt = _arg_value("--ckpt", None)
    if ckpt is None:
        ckpt = eval_section.get("checkpoint_path", None)
    if ckpt is None:
        raise ValueError("No checkpoint provided. Use --ckpt or eval.checkpoint_path in eval.yaml")
    return ckpt, f"local:{ckpt}"


def main():
    eval_fname = _arg_value("--fname", "examples/eeg/cfgs/eval.yaml")
    eval_cfg = OmegaConf.load(eval_fname)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wb_run = None
    wb_cfg = eval_cfg.get("wandb", None)
    wb_enabled = bool(wb_cfg and wb_cfg.get("enabled", False))
    if wb_enabled:
        if wandb is None:
            raise ImportError("wandb is enabled in config but package is not installed")
        wb_run = wandb.init(
            project=wb_cfg.get("project", "eb-jepa-eeg"),
            entity=wb_cfg.get("entity", None),
            name=wb_cfg.get("run_name", None),
            group=wb_cfg.get("group", None),
            job_type=wb_cfg.get("job_type", "eval"),
            tags=list(wb_cfg.get("tags", ["evaluation"])),
            config=OmegaConf.to_container(eval_cfg, resolve=True),
        )

    ckpt, ckpt_source = _load_checkpoint_with_optional_artifact(eval_cfg, wb_run, device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])

    if wb_run is not None:
        wandb.log({
            "eval/checkpoint_path": ckpt,
            "eval/checkpoint_source": ckpt_source,
        })
        source_train_run_id = eval_cfg.get("eval", {}).get("source_train_run_id", None)
        if source_train_run_id:
            wandb.log({"eval/parent_train_run_id": source_train_run_id})
            wb_run.summary["parent_train_run_id"] = source_train_run_id
        ckpt_train_run_id = state.get("wandb_run_id", None)
        if ckpt_train_run_id:
            wandb.log({"eval/checkpoint_wandb_run_id": ckpt_train_run_id})
            wb_run.summary["checkpoint_wandb_run_id"] = ckpt_train_run_id

    encoder = build_encoder(cfg.model).to(device)
    enc_state = state.get("encoder_state_dict", state.get("encoder", None))
    if enc_state is None:
        raise KeyError("Checkpoint does not contain encoder state dict")
    encoder.load_state_dict(enc_state); encoder.eval()

    try:
        print("[eeg-eval] extracting TRAIN embeddings (fit set)...", flush=True)
        Xtr, ytr = extract_features(encoder, "train", device, cfg.data)
        print("[eeg-eval] extracting EVAL embeddings (held-out patients)...", flush=True)
        Xev, yev = extract_features(encoder, "eval", device, cfg.data)
        if wb_run is not None:
            wandb.log(
                {
                    "eval/n_train_embeddings": int(len(ytr)),
                    "eval/n_eval_embeddings": int(len(yev)),
                    "eval/embedding_dim": int(Xtr.shape[1]) if Xtr.ndim == 2 else -1,
                }
            )
        metrics = probe(Xtr, ytr, Xev, yev)
        print("[eeg-eval]", metrics)
        if wb_run is not None and isinstance(metrics, dict):
            to_log = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    to_log[f"eval/{key}"] = float(value)
                elif key == "fractions" and isinstance(value, dict):
                    for frac_key, frac_metrics in value.items():
                        if isinstance(frac_metrics, dict):
                            for mk, mv in frac_metrics.items():
                                if isinstance(mv, (int, float, np.integer, np.floating)):
                                    frac_name = str(frac_key).replace(".", "")
                                    to_log[f"eval/{frac_name}/{mk}"] = float(mv)
                elif key == "confusion_matrix" and isinstance(value, (list, tuple, np.ndarray)):
                    try:
                        cm = np.asarray(value)
                        if cm.ndim == 2 and cm.shape[0] == 2 and cm.shape[1] == 2:
                            to_log["eval/confusion_matrix"] = wandb.Table(
                                columns=["pred_normal", "pred_abnormal"],
                                data=cm.tolist(),
                            )
                    except Exception:
                        pass
            if to_log:
                wandb.log(to_log)
    finally:
        if wb_run is not None:
            wb_run.finish()


if __name__ == "__main__":
    main()
