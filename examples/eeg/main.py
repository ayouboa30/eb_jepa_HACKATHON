"""EEG — SSL pretraining entrypoint (self-supervised representation learning).

Research question: can two-view invariance learning on unlabeled EEG learn
features that linearly separate *normal vs abnormal* recordings, generalizing
to held-out (patient-disjoint) subjects?

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement
are marked `# TODO` below — that is the whole point of the track:
  1. the 1D encoder over [B, C=19, T]
  2. the SSL objective (two-view VICReg  *or*  predictive JEPA)
The downstream probe + metric is the third `# TODO`, in eval.py.

Run:  python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

try:
    import wandb
except ImportError:  # optional dependency
    wandb = None

from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP), RNNPredictor (GRU)
#   eb_jepa.losses:        VICRegLoss (inv+var+cov), VCLoss (variance+covariance)


import torch.nn as nn
from eb_jepa.architectures import Projector
from eb_jepa.losses import VICRegLoss


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _parse_loss_logs(logs: str) -> dict:
    metrics = {}
    patterns = {
        "train/loss_pred": r"inv:\s*([0-9eE+\-.]+)",
        "train/loss_variance": r"var:\s*([0-9eE+\-.]+)",
        "train/loss_covariance": r"cov:\s*([0-9eE+\-.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, logs)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


@torch.no_grad()
def _compute_repr_stats(encoder: torch.nn.Module, batch, device: torch.device) -> dict:
    if torch.is_tensor(batch):
        x = batch.to(device)
    else:
        x = batch[0].to(device)
    z = encoder.represent(x).detach()
    if z.ndim != 2 or z.shape[0] < 2:
        return {}
    std_per_dim = z.std(dim=0, unbiased=False)
    dead_pct = (std_per_dim < 1e-3).float().mean() * 100.0
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    offdiag = cov - torch.diag(torch.diag(cov))
    s = torch.linalg.svdvals(zc)
    s_sum = float(s.sum().item()) + 1e-12
    p = s / s_sum
    effective_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())
    return {
        "repr/dead_dimensions_pct": float(dead_pct.item()),
        "repr/effective_rank": effective_rank,
        "repr/mean_std": float(std_per_dim.mean().item()),
        "repr/min_std": float(std_per_dim.min().item()),
        "repr/max_std": float(std_per_dim.max().item()),
        "repr/covariance_offdiag_mean": float(offdiag.abs().mean().item()),
        "repr/embedding_norm": float(z.norm(dim=1).mean().item()),
        "repr/embedding_variance": float(z.var(unbiased=False).item()),
    }

class EEGEncoder1D(nn.Module):
    def __init__(self, in_channels, base_filters, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Conv1d(base_filters, base_filters*2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters*2),
            nn.GELU(),
            nn.Conv1d(base_filters*2, base_filters*4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters*4),
            nn.GELU(),
            nn.Conv1d(base_filters*4, out_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def represent(self, x):
        features = self.net(x)
        return self.pool(features).squeeze(-1)

    def forward(self, x):
        return self.represent(x)

class VICRegSSL(nn.Module):
    def __init__(self, encoder, cfg):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(cfg.projector_spec)
        self.loss_fn = VICRegLoss(std_coeff=cfg.std_coeff, cov_coeff=cfg.cov_coeff)
        
    def compute_loss(self, batch):
        v1, v2 = batch
        r1 = self.encoder.represent(v1)
        r2 = self.encoder.represent(v2)
        z1 = self.projector(r1)
        z2 = self.projector(r2)
        loss_dict = self.loss_fn(z1, z2)
        loss = loss_dict["loss"]
        logs = f"inv: {loss_dict['invariance_loss']:.3f} var: {loss_dict['var_loss']:.3f} cov: {loss_dict['cov_loss']:.3f}"
        return loss, logs

from examples.eeg.architectures import EEGVideoJEPAEncoder, VideoJEPASSL, EEGImageJEPAEncoder, ImageJEPASSL

# --------------------------------------------------------------------------- #
# 1) ENCODER
# --------------------------------------------------------------------------- #
def build_encoder(cfg):
    arch = cfg.get("architecture", "vicreg")
    if arch == "video_jepa":
        return EEGVideoJEPAEncoder(in_channels=cfg.in_channels, base_filters=cfg.base_filters, out_dim=cfg.out_dim)
    elif arch == "image_jepa":
        return EEGImageJEPAEncoder(embed_dim=cfg.out_dim)
    else:
        return EEGEncoder1D(
            in_channels=cfg.in_channels, 
            base_filters=cfg.base_filters, 
            out_dim=cfg.out_dim
        )

# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE
# --------------------------------------------------------------------------- #
def build_ssl(encoder, cfg):
    arch = cfg.get("architecture", "vicreg")
    if arch == "video_jepa":
        return VideoJEPASSL(encoder, cfg)
    elif arch == "image_jepa":
        return ImageJEPASSL(encoder, cfg)
    else:
        return VICRegSSL(encoder, cfg)


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/eeg/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))

    wb_run = None
    wb_cfg = cfg.get("wandb", None)
    wb_enabled = bool(wb_cfg and wb_cfg.get("enabled", False))
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    if wb_enabled:
        if wandb is None:
            raise ImportError("wandb is enabled in config but package is not installed")
        wb_run = wandb.init(
            project=wb_cfg.get("project", "eb-jepa-eeg"),
            entity=wb_cfg.get("entity", None),
            name=wb_cfg.get("run_name", None),
            group=wb_cfg.get("group", None),
            job_type=wb_cfg.get("job_type", "train"),
            tags=list(wb_cfg.get("tags", [])),
            config=run_cfg,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = EEGConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    num_params = sum(p.numel() for p in ssl.parameters())

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)
    git_commit = _get_git_commit()
    log_interval = int(wb_cfg.get("log_interval", 50)) if wb_cfg else 50
    save_artifact = bool(wb_cfg.get("save_artifact", False)) if wb_cfg else False
    artifact_name = wb_cfg.get("artifact_name", "model-temporal-jepa-tuab") if wb_cfg else "model-temporal-jepa-tuab"
    best_mode = wb_cfg.get("best_mode", "min") if wb_cfg else "min"
    best_metric_name = wb_cfg.get("best_metric", "train/epoch_loss") if wb_cfg else "train/epoch_loss"
    global_step = 0
    best_metric = float("inf") if best_mode == "min" else float("-inf")

    if wb_run is not None:
        wandb.log(
            {
                "system/num_parameters": int(num_params),
                "system/device": str(device),
                "system/python_version": sys.version.split()[0],
                "system/torch_version": torch.__version__,
                "system/cuda_available": bool(torch.cuda.is_available()),
                "system/git_commit": git_commit,
                "data/data_root": str(dcfg.data_root),
                "data/split": str(dcfg.split),
                "data/n_channels": int(dcfg.n_channels),
                "data/sfreq": int(dcfg.sfreq),
                "data/window_sec": float(dcfg.window_sec),
            }
        )

    try:
        for epoch in range(cfg.optim.epochs):
            ssl.train()
            epoch_loss_sum = 0.0
            epoch_steps = 0
            last_batch = None
            for batch in loader:
                last_batch = batch
                batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
                opt.zero_grad(set_to_none=True)
                loss, logs = ssl.compute_loss(batch)
                loss.backward(); opt.step()
                loss_value = float(loss.item())
                epoch_loss_sum += loss_value
                epoch_steps += 1
                if wb_run is not None and (global_step % max(log_interval, 1) == 0):
                    log_payload = {
                        "train/loss_total": loss_value,
                        "train/lr": float(opt.param_groups[0]["lr"]),
                        "train/epoch": int(epoch),
                        "train/global_step": int(global_step),
                    }
                    log_payload.update(_parse_loss_logs(logs))
                    wandb.log(log_payload, step=global_step)
                global_step += 1
            epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
            print(f"[eeg] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)

            ckpt_payload = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "model_state_dict": ssl.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": None,
                "cfg": run_cfg,
                "best_metric": float(best_metric),
                "wandb_run_id": wb_run.id if wb_run is not None else None,
                "git_commit": git_commit,
            }
            latest_path = os.path.join(ckpt_dir, "latest.pth.tar")
            best_path = os.path.join(ckpt_dir, "best.pth.tar")
            torch.save(ckpt_payload, latest_path)

            is_best = (epoch_loss <= best_metric) if best_mode == "min" else (epoch_loss >= best_metric)
            if is_best:
                best_metric = epoch_loss
                ckpt_payload["best_metric"] = float(best_metric)
                torch.save(ckpt_payload, best_path)

            if wb_run is not None:
                epoch_payload = {
                    "train/epoch_loss": float(epoch_loss),
                    "train/epoch": int(epoch),
                    "train/global_step": int(global_step),
                    best_metric_name: float(epoch_loss),
                    "train/best_metric": float(best_metric),
                    "train/is_best_checkpoint": int(is_best),
                }
                if last_batch is not None:
                    epoch_payload.update(_compute_repr_stats(encoder, last_batch, device))
                wandb.log(epoch_payload, step=global_step)
        print(f"[eeg] done -> {ckpt_dir}/latest.pth.tar")

        if wb_run is not None and save_artifact:
            best_path = Path(ckpt_dir) / "best.pth.tar"
            if best_path.exists():
                art = wandb.Artifact(
                    artifact_name,
                    type="model",
                    metadata={
                        "model_name": cfg.get("experiment", {}).get("model_name", cfg.model.get("architecture", "unknown")),
                        "dataset_name": cfg.get("experiment", {}).get("dataset_name", "TUAB"),
                        "split": cfg.get("experiment", {}).get("split", "train"),
                        "best_metric": float(best_metric),
                        "best_metric_name": best_metric_name,
                        "wandb_run_id": wb_run.id,
                        "git_commit": git_commit,
                    },
                )
                art.add_file(str(best_path), name="best.pth.tar")
                art.add_file(config_path, name="config.yaml")
                wb_run.log_artifact(art, aliases=["best", "latest"])
    finally:
        if wb_run is not None:
            wb_run.finish()


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/train.yaml"
    run(fname=fname)
