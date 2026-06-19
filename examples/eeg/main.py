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
import sys

import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP), RNNPredictor (GRU)
#   eb_jepa.losses:        VICRegLoss (inv+var+cov), VCLoss (variance+covariance)


import torch.nn as nn
from eb_jepa.architectures import Projector
from eb_jepa.losses import VICRegLoss

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = EEGConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        for batch in loader:
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
        print(f"[eeg] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[eeg] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/train.yaml"
    run(fname=fname)
