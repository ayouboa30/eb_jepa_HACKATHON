"""Expérimentation Data Augmentation VAE vs Baseline sur TUAB."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from eb_jepa.models.spectral_state_vae import SpectralStateVAE

# Simple 1D Encoder for classification
class EEGClassifier(nn.Module):
    def __init__(self, in_channels=19, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(-1) # [B, 128]
        return self.fc(x)

def load_data(split, frac=1.0, data_root="", manifest=""):
    cfg = EEGConfig(
        data_root=data_root,
        manifest_path=manifest,
        split=split,
        mode="probe", # window level
        n_channels=19,
        sfreq=200,
        window_sec=10.0,
        n_windows=2, # REDUCED from 16 to 2 for speed
        epoch_size=int(2000 * frac) if split == "train" else 500,
        apply_signal_preprocessing=False
    )
    ds = EEGDataset(cfg)
    if frac < 1.0: # subset BOTH train and eval
        n_items = max(1, int(len(ds.items) * frac))
        ds.items = ds.items[:n_items]
    return ds

def train_classifier(train_loader, eval_loader, device, augment_vae=None, epochs=5):
    model = EEGClassifier(in_channels=19, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        model.train()
        for wins, labels, ok in train_loader:
            # wins: [B, N, C, T]
            B, N = wins.shape[0], wins.shape[1]
            x = wins.reshape(B * N, 19, 2000).to(device)
            y = labels.unsqueeze(1).repeat(1, N).flatten().long().to(device)
            valid = ok.unsqueeze(1).repeat(1, N).flatten().bool()
            x, y = x[valid], y[valid]
            if len(x) == 0: continue
            
            # Data Augmentation (double the batch size)
            if augment_vae is not None:
                with torch.no_grad():
                    # We need to chunk 2000 into 10 x 200 to pass to VAE, then stitch back.
                    # Or simpler: just augment the first 200 samples and copy?
                    # Let's chunk:
                    x_chunked = x.reshape(-1, 19, 200) # [B*10, 19, 200]
                    from eb_jepa.models.spectral_state_vae import spectral_features
                    _, log_amp, cos_phase, sin_phase = spectral_features(x_chunked)
                    out = augment_vae(x_chunked, log_amp, cos_phase, sin_phase, torch.zeros(len(x_chunked), 1).to(device), sample=True)
                    x_aug = out["F_hat"].reshape(x.shape)
                
                # Concat
                x = torch.cat([x, x_aug], dim=0)
                y = torch.cat([y, y], dim=0)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for wins, labels, ok in eval_loader:
            B, N = wins.shape[0], wins.shape[1]
            x = wins.reshape(B * N, 19, 2000).to(device)
            y = labels.unsqueeze(1).repeat(1, N).flatten().long().to(device)
            valid = ok.unsqueeze(1).repeat(1, N).flatten().bool()
            x, y = x[valid], y[valid]
            if len(x) == 0: continue
            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # Load VAE
    vae = SpectralStateVAE(channels=19, time_samples=200, latent_dim=64, hidden_dim=128).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    
    fractions = [0.01, 0.05, 0.10, 0.25]
    results = {}
    
    # Eval Dataset is constant (but subset to 10% for speed)
    eval_ds = load_data("eval", frac=0.10, data_root=args.data_root, manifest=args.manifest)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=32, shuffle=False, num_workers=0)
    
    for frac in fractions:
        print(f"\n=== Training with {frac*100}% of data ===", flush=True)
        train_ds = load_data("train", frac=frac, data_root=args.data_root, manifest=args.manifest)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        
        # Baseline
        print("  -> Baseline (No Augmentation)", flush=True)
        base_acc, base_rec = train_classifier(train_loader, eval_loader, device, augment_vae=None, epochs=2)
        print(f"     Acc: {base_acc:.4f}, Recall: {base_rec:.4f}", flush=True)
        
        # Augmented
        print("  -> Augmented (VAE * 2)", flush=True)
        aug_acc, aug_rec = train_classifier(train_loader, eval_loader, device, augment_vae=vae, epochs=2)
        print(f"     Acc: {aug_acc:.4f}, Recall: {aug_rec:.4f}", flush=True)
        
        results[frac] = {
            "base_acc": base_acc, "base_rec": base_rec,
            "aug_acc": aug_acc, "aug_rec": aug_rec
        }
        
    print("\n=== FINAL RESULTS ===", flush=True)
    print("Frac | Base Acc | Aug Acc | Base Rec | Aug Rec", flush=True)
    for frac in fractions:
        r = results[frac]
        print(f"{frac:4.2f} |  {r['base_acc']:.4f}  | {r['aug_acc']:.4f}  |  {r['base_rec']:.4f}  | {r['aug_rec']:.4f}", flush=True)

if __name__ == "__main__":
    main()
