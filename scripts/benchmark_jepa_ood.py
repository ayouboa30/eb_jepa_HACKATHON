import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset, _load_manifest
from eb_jepa.models.spectral_state_vae import SpectralStateVAE
from examples.eeg.main import build_encoder

def extract_features(loader, encoder, device, vae=None):
    encoder.eval()
    if vae:
        vae.eval()
        
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]
            x = x.to(device)
            
            if len(x.shape) == 4:
                b, nw, c, t = x.shape
                x = x.view(b * nw, c, t)
            else:
                b, nw = x.shape[0], 1
                
            if vae is not None:
                x_chunked = x.reshape(-1, 19, 200)
                from eb_jepa.models.spectral_state_vae import spectral_features
                _, log_amp, cos_phase, sin_phase = spectral_features(x_chunked)
                out = vae(x_chunked, log_amp, cos_phase, sin_phase, torch.zeros(len(x_chunked), 1).to(device), sample=True)
                x = out["F_hat"].reshape(x.shape)
                
            features = encoder.represent(x)
            
            if nw > 1:
                features = features.view(b, nw, -1).mean(dim=1)
                
            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())
            
    if len(all_features) == 0:
        return np.array([]), np.array([])
        
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

def load_fraction_data(data_root, manifest_path, fraction=1.0, split_type="train", seed=42):
    items = _load_manifest(manifest_path, split_type)
    patient_to_items = defaultdict(list)
    for item in items:
        patient_to_items[item.patient_id].append(item)
        
    unique_patients = sorted(list(patient_to_items.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_patients)
    
    n_patients = max(1, int(len(unique_patients) * fraction))
    selected_patients = set(unique_patients[:n_patients])
    
    selected_items = [item for item in items if item.patient_id in selected_patients]
    
    cfg = EEGConfig(
        data_root=data_root,
        manifest_path=manifest_path,
        split=split_type,
        mode="probe",
        n_channels=19,
        sfreq=200,
        window_sec=10.0,
        n_windows=2,
        epoch_size=len(selected_items),
        apply_signal_preprocessing=False
    )
    ds = EEGDataset(cfg)
    ds.items = selected_items
    return ds

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

class MLPDecider(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp_decider(X_train, y_train, X_eval, device, epochs=50):
    if len(X_train) == 0:
        return np.zeros(len(X_eval))
        
    in_dim = X_train.shape[1]
    model = MLPDecider(in_dim=in_dim, hidden_dim=128, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        eval_t = torch.from_numpy(X_eval).float().to(device)
        logits = model(eval_t)
        preds = logits.argmax(dim=-1).cpu().numpy()
        
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--jepa-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    cfg = OmegaConf.create(ckpt["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    
    vae = SpectralStateVAE(channels=19, time_samples=200, latent_dim=256, hidden_dim=512).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    
    print("Extracting evaluation features (OOD cohort)...")
    eval_ds = load_fraction_data(args.data_root, args.manifest, fraction=1.0, split_type="eval")
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0)
    X_eval, y_eval = extract_features(eval_loader, encoder, device)
    print(f"Eval features shape: {X_eval.shape}")
    
    fractions = [0.01, 0.05, 0.25, 1.0]
    
    print("\nStarting Benchmark OOD Inter-Cohorte (Few-Label)")
    print("="*60)
    
    for frac in fractions:
        print(f"\n--- Fraction: {frac*100}% of training patients ---")
        train_ds = load_fraction_data(args.data_root, args.manifest, fraction=frac, split_type="train")
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        
        print(f"Loaded {len(train_ds.items)} windows from training cohort.")
        
        X_train_clean, y_train = extract_features(train_loader, encoder, device, vae=None)
        preds_base = train_mlp_decider(X_train_clean, y_train, X_eval, device, epochs=100)
        acc_base = accuracy_score(y_eval, preds_base)
        rec_base = recall_score(y_eval, preds_base, average='macro')
        
        X_train_aug, y_train_aug = extract_features(train_loader, encoder, device, vae=vae)
        X_train_combined = np.concatenate([X_train_clean, X_train_aug], axis=0)
        y_train_combined = np.concatenate([y_train, y_train_aug], axis=0)
        
        preds_aug = train_mlp_decider(X_train_combined, y_train_combined, X_eval, device, epochs=100)
        acc_aug = accuracy_score(y_eval, preds_aug)
        rec_aug = recall_score(y_eval, preds_aug, average='macro')
        
        print(f"Method: JEPA sans aug      -> Acc: {acc_base:.4f}, Recall: {rec_base:.4f}")
        print(f"Method: JEPA + SpectralVAE -> Acc: {acc_aug:.4f}, Recall: {rec_aug:.4f}")

if __name__ == "__main__":
    main()
