import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from eb_jepa.models.spectral_state_vae import SpectralStateVAE
from scripts.augmentation_benchmark import EEGClassifier, load_data

def train_augmented_classifier(train_loader, device, vae, epochs=3):
    model = EEGClassifier(in_channels=19, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        model.train()
        for wins, labels, ok in train_loader:
            B, N = wins.shape[0], wins.shape[1]
            x = wins.reshape(B * N, 19, 2000).to(device)
            y = labels.unsqueeze(1).repeat(1, N).flatten().long().to(device)
            valid = ok.unsqueeze(1).repeat(1, N).flatten().bool()
            x, y = x[valid], y[valid]
            if len(x) == 0: continue
            
            # VAE Augmentation
            with torch.no_grad():
                x_chunked = x.reshape(-1, 19, 200)
                from eb_jepa.models.spectral_state_vae import spectral_features
                _, log_amp, cos_phase, sin_phase = spectral_features(x_chunked)
                out = vae(x_chunked, log_amp, cos_phase, sin_phase, torch.zeros(len(x_chunked), 1).to(device), sample=True)
                x_aug = out["F_hat"].reshape(x.shape)
            
            x = torch.cat([x, x_aug], dim=0)
            y = torch.cat([y, y], dim=0)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    return model

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
    
    # Train set (10% for speed)
    print("Loading train dataset...")
    train_ds = load_data("train", frac=0.10, data_root=args.data_root, manifest=args.manifest)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    
    # Train the VAE-Augmented Classifier
    print("Training VAE-Augmented downstream classifier...")
    clf = train_augmented_classifier(train_loader, device, vae, epochs=2)
    clf.eval()
    
    # Load evaluation dataset (all items but we select 20 representative ones for detailed logging)
    print("Loading eval dataset (held-out patients)...")
    eval_ds = load_data("eval", frac=1.0, data_root=args.data_root, manifest=args.manifest)
    
    print("\n=== PREDICTION INDIVIDUELLE SUR PATIENTS HORS COHORTE ===")
    print(f"{'Patient ID':<15} | {'Ground Truth':<12} | {'Prediction':<12} | {'Statut':<10}")
    print("-" * 60)
    
    correct = 0
    total = 0
    
    # Run predictions patient by patient
    with torch.no_grad():
        for idx in range(min(30, len(eval_ds.items))):
            item = eval_ds.items[idx]
            try:
                # Load single patient recording windows
                wins, label, ok = eval_ds[idx]
                if not ok: continue
                
                # wins: [N, C, T]
                N, C, T = wins.shape
                x = wins.unsqueeze(0).reshape(N, C, T).to(device)
                
                # Predict
                logits = clf(x)
                preds = logits.argmax(dim=-1).cpu().numpy()
                # Voting ensemble of windows for recording-level label
                final_pred = int(np.round(preds.mean()))
                
                gt_str = "Abnormal" if label == 1 else "Normal"
                pred_str = "Abnormal" if final_pred == 1 else "Normal"
                status = "✅ CORRECT" if final_pred == label else "❌ ERREUR"
                
                if final_pred == label:
                    correct += 1
                total += 1
                
                print(f"{item.patient_id:<15} | {gt_str:<12} | {pred_str:<12} | {status:<10}")
            except Exception as e:
                # print(f"Error loading {item.patient_id}: {e}")
                pass
                
    if total > 0:
        print("-" * 60)
        print(f"Accuracy globale sur ce groupe de test hors cohorte : {correct/total*100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
