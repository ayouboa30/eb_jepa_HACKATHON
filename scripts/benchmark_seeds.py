import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset, _load_manifest
from eb_jepa.models.spectral_state_vae import SpectralStateVAE
from scripts.augmentation_benchmark import EEGClassifier, train_classifier

def load_data_70_30(data_root, manifest_path, split_type="train", frac=1.0, seed=42):
    # Load all items from train and eval to pool them
    items_train = _load_manifest(manifest_path, "train")
    items_eval = _load_manifest(manifest_path, "eval")
    all_items = items_train + items_eval
    
    # Group by patient_id to prevent leak
    patient_to_items = defaultdict(list)
    for item in all_items:
        patient_to_items[item.patient_id].append(item)
        
    unique_patients = sorted(list(patient_to_items.keys()))
    
    # Shuffle patients deterministically
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_patients)
    
    # Strict 70% Train, 30% Eval split on PATIENTS
    split_idx = int(len(unique_patients) * 0.7)
    train_patients = set(unique_patients[:split_idx])
    eval_patients = set(unique_patients[split_idx:])
    
    selected_patients = train_patients if split_type == "train" else eval_patients
    selected_items = []
    for p in selected_patients:
        selected_items.extend(patient_to_items[p])
        
    # Shuffle items within the split
    rng.shuffle(selected_items)
    
    # Apply fraction subsetting if requested
    if frac < 1.0:
        n_items = max(1, int(len(selected_items) * frac))
        selected_items = selected_items[:n_items]
        
    cfg = EEGConfig(
        data_root=data_root,
        manifest_path=manifest_path,
        split="train" if split_type == "train" else "eval",
        mode="probe",
        n_channels=19,
        sfreq=200,
        window_sec=10.0,
        n_windows=2,
        epoch_size=int(2000 * frac) if split_type == "train" else 500,
        apply_signal_preprocessing=False
    )
    ds = EEGDataset(cfg)
    ds.items = selected_items
    
    # Report class distribution
    labels = [item.label for item in selected_items]
    normals = sum(1 for l in labels if l == 0)
    abnormals = sum(1 for l in labels if l == 1)
    print(f"[{split_type}] loaded {len(selected_items)} windows from {len(selected_patients)} patients (Normal: {normals}, Abnormal: {abnormals})")
    
    return ds

def run_seed(seed, train_loader, eval_loader, device, vae):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    base_acc, base_rec = train_classifier(train_loader, eval_loader, device, augment_vae=None, epochs=3)
    aug_acc, aug_rec = train_classifier(train_loader, eval_loader, device, augment_vae=vae, epochs=3)
    
    return base_acc, base_rec, aug_acc, aug_rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running seeds benchmark on {device} with 70/30 split (fraction={args.fraction*100}%)")
    
    # Load VAE
    vae = SpectralStateVAE(channels=19, time_samples=200, latent_dim=64, hidden_dim=128).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    
    # Load 70/30 Split datasets
    train_ds = load_data_70_30(args.data_root, args.manifest, "train", frac=args.fraction, seed=42)
    eval_ds = load_data_70_30(args.data_root, args.manifest, "eval", frac=1.0, seed=42)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=32, shuffle=False, num_workers=0)
    
    seeds = [42, 100, 2026]
    
    base_accs, base_recs = [], []
    aug_accs, aug_recs = [], []
    
    for seed in seeds:
        print(f"\nRunning seed {seed}...", flush=True)
        base_acc, base_rec, aug_acc, aug_rec = run_seed(seed, train_loader, eval_loader, device, vae)
        print(f"  Seed {seed} -> Base Acc: {base_acc:.4f}, Rec: {base_rec:.4f} | Aug Acc: {aug_acc:.4f}, Rec: {aug_rec:.4f}")
        base_accs.append(base_acc)
        base_recs.append(base_rec)
        aug_accs.append(aug_acc)
        aug_recs.append(aug_rec)
        
    print("\n=== BENCHMARK 70/30 SEEDS SUMMARY ===", flush=True)
    print(f"Baseline Accuracy:  {np.mean(base_accs):.4f} +/- {np.var(base_accs):.6f}")
    print(f"Baseline Recall:    {np.mean(base_recs):.4f} +/- {np.var(base_recs):.6f}")
    print(f"Augmented Accuracy: {np.mean(aug_accs):.4f} +/- {np.var(aug_accs):.6f}")
    print(f"Augmented Recall:   {np.mean(aug_recs):.4f} +/- {np.var(aug_recs):.6f}")

if __name__ == "__main__":
    main()
