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

def load_few_shot_data(data_root, manifest_path, num_shots=15, split_type="train", seed=42):
    items_train = _load_manifest(manifest_path, "train")
    items_eval = _load_manifest(manifest_path, "eval")
    all_items = items_train + items_eval
    
    patient_to_items = defaultdict(list)
    for item in all_items:
        patient_to_items[item.patient_id].append(item)
        
    unique_patients = sorted(list(patient_to_items.keys()))
    
    rng = np.random.default_rng(seed)
    # The evaluation set patient split should be consistent across all seeds.
    # So we use a fixed seed (e.g. 42) for splitting the patients 70/30
    rng_split = np.random.default_rng(42)
    rng_split.shuffle(unique_patients)
    split_idx = int(len(unique_patients) * 0.7)
    
    if split_type == "train":
        train_patients = set(unique_patients[:split_idx])
        selected_items = []
        for p in train_patients:
            selected_items.extend(patient_to_items[p])
            
        rng.shuffle(selected_items)
        
        normal_items = [item for item in selected_items if item.label == 0]
        abnormal_items = [item for item in selected_items if item.label == 1]
        
        n_normal = num_shots // 2
        n_abnormal = num_shots - n_normal
        
        few_shot_items = normal_items[:n_normal] + abnormal_items[:n_abnormal]
        rng.shuffle(few_shot_items)
        selected_items = few_shot_items
        
    else:
        eval_patients = set(unique_patients[split_idx:])
        selected_items = []
        for p in eval_patients:
            selected_items.extend(patient_to_items[p])
        rng.shuffle(selected_items)
        selected_items = selected_items[:500]
        
    cfg = EEGConfig(
        data_root=data_root,
        manifest_path=manifest_path,
        split="train" if split_type == "train" else "eval",
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
    
    labels = [item.label for item in selected_items]
    normals = sum(1 for l in labels if l == 0)
    abnormals = sum(1 for l in labels if l == 1)
    print(f"[{split_type}] loaded {len(selected_items)} windows (Normal: {normals}, Abnormal: {abnormals})")
    
    return ds

def run_seed(seed, train_loader, eval_loader, device, vae):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    base_acc, base_rec = train_classifier(train_loader, eval_loader, device, augment_vae=None, epochs=5)
    aug_acc, aug_rec = train_classifier(train_loader, eval_loader, device, augment_vae=vae, epochs=5)
    
    return base_acc, base_rec, aug_acc, aug_rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--num-shots", type=int, default=15)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running FEW-SHOT seeds benchmark ({args.num_shots} shots) on {device}")
    
    vae = SpectralStateVAE(channels=19, time_samples=200, latent_dim=64, hidden_dim=128).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    
    eval_ds = load_few_shot_data(args.data_root, args.manifest, split_type="eval", seed=42)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=32, shuffle=False, num_workers=0)
    
    seeds = [42, 100, 2026, 7, 88]
    
    base_accs, base_recs = [], []
    aug_accs, aug_recs = [], []
    
    for seed in seeds:
        print(f"\nRunning seed {seed}...", flush=True)
        train_ds = load_few_shot_data(args.data_root, args.manifest, num_shots=args.num_shots, split_type="train", seed=seed)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=15, shuffle=True, num_workers=0)
        
        base_acc, base_rec, aug_acc, aug_rec = run_seed(seed, train_loader, eval_loader, device, vae)
        print(f"  Seed {seed} -> Base Acc: {base_acc:.4f}, Rec: {base_rec:.4f} | Aug Acc: {aug_acc:.4f}, Rec: {aug_rec:.4f}")
        
        base_accs.append(base_acc)
        base_recs.append(base_rec)
        aug_accs.append(aug_acc)
        aug_recs.append(aug_rec)
        
    print("\n=== FEW-SHOT BENCHMARK SUMMARY ===", flush=True)
    print(f"Baseline Accuracy:  {np.mean(base_accs):.4f} +/- {np.var(base_accs):.6f}")
    print(f"Baseline Recall:    {np.mean(base_recs):.4f} +/- {np.var(base_recs):.6f}")
    print(f"Augmented Accuracy: {np.mean(aug_accs):.4f} +/- {np.var(aug_accs):.6f}")
    print(f"Augmented Recall:   {np.mean(aug_recs):.4f} +/- {np.var(aug_recs):.6f}")

if __name__ == "__main__":
    main()
