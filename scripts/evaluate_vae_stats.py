"""Évaluation statistique du VAE sur 10 tests et génération aléatoire."""

import sys
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.models.spectral_state_vae import SpectralStateVAE, spectral_features, compute_bandpower
from scripts.test_spectral_state_vae import load_edf_pairs
from scripts.visualize_eeg_patient import verify_physical_constraints

def evaluate_reconstruction(model, current, future, device):
    """Calcule les statistiques de reconstruction sur un batch."""
    model.eval()
    with torch.no_grad():
        current = current.to(device)
        future = future.to(device)
        spectrum, log_amp, cos_phase, sin_phase = spectral_features(current)
        _, target_log_amp, target_cos, target_sin = spectral_features(future)
        
        output = model(current, log_amp, cos_phase, sin_phase, torch.zeros(current.shape[0], 1).to(device))
        
        F_hat = output["F_hat"]
        log_amp_hat = output["log_amp_hat"]
        
        time_mse = torch.nn.functional.mse_loss(F_hat, future, reduction='none').mean(dim=[1,2])
        amp_mse = torch.nn.functional.mse_loss(log_amp_hat, target_log_amp, reduction='none').mean(dim=[1,2])
        
        bp_hat = compute_bandpower(log_amp_hat)
        bp_target = compute_bandpower(target_log_amp)
        
        bp_errors = {}
        for band in bp_hat:
            bp_errors[band] = torch.abs(bp_hat[band] - bp_target[band]).mean(dim=1) # mean over channels
            
        return time_mse, amp_mse, bp_errors, F_hat

def main():
    device = torch.device("cpu")
    print("Loading model...")
    model = SpectralStateVAE(channels=19, time_samples=200, latent_dim=64, hidden_dim=128).to(device)
    
    checkpoint_path = REPO_ROOT / "results" / "vae_checkpoint_final.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint {checkpoint_path}")
    else:
        print("WARNING: Checkpoint not found! Using random weights.")
        
    print("\n--- 1. Évaluation de la Reconstruction (10 patients) ---")
    manifest = "/lustre/work/vivatech-equipe7/aouladali/eeg_audit/output/edf_inventory.csv"
    try:
        current, future = load_edf_pairs(str(manifest), batch_size=10, channels=19, samples=200, max_files=10, seed=42)
    except Exception as e:
        print(f"Failed to load EDF data: {e}")
        return

    time_mse, amp_mse, bp_errors, F_hat = evaluate_reconstruction(model, current, future, device)
    
    print("\n--- Statistiques MSE ---")
    print(f"Time MSE: {time_mse.mean().item():.4f} +/- {time_mse.var().item():.4f}")
    print(f"Amp MSE:  {amp_mse.mean().item():.4f} +/- {amp_mse.var().item():.4f}")
    
    print("\n--- Statistiques Bandpower (Erreur Absolue) ---")
    for band in bp_errors:
        print(f"{band.capitalize()}: {bp_errors[band].mean().item():.4f} +/- {bp_errors[band].var().item():.4f}")
        
    print("\n--- 2. Génération Aléatoire Pure (z ~ N(0,1)) ---")
    model.eval()
    with torch.no_grad():
        z = torch.randn(5, 64).to(device) # 5 random samples
        tau = torch.zeros(5, 1).to(device)
        output = model.decode(z, tau)
        F_gen = output["F_hat"]
        
    print("Check physics sur 5 signaux générés:")
    for i in range(5):
        stats = verify_physical_constraints(F_gen[i:i+1])
        print(f"Sample {i+1}:")
        print(f"  Stationnarité: {stats['stationarity_ratio']:.3f} (ok: {stats['stationarity_ok']})")
        print(f"  PtP Max:       {stats['ptp_max']:.3f} (ok: {stats['amplitude_ok']})")
        print(f"  NaN/Inf:       {'Non' if (stats['no_nan'] and stats['no_inf']) else 'OUI'}")

if __name__ == "__main__":
    main()
