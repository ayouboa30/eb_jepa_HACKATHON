"""Compare the Power Spectral Density (PSD) of real EEG signals vs. VAE predictions.

If pyedflib is available (e.g. on Dalia), it can extract segments from raw EDF files.
Otherwise, it can load pre-extracted segments from a .pt file.

Usage:
    python scripts/compare_psd.py --real-source pt --input-file results/real_eeg_samples.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.models.spectral_state_vae import SpectralStateVAE, spectral_features, compute_bandpower

# Import matplotlib safely for headless execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_from_edf_manifest(manifest_path: str, channels: int, samples: int, max_files: int) -> torch.Tensor:
    """Read a few consecutive 2-second segments from audited EDF files."""
    import pyedflib
    print(f"Loading real EEG segments from EDF manifest: {manifest_path}")
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train" or row["split"] == "eval"]
    if not rows:
        raise ValueError(f"No recordings found in {manifest_path}")
    
    count = min(max_files, len(rows))
    rows = rows[:count]
    segments = []
    
    for row in rows:
        indices = [int(value) for value in row["channel_indices"].split(";")]
        if len(indices) != channels:
            continue
        reader = pyedflib.EdfReader(row["path"])
        try:
            # Read first 2 seconds (400 samples at 200 Hz)
            seg = torch.stack([
                torch.from_numpy(reader.readSignal(index, 0, 2 * samples)).float()
                for index in indices
            ])
            # Z-score normalize
            seg = (seg - seg.mean(dim=-1, keepdim=True)) / (seg.std(dim=-1, keepdim=True) + 1e-6)
            segments.append(seg)
        except Exception as e:
            print(f"Error reading {row['path']}: {e}")
        finally:
            try:
                reader.close()
            except AttributeError:
                reader._close()
                
    if not segments:
        raise RuntimeError("No segments successfully read from EDF files")
    return torch.stack(segments)


def compute_psd(signals: torch.Tensor, fs: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD of shape [B, C, F] and return frequency axis and PSD array."""
    # signals: [B, C, T]
    B, C, T = signals.shape
    psd_list = []
    for b in range(B):
        chan_psds = []
        for c in range(C):
            f, pxx = signal.periodogram(signals[b, c].cpu().numpy(), fs=fs)
            chan_psds.append(pxx)
        psd_list.append(np.stack(chan_psds))
    return f, np.stack(psd_list)


def compare_and_plot(args):
    print("Running PSD Comparison...")
    
    # 1. Load Real Signals
    if args.real_source == "edf":
        try:
            import pyedflib
        except ImportError:
            print("ERROR: pyedflib is required for EDF source, but not installed.")
            sys.exit(1)
        if not args.manifest:
            print("ERROR: --manifest is required with --real-source edf")
            sys.exit(1)
        # Load double window segments [B, C, 2*T]
        segments = load_from_edf_manifest(args.manifest, args.channels, args.samples, args.max_edf_files)
        F_t = segments[:, :, :args.samples]
        F_next = segments[:, :, args.samples:]
        if args.input_file:
            input_path = Path(args.input_file)
            input_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving extracted segments to {input_path} for offline use...")
            torch.save({"F_t": F_t, "F_next": F_next}, input_path)
    elif args.real_source == "pt":
        if not args.input_file or not Path(args.input_file).exists():
            # If default results/real_eeg_samples.pt does not exist, let's create a synthetic fallback
            print(f"File {args.input_file} not found. Creating synthetic mock data for local fallback comparison.")
            from scripts.generate_synthetic_eeg import make_synthetic_inputs
            F_t = make_synthetic_inputs(8, args.channels, args.samples, seed=args.seed)
            F_next = make_synthetic_inputs(8, args.channels, args.samples, seed=args.seed + 1)
            # Save it so we can re-use it
            Path(args.input_file).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"F_t": F_t, "F_next": F_next}, args.input_file)
        else:
            print(f"Loading real EEG segments from {args.input_file}")
            data = torch.load(args.input_file, map_location="cpu")
            if isinstance(data, dict):
                F_t = data["F_t"]
                F_next = data["F_next"]
            else:
                F_t = data
                F_next = data  # fallback if it's just a raw tensor
            if F_t.ndim == 2:
                F_t = F_t.unsqueeze(0)
                F_next = F_next.unsqueeze(0)
    else:
        raise ValueError(f"Unknown real-source: {args.real_source}")

    print(f"Loaded signals. F_t shape: {F_t.shape}, F_next shape: {F_next.shape}")

    # 2. Run VAE on F_t to predict F_hat_next
    model = SpectralStateVAE(
        channels=args.channels,
        time_samples=args.samples,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading VAE checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        print("Using randomly initialized model for evaluation (or checkpoint not specified/found).")
    
    model.eval()
    with torch.no_grad():
        _, log_amp, cos_phase, sin_phase = spectral_features(F_t)
        tau = torch.zeros(F_t.shape[0], 1, dtype=torch.float32)
        output = model(F_t, log_amp, cos_phase, sin_phase, tau, sample=False)
        F_hat_next = output["F_hat"]
        
        # Calculate quantitative bandpowers using VAE log amplitude
        bp_hat = compute_bandpower(output["log_amp_hat"])
        _, target_log_amp, _, _ = spectral_features(F_next)
        bp_target = compute_bandpower(target_log_amp)

    # 3. Print quantitative bandpowers
    print("\n--- Quantitative Bandpowers (mean log-amplitude over batch/channels) ---")
    for band in bp_hat:
        target_val = bp_target[band].mean().item()
        hat_val = bp_hat[band].mean().item()
        diff = abs(target_val - hat_val)
        print(f"{band.capitalize():<8} | Target: {target_val:8.4f} | Predicted: {hat_val:8.4f} | Diff: {diff:8.4f}")

    # 4. Compute PSD curves
    f_axis, psd_real = compute_psd(F_next, fs=200.0)
    _, psd_syn = compute_psd(F_hat_next, fs=200.0)
    
    # Average across batches and channels
    mean_psd_real = psd_real.mean(axis=(0, 1))
    mean_psd_syn = psd_syn.mean(axis=(0, 1))
    
    # 5. Plot PSD Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(f_axis, 10 * np.log10(mean_psd_real + 1e-12), label="Real/Target EEG", color="blue", linewidth=2)
    plt.plot(f_axis, 10 * np.log10(mean_psd_syn + 1e-12), label="VAE Generated EEG", color="red", linestyle="--", linewidth=2)
    
    # Add bands shading
    bands = {
        "Delta (1-4 Hz)": (1, 4, "yellow"),
        "Theta (4-8 Hz)": (4, 8, "orange"),
        "Alpha (8-12 Hz)": (8, 12, "green"),
        "Beta (12-30 Hz)": (12, 30, "cyan"),
        "Gamma (30-100 Hz)": (30, 100, "purple"),
    }
    for name, (start, end, color) in bands.items():
        plt.axvspan(start, end, alpha=0.1, color=color, label=name)
        
    plt.title("EEG Power Spectral Density (PSD) Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.xlim(0, 100)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    
    output_path = Path(args.output_plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved PSD comparison plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-source", choices=("edf", "pt"), default="pt", help="Source for real EEG signals")
    parser.add_argument("--manifest", type=str, help="Path to audited EDF manifest")
    parser.add_argument("--input-file", type=str, default="results/real_eeg_samples.pt", help="Path to input .pt file of real signals")
    parser.add_argument("--checkpoint", type=str, help="Path to VAE checkpoint")
    parser.add_argument("--output-plot", type=str, default="results/psd_comparison.png", help="Path to output PSD plot image")
    parser.add_argument("--max-edf-files", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=19)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    
    compare_and_plot(parser.parse_args())
