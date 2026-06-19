"""Generate synthetic EEG pairs (F_t, F_hat_next) using the VAE model.

Usage:
    python scripts/generate_synthetic_eeg.py --output results/synthetic_eeg_pairs.pt --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.models.spectral_state_vae import SpectralStateVAE, spectral_features


def make_synthetic_inputs(batch_size: int, channels: int, samples: int, seed: int):
    """Create continuous multichannel oscillations with noise."""
    generator = torch.Generator().manual_seed(seed)
    time_axis = torch.arange(samples, dtype=torch.float32) / samples
    signals = []
    for _ in range(batch_size):
        channel_signals = []
        base_frequency = 8.0 + 4.0 * torch.rand((), generator=generator)
        for channel in range(channels):
            phase = 2.0 * torch.pi * torch.rand((), generator=generator)
            amplitude = 0.7 + 0.6 * torch.rand((), generator=generator)
            alpha = amplitude * torch.sin(2.0 * torch.pi * base_frequency * time_axis + phase)
            slow = 0.25 * torch.sin(
                2.0 * torch.pi * (2.0 + 0.03 * channel) * time_axis + 0.5 * phase
            )
            noise = 0.02 * torch.randn(samples, generator=generator)
            channel_signals.append(alpha + slow + noise)
        signals.append(torch.stack(channel_signals))
    signals = torch.stack(signals)
    # Z-score normalize
    signals = (signals - signals.mean(dim=-1, keepdim=True)) / (
        signals.std(dim=-1, keepdim=True) + 1e-6
    )
    return signals


def generate_pairs(args):
    print(f"Generating synthetic EEG using seed={args.seed}...")
    torch.manual_seed(args.seed)

    # 1. Initialize VAE
    model = SpectralStateVAE(
        channels=args.channels,
        time_samples=args.samples,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        print("Using randomly initialized VAE weights.")

    model.eval()

    # 2. Get inputs
    if args.source == "synthetic":
        F_t = make_synthetic_inputs(args.batch_size, args.channels, args.samples, args.seed)
    elif args.source == "pt":
        if not args.input_file:
            raise ValueError("--input-file is required when --source is pt")
        print(f"Loading input signals from {args.input_file}...")
        F_t = torch.load(args.input_file, map_location="cpu")
        if F_t.ndim == 2:  # Assume [C, T], add batch dim
            F_t = F_t.unsqueeze(0)
        # Ensure dimensions match
        if F_t.shape[1] != args.channels or F_t.shape[2] != args.samples:
            raise ValueError(f"Input shape {F_t.shape} does not match channels={args.channels}, samples={args.samples}")
    else:
        raise ValueError(f"Unknown source: {args.source}")

    # 3. Compute spectral features for VAE inputs
    with torch.no_grad():
        _, log_amp, cos_phase, sin_phase = spectral_features(F_t)
        tau = torch.zeros(F_t.shape[0], 1, dtype=torch.float32)
        
        # 4. Run model forward to decode next step prediction
        output = model(F_t, log_amp, cos_phase, sin_phase, tau, sample=False)
        F_hat_next = output["F_hat"]

    # 5. Save generated pairs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as a dict containing inputs and predictions
    data_to_save = {
        "F_t": F_t,
        "F_hat_next": F_hat_next,
        "log_amp_hat_next": output["log_amp_hat"],
        "cos_hat_next": output["cos_hat"],
        "sin_hat_next": output["sin_hat"],
    }
    torch.save(data_to_save, output_path)
    print(f"Successfully generated and saved {F_t.shape[0]} pairs to {output_path}")
    print(f"F_t shape: {F_t.shape}, F_hat_next shape: {F_hat_next.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="results/synthetic_eeg_pairs.pt", help="Path to output .pt file")
    parser.add_argument("--source", choices=("synthetic", "pt"), default="synthetic", help="Source of input signals")
    parser.add_argument("--input-file", type=str, help="Path to input .pt file of signals (required if source=pt)")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of pairs to generate")
    parser.add_argument("--channels", type=int, default=19, help="Number of channels")
    parser.add_argument("--samples", type=int, default=200, help="Number of time samples")
    parser.add_argument("--latent-dim", type=int, default=32, help="VAE latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="VAE hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    generate_pairs(parser.parse_args())
