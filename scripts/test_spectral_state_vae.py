"""CPU smoke test for the spectral EEG VAE, with synthetic or audited TUAB data.

Synthetic (default)::
    python scripts/test_spectral_state_vae.py --steps 20 --batch-size 4

Audited EDF (reads only a few 2-second segments)::
    python scripts/test_spectral_state_vae.py --source edf \
      --manifest results/eeg_audit/edf_inventory.csv --max-edf-files 3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.models.spectral_state_vae import (  # noqa: E402
    SpectralStateVAE,
    spectral_features,
    spectral_vae_loss,
    training_physics_weight,
)


def make_synthetic_pairs(batch_size: int, channels: int, samples: int, seed: int):
    """Create continuous two-second multichannel oscillations with mild noise."""
    generator = torch.Generator().manual_seed(seed)
    total = 2 * samples
    time_axis = torch.arange(total, dtype=torch.float32) / samples
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
            noise = 0.02 * torch.randn(total, generator=generator)
            channel_signals.append(alpha + slow + noise)
        signals.append(torch.stack(channel_signals))
    pair = torch.stack(signals)
    pair = (pair - pair.mean(dim=-1, keepdim=True)) / (
        pair.std(dim=-1, keepdim=True) + 1e-6
    )
    return pair[:, :, :samples], pair[:, :, samples:]


def load_edf_pairs(
    manifest_path: str,
    batch_size: int,
    channels: int,
    samples: int,
    max_files: int,
    seed: int,
):
    """Read one random consecutive 2-second pair from a few audited EDF files."""
    try:
        import pyedflib
    except ImportError as exc:
        raise ImportError("EDF mode requires pyedflib") from exc

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    if not rows:
        raise ValueError(f"No train recordings in {manifest_path}")
    count = min(max(batch_size, 1), max_files, len(rows))
    rows = rows[:count]
    generator = torch.Generator().manual_seed(seed)
    pairs = []
    for row in rows:
        source_sfreq = float(row["source_sfreq"])
        if abs(source_sfreq - samples) > 1e-6:
            raise ValueError(
                f"Smoke test expects 200 Hz audited EDF, got {source_sfreq} for {row['path']}"
            )
        indices = [int(value) for value in row["channel_indices"].split(";")]
        if len(indices) != channels:
            raise ValueError(f"Expected {channels} channels for {row['path']}")
        duration = float(row["duration_sec"])
        start_sec = float(torch.rand((), generator=generator)) * max(0.0, duration - 2.0)
        start = int(start_sec * source_sfreq)
        reader = pyedflib.EdfReader(row["path"])
        try:
            segment = torch.stack(
                [
                    torch.from_numpy(reader.readSignal(index, start, 2 * samples)).float()
                    for index in indices
                ]
            )
        finally:
            try:
                reader.close()
            except AttributeError:
                reader._close()
        if segment.shape != (channels, 2 * samples):
            raise RuntimeError(f"Short EDF read {tuple(segment.shape)} for {row['path']}")
        segment = (segment - segment.mean(dim=-1, keepdim=True)) / (
            segment.std(dim=-1, keepdim=True) + 1e-6
        )
        pairs.append(segment)
    pair = torch.stack(pairs)
    while pair.shape[0] < batch_size:
        pair = torch.cat([pair, pair[: batch_size - pair.shape[0]]], dim=0)
    return pair[:, :, :samples], pair[:, :, samples:]


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite values detected in {name}")


def run_smoke_test(args) -> list[float]:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(args.cpu_threads, 8)))

    if args.source == "synthetic":
        current, future = make_synthetic_pairs(
            args.batch_size, args.channels, args.samples, args.seed
        )
    else:
        if not args.manifest:
            raise ValueError("--manifest is required with --source edf")
        current, future = load_edf_pairs(
            args.manifest,
            args.batch_size,
            args.channels,
            args.samples,
            args.max_edf_files,
            args.seed,
        )
    current, future = current.to(device), future.to(device)

    spectrum, log_amp, cos_phase, sin_phase = spectral_features(current)
    _, target_log_amp, target_cos, target_sin = spectral_features(future)
    roundtrip = torch.fft.irfft(torch.fft.rfft(current, dim=-1), n=args.samples, dim=-1)
    roundtrip_error = (roundtrip - current).abs().max().item()
    if roundtrip_error > 1e-5:
        raise AssertionError(f"FFT round-trip error too large: {roundtrip_error}")
    expected_freq = args.samples // 2 + 1
    if spectrum.shape != (args.batch_size, args.channels, expected_freq):
        raise AssertionError(f"Unexpected rFFT shape: {tuple(spectrum.shape)}")
    for name, value in (
        ("log_amp", log_amp),
        ("cos_phase", cos_phase),
        ("sin_phase", sin_phase),
    ):
        assert_finite(name, value)

    model = SpectralStateVAE(
        channels=args.channels,
        time_samples=args.samples,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    tau = torch.zeros(args.batch_size, 1, dtype=torch.float32, device=device)
    losses = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(current, log_amp, cos_phase, sin_phase, tau)
        physics_weight = training_physics_weight(
            step,
            warmup_epochs=args.warmup_epochs,
            ramp_epochs=args.ramp_epochs,
            min_scale=args.physics_min_scale,
            max_scale=args.physics_max_scale,
        )
        total, parts = spectral_vae_loss(
            output,
            future,
            target_log_amp,
            target_cos,
            target_sin,
            lambda_time=args.lambda_time,
            lambda_amp=args.lambda_amp,
            lambda_phase=0.0 if args.disable_phase else args.lambda_phase,
            lambda_bandpower=args.lambda_bandpower,
            physics_weight=physics_weight,
            beta=args.beta,
        )
        assert output["F_hat"].shape == future.shape
        assert output["log_amp_hat"].shape == target_log_amp.shape
        assert_finite("F_hat_next", output["F_hat"])
        assert_finite("loss", total)
        loss_value = total.detach().item()
        part_values = {name: value.item() for name, value in parts.items()}
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(loss_value)
        print(
            f"step={step:02d} loss={part_values['loss']:.6f} "
            f"time={part_values['time']:.6f} amp={part_values['amp']:.6f} "
            f"phase={part_values['phase']:.6f} bandpower={part_values['bandpower']:.6f} "
            f"kl={part_values['kl']:.6f} phys_w={physics_weight:.3f} "
            f"F_t={tuple(current.shape)} X_t={tuple(spectrum.shape)} "
            f"F_hat_next={tuple(output['F_hat'].shape)}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        f"done source={args.source} device={device} steps={args.steps} elapsed={elapsed:.2f}s "
        f"fft_max_error={roundtrip_error:.3e} initial={losses[0]:.6f} "
        f"final={losses[-1]:.6f} parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    if getattr(args, "save_checkpoint", None):
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"Checkpoint saved to {args.save_checkpoint}")
    if args.steps >= 5 and sum(losses[-3:]) / 3 >= sum(losses[:3]) / 3:
        raise AssertionError("Smoke-test loss did not decrease between the first and last steps")
    return losses


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "edf"), default="synthetic")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--manifest")
    parser.add_argument("--max-edf-files", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--channels", type=int, default=19)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-time", type=float, default=1.0)
    parser.add_argument("--lambda-amp", type=float, default=1.0)
    parser.add_argument("--lambda-phase", type=float, default=0.1)
    parser.add_argument("--lambda-bandpower", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=100)
    parser.add_argument("--ramp-epochs", type=int, default=50)
    parser.add_argument("--physics-min-scale", type=float, default=0.5)
    parser.add_argument("--physics-max-scale", type=float, default=1.0)
    parser.add_argument("--disable-phase", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-checkpoint", type=str, default=None, help="Chemin pour sauvegarder le modèle")
    return parser.parse_args()


if __name__ == "__main__":
    run_smoke_test(parse_args())
