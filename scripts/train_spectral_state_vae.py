"""Training script for SpectralStateVAE on Dalia (GPU)."""

import argparse
import time
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.models.spectral_state_vae import (
    SpectralStateVAE,
    spectral_features,
    spectral_vae_loss,
    training_physics_weight,
)

class SimpleEDFDataset(Dataset):
    def __init__(self, manifest_path: str, channels: int, samples: int, max_files: int = -1):
        try:
            import pyedflib
            self.pyedflib = pyedflib
        except ImportError as exc:
            raise ImportError("EDF mode requires pyedflib") from exc
            
        self.channels = channels
        self.samples = samples
        
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            # Try to load train data, fallback to anything if empty
            rows = list(csv.DictReader(handle))
            train_rows = [r for r in rows if r["split"] == "train"]
            if train_rows:
                self.rows = train_rows
            else:
                self.rows = rows
                
        if max_files > 0:
            self.rows = self.rows[:max_files]
            
        print(f"Dataset initialized with {len(self.rows)} EDF files.")

    def __len__(self):
        # We can simulate a larger dataset by returning a large number
        # Each getitem will just pick a random file and a random 2-second segment
        return len(self.rows) * 100 

    def __getitem__(self, idx):
        # Pick a random file
        row = random.choice(self.rows)
        source_sfreq = float(row["source_sfreq"])
        indices = [int(value) for value in row["channel_indices"].split(";")]
        duration = float(row["duration_sec"])
        
        # Random start time for a 2-second window
        start_sec = random.uniform(0.0, max(0.0, duration - 2.0))
        start = int(start_sec * source_sfreq)
        
        reader = self.pyedflib.EdfReader(row["path"])
        try:
            segment = torch.stack(
                [
                    torch.from_numpy(reader.readSignal(index, start, 2 * self.samples)).float()
                    for index in indices
                ]
            )
        finally:
            try:
                reader.close()
            except AttributeError:
                reader._close()
                
        # Z-score normalization
        segment = (segment - segment.mean(dim=-1, keepdim=True)) / (
            segment.std(dim=-1, keepdim=True) + 1e-6
        )
        
        # Return F_t and F_{t+1} (1 second each)
        return segment[:, :self.samples], segment[:, self.samples:]


def run_training(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")

    # 1. Dataset
    if args.source == "synthetic":
        raise NotImplementedError("Only EDF source supported for scale-up training script.")
    else:
        if not args.manifest:
            raise ValueError("--manifest is required when source='edf'")
        dataset = SimpleEDFDataset(
            manifest_path=args.manifest,
            channels=args.channels,
            samples=args.samples,
            max_files=args.max_edf_files,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

    # 2. Model
    model = SpectralStateVAE(
        channels=args.channels,
        time_samples=args.samples,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {args.steps} steps...", flush=True)

    # 3. Training Loop
    started = time.perf_counter()
    model.train()
    
    losses = []
    step = 0
    iterator = iter(loader)
    
    current_epoch = -1
    phys_w = 0.0
    
    while step < args.steps:
        epoch = step // args.steps_per_epoch
        if epoch != current_epoch:
            phys_w = training_physics_weight(
                epoch,
                warmup_epochs=args.warmup_epochs,
                ramp_epochs=args.ramp_epochs,
                min_scale=args.physics_min_scale,
                max_scale=args.physics_max_scale,
            )
            current_epoch = epoch

        try:
            current, future = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            current, future = next(iterator)

        current, future = current.to(device), future.to(device)

        # Features
        spectrum, log_amp, cos_phase, sin_phase = spectral_features(current)
        _, target_log_amp, target_cos, target_sin = spectral_features(future)
        tau = torch.zeros(current.shape[0], 1, device=device)

        # Forward
        optimizer.zero_grad(set_to_none=True)
        output = model(current, log_amp, cos_phase, sin_phase, tau)
        
        # Loss
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
            physics_weight=phys_w,
            beta=args.beta,
        )
        
        loss_value = total.detach().item()
        
        # Backward
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        losses.append(loss_value)
        
        # Logging
        if step % 10 == 0 or step == args.steps - 1:
            part_values = {name: value.item() for name, value in parts.items()}
            print(
                f"ep={epoch} step={step:05d} phys_w={phys_w:.3f} loss={part_values['loss']:.4f} "
                f"time={part_values['time']:.4f} amp={part_values['amp']:.4f} "
                f"phase={part_values['phase']:.4f} bp={part_values['bandpower']:.4f} "
                f"kl={part_values['kl']:.4f}",
                flush=True
            )
            
        # Checkpointing
        if args.save_interval > 0 and step > 0 and step % args.save_interval == 0:
            if args.save_checkpoint:
                ckpt_path = f"{args.save_checkpoint}_step{step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")

        step += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
        
    elapsed = time.perf_counter() - started
    print(f"Training completed. Elapsed: {elapsed:.2f}s")

    if args.save_checkpoint:
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"Final checkpoint saved to {args.save_checkpoint}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "edf"), default="synthetic")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--manifest")
    parser.add_argument("--max-edf-files", type=int, default=-1, help="Number of files to use (-1 for all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=19)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--steps-per-epoch", type=int, default=100, help="Steps per epoch for schedule")
    parser.add_argument("--warmup-epochs", type=int, default=100, help="Epochs before physical ramp")
    parser.add_argument("--ramp-epochs", type=int, default=50, help="Epochs for physical ramp")
    parser.add_argument("--physics-min-scale", type=float, default=0.5)
    parser.add_argument("--physics-max-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-time", type=float, default=1.0)
    parser.add_argument("--lambda-amp", type=float, default=1.0)
    parser.add_argument("--lambda-phase", type=float, default=0.1)
    parser.add_argument("--lambda-bandpower", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1e-5)
    parser.add_argument("--disable-phase", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-checkpoint", type=str, default="results/vae_checkpoint.pt")
    parser.add_argument("--save-interval", type=int, default=1000, help="Steps between checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
