"""Minimal CPU-friendly VAE that predicts the next EEG window in Fourier space."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def spectral_features(signal: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return rFFT, log-amplitude, and circular phase features for ``[B,C,T]``."""
    if signal.ndim != 3:
        raise ValueError(f"Expected signal with shape [B, C, T], got {tuple(signal.shape)}")
    spectrum = torch.fft.rfft(signal, dim=-1)
    amplitude = torch.abs(spectrum)
    log_amplitude = torch.log(amplitude + eps)
    phase = torch.angle(spectrum)
    return spectrum, log_amplitude, torch.cos(phase), torch.sin(phase)


def _as_tau(tau: Tensor | float, batch_size: int, *, device, dtype) -> Tensor:
    tau = torch.as_tensor(tau, device=device, dtype=dtype)
    if tau.ndim == 0:
        tau = tau.expand(batch_size).unsqueeze(1)
    elif tau.ndim == 1:
        tau = tau.unsqueeze(1)
    if tau.shape != (batch_size, 1):
        raise ValueError(f"tau must be scalar, [B], or [B,1], got {tuple(tau.shape)}")
    return tau


class SpectralStateVAE(nn.Module):
    """MLP VAE mapping the current EEG state to the next spectral state."""

    def __init__(
        self,
        channels: int = 19,
        time_samples: int = 200,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        eps: float = 1e-8,
        phase_eps: float = 1e-12,
        min_log_amp: float = -12.0,
        max_log_amp: float = 6.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.time_samples = time_samples
        self.freq_bins = time_samples // 2 + 1
        self.latent_dim = latent_dim
        self.eps = eps
        self.phase_eps = phase_eps
        self.min_log_amp = min_log_amp
        self.max_log_amp = max_log_amp

        temporal_size = channels * time_samples
        spectral_size = channels * self.freq_bins
        encoder_input = temporal_size + 3 * spectral_size + 1
        decoder_output = 3 * spectral_size

        self.encoder = nn.Sequential(
            nn.Linear(encoder_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, decoder_output),
        )

    def encode(
        self,
        signal: Tensor,
        log_amp: Tensor,
        cos_phase: Tensor,
        sin_phase: Tensor,
        tau: Tensor | float,
    ) -> Tuple[Tensor, Tensor]:
        batch = signal.shape[0]
        expected_signal = (batch, self.channels, self.time_samples)
        expected_spectral = (batch, self.channels, self.freq_bins)
        if tuple(signal.shape) != expected_signal:
            raise ValueError(f"Expected F_t {expected_signal}, got {tuple(signal.shape)}")
        for name, value in (
            ("log_amp", log_amp),
            ("cos_phase", cos_phase),
            ("sin_phase", sin_phase),
        ):
            if tuple(value.shape) != expected_spectral:
                raise ValueError(f"Expected {name} {expected_spectral}, got {tuple(value.shape)}")
        tau = _as_tau(tau, batch, device=signal.device, dtype=signal.dtype)
        features = torch.cat(
            [
                signal.flatten(1),
                log_amp.flatten(1),
                cos_phase.flatten(1),
                sin_phase.flatten(1),
                tau,
            ],
            dim=1,
        )
        hidden = self.encoder(features)
        return self.mu_head(hidden), self.logvar_head(hidden).clamp(-12.0, 12.0)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor, sample: bool = True) -> Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: Tensor, tau: Tensor | float) -> Dict[str, Tensor]:
        batch = z.shape[0]
        tau = _as_tau(tau, batch, device=z.device, dtype=z.dtype)
        decoded = self.decoder(torch.cat([z, tau], dim=1))
        decoded = decoded.reshape(batch, 3, self.channels, self.freq_bins)
        raw_log_amp, raw_cos, raw_sin = decoded.unbind(dim=1)
        log_amp = raw_log_amp.clamp(self.min_log_amp, self.max_log_amp)
        phase_norm = torch.sqrt(raw_cos.square() + raw_sin.square() + self.phase_eps)
        cos_phase = raw_cos / phase_norm
        sin_phase = raw_sin / phase_norm
        unit_phase = torch.complex(cos_phase, sin_phase)
        spectrum = torch.exp(log_amp).to(unit_phase.dtype) * unit_phase
        signal = torch.fft.irfft(spectrum, n=self.time_samples, dim=-1)
        return {
            "F_hat": signal,
            "X_hat": spectrum,
            "log_amp_hat": log_amp,
            "cos_hat": cos_phase,
            "sin_hat": sin_phase,
        }

    def forward(
        self,
        signal: Tensor,
        log_amp: Tensor,
        cos_phase: Tensor,
        sin_phase: Tensor,
        tau: Tensor | float,
        *,
        sample: bool | None = None,
    ) -> Dict[str, Tensor]:
        mu, logvar = self.encode(signal, log_amp, cos_phase, sin_phase, tau)
        z = self.reparameterize(mu, logvar, self.training if sample is None else sample)
        output = self.decode(z, tau)
        output.update({"mu": mu, "logvar": logvar, "z": z})
        return output


def compute_bandpower(
    log_amplitude: Tensor,
    *,
    sample_rate: float = 200.0,
    time_samples: int = 200,
) -> Dict[str, Tensor]:
    """Return per-band power estimated from the one-sided PSD.

    The input is ``log|X|`` from ``rfft``. We convert back to power, build a
    one-sided PSD, and integrate in standard EEG bands.
    """
    if log_amplitude.ndim != 3:
        raise ValueError(f"Expected log_amplitude [B,C,F], got {tuple(log_amplitude.shape)}")
    freq_axis = torch.fft.rfftfreq(time_samples, d=1.0 / sample_rate).to(
        device=log_amplitude.device, dtype=log_amplitude.dtype
    )
    power = torch.exp(2.0 * log_amplitude).clamp_min(0.0)
    if power.shape[-1] != freq_axis.shape[0]:
        raise ValueError(
            f"Expected {freq_axis.shape[0]} frequency bins for T={time_samples}, "
            f"got {power.shape[-1]}"
        )
    if time_samples % 2 == 0:
        scale = torch.ones_like(power)
        if power.shape[-1] > 2:
            scale[..., 1:-1] = 2.0
    else:
        scale = torch.ones_like(power)
        if power.shape[-1] > 1:
            scale[..., 1:] = 2.0
    one_sided_bandpower_density = power * scale / float(time_samples) ** 2

    band_edges = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 75.0),
    }
    res: Dict[str, Tensor] = {}
    for name, (low, high) in band_edges.items():
        mask = (freq_axis >= low) & (freq_axis < high)
        if mask.any():
            res[name] = one_sided_bandpower_density[..., mask].sum(dim=-1)
        else:
            res[name] = torch.zeros_like(one_sided_bandpower_density[..., 0])
    return res


def training_physics_weight(
    epoch: int,
    *,
    warmup_epochs: int = 100,
    ramp_epochs: int = 50,
    min_scale: float = 0.5,
    max_scale: float = 1.0,
    generator: Tensor | None = None,
) -> float:
    """Warm up on pure reconstruction, then ramp physical penalties with per-epoch jitter."""
    if epoch < warmup_epochs:
        return 0.0
    ramp_progress = 1.0 if ramp_epochs <= 0 else min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)
    if generator is None:
        u = torch.rand((), dtype=torch.float32).item()
    else:
        u = torch.rand((), generator=generator, dtype=torch.float32).item()
    return (min_scale + (max_scale - min_scale) * u) * ramp_progress


def spectral_vae_loss(
    output: Dict[str, Tensor],
    target_signal: Tensor,
    target_log_amp: Tensor,
    target_cos: Tensor,
    target_sin: Tensor,
    *,
    lambda_time: float = 1.0,
    lambda_amp: float = 1.0,
    lambda_phase: float = 0.1,
    lambda_bandpower: float = 0.0,
    physics_weight: float = 1.0,
    beta: float = 1e-4,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Composite temporal, amplitude, circular-phase, bandpower and KL objective."""
    loss_time = F.mse_loss(output["F_hat"], target_signal)
    loss_amp = F.mse_loss(output["log_amp_hat"], target_log_amp)
    loss_phase = F.mse_loss(output["cos_hat"], target_cos) + F.mse_loss(
        output["sin_hat"], target_sin
    )

    bp_hat = compute_bandpower(output["log_amp_hat"], time_samples=target_signal.shape[-1])
    bp_target = compute_bandpower(target_log_amp, time_samples=target_signal.shape[-1])

    loss_bp = sum(F.mse_loss(bp_hat[band], bp_target[band]) for band in bp_hat)

    mu, logvar = output["mu"], output["logvar"]
    loss_kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    total = (
        lambda_time * loss_time
        + lambda_amp * loss_amp
        + physics_weight * (lambda_phase * loss_phase + lambda_bandpower * loss_bp)
        + beta * loss_kl
    )
    return total, {
        "loss": total.detach().clone(),
        "time": loss_time.detach().clone(),
        "amp": loss_amp.detach().clone(),
        "phase": loss_phase.detach().clone(),
        "bandpower": loss_bp.detach().clone(),
        "kl": loss_kl.detach().clone(),
    }
