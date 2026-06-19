"""Standalone research models that are not part of the core JEPA API."""

from .spectral_state_vae import SpectralStateVAE, spectral_features, spectral_vae_loss

__all__ = ["SpectralStateVAE", "spectral_features", "spectral_vae_loss"]
