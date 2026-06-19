import torch

from eb_jepa.models.spectral_state_vae import (
    SpectralStateVAE,
    spectral_features,
    spectral_vae_loss,
    training_physics_weight,
)


def test_rfft_roundtrip_and_frequency_dimension():
    signal = torch.randn(2, 19, 200)
    spectrum, log_amp, cos_phase, sin_phase = spectral_features(signal)
    reconstructed = torch.fft.irfft(spectrum, n=200, dim=-1)

    assert spectrum.shape == (2, 19, 101)
    assert log_amp.shape == cos_phase.shape == sin_phase.shape == spectrum.shape
    assert torch.allclose(reconstructed, signal, atol=1e-5, rtol=1e-5)
    assert all(torch.isfinite(value).all() for value in (log_amp, cos_phase, sin_phase))


def test_model_output_shapes_phase_unit_circle_and_finite_loss():
    torch.manual_seed(0)
    current = torch.randn(2, 19, 200)
    future = torch.randn(2, 19, 200)
    _, log_amp, cos_phase, sin_phase = spectral_features(current)
    _, next_log_amp, next_cos, next_sin = spectral_features(future)
    model = SpectralStateVAE(latent_dim=32, hidden_dim=128)

    output = model(current, log_amp, cos_phase, sin_phase, torch.zeros(2, 1), sample=False)
    loss, parts = spectral_vae_loss(
        output, future, next_log_amp, next_cos, next_sin
    )

    assert output["F_hat"].shape == future.shape
    assert output["X_hat"].shape == (2, 19, 101)
    assert output["log_amp_hat"].shape == (2, 19, 101)
    phase_radius = torch.sqrt(output["cos_hat"].square() + output["sin_hat"].square())
    assert torch.allclose(phase_radius, torch.ones_like(phase_radius), atol=2e-4)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_reparameterization_shape_and_gradients():
    mu = torch.zeros(4, 32, requires_grad=True)
    logvar = torch.zeros(4, 32, requires_grad=True)
    z = SpectralStateVAE.reparameterize(mu, logvar, sample=True)
    z.square().mean().backward()

    assert z.shape == mu.shape
    assert mu.grad is not None
    assert logvar.grad is not None


def test_physics_weight_schedule_strict():
    # Before 100 epochs
    assert training_physics_weight(0, warmup_epochs=100) == 0.0
    assert training_physics_weight(99, warmup_epochs=100) == 0.0

    # During ramp (e.g. epochs 100 to 149 for ramp=50)
    # The weight must be ramp_progress * u, where u is in [0.5, 1.0]
    for ep in range(100, 150):
        w = training_physics_weight(ep, warmup_epochs=100, ramp_epochs=50, min_scale=0.5, max_scale=1.0)
        ramp_progress = (ep - 100 + 1) / 50.0
        assert 0.5 * ramp_progress <= w <= 1.0 * ramp_progress

    # After ramp
    for ep in range(150, 200):
        w = training_physics_weight(ep, warmup_epochs=100, ramp_epochs=50, min_scale=0.5, max_scale=1.0)
        assert 0.5 <= w <= 1.0


def test_reconstruction_and_no_nan_with_schedule():
    torch.manual_seed(42)
    signal = torch.randn(2, 19, 200)
    future = torch.randn(2, 19, 200)
    _, log_amp, cos_phase, sin_phase = spectral_features(signal)
    _, next_log_amp, next_cos, next_sin = spectral_features(future)
    model = SpectralStateVAE(latent_dim=32, hidden_dim=128)

    output = model(signal, log_amp, cos_phase, sin_phase, torch.zeros(2, 1), sample=False)
    
    # Test with physics_weight = 0 (warmup)
    loss_0, parts_0 = spectral_vae_loss(
        output, future, next_log_amp, next_cos, next_sin, physics_weight=0.0
    )
    assert torch.isfinite(loss_0)
    assert parts_0["time"] > 0
    assert parts_0["amp"] > 0
    
    # Test with physics_weight = 0.75 (during/after ramp)
    loss_w, parts_w = spectral_vae_loss(
        output, future, next_log_amp, next_cos, next_sin, physics_weight=0.75
    )
    assert torch.isfinite(loss_w)
    
    # Check that lambda_time and lambda_amp are unaffected
    assert torch.allclose(parts_0["time"], parts_w["time"])
    assert torch.allclose(parts_0["amp"], parts_w["amp"])
    assert torch.allclose(parts_0["kl"], parts_w["kl"])
    
    # Check that physics terms are zeroed when weight is 0 in the total loss
    # Actually parts_0["phase"] still holds the *unweighted* base loss value, 
    # but the total loss `loss_0` shouldn't include it.
    total_reconstructed_0 = (
        1.0 * parts_0["time"] + 1.0 * parts_0["amp"] + 1e-4 * parts_0["kl"]
    )
    assert torch.allclose(loss_0, total_reconstructed_0)
    
    # Total with weight should include phase and bandpower
    total_reconstructed_w = (
        total_reconstructed_0 + 0.75 * (0.1 * parts_w["phase"] + 0.0 * parts_w["bandpower"])
    )
    assert torch.allclose(loss_w, total_reconstructed_w)
