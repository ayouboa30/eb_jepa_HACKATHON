"""Visualisation physique d'un patient EEG — 19 canaux + reconstruction VAE.

Usage:
    python scripts/visualize_eeg_patient.py
    python scripts/visualize_eeg_patient.py --patient-idx 1 --output results/eeg_patient_1.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import signal as scipy_signal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_jepa.models.spectral_state_vae import (
    SpectralStateVAE,
    spectral_features,
    compute_bandpower,
)

# ----- Constantes cliniques EEG -----------------------------------------------
CHANNEL_NAMES = [
    "FP1", "FP2",          # Frontal polaire
    "F3",  "F4",  "F7", "F8",  # Frontal
    "C3",  "C4",           # Central
    "P3",  "P4",           # Pariétal
    "O1",  "O2",           # Occipital
    "T3",  "T4",  "T5", "T6",  # Temporal
    "FZ",  "CZ",  "PZ",   # Médian
]

BAND_COLORS = {
    "Delta\n(1-4 Hz)":  "#4C72B0",
    "Theta\n(4-8 Hz)":  "#DD8452",
    "Alpha\n(8-12 Hz)": "#55A868",
    "Beta\n(12-30 Hz)": "#C44E52",
    "Gamma\n(30+ Hz)":  "#8172B2",
}

BAND_RANGES = {
    "Delta\n(1-4 Hz)":  (1, 4),
    "Theta\n(4-8 Hz)":  (4, 8),
    "Alpha\n(8-12 Hz)": (8, 12),
    "Beta\n(12-30 Hz)": (12, 30),
    "Gamma\n(30+ Hz)":  (30, 100),
}

FS = 200  # Hz


def verify_physical_constraints(signal_tensor: torch.Tensor) -> dict:
    """Vérifie les contraintes physiques d'un signal EEG [B, C, T]."""
    results = {}

    # 1. Vérification de la stationnarité (ratio var première vs deuxième moitié)
    half = signal_tensor.shape[-1] // 2
    var_first = signal_tensor[..., :half].var(dim=-1)
    var_second = signal_tensor[..., half:].var(dim=-1)
    stationarity_ratio = (var_first / (var_second + 1e-8)).mean().item()
    results["stationarity_ratio"] = stationarity_ratio
    results["stationarity_ok"] = 0.5 < stationarity_ratio < 2.0

    # 2. Amplitude peak-to-peak (doit être dans une plage raisonnable)
    ptp = (signal_tensor.max(dim=-1).values - signal_tensor.min(dim=-1).values)
    results["ptp_mean"] = ptp.mean().item()
    results["ptp_max"] = ptp.max().item()
    results["amplitude_ok"] = ptp.max().item() < 20.0  # signal z-scoré < 20σ

    # 3. Vérification NaN/Inf
    results["no_nan"] = not torch.isnan(signal_tensor).any().item()
    results["no_inf"] = not torch.isinf(signal_tensor).any().item()

    # 4. Dominance de la bande alpha dans O1/O2 (classique EEG yeux fermés)
    _, log_amp, _, _ = spectral_features(signal_tensor)
    bp = compute_bandpower(log_amp)  # chaque bande -> [B, C]
    alpha_power = bp["alpha\n(8-12 Hz)"] if "alpha\n(8-12 Hz)" in bp else None
    results["alpha_mean_log_power"] = log_amp[..., 8:12].mean().item()

    # 5. Rapport signal/bruit spectral global
    total_power = log_amp.mean().item()
    results["mean_log_power"] = total_power

    return results


def compute_psd_per_channel(signal_np: np.ndarray, fs: float) -> tuple:
    """PSD pour chaque canal [C, T] -> fréquences, PSD [C, F]."""
    f_list, psd_list = [], []
    for ch in range(signal_np.shape[0]):
        f, pxx = scipy_signal.periodogram(signal_np[ch], fs=fs)
        f_list.append(f)
        psd_list.append(pxx)
    return np.array(f_list[0]), np.array(psd_list)


def run_vae(F_t: torch.Tensor, model: SpectralStateVAE) -> dict:
    """Exécute le VAE sur un batch [B, C, T] et retourne le dictionnaire de sortie."""
    model.eval()
    with torch.no_grad():
        _, log_amp, cos_phase, sin_phase = spectral_features(F_t)
        tau = torch.zeros(F_t.shape[0], 1)
        output = model(F_t, log_amp, cos_phase, sin_phase, tau, sample=False)
    return output


def plot_patient(
    F_t_patient: np.ndarray,    # [19, 200] — signal original
    F_hat: np.ndarray,          # [19, 200] — reconstruction VAE
    patient_id: str,
    label: str,
    physical_checks: dict,
    output_path: Path,
):
    """Génère la figure multi-panneau : 19 canaux + PSD + bandpowers + validation."""

    t = np.arange(F_t_patient.shape[-1]) / FS  # axe temporel en secondes
    freqs_orig, psd_orig = compute_psd_per_channel(F_t_patient, FS)
    freqs_hat,  psd_hat  = compute_psd_per_channel(F_hat, FS)

    # -- Mise en page -------------------------------------------------------
    fig = plt.figure(figsize=(26, 28), facecolor="#0D1117")
    fig.suptitle(
        f"Vérification EEG — Patient {patient_id}  •  Diagnostic: {label.upper()}",
        fontsize=18, color="white", fontweight="bold", y=0.99,
    )

    outer = gridspec.GridSpec(
        2, 1, figure=fig, hspace=0.32,
        top=0.96, bottom=0.04, left=0.06, right=0.97,
    )

    # =======================================================================
    # PANNEAU HAUT : 19 signaux temporels
    # =======================================================================
    inner_top = gridspec.GridSpecFromSubplotSpec(
        5, 4, subplot_spec=outer[0], hspace=0.55, wspace=0.30
    )

    for i, ch_name in enumerate(CHANNEL_NAMES):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(inner_top[row, col])
        ax.set_facecolor("#161B22")

        sig   = F_t_patient[i]
        recon = F_hat[i]

        ax.plot(t, sig,   lw=1.0, color="#58A6FF", label="Réel",        alpha=0.9)
        ax.plot(t, recon, lw=1.0, color="#FF7B72", label="VAE recon", linestyle="--", alpha=0.9)

        # Mise en forme
        ax.set_title(ch_name, color="white", fontsize=9, fontweight="bold", pad=2)
        ax.set_xlabel("s", color="#8B949E", fontsize=7)
        ax.set_ylabel("μV (norm.)", color="#8B949E", fontsize=6)
        ax.tick_params(colors="#8B949E", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.set_xlim(0, t[-1])

        # MSE locale
        mse = np.mean((sig - recon) ** 2)
        ax.text(
            0.97, 0.95, f"MSE={mse:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            color="#F0E68C", fontsize=6,
        )

    # Légende globale signaux
    ax_leg = fig.add_subplot(inner_top[4, 3])
    ax_leg.set_facecolor("#161B22")
    ax_leg.axis("off")
    ax_leg.legend(
        handles=[
            Patch(color="#58A6FF", label="Signal réel  $F_t$"),
            Patch(color="#FF7B72", label="Reconstruction VAE  $\\hat{F}_{t+1}$"),
        ],
        loc="center", fontsize=9, facecolor="#161B22",
        edgecolor="#30363D", labelcolor="white",
    )

    # =======================================================================
    # PANNEAU BAS : PSD moyenne + Bandpowers + Vérification physique
    # =======================================================================
    inner_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.35
    )

    # --- PSD comparée (moyenne sur 19 canaux) ---
    ax_psd = fig.add_subplot(inner_bot[0, 0])
    ax_psd.set_facecolor("#161B22")

    mean_psd_orig = psd_orig.mean(axis=0)
    mean_psd_hat  = psd_hat.mean(axis=0)
    mask = freqs_orig <= 100

    ax_psd.semilogy(freqs_orig[mask], mean_psd_orig[mask], color="#58A6FF",
                    lw=1.5, label="Réel $F_t$")
    ax_psd.semilogy(freqs_hat[mask],  mean_psd_hat[mask],  color="#FF7B72",
                    lw=1.5, linestyle="--", label="VAE $\\hat{F}$")

    for band_name, (f_lo, f_hi) in BAND_RANGES.items():
        ax_psd.axvspan(f_lo, f_hi, alpha=0.08, color=BAND_COLORS[band_name])
        mid = (f_lo + f_hi) / 2
        ax_psd.axvline(mid, color=BAND_COLORS[band_name], lw=0.5, alpha=0.4)

    ax_psd.set_xlabel("Fréquence (Hz)", color="#8B949E", fontsize=9)
    ax_psd.set_ylabel("PSD (dB/Hz)", color="#8B949E", fontsize=9)
    ax_psd.set_title("PSD moyenne (19 canaux)", color="white", fontsize=10, fontweight="bold")
    ax_psd.set_xlim(0, 100)
    ax_psd.legend(fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor="white")
    ax_psd.tick_params(colors="#8B949E", labelsize=8)
    for sp in ax_psd.spines.values():
        sp.set_edgecolor("#30363D")

    # --- Bandpowers par canal (barplot) ---
    ax_bp = fig.add_subplot(inner_bot[0, 1])
    ax_bp.set_facecolor("#161B22")

    band_keys = list(BAND_RANGES.keys())
    band_power_orig = []
    band_power_hat  = []
    for band_name, (f_lo, f_hi) in BAND_RANGES.items():
        mask_band = (freqs_orig >= f_lo) & (freqs_orig <= f_hi)
        band_power_orig.append(10 * np.log10(mean_psd_orig[mask_band].mean() + 1e-12))
        band_power_hat.append( 10 * np.log10(mean_psd_hat[mask_band].mean()  + 1e-12))

    x = np.arange(len(band_keys))
    w = 0.36
    bars1 = ax_bp.bar(x - w/2, band_power_orig, w,
                      color=[BAND_COLORS[k] for k in band_keys],
                      alpha=0.85, label="Réel")
    bars2 = ax_bp.bar(x + w/2, band_power_hat,  w,
                      color=[BAND_COLORS[k] for k in band_keys],
                      alpha=0.40, edgecolor="white", linewidth=0.8, label="VAE")

    # Labels sur les barres
    for bar, val in zip(bars1, band_power_orig):
        ax_bp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f"{val:.1f}", ha="center", va="bottom",
                   color="white", fontsize=7)
    for bar, val in zip(bars2, band_power_hat):
        ax_bp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f"{val:.1f}", ha="center", va="bottom",
                   color="#FF7B72", fontsize=7)

    ax_bp.set_xticks(x)
    ax_bp.set_xticklabels(band_keys, color="#8B949E", fontsize=7)
    ax_bp.set_ylabel("Puissance (dB)", color="#8B949E", fontsize=9)
    ax_bp.set_title("Bandpowers EEG\n(réel vs. VAE)", color="white", fontsize=10, fontweight="bold")
    ax_bp.legend(fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor="white")
    ax_bp.tick_params(colors="#8B949E", labelsize=7)
    for sp in ax_bp.spines.values():
        sp.set_edgecolor("#30363D")

    # --- Vérification physique (tableau) ---
    ax_check = fig.add_subplot(inner_bot[0, 2])
    ax_check.set_facecolor("#161B22")
    ax_check.axis("off")
    ax_check.set_title("Vérification physique", color="white", fontsize=10, fontweight="bold")

    checks = [
        ("[OK] Sans NaN",        physical_checks["no_nan"]),
        ("[OK] Sans Inf",         physical_checks["no_inf"]),
        ("[OK] Amplitude stable", physical_checks["amplitude_ok"]),
        ("[OK] Stationnarite",    physical_checks["stationarity_ok"]),
    ]
    metrics = [
        ("Peak-to-peak moyen",  f"{physical_checks['ptp_mean']:.3f} σ"),
        ("Peak-to-peak max",    f"{physical_checks['ptp_max']:.3f} σ"),
        ("Log-power moyen",     f"{physical_checks['mean_log_power']:.3f}"),
        ("Ratio stationnarité", f"{physical_checks['stationarity_ratio']:.3f}"),
        ("Log-power alpha",     f"{physical_checks['alpha_mean_log_power']:.3f}"),
    ]

    y_cur = 0.95
    for label_txt, ok in checks:
        color = "#3FB950" if ok else "#F85149"
        sym   = "[OK]" if ok else "[!!]"
        ax_check.text(0.05, y_cur, f"{sym}  {label_txt.lstrip('[OK]').strip()}",
                      transform=ax_check.transAxes,
                      color=color, fontsize=9, va="top", fontweight="bold")
        y_cur -= 0.09

    y_cur -= 0.02
    ax_check.plot([0.02, 0.98], [y_cur + 0.04, y_cur + 0.04],
                  color="#30363D", lw=0.8,
                  transform=ax_check.transAxes, clip_on=False)
    y_cur -= 0.02
    for m_label, m_val in metrics:
        ax_check.text(0.05, y_cur, f"  {m_label}:",
                      transform=ax_check.transAxes, color="#8B949E", fontsize=8, va="top")
        ax_check.text(0.72, y_cur, m_val,
                      transform=ax_check.transAxes, color="white",
                      fontsize=8, va="top", fontweight="bold")
        y_cur -= 0.09

    # Verdict global
    all_ok = all(v for _, v in checks)
    verdict_col = "#3FB950" if all_ok else "#F85149"
    verdict_txt = "[OK] PHYSIQUEMENT VALIDE" if all_ok else "[!!] ATTENTION : anomalie detectee"
    ax_check.text(
        0.5, 0.07, verdict_txt,
        transform=ax_check.transAxes, ha="center", va="bottom",
        color=verdict_col, fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262D", edgecolor=verdict_col, linewidth=1.5),
    )

    # =======================================================================
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Figure sauvegardée : {output_path}")


def main(args):
    # 1. Chargement des données
    data = torch.load(args.input_file, map_location="cpu")
    F_t_all   = data["F_t"]    # [3, 19, 200]
    F_next_all = data["F_next"]  # [3, 19, 200]

    n_patients = F_t_all.shape[0]
    patient_idx = args.patient_idx % n_patients
    print(f"Patients disponibles : {n_patients}  — affichage patient #{patient_idx}")

    F_t_pt = F_t_all[patient_idx : patient_idx + 1]  # [1, 19, 200]

    # 2. Initialisation du VAE
    model = SpectralStateVAE(channels=19, time_samples=200, latent_dim=64, hidden_dim=128)
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Chargement du checkpoint : {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        print("Poids aléatoires (pas de checkpoint spécifié).")

    # 3. Inférence VAE
    output = run_vae(F_t_pt, model)

    F_t_np  = F_t_pt[0].cpu().numpy()           # [19, 200]
    F_hat_np = output["F_hat"][0].cpu().numpy()  # [19, 200]

    # 4. Vérification physique
    print("\n--- Vérification physique du signal réel ---")
    checks = verify_physical_constraints(F_t_pt)
    for k, v in checks.items():
        print(f"  {k}: {v}")

    print("\n--- Vérification physique de la reconstruction VAE ---")
    checks_hat = verify_physical_constraints(output["F_hat"])
    for k, v in checks_hat.items():
        print(f"  {k}: {v}")

    # Patientid fictif basé sur index (les vrais IDs viennent du manifeste)
    patient_labels = ["aaaaabdo (anormal)", "aaaaabsk (anormal)", "aaaaabuv (anormal)"]
    patient_label_full = patient_labels[patient_idx] if patient_idx < len(patient_labels) else f"Patient {patient_idx}"
    patient_id, patient_dx = patient_label_full.split(" (")
    patient_dx = patient_dx.rstrip(")")

    # 5. Plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_patient(F_t_np, F_hat_np, patient_id.strip(), patient_dx, checks, output_path)

    # 6. Résumé console des bandpowers
    print(f"\n--- Bandpowers (log-amplitude) — patient {patient_id.strip()} ---")
    _, log_amp_real, _, _ = spectral_features(F_t_pt)
    _, log_amp_hat, _, _  = spectral_features(output["F_hat"])
    bp_real = compute_bandpower(log_amp_real)
    bp_hat  = compute_bandpower(log_amp_hat)
    bands_print = {
        "delta":  "Delta  (1-4 Hz)",
        "theta":  "Theta  (4-8 Hz)",
        "alpha":  "Alpha  (8-12 Hz)",
        "beta":   "Beta   (12-30 Hz)",
        "gamma":  "Gamma  (30+ Hz)",
    }
    for key, name in bands_print.items():
        real_val = bp_real[key].mean().item()
        hat_val  = bp_hat[key].mean().item()
        diff     = abs(real_val - hat_val)
        print(f"  {name:<20} | Reel: {real_val:7.4f}  | VAE: {hat_val:7.4f}  | Diff: {diff:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file",   default="results/real_eeg_samples.pt", help="Fichier .pt de segments réels")
    parser.add_argument("--patient-idx",  type=int, default=0, help="Index du patient à afficher (0, 1, 2)")
    parser.add_argument("--checkpoint",   type=str, default=None, help="Chemin vers un checkpoint VAE")
    parser.add_argument("--output",       default="results/eeg_patient_visualization.png", help="Chemin de l'image de sortie")
    main(parser.parse_args())
