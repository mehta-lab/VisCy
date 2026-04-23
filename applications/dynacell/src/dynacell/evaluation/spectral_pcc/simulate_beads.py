"""Simulate fluorescent beads with controlled bleaching for metric validation.

Generates a multi-bead phantom, convolves with a physically accurate OTF
(via waveorder), adds Poisson noise with exponential bleaching, and evaluates
all spectral PCC variants to validate metric behavior under known conditions.

Uses Hydra for configuration. Stages can be run independently::

    uv run python evaluation/spectral_pcc/simulate_beads.py              # all
    uv run python evaluation/spectral_pcc/simulate_beads.py stage=plot   # re-plot only
"""

import dataclasses
import logging
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SimulationData:
    """Intermediate simulation outputs, cached as .npz between stages."""

    clean: np.ndarray  # (Y,X) or (Z,Y,X), float32, normalized [0,1]
    series: np.ndarray  # (T,...), float32, Poisson-noisy bleached images
    prediction: np.ndarray  # same shape as clean, float32
    true_snr: np.ndarray  # (T,), float64


def _resolve_shape(cfg: DictConfig) -> tuple[int, ...]:
    """Return image shape based on ``cfg.phantom.ndim``."""
    if cfg.phantom.ndim == 2:
        return tuple(cfg.phantom.shape_2d)
    return tuple(cfg.phantom.shape_3d)


def _resolve_spacing(cfg: DictConfig) -> list[float]:
    """Return pixel spacing based on ``cfg.phantom.ndim``."""
    if cfg.phantom.ndim == 2:
        return list(cfg.phantom.spacing_2d)
    return list(cfg.phantom.spacing_3d)


def _save_simulation(sim: SimulationData, output_dir: Path) -> None:
    """Save simulation arrays to compressed .npz."""
    np.savez_compressed(
        output_dir / "simulation.npz",
        clean=sim.clean,
        series=sim.series,
        prediction=sim.prediction,
        true_snr=sim.true_snr,
    )


def _load_simulation(output_dir: Path) -> SimulationData:
    """Load cached simulation data from .npz.

    Raises
    ------
    FileNotFoundError
        If no cached simulation exists.
    """
    path = output_dir / "simulation.npz"
    if not path.exists():
        raise FileNotFoundError(f"No cached simulation at {path}. Run with stage=all or stage=simulate first.")
    data = np.load(path)
    return SimulationData(
        clean=data["clean"],
        series=data["series"],
        prediction=data["prediction"],
        true_snr=data["true_snr"],
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def generate_multi_bead_phantom(
    shape: tuple[int, ...],
    spacing: list[float],
    n_beads: int = 30,
    sphere_radius: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Generate an image with multiple fluorescent beads at random positions.

    Parameters
    ----------
    shape : tuple
        (Y, X) for 2D or (Z, Y, X) for 3D.
    spacing : list[float]
        Pixel/voxel spacing in physical units.
    n_beads : int
        Number of beads to place.
    sphere_radius : float
        Bead radius in physical units (0.01 = sub-resolution).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Phantom with beads (float32).
    """
    rng = np.random.default_rng(seed)
    ndim = len(shape)

    if ndim == 2:
        from waveorder.models import isotropic_fluorescent_thin_3d as model

        single = model.generate_test_phantom(shape, spacing[0], sphere_radius)
        single = single.numpy()
    else:
        from waveorder.models import isotropic_fluorescent_thick_3d as model

        single = model.generate_test_phantom(shape, spacing[1], spacing[0], sphere_radius)
        single = single.numpy()

    # Place beads at random positions via circular shifts
    phantom = np.zeros(shape, dtype=np.float32)
    center = np.array(shape) // 2
    for _ in range(n_beads):
        shift = rng.integers(-center, center, size=ndim)
        shifted = np.roll(single, shift, axis=tuple(range(ndim)))
        phantom += shifted

    # Normalize to [0, 1]
    pmax = phantom.max()
    if pmax > 0:
        phantom /= pmax
    return phantom


def apply_otf(
    phantom: np.ndarray,
    spacing: list[float],
    wavelength_emission: float = 0.698,
    numerical_aperture: float = 1.35,
    index_of_refraction: float = 1.3,
) -> np.ndarray:
    """Convolve phantom with widefield fluorescence OTF.

    Parameters
    ----------
    phantom : np.ndarray
        Input phantom (2D or 3D).
    spacing : list[float]
        Pixel/voxel spacing.
    wavelength_emission : float
        Emission wavelength in same units as spacing.
    numerical_aperture : float
        Detection NA.
    index_of_refraction : float
        Refractive index of medium.

    Returns
    -------
    np.ndarray
        OTF-convolved image (float32, non-negative).
    """
    ndim = phantom.ndim
    phantom_t = torch.from_numpy(phantom)

    if ndim == 2:
        from waveorder.models import isotropic_fluorescent_thin_3d as model

        otf = model.calculate_transfer_function(
            phantom.shape,
            spacing[0],
            [0.0],  # single focal plane
            wavelength_emission=wavelength_emission,
            index_of_refraction_media=index_of_refraction,
            numerical_aperture_detection=numerical_aperture,
        )
        data = model.apply_transfer_function(phantom_t, otf, background=0)
        result = data[0].numpy()  # extract z=0 slice
    else:
        from waveorder.models import isotropic_fluorescent_thick_3d as model

        otf = model.calculate_transfer_function(
            phantom.shape,
            spacing[1],
            spacing[0],
            wavelength_emission=wavelength_emission,
            z_padding=0,
            index_of_refraction_media=index_of_refraction,
            numerical_aperture_detection=numerical_aperture,
        )
        data = model.apply_transfer_function(phantom_t, otf, z_padding=0, background=0)
        result = data.numpy()

    # Ensure non-negative and float32
    result = np.maximum(result, 0).astype(np.float32)
    # Normalize to [0, 1]
    rmax = result.max()
    if rmax > 0:
        result /= rmax
    return result


def simulate_bleaching_series(
    clean_norm: np.ndarray,
    n_timepoints: int = 125,
    initial_counts: float = 10000.0,
    bleach_tau: float = 12.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create Poisson-noise bleaching time series.

    Parameters
    ----------
    clean_norm : np.ndarray
        OTF-convolved image normalized to [0, 1].
    n_timepoints : int
        Number of timepoints.
    initial_counts : float
        Peak photon counts at t=0.
    bleach_tau : float
        Exponential decay time constant (in timepoint units).
    seed : int
        Random seed.

    Returns
    -------
    series : np.ndarray
        Shape (T, ...) with Poisson-noisy bleaching images.
    true_snr : np.ndarray
        Shape (T,) with known peak SNR at each timepoint.
    """
    rng = np.random.default_rng(seed)
    t_vals = np.arange(n_timepoints, dtype=np.float64)
    signal_levels = initial_counts * np.exp(-t_vals / bleach_tau)
    true_snr = np.sqrt(signal_levels)

    series = np.zeros((n_timepoints, *clean_norm.shape), dtype=np.float32)
    for t in range(n_timepoints):
        lam = np.maximum(signal_levels[t] * clean_norm, 0).astype(np.float64)
        series[t] = rng.poisson(lam).astype(np.float32)

    return series, true_snr


def generate_shading_field(
    shape: tuple[int, ...],
    sigma_px: float | tuple[float, ...] = 50.0,
    seed: int = 123,
) -> np.ndarray:
    """Generate a smooth, non-negative shading field.

    Models realistic illumination non-uniformity / autofluorescence
    background: strictly positive, smoothly varying across the FOV.
    Normalized to [0, 1] range so that ``beta * initial_counts`` gives
    the peak background level in photon counts.

    Parameters
    ----------
    shape : tuple
        Image shape (Y, X) or (Z, Y, X).
    sigma_px : float or tuple of float
        Gaussian blur sigma in pixels (scalar or per-axis).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Smooth field in [0, 1] (float32).
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    s = rng.standard_normal(shape).astype(np.float32)
    s = gaussian_filter(s, sigma=sigma_px)
    # Normalize to [0, 1] — non-negative background
    s = (s - s.min()) / (s.max() - s.min() + 1e-10)
    return s


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def _compute_radial_otf(
    shape: tuple[int, int],
    spacing_yx: list[float],
    wavelength_emission: float = 0.698,
    numerical_aperture: float = 1.35,
    index_of_refraction: float = 1.3,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial OTF profile from waveorder transfer function.

    Returns (bin_centers, radial_otf_normalized).
    """
    from waveorder.models import isotropic_fluorescent_thin_3d as thin_model

    otf_3d = thin_model.calculate_transfer_function(
        shape,
        spacing_yx[0],
        [0.0],
        wavelength_emission=wavelength_emission,
        index_of_refraction_media=index_of_refraction,
        numerical_aperture_detection=numerical_aperture,
    )
    otf_mag = np.abs(otf_3d[0].numpy())

    fy = np.fft.fftfreq(shape[0], d=spacing_yx[0])
    fx = np.fft.fftfreq(shape[1], d=spacing_yx[1])
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    kr = np.sqrt(fy_grid**2 + fx_grid**2)

    bin_edges = np.linspace(0, kr.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    radial_otf = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (kr >= bin_edges[i]) & (kr < bin_edges[i + 1])
        if mask.sum() > 0:
            radial_otf[i] = otf_mag[mask].mean()

    otf_max = radial_otf.max()
    if otf_max > 0:
        radial_otf /= otf_max
    return bin_centers, radial_otf


def plot_diagnostic_spectra(
    clean: np.ndarray,
    series: np.ndarray,
    prediction: np.ndarray,
    spacing: list[float],
    true_snr: np.ndarray,
    output_path: Path,
    spectral_pcc_kwargs: dict | None = None,
    n_snapshots: int = 6,
    wavelength_emission: float = 0.698,
    numerical_aperture: float = 1.35,
) -> None:
    """Diagnostic visualization of bleaching simulation.

    Row 0: 2D image slices (clean + selected noisy timepoints).
    Row 1: Radial power spectra with OTF overlay.
    Row 2: DCR-filtered power spectra + cutoff line.
    Row 3: FSC-filtered power spectra + cutoff line.
    Row 4: Spectral_PCC weighted w*P (subtract-normalize).
    Row 5: SNR² weighted w*P.
    Row 6: LogSNR weighted w*P.
    Row 7: Weight curves (linear scale).
    Row 8: FRC curve (linear [0,1] scale).
    Row 9: FRCW-weighted w*P.
    Row 10: Cumulative weight mass.
    """
    from cubic.metrics.bandlimited import (
        _apply_lowpass,
        estimate_cutoff,
        estimate_noise_floor,
        otf_cutoff,
        radial_power_spectrum,
        spectral_weights,
    )

    T = len(series)
    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

    def to_2d(img):
        return img[img.shape[0] // 2] if img.ndim == 3 else img

    sp_2d = spacing[-2:]
    nyquist = 0.5 / sp_2d[0]  # Nyquist frequency

    n_cols = n_snapshots + 1  # +1 for clean
    n_rows = 11
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    # Share x-axis across all spectrum/weight rows (rows 1–6), using col 0 as reference
    for row in range(1, n_rows):
        for col in range(n_cols):
            if row == 1 and col == 0:
                continue
            axes[row, col].sharex(axes[1, 0])
    # Share y-axis across row 1 (power spectra), skip col 0 (has OTF twin axis)
    for j in range(2, n_cols):
        axes[1, j].sharey(axes[1, 1])

    # Pre-compute OTF cutoff (fixed for all timepoints)
    cutoff_otf_val = otf_cutoff(numerical_aperture, wavelength_emission)
    otf_cutoff_norm = cutoff_otf_val / nyquist
    # x-axis extends to the true OTF cutoff
    x_max = max(1.05, otf_cutoff_norm)

    to_2d(clean).shape

    # --- Row 0: 2D image slices (each panel auto-scaled) ---
    clean_2d = to_2d(clean).astype(np.float32)
    axes[0, 0].imshow(clean_2d, cmap="gray")
    axes[0, 0].set_title("Clean (no noise)", fontsize=9)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    for j, t_idx in enumerate(indices):
        noisy_2d = to_2d(series[t_idx]).astype(np.float32)
        ax = axes[0, j + 1]
        ax.imshow(noisy_2d, cmap="gray")
        ax.set_title(f"t={t_idx} SNR={true_snr[t_idx]:.1f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel("Image\n(auto-scaled)")

    # --- Row 1: Power spectra on log scale ---
    radii_c_raw, power_c = radial_power_spectrum(clean_2d, spacing=sp_2d)
    radii_c = radii_c_raw / nyquist  # normalize to [0, 1]

    # Normalize all power spectra by clean max so y-axis peaks at 1.0
    power_c_max = float(power_c.max()) if power_c.max() > 0 else 1.0
    power_c_norm = power_c / power_c_max

    freq_label = "Freq / Nyquist"

    # Clean panel: normalized power (log)
    axes[1, 0].semilogy(radii_c, power_c_norm, "k-", linewidth=1, label="Power")
    axes[1, 0].axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Nyquist")
    axes[1, 0].set_ylim(bottom=1e-18, top=2.0)
    axes[1, 0].set_xlim(0, x_max)
    axes[1, 0].set_title("Clean (no noise)", fontsize=9)
    axes[1, 0].set_xlabel(freq_label)
    axes[1, 0].legend(fontsize=7, loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    # Cache raw noisy power spectra for reuse as reference in filtered rows
    noisy_radii_norm = {}  # j -> normalized radii
    noisy_power_norm = {}  # j -> normalized power

    for j, t_idx in enumerate(indices):
        noisy_2d = to_2d(series[t_idx]).astype(np.float32)
        radii_raw, power = radial_power_spectrum(noisy_2d, spacing=sp_2d)
        radii = radii_raw / nyquist
        power_norm = power / power_c_max
        noisy_radii_norm[j] = radii
        noisy_power_norm[j] = power_norm
        ax = axes[1, j + 1]
        ax.semilogy(radii, power_norm, "C0-", linewidth=1, alpha=0.8, label="Noisy")
        ax.semilogy(radii_c, power_c_norm, "k--", linewidth=1, alpha=0.4, label="Clean")
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(bottom=1e-18, top=2.0)
        ax.set_title(f"t={t_idx} SNR={true_snr[t_idx]:.1f}", fontsize=9)
        ax.set_xlabel(freq_label)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_ylabel("Power")

    # --- Rows 2–4: Band-limited filtered power spectra ---
    # Helper to safely estimate cutoff and filter
    def _safe_filter_spectrum(image_2d, method, sp, na, wl):
        """Estimate cutoff and return (radii_norm, power_norm, cutoff_norm).

        Radii and cutoff are normalized by Nyquist.
        Returns (None, None, None) if cutoff estimation fails.
        """
        try:
            kw = {"spacing": sp, "method": method}
            if method in ("dcr",):
                kw["dcr_kwargs"] = {
                    "num_radii": 100,
                    "num_highpass": 10,
                    "windowing": True,
                    "refine": True,
                }
            if method in ("frc",):
                kw["frc_kwargs"] = {"bin_delta": 1, "backend": "hist"}
            if method == "otf":
                kw["numerical_aperture"] = na
                kw["wavelength_emission"] = wl
            cutoff_val = estimate_cutoff(image_2d, **kw)
        except Exception:
            return None, None, None

        # Guard against degenerate cutoffs
        if cutoff_val <= 0 or cutoff_val > nyquist:
            return None, None, None

        filtered = _apply_lowpass(image_2d, cutoff_val, spacing=sp, order=2)
        radii_f, power_f = radial_power_spectrum(filtered, spacing=sp)
        return radii_f / nyquist, power_f / power_c_max, cutoff_val / nyquist

    frc_label = "FRC" if clean.ndim == 2 else "FSC"
    bl_configs = [
        (2, "DCR", "dcr", "C2"),
        (3, frc_label, "frc", "C3"),
    ]

    for row_idx, label, method, color in bl_configs:
        # Clean panel: filter clean image with cutoff estimated from clean
        r_f, p_f, c_val = _safe_filter_spectrum(
            clean_2d,
            method,
            sp_2d,
            numerical_aperture,
            wavelength_emission,
        )
        ax = axes[row_idx, 0]
        if r_f is not None:
            ax.semilogy(r_f, p_f, "k-", linewidth=1, label="Filtered")
            ax.semilogy(radii_c, power_c_norm, "k--", linewidth=1, alpha=0.3, label="Raw")
            ax.axvline(c_val, color="k", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_title(f"{label} clean (fc={c_val:.2f})", fontsize=9)
        else:
            ax.text(
                0.5,
                0.5,
                "cutoff failed",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="red",
            )
            ax.set_title(f"{label} clean", fontsize=9)
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(bottom=1e-18, top=2.0)
        ax.set_xlabel(freq_label)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Noisy timepoint panels
        for j, t_idx in enumerate(indices):
            noisy_2d = to_2d(series[t_idx]).astype(np.float32)
            r_f, p_f, c_val = _safe_filter_spectrum(
                noisy_2d,
                method,
                sp_2d,
                numerical_aperture,
                wavelength_emission,
            )
            ax = axes[row_idx, j + 1]
            if r_f is not None:
                ax.semilogy(
                    noisy_radii_norm[j],
                    noisy_power_norm[j],
                    "C0--",
                    linewidth=1,
                    alpha=0.3,
                    label="Noisy",
                )
                ax.semilogy(r_f, p_f, f"{color}-", linewidth=1, alpha=0.8, label="Filtered")
                ax.semilogy(radii_c, power_c_norm, "k--", linewidth=1, alpha=0.3, label="Clean")
                ax.axvline(
                    c_val,
                    color=color,
                    linestyle=":",
                    linewidth=1,
                    alpha=0.6,
                    label="Cutoff",
                )
                ax.set_title(f"t={t_idx} fc={c_val:.2f}", fontsize=9)
                if j == 0:
                    ax.legend(fontsize=6, loc="upper right")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "cutoff failed",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="red",
                )
                ax.set_title(f"t={t_idx}", fontsize=9)
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(bottom=1e-18, top=2.0)
            ax.set_xlabel(freq_label)
            ax.grid(True, alpha=0.3)

        axes[row_idx, 0].set_ylabel(f"Power ({label})")

    # --- Rows 4–6: Weighted power spectra (Spectral_PCC, SNR², LogSNR) ---
    from dynacell.evaluation.spectral_pcc.evaluate import _snr_adaptive_weights

    bd = spectral_pcc_kwargs.get("bin_delta", 1.0) if spectral_pcc_kwargs else 1.0
    tf = spectral_pcc_kwargs.get("tail_fraction", 0.2) if spectral_pcc_kwargs else 0.2

    nf_c = estimate_noise_floor(radii_c_raw, power_c, tail_fraction=tf)

    def _sum_norm(w):
        s = float(np.sum(w))
        return w / s if s > 0 else w

    # Weight configs: (row, title, weight_fn, color)
    def _w_spectral(power, nf, radii):
        return spectral_weights(radii, power, nf)

    def _w_snr2(power, nf, radii):
        return _snr_adaptive_weights(power, nf, radii=radii, method="snr_squared")

    def _w_logsnr(power, nf, radii):
        return _snr_adaptive_weights(power, nf, radii=radii, method="log_snr")

    w_configs = [
        (4, "Spectral_PCC", _w_spectral, "C1"),
        (5, "SNR²_PCC", _w_snr2, "C7"),
        (6, "LogSNR_PCC", _w_logsnr, "C4"),
    ]

    # Store weights for the weight-curves row below
    w_clean_all = {}

    for row_idx, title, w_fn, color in w_configs:
        # Clean panel
        w_c = w_fn(power_c, nf_c, radii_c_raw)
        w_c_norm = _sum_norm(w_c)
        w_clean_all[row_idx] = (w_c, w_c_norm)
        axes[row_idx, 0].semilogy(radii_c, w_c_norm * power_c_norm, "k-", linewidth=1, label="w*P")
        axes[row_idx, 0].semilogy(radii_c, power_c_norm, "k--", linewidth=1, alpha=0.3, label="Raw")
        axes[row_idx, 0].set_ylim(bottom=1e-18, top=2.0)
        axes[row_idx, 0].set_title(title, fontsize=9)
        axes[row_idx, 0].set_xlabel(freq_label)
        axes[row_idx, 0].axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        axes[row_idx, 0].legend(fontsize=6, loc="upper right")
        axes[row_idx, 0].grid(True, alpha=0.3)

        # Noisy panels
        for j, t_idx in enumerate(indices):
            noisy_2d = to_2d(series[t_idx]).astype(np.float32)
            radii_raw, power = radial_power_spectrum(noisy_2d, spacing=sp_2d, bin_delta=bd)
            radii = radii_raw / nyquist
            power_norm = power / power_c_max
            nf = estimate_noise_floor(radii_raw, power, tail_fraction=tf)

            w_sub = w_fn(power, nf, radii_raw)
            w_sub_norm = _sum_norm(w_sub)
            ax = axes[row_idx, j + 1]
            ax.semilogy(
                noisy_radii_norm[j],
                noisy_power_norm[j],
                "C0--",
                linewidth=1,
                alpha=0.3,
                label="Noisy",
            )
            ax.semilogy(
                radii,
                w_sub_norm * power_norm,
                f"{color}-",
                linewidth=1,
                alpha=0.8,
                label="w*P",
            )
            ax.semilogy(radii_c, power_c_norm, "k--", linewidth=1, alpha=0.3, label="Clean")
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(bottom=1e-18, top=2.0)
            ax.set_title(f"t={t_idx}", fontsize=9)
            ax.set_xlabel(freq_label)
            if j == 0:
                ax.legend(fontsize=6, loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[row_idx, 0].set_ylabel(f"Power ({title.split('_')[0]})")

    # --- Row 7: Weight curves (linear scale, all three variants) ---
    w_colors = [("C1", "Spectral"), ("C7", "SNR²"), ("C4", "LogSNR")]
    for (row_idx, _, w_fn, _), (wc, wlabel) in zip(w_configs, w_colors):
        w_raw, _ = w_clean_all[row_idx]
        w_max_norm = w_raw / (w_raw.max() + 1e-30)
        axes[7, 0].plot(radii_c, w_max_norm, f"{wc}-", linewidth=1, label=wlabel)

    axes[7, 0].set_title("Weight curves", fontsize=9)
    axes[7, 0].set_ylim(-0.05, 1.05)
    axes[7, 0].set_xlabel(freq_label)
    axes[7, 0].axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[7, 0].legend(fontsize=6, loc="upper right")
    axes[7, 0].grid(True, alpha=0.3)

    for j, t_idx in enumerate(indices):
        noisy_2d = to_2d(series[t_idx]).astype(np.float32)
        radii_raw, power = radial_power_spectrum(noisy_2d, spacing=sp_2d, bin_delta=bd)
        radii = radii_raw / nyquist
        nf = estimate_noise_floor(radii_raw, power, tail_fraction=tf)

        ax = axes[7, j + 1]
        for (_, _, w_fn, _), (wc, wlabel) in zip(w_configs, w_colors):
            w_raw = w_fn(power, nf, radii_raw)
            w_max_norm = w_raw / (w_raw.max() + 1e-30)
            ax.plot(radii, w_max_norm, f"{wc}-", linewidth=1, label=wlabel)

        ax.set_title(f"t={t_idx}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(freq_label)
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[7, 0].set_ylabel("Weight (max=1)")

    # --- Row 8: FRC curve (linear [0,1] scale) ---
    from cubic.metrics.bandlimited import frc_weights
    from cubic.metrics.spectral.frc import calculate_frc as _calculate_frc

    frcw_threshold = spectral_pcc_kwargs.get("frcw_threshold", 0.143) if spectral_pcc_kwargs else 0.143

    # FRC curve for clean image
    frc_result_c = _calculate_frc(
        clean_2d,
        image2=None,
        backend="hist",
        bin_delta=bd,
        zero_padding=False,
        disable_hamming=False,
        average=True,
    )
    frc_curve_c = frc_result_c.correlation["correlation"]
    frc_freq_c = frc_result_c.correlation["frequency"]
    axes[8, 0].plot(frc_freq_c, frc_curve_c, "k-", linewidth=1, label="FRC")
    axes[8, 0].axhline(
        frcw_threshold,
        color="r",
        linestyle="--",
        linewidth=0.8,
        label=f"tau={frcw_threshold}",
    )
    axes[8, 0].set_ylim(-0.1, 1.05)
    axes[8, 0].set_title("FRC (clean)", fontsize=9)
    axes[8, 0].set_xlabel("Freq (normalized)")
    axes[8, 0].legend(fontsize=6, loc="upper right")
    axes[8, 0].grid(True, alpha=0.3)

    for j, t_idx in enumerate(indices):
        noisy_2d = to_2d(series[t_idx]).astype(np.float32)
        frc_result_n = _calculate_frc(
            noisy_2d,
            image2=None,
            backend="hist",
            bin_delta=bd,
            zero_padding=False,
            disable_hamming=False,
            average=True,
        )
        frc_curve_n = frc_result_n.correlation["correlation"]
        frc_freq_n = frc_result_n.correlation["frequency"]
        ax = axes[8, j + 1]
        ax.plot(frc_freq_n, frc_curve_n, "C5-", linewidth=1, label="FRC")
        ax.plot(frc_freq_c, frc_curve_c, "k--", linewidth=1, alpha=0.3, label="Clean")
        ax.axhline(frcw_threshold, color="r", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"t={t_idx}", fontsize=9)
        ax.set_xlabel("Freq (normalized)")
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[8, 0].set_ylabel("FRC")

    # --- Row 9: FRCW-weighted w*P ---
    w_frcw_c = frc_weights(clean_2d, bin_delta=bd)
    w_frcw_c_sn = w_frcw_c / (np.sum(w_frcw_c) + 1e-30)  # sum-normalized
    # Map FRCW weights (index-unit bins) to the Nyquist-normalized radii
    from cubic.metrics.spectral.radial import radial_edges as _radial_edges

    frcw_edges_c, frcw_radii_c = _radial_edges(clean_2d.shape, bin_delta=bd, spacing=None)
    frcw_radii_c_norm = frcw_radii_c / (0.5 * clean_2d.shape[0])  # normalize by Nyquist index
    # Trim to weight length
    frcw_radii_c_norm[: len(w_frcw_c)]
    # Need power on index-unit bins for overlay
    radii_idx_c, power_idx_c = radial_power_spectrum(clean_2d, spacing=sp_2d, bin_delta=bd)
    power_idx_c / power_c_max
    # Use physical-unit radii for x-axis consistency with other rows
    axes[9, 0].semilogy(
        radii_c[: len(w_frcw_c_sn)],
        w_frcw_c_sn * power_c_norm[: len(w_frcw_c_sn)],
        "k-",
        linewidth=1,
        label="w*P",
    )
    axes[9, 0].semilogy(radii_c, power_c_norm, "k--", linewidth=1, alpha=0.3, label="Raw")
    axes[9, 0].set_ylim(bottom=1e-18, top=2.0)
    axes[9, 0].set_title("FRCW", fontsize=9)
    axes[9, 0].set_xlabel(freq_label)
    axes[9, 0].axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[9, 0].legend(fontsize=6, loc="upper right")
    axes[9, 0].grid(True, alpha=0.3)

    for j, t_idx in enumerate(indices):
        noisy_2d = to_2d(series[t_idx]).astype(np.float32)
        radii_raw, power = radial_power_spectrum(noisy_2d, spacing=sp_2d, bin_delta=bd)
        radii = radii_raw / nyquist
        power_norm = power / power_c_max
        w_frcw = frc_weights(noisy_2d, bin_delta=bd)
        w_frcw_sn = w_frcw / (np.sum(w_frcw) + 1e-30)
        ax = axes[9, j + 1]
        ax.semilogy(
            noisy_radii_norm[j],
            noisy_power_norm[j],
            "C0--",
            linewidth=1,
            alpha=0.3,
            label="Noisy",
        )
        ax.semilogy(
            radii[: len(w_frcw_sn)],
            w_frcw_sn * power_norm[: len(w_frcw_sn)],
            "C5-",
            linewidth=1,
            alpha=0.8,
            label="w*P",
        )
        ax.semilogy(radii_c, power_c_norm, "k--", linewidth=1, alpha=0.3, label="Clean")
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(bottom=1e-18, top=2.0)
        ax.set_title(f"t={t_idx}", fontsize=9)
        ax.set_xlabel(freq_label)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[9, 0].set_ylabel("Power (FRCW)")

    # --- Row 10: Cumulative weight mass ---
    # Need bin pixel counts for shell-volume correction
    from cubic.metrics.spectral.radial import radial_bin_id, radial_edges

    edges_cpu, _ = radial_edges(to_2d(clean).shape, bin_delta=bd, spacing=sp_2d)
    bid = radial_bin_id(to_2d(clean).shape, edges_cpu, spacing=sp_2d)
    n_pixels = np.bincount(bid[bid >= 0], minlength=len(radii_c_raw))
    n_pix = n_pixels[: len(radii_c_raw)]

    def _cum_mass(w, n_pix_arr):
        mass = w * n_pix_arr[: len(w)]
        s = mass.sum()
        if s <= 0:
            return np.zeros_like(w)
        return np.cumsum(mass) / s

    # Clean panel: all 3 weight variants + FRCW
    cum_spectral_c = _cum_mass(w_clean_all[4][0], n_pix)  # subtract-normalize
    cum_snr2_c = _cum_mass(w_clean_all[5][0], n_pix)  # SNR²
    cum_frcw_c = _cum_mass(w_frcw_c, n_pix[: len(w_frcw_c)])  # FRCW
    axes[10, 0].plot(radii_c, cum_spectral_c, "C1-", linewidth=1, label="Spectral")
    axes[10, 0].plot(radii_c, cum_snr2_c, "C7-", linewidth=1, label="SNR²")
    axes[10, 0].plot(radii_c[: len(cum_frcw_c)], cum_frcw_c, "C5-", linewidth=1, label="FRCW")
    axes[10, 0].axhline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[10, 0].set_ylim(-0.05, 1.05)
    axes[10, 0].set_title("Cumulative mass", fontsize=9)
    axes[10, 0].set_xlabel(freq_label)
    axes[10, 0].axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[10, 0].legend(fontsize=6, loc="lower right")
    axes[10, 0].grid(True, alpha=0.3)

    for j, t_idx in enumerate(indices):
        noisy_2d = to_2d(series[t_idx]).astype(np.float32)
        radii_raw, power = radial_power_spectrum(noisy_2d, spacing=sp_2d, bin_delta=bd)
        radii = radii_raw / nyquist
        nf = estimate_noise_floor(radii_raw, power, tail_fraction=tf)

        # Spectral weights
        w_sp = spectral_weights(radii_raw, power, nf)
        # SNR²
        w_s2 = _snr_adaptive_weights(power, nf, radii=radii_raw, method="snr_squared")
        # FRCW
        w_frcw_j = frc_weights(noisy_2d, bin_delta=bd)

        ax = axes[10, j + 1]
        n_pix_j = n_pix[: len(w_sp)]
        ax.plot(radii, _cum_mass(w_sp, n_pix_j), "C1-", linewidth=1, label="Spectral")
        ax.plot(radii, _cum_mass(w_s2, n_pix_j), "C7-", linewidth=1, label="SNR²")
        ax.plot(
            radii[: len(w_frcw_j)],
            _cum_mass(w_frcw_j, n_pix_j[: len(w_frcw_j)]),
            "C5-",
            linewidth=1,
            label="FRCW",
        )
        ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"t={t_idx}", fontsize=9)
        ax.set_xlabel(freq_label)
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        if j == 0:
            ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)

    axes[10, 0].set_ylabel("Cum. weight")

    fig.suptitle("Diagnostic: power spectra & metric weights vs bleaching", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", output_path)


def plot_raw_power_and_otf(
    clean: np.ndarray,
    spacing: list[float],
    output_path: Path,
    wavelength_emission: float = 0.698,
    numerical_aperture: float = 1.35,
    index_of_refraction: float = 1.3,
) -> None:
    """Two-panel plot showing raw (unnormalized) power spectrum and OTF profile."""
    from cubic.metrics.bandlimited import radial_power_spectrum

    clean_2d = clean[clean.shape[0] // 2] if clean.ndim == 3 else clean
    clean_2d = clean_2d.astype(np.float32)
    sp_2d = spacing[-2:]

    # Raw power spectrum (no normalization)
    radii, power = radial_power_spectrum(clean_2d, spacing=sp_2d)

    # Radial OTF profile (reuse existing helper)
    bin_centers, radial_otf = _compute_radial_otf(
        clean_2d.shape,
        sp_2d,
        wavelength_emission=wavelength_emission,
        numerical_aperture=numerical_aperture,
        index_of_refraction=index_of_refraction,
    )
    # Undo the normalization — plot_raw_power_and_otf expects unnormalized
    # (the existing function normalizes to max=1, which is fine for overlay)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Panel 1: Raw power spectrum
    ax1.semilogy(radii, power, "k-", linewidth=1.5)
    ax1.set_xlabel("Spatial frequency (cy/μm)")
    ax1.set_ylabel("Power (a.u.)")
    ax1.set_title("Clean power spectrum (raw)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Raw OTF profile
    ax2.plot(bin_centers, radial_otf, "r-", linewidth=1.5)
    ax2.set_xlabel("Spatial frequency (cy/μm)")
    ax2.set_ylabel("|OTF| magnitude")
    ax2.set_title(f"OTF profile (NA={numerical_aperture}, λ={wavelength_emission} μm)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", output_path)


def plot_simulation_metrics(
    df: pd.DataFrame,
    output_path: Path,
    ndim: int = 2,
    n_beads: int = 30,
    bleach_tau: float = 12.0,
    dpi: int = 150,
) -> None:
    """Plot metric trends vs timepoint from simulation results."""
    plot_cols = [
        c
        for c in df.columns
        if c
        not in (
            "timepoint",
            "true_SNR",
            "signal_level",
            "zero_frac",
            "DCR_r0",
        )
        and not c.startswith("EV_")
    ]
    n = len(plot_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.asarray(axes).flatten()
    t_vals = df["timepoint"].values

    for i, col in enumerate(plot_cols):
        ax = axes[i]
        vals = df[col].values
        ax.plot(t_vals, vals, marker="o", markersize=1.5, linewidth=1)
        mask = np.isfinite(vals)
        if mask.sum() > 1:
            slope, intercept = np.polyfit(t_vals[mask], vals[mask], 1)
            ax.plot(t_vals, slope * t_vals + intercept, "r--", linewidth=1)
            y0 = intercept + slope * t_vals[0]
            yT = intercept + slope * t_vals[-1]
            drop = (y0 - yT) / y0 * 100 if y0 > 0 else 0
            cv = np.std(vals[mask]) / np.mean(vals[mask]) * 100 if np.mean(vals[mask]) != 0 else 0
            ax.set_title(f"{col}\ndrop={drop:.1f}% CV={cv:.1f}%", fontsize=8)
        else:
            ax.set_title(col, fontsize=8)
        ax.set_xlabel("Timepoint")
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Simulated beads ({ndim}D, {n_beads} beads, tau={bleach_tau})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", output_path)


def plot_pcc_comparison(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
    df_noshade: pd.DataFrame | None = None,
    series: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
    spacing: list[float] | None = None,
    nbins_low_sweep: list[int] | None = None,
    title: str | None = None,
    pcc_label: str | None = None,
    sweep_values: dict[int, np.ndarray] | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Single-panel comparison of PCC variants with nbins_low sweep.

    Shows no-shading baselines, shading PCC, and a sweep of Spectral_PCC
    over nbins_low values to illustrate how low-k exclusion removes the
    shading plateau.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics from the shading run (with current nbins_low).
    output_path : Path
        Where to save the figure.
    dpi : int
        Figure resolution.
    df_noshade : pd.DataFrame, optional
        Metrics from the no-shading run (baselines).
    series, prediction : np.ndarray, optional
        Cached simulation data for on-the-fly sweep computation.
    spacing : list[float], optional
        Pixel spacing for spectral_pcc calls.
    nbins_low_sweep : list[int], optional
        Values of nbins_low to sweep. Defaults to [0, 1, 2, 3, 4, 5].
    title : str, optional
        Plot title. Defaults to simulation-specific title.
    pcc_label : str, optional
        Label for the PCC_2D line. Defaults to ``"PCC (shading)"``.
    sweep_values : dict[int, np.ndarray], optional
        Pre-computed sweep: ``{nbins_low: array_of_values}``. When provided,
        skips on-the-fly spectral_pcc computation.
    """
    from cubic.metrics.bandlimited import spectral_pcc as _spcc

    t = df["timepoint"].values
    fig, ax = plt.subplots(figsize=figsize or (6, 3.5))

    # --- No-shading baselines (solid, muted) ---
    if df_noshade is not None:
        t_ns = df_noshade["timepoint"].values
        ax.plot(
            t_ns,
            df_noshade["PCC_2D"],
            color="0.55",
            ls="-",
            lw=1.5,
            label="PCC (no shading)",
        )
        ax.plot(
            t_ns,
            df_noshade["Spectral_PCC_2D"],
            color="0.35",
            ls="-",
            lw=1.5,
            label="Spectral_PCC (no shading)",
        )

    # --- PCC baseline (solid, prominent) ---
    ax.plot(
        t,
        df["PCC_2D"],
        color="0.55",
        ls="-",
        lw=2.0,
        label=pcc_label or "PCC (shading)",
    )

    # --- Pre-computed Spectral_PCC from df (only when no sweep provides it) ---
    if "Spectral_PCC_2D" in df.columns and sweep_values is None and series is None:
        ax.plot(t, df["Spectral_PCC_2D"], color="0.25", ls="-", lw=2.0, label="Spectral_PCC")

    # --- nbins_low sweep (sequential colormap, thinner) ---
    if sweep_values is not None:
        # Pre-computed sweep — no spectral_pcc calls needed
        if nbins_low_sweep is None:
            nbins_low_sweep = sorted(sweep_values.keys())
        cmap = plt.cm.plasma_r
        n_vals = len(nbins_low_sweep)
        for i, nbl in enumerate(nbins_low_sweep):
            color = cmap(0.15 + 0.75 * i / max(n_vals - 1, 1))
            ls = "-" if nbl == 0 else "--"
            lw = 2.0 if nbl == 0 else 0.9
            label = "Spectral_PCC" if nbl == 0 else f"Spectral_PCC (nbins_low={nbl})"
            ax.plot(t, sweep_values[nbl], color=color, ls=ls, lw=lw, label=label)
    elif series is not None and prediction is not None and spacing is not None:
        if nbins_low_sweep is None:
            nbins_low_sweep = list(range(11))  # 0..10

        cmap = plt.cm.plasma_r
        n_vals = len(nbins_low_sweep)
        n_tp = len(t)

        for i, nbl in enumerate(nbins_low_sweep):
            color = cmap(0.15 + 0.75 * i / max(n_vals - 1, 1))
            vals = np.empty(n_tp)
            for ti in range(n_tp):
                vals[ti] = _spcc(
                    prediction,
                    series[ti],
                    spacing=spacing,
                    nbins_low=nbl,
                )
            ls = "-" if nbl == 0 else "--"
            lw = 2.0 if nbl == 0 else 0.9
            label = "Spectral_PCC" if nbl == 0 else f"Spectral_PCC (nbins_low={nbl})"
            ax.plot(t, vals, color=color, ls=ls, lw=lw, label=label)

    ax.set_xlabel("Timepoint", fontsize=8)
    ax.set_ylabel("PCC", fontsize=8)
    ax.set_title(title or "Simulated beads with shading (beta=0.01) — PCC variants", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", output_path)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def run_simulate(cfg: DictConfig) -> SimulationData:
    """Stage 1: Generate phantom, apply OTF, simulate bleaching series."""
    shape = _resolve_shape(cfg)
    spacing = _resolve_spacing(cfg)
    ndim = cfg.phantom.ndim
    optics = cfg.optics

    log.info("Generating %dD multi-bead phantom (%d beads)...", ndim, cfg.phantom.n_beads)
    phantom = generate_multi_bead_phantom(
        shape,
        spacing,
        n_beads=cfg.phantom.n_beads,
        sphere_radius=cfg.phantom.sphere_radius,
        seed=cfg.phantom.seed,
    )
    log.info("  Phantom shape: %s, max: %.4f", phantom.shape, phantom.max())

    log.info(
        "Applying OTF (NA=%.2f, λ=%.3f μm)...",
        optics.numerical_aperture,
        optics.wavelength_emission,
    )
    clean = apply_otf(
        phantom,
        spacing,
        wavelength_emission=optics.wavelength_emission,
        numerical_aperture=optics.numerical_aperture,
        index_of_refraction=optics.index_of_refraction,
    )
    log.info("  Clean image shape: %s, max: %.4f", clean.shape, clean.max())

    # Apply illumination shading if configured
    alpha = float(cfg.shading.alpha)
    beta = float(cfg.shading.beta)
    initial_counts = float(cfg.bleaching.initial_counts)
    clean_for_sim = clean

    if alpha > 0 or beta > 0:
        sigma_px = tuple(cfg.shading.sigma_um / sp for sp in spacing)
        shading = generate_shading_field(shape, sigma_px=sigma_px, seed=cfg.shading.seed)
        log.info(
            "  Shading: alpha=%.2f, beta=%.2f, sigma=%.1f μm",
            alpha,
            beta,
            cfg.shading.sigma_um,
        )
        if alpha > 0:
            gain = np.clip(1 + alpha * shading, 0.1, None).astype(np.float32)
            clean_for_sim = clean * gain

    prediction = (clean_for_sim * initial_counts).astype(np.float32)

    log.info(
        "Simulating bleaching series (%d timepoints, tau=%.0f)...",
        cfg.bleaching.n_timepoints,
        cfg.bleaching.bleach_tau,
    )
    series, true_snr = simulate_bleaching_series(
        clean_for_sim,
        n_timepoints=cfg.bleaching.n_timepoints,
        initial_counts=initial_counts,
        bleach_tau=cfg.bleaching.bleach_tau,
        seed=cfg.bleaching.seed,
    )
    log.info("  Series shape: %s", series.shape)

    # Additive background (constant across time, fraction of initial peak)
    if beta > 0:
        bg = (beta * initial_counts * shading).astype(np.float32)
        for t in range(len(series)):
            series[t] += bg
        prediction = prediction + bg
        log.info("  Added shading background (beta=%.2f)", beta)

    return SimulationData(
        clean=clean,
        series=series,
        prediction=prediction,
        true_snr=true_snr,
    )


def run_evaluate(
    cfg: DictConfig,
    sim: SimulationData,
    output_dir: Path,
) -> pd.DataFrame:
    """Stage 2: Compute per-timepoint metrics and save CSV."""
    spacing = _resolve_spacing(cfg)
    ndim = cfg.phantom.ndim
    initial_counts = cfg.bleaching.initial_counts
    bleach_tau = cfg.bleaching.bleach_tau

    spectral_pcc_kwargs = OmegaConf.to_container(cfg.metrics.spectral_pcc, resolve=True)
    dcr_kwargs = OmegaConf.to_container(cfg.metrics.dcr, resolve=True)
    bandlimited_kwargs = OmegaConf.to_container(cfg.metrics.bandlimited, resolve=True)
    optics_dict = OmegaConf.to_container(cfg.optics, resolve=True)

    n_timepoints = len(sim.series)

    # Compute frozen FRCW weights from first K=5 frames (median)
    from cubic.metrics.bandlimited import frc_weights
    from scipy.ndimage import median_filter

    K = min(5, n_timepoints)
    frcw_per_frame = []
    frcw_kw_frozen = {"bin_delta": spectral_pcc_kwargs.get("bin_delta", 1.0)}
    nbins_low = spectral_pcc_kwargs.get("frcw_nbins_low", 3)
    smooth_window = spectral_pcc_kwargs.get("frcw_smooth_window", 5)
    for t_ref in range(K):
        gt_t = sim.series[t_ref]
        if ndim == 3:
            gt_t = gt_t[gt_t.shape[0] // 2]
        gt_t = gt_t.astype(np.float32)
        frcw_per_frame.append(frc_weights(gt_t, **frcw_kw_frozen))
    frozen_frcw = np.median(np.stack(frcw_per_frame), axis=0)
    # Re-smooth + monotone after median for maximal stability
    sw = smooth_window | 1
    sw = max(3, min(sw, len(frozen_frcw) | 1))
    frozen_frcw = median_filter(frozen_frcw, size=sw)
    frozen_frcw = np.maximum.accumulate(frozen_frcw[::-1])[::-1]
    frozen_frcw[:nbins_low] = 0
    log.info(
        "Frozen FRCW: %d/%d nonzero, total mass=%.3f",
        (frozen_frcw > 0).sum(),
        len(frozen_frcw),
        frozen_frcw.sum(),
    )

    log.info("Computing metrics...")
    rows = []
    for t in range(n_timepoints):
        if (t + 1) % 25 == 0 or t == 0:
            log.info(
                "  timepoint %d / %d (true SNR=%.1f)",
                t + 1,
                n_timepoints,
                sim.true_snr[t],
            )

        gt = sim.series[t]
        pred = sim.prediction

        if ndim == 2:
            from dynacell.evaluation.spectral_pcc.evaluate import (
                compute_gt_reliability,
                compute_timepoint_metrics_2d,
                corr_coef,
                psnr,
            )

            gt_f = gt.astype(np.float32)
            pred_f = pred.astype(np.float32)
            data_range = float(gt_f.max() - gt_f.min()) if gt_f.max() > gt_f.min() else 1.0

            m: dict[str, float] = {
                "PCC_2D": float(corr_coef(gt_f, pred_f)),
                "PSNR_2D": float(psnr(gt_f, pred_f, data_range=data_range)),
            }

            m_2d = compute_timepoint_metrics_2d(
                gt,
                pred,
                spacing,
                dcr_kwargs,
                spectral_pcc_kwargs=spectral_pcc_kwargs,
                bandlimited_kwargs=bandlimited_kwargs,
                optics=optics_dict,
                frozen_frcw_weights=frozen_frcw,
            )
            for k, v in m_2d.items():
                if k not in m:
                    m[k] = v

            a0, r0 = compute_gt_reliability(gt, spacing, dcr_kwargs)
            m["DCR_A0"] = a0
            m["DCR_r0"] = r0
        else:
            from dynacell.evaluation.spectral_pcc.evaluate import (
                compute_gt_reliability,
                compute_timepoint_metrics,
            )

            fsc_kwargs = OmegaConf.to_container(cfg.metrics.fsc, resolve=True)
            m = compute_timepoint_metrics(
                gt,
                pred,
                spacing,
                fsc_kwargs,
                dcr_kwargs,
                spectral_pcc_kwargs=spectral_pcc_kwargs,
            )
            mid_z = gt.shape[0] // 2
            a0, r0 = compute_gt_reliability(gt[mid_z], spacing[1:], dcr_kwargs)
            m["DCR_A0"] = a0
            m["DCR_r0"] = r0

        m["timepoint"] = t
        m["true_SNR"] = sim.true_snr[t]
        m["signal_level"] = initial_counts * np.exp(-t / bleach_tau)
        rows.append(m)

    df = pd.DataFrame(rows)

    # Compute DCR_w reliability weights
    if "DCR_A0" in df.columns:
        a0_vals = df["DCR_A0"].values
        k_ref = 5
        a_good = float(np.median(a0_vals[:k_ref]))
        a_bad = float(np.median(a0_vals[-k_ref:]))
        eps = 1e-6
        if a_good <= 0:
            df["DCR_w"] = 0.0
        elif (a_good - a_bad) < eps:
            df["DCR_w"] = 1.0
        else:
            w = np.clip((a0_vals - a_bad) / (a_good - a_bad), 0.0, 1.0)
            w = np.where(np.isfinite(a0_vals), w, 0.0)
            df["DCR_w"] = w

    # Reorder columns
    cols = ["timepoint", "true_SNR", "signal_level"] + [
        c for c in df.columns if c not in ("timepoint", "true_SNR", "signal_level")
    ]
    df = df[cols]

    csv_path = output_dir / "simulation_metrics.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %s", csv_path)

    return df


def run_plots(
    cfg: DictConfig,
    sim: SimulationData,
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Stage 3: Generate all plots from simulation data and metrics."""
    spacing = _resolve_spacing(cfg)
    optics = cfg.optics
    initial_counts = cfg.bleaching.initial_counts
    spectral_pcc_kwargs = OmegaConf.to_container(cfg.metrics.spectral_pcc, resolve=True)

    plot_simulation_metrics(
        df,
        output_dir / "simulation_metrics.png",
        ndim=cfg.phantom.ndim,
        n_beads=cfg.phantom.n_beads,
        bleach_tau=cfg.bleaching.bleach_tau,
        dpi=cfg.plot.dpi,
    )

    plot_raw_power_and_otf(
        sim.clean * initial_counts,
        spacing,
        output_dir / "raw_power_and_otf.png",
        wavelength_emission=optics.wavelength_emission,
        numerical_aperture=optics.numerical_aperture,
        index_of_refraction=optics.index_of_refraction,
    )

    plot_diagnostic_spectra(
        sim.clean * initial_counts,
        sim.series,
        sim.prediction,
        spacing,
        sim.true_snr,
        output_dir / "diagnostic_spectra.png",
        spectral_pcc_kwargs=spectral_pcc_kwargs,
        n_snapshots=cfg.plot.n_snapshots,
        wavelength_emission=optics.wavelength_emission,
        numerical_aperture=optics.numerical_aperture,
    )

    # Load no-shading baseline CSV if available
    noshade_path = output_dir.parent / "output_simulation" / "simulation_metrics.csv"
    df_noshade = pd.read_csv(noshade_path) if noshade_path.exists() else None

    plot_pcc_comparison(
        df,
        output_dir / "pcc_comparison.png",
        dpi=cfg.plot.dpi,
        df_noshade=df_noshade,
        series=sim.series,
        prediction=sim.prediction,
        spacing=spacing,
    )


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.2", config_path="../_configs/spectral_pcc", config_name="simulate")
def main(cfg: DictConfig) -> None:
    """Simulate fluorescent beads and evaluate spectral PCC metrics."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = cfg.stage

    # Stage 1: Simulate
    sim_data = None
    if stage in ("all", "simulate"):
        sim_data = run_simulate(cfg)
        _save_simulation(sim_data, output_dir)
        log.info("Saved simulation.npz")
        if stage == "simulate":
            return

    # Load from .npz if we didn't just simulate
    if sim_data is None:
        sim_data = _load_simulation(output_dir)

    # Stage 2: Evaluate
    df = None
    if stage in ("all", "evaluate"):
        df = run_evaluate(cfg, sim_data, output_dir)

    # Load CSV if we didn't just evaluate
    if df is None:
        csv_path = output_dir / "simulation_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No metrics CSV at {csv_path}. Run with stage=all or stage=evaluate first.")
        df = pd.read_csv(csv_path)

    # Stage 3: Plot (runs for all, evaluate, and plot)
    run_plots(cfg, sim_data, df, output_dir)


if __name__ == "__main__":
    main()
