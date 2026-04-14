"""Per-position time-series evaluation of virtual staining predictions.

Computes pixel-level quality metrics (PCC, PSNR, SSIM) and resolution
metrics (FSC, DCR) at each timepoint from OME-Zarr stores, producing
per-position CSVs and plots.
"""

import logging
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iohub.ngff import open_ome_zarr
from matplotlib.gridspec import GridSpec
from omegaconf import DictConfig
from scipy.stats import median_abs_deviation

try:
    from cubic.cuda import ascupy, asnumpy, get_array_module, get_device, to_same_device
    from cubic.metrics import dcr_resolution, fsc_resolution, skimage_metrics
    from cubic.metrics.bandlimited import (
        _APODIZATION_FNS,
        band_limited_pcc,
        band_limited_ssim,
        estimate_noise_floor,
        frc_weights,
        otf_cutoff,
        radial_power_spectrum,
        spectral_weights,
    )
    from cubic.metrics.bandlimited import (
        spectral_pcc as _spectral_pcc,
    )
    from cubic.metrics.bandlimited import (
        spectral_pcc_frcw as _spectral_pcc_frcw,
    )
    from cubic.metrics.spectral.dcr import dcr_curve
    from cubic.metrics.spectral.radial import radial_bin_id, radial_edges
except ImportError:
    ascupy = None  # type: ignore[assignment]
    asnumpy = None  # type: ignore[assignment]
    get_device = None  # type: ignore[assignment]
    to_same_device = None  # type: ignore[assignment]
    get_array_module = None  # type: ignore[assignment]
    dcr_resolution = None  # type: ignore[assignment]
    fsc_resolution = None  # type: ignore[assignment]
    skimage_metrics = None  # type: ignore[assignment]
    _APODIZATION_FNS = None  # type: ignore[assignment]
    otf_cutoff = None  # type: ignore[assignment]
    frc_weights = None  # type: ignore[assignment]
    band_limited_pcc = None  # type: ignore[assignment]
    spectral_weights = None  # type: ignore[assignment]
    band_limited_ssim = None  # type: ignore[assignment]
    estimate_noise_floor = None  # type: ignore[assignment]
    radial_power_spectrum = None  # type: ignore[assignment]
    _spectral_pcc = None  # type: ignore[assignment]
    _spectral_pcc_frcw = None  # type: ignore[assignment]
    dcr_curve = None  # type: ignore[assignment]
    radial_edges = None  # type: ignore[assignment]
    radial_bin_id = None  # type: ignore[assignment]


def corr_coef(a, b, mask=None):
    """Pearson correlation coefficient (numpy/cupy, with optional mask)."""
    if get_device(a) != get_device(b):
        raise ValueError(f"Images must be on same device, got {get_device(a)} and {get_device(b)}")
    if a.shape != b.shape:
        raise ValueError(f"Inputs must be same shape, got {a.shape} and {b.shape}")
    if mask is not None:
        a = a[mask]
        b = b[mask]
    num = (a - a.mean()) * (b - b.mean())
    denom = a.std() * b.std()
    return float(num.mean() / denom) if float(denom) > 0 else float("nan")


def psnr(image_true, image_test, data_range=None, mask=None):
    """Peak signal to noise ratio (PSNR)."""
    return float(skimage_metrics.psnr(image_true, image_test, data_range=data_range, mask=mask))


def ssim(im1, im2, data_range=None):
    """Mean structural similarity index (SSIM)."""
    return float(skimage_metrics.ssim(im1, im2, data_range=data_range))


log = logging.getLogger(__name__)


def _wiener_spectral_weights(
    power: np.ndarray,
    noise_floor: float,
    radii: np.ndarray | None = None,
    cutoff: float | None = None,
) -> np.ndarray:
    """Wiener-style per-bin weights: P² / (P² + N²).

    Unlike subtract-and-normalize weights, these are inherently
    bounded [0, 1] and degrade smoothly as signal dims.
    """
    n2 = noise_floor**2
    w = power**2 / (power**2 + n2)
    if cutoff is not None and radii is not None:
        w[radii > cutoff] = 0.0
    return w.astype(np.float32)


def _snr_adaptive_weights(
    power: np.ndarray,
    noise_floor: float,
    radii: np.ndarray | None = None,
    cutoff: float | None = None,
    method: str = "snr_squared",
) -> np.ndarray:
    """SNR-adaptive per-bin weights that strongly favor high-SNR bins.

    Unlike Wiener weights (which saturate near 1 for SNR>3), these
    provide strong differentiation across the full SNR range.

    Methods
    -------
    snr_squared : w = max(0, SNR - 1)^2. 10000:1 ratio at SNR=100 vs 1.
    log_snr : w = max(0, log2(SNR)). 6.6:1 ratio at SNR=100 vs 1.
    """
    snr = power / max(noise_floor, 1e-30)
    if method == "snr_squared":
        w = np.maximum(snr - 1.0, 0.0) ** 2
    elif method == "log_snr":
        w = np.maximum(np.log2(np.maximum(snr, 1.0)), 0.0)
    else:
        raise ValueError(f"Unknown SNR-adaptive method: {method!r}")
    if cutoff is not None and radii is not None:
        w[radii > cutoff] = 0.0
    return w.astype(np.float32)


def _spectral_pcc_fixed_noise(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: list[float],
    noise_floor: float,
    bin_delta: float = 1.0,
    cutoff: float | None = None,
    apodization: str = "tukey",
    weighting: str = "subtract",
    nbins_low: int = 0,
) -> float:
    """Spectral PCC with a pre-computed (frozen) noise floor.

    Same as ``spectral_pcc`` but uses ``noise_floor`` instead of
    estimating it from the target's high-frequency tail. This prevents
    the noise floor from tracking signal down under photobleaching.
    """
    from cubic.metrics.bandlimited import _APODIZATION_FNS, _normalize_spacing

    spacing_seq = _normalize_spacing(spacing, prediction.ndim)
    apo_fn = _APODIZATION_FNS[apodization]

    pred = prediction.astype(np.float32) - np.mean(prediction)
    targ = target.astype(np.float32) - np.mean(target)
    pred = apo_fn(pred)
    targ = apo_fn(targ)

    F_pred = np.fft.fftn(pred)
    F_targ = np.fft.fftn(targ)

    # Power spectrum of target for weights (but use frozen noise floor)
    radii, power = radial_power_spectrum(target, spacing=spacing_seq, bin_delta=bin_delta)
    if weighting == "wiener":
        w_bins = _wiener_spectral_weights(power, noise_floor, radii=radii, cutoff=cutoff)
    elif weighting in ("snr_squared", "log_snr"):
        w_bins = _snr_adaptive_weights(power, noise_floor, radii=radii, cutoff=cutoff, method=weighting)
    else:
        w_bins = spectral_weights(radii, power, noise_floor, cutoff=cutoff)

    # Low-k exclusion (DC / illumination / background)
    _nbl = min(nbins_low, len(w_bins))
    if _nbl > 0:
        w_bins[:_nbl] = 0.0
    if float(w_bins.max().item()) == 0.0:
        return 0.0

    edges_cpu, _ = radial_edges(prediction.shape, bin_delta=bin_delta, spacing=spacing_seq)
    edges = to_same_device(edges_cpu, prediction)
    bid = radial_bin_id(prediction.shape, edges, spacing=spacing_seq)

    xp = get_array_module(prediction)
    w_bins_dev = xp.asarray(w_bins) if xp is not np else w_bins

    W = np.zeros_like(bid, dtype=np.float32)
    valid = bid >= 0
    W[valid] = w_bins_dev[bid[valid]]

    cross = np.real(F_pred.ravel() * np.conj(F_targ.ravel()))
    num = float(asnumpy(np.sum(W * cross)))
    denom_pred = float(asnumpy(np.sum(W * np.abs(F_pred.ravel()) ** 2)))
    denom_targ = float(asnumpy(np.sum(W * np.abs(F_targ.ravel()) ** 2)))
    denom = np.sqrt(denom_pred * denom_targ)

    if denom < 1e-12:
        return 0.0
    return float(np.clip(num / denom, -1.0, 1.0))


def _prepare_masked_inputs(
    gt_f: np.ndarray,
    pred_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, float]:
    """Create foreground mask and mean-filled arrays for FFT metrics.

    GT may have zero-valued voxels from registration corrections. For pixel
    metrics, a boolean mask excludes these regions. For FFT metrics, zeros
    are replaced with the per-image foreground mean so that after internal
    mean subtraction they become spectrally invisible.
    """
    mask_bool = gt_f > 0
    has_zeros = not bool(mask_bool.all())
    if has_zeros:
        gt_filled = gt_f.copy()
        pred_filled = pred_f.copy()
        gt_filled[~mask_bool] = float(gt_f[mask_bool].mean())
        pred_filled[~mask_bool] = float(pred_f[mask_bool].mean())
        data_range = float(gt_f[mask_bool].max() - gt_f[mask_bool].min())
        zero_frac = 1.0 - float(mask_bool.sum()) / float(mask_bool.size)
        return gt_filled, pred_filled, mask_bool, data_range, zero_frac
    data_range = float(gt_f.max() - gt_f.min())
    return gt_f, pred_f, None, data_range, 0.0


def estimate_gt_noise_floor(
    gt: np.ndarray,
    spacing: list[float],
    spectral_pcc_kwargs: dict,
) -> float:
    """Estimate the spectral noise floor from a GT volume.

    Call this once on t=0 (high-SNR) and reuse for all timepoints.
    Handles zero-padded registration artifacts via mean-fill.
    """
    gt_f = ascupy(gt.astype(np.float32))
    # Mean-fill zeros before power spectrum estimation
    mask = gt_f > 0
    if not bool(mask.all()):
        gt_f = gt_f.copy()
        gt_f[~mask] = float(gt_f[mask].mean())
    bin_delta = spectral_pcc_kwargs.get("bin_delta", 1.0)
    tail_fraction = spectral_pcc_kwargs.get("tail_fraction", 0.2)
    radii, power = radial_power_spectrum(gt_f, spacing=spacing, bin_delta=bin_delta)
    return estimate_noise_floor(radii, power, tail_fraction=tail_fraction)


def compute_gt_reliability(
    gt_2d: np.ndarray,
    spacing_2d: list[float],
    dcr_kwargs: dict,
) -> tuple[float, float]:
    """Compute DCR A₀ and r₀ from a 2D GT slice for reliability estimation.

    Runs DCR step-2 (unfiltered decorrelation curve) on the GT mid-Z slice
    and extracts the peak amplitude (A₀) and peak location (r₀). A₀ tracks
    image SNR/reliability: high when structure beats noise, ~0 when noise
    dominates.

    Parameters
    ----------
    gt_2d : np.ndarray
        Ground truth 2D slice (Y, X).
    spacing_2d : list[float]
        Pixel spacing [y, x] in physical units.
    dcr_kwargs : dict
        DCR configuration from Hydra config.

    Returns
    -------
    tuple[float, float]
        (A0, r0). Returns (0.0, 0.0) if no peak found or image is empty.
    """
    gt_f = ascupy(gt_2d.astype(np.float32))
    mask = np.isfinite(gt_f) & (gt_f != 0)
    if mask.sum() == 0:
        return 0.0, 0.0
    if not bool(mask.all()):
        gt_f = gt_f.copy()
        gt_f[~mask] = float(gt_f[mask].mean())
    # Use default highpass sweep; take the first valid peak (highest A₀)
    kw = {
        k: v
        for k, v in dcr_kwargs.items()
        if k in ("num_radii", "num_highpass", "windowing", "refine", "min_amplitude")
    }
    _resolution, _radii, _curves, all_peaks = dcr_curve(gt_f, spacing=spacing_2d, **kw)
    # Find first peak with valid amplitude (skip failed peaks at A=0)
    if len(all_peaks) > 0:
        valid = all_peaks[:, 1] > 0
        if valid.any():
            idx = int(np.argmax(valid))  # first valid
            return float(all_peaks[idx, 1]), float(all_peaks[idx, 0])
    return 0.0, 0.0


def _butterworth_lp(k_rad: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    """Amplitude Butterworth low-pass: H(k) = 1 / sqrt(1 + (k/k_c)^(2n))."""
    return 1.0 / np.sqrt(1.0 + (k_rad / max(cutoff, 1e-30)) ** (2 * order))


def _trimmed_mad_sigma2(arr: np.ndarray, trim_quantile: float = 0.85) -> tuple[float, int]:
    """Estimate noise variance via trimmed MAD.

    Trims top (1-trim_quantile) of |arr| by absolute magnitude to exclude
    structure, then computes (1.4826 * MAD)^2 on the remaining pixels.

    Returns (sigma2, n_kept).
    """
    flat = asnumpy(arr).ravel()
    threshold = np.quantile(np.abs(flat), trim_quantile)
    kept = flat[np.abs(flat) <= threshold]
    n_kept = len(kept)
    if n_kept < 10:
        return float(np.var(flat)), n_kept
    mad = float(median_abs_deviation(kept, scale="normal"))
    return mad**2, n_kept


def multiband_ev_score(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: list[float],
    band_edges: list[float] | None = None,
    filter_order: int = 2,
    apodization: str = "tukey",
    noise_corrected: bool = True,
) -> tuple[float, dict[str, object]]:
    """Multi-band explainable-variance score.

    Decomposes pred/target into radial frequency bands, estimates per-band
    noise and explainable variance, and returns an EV-weighted aggregate.

    Parameters
    ----------
    prediction, target : np.ndarray
        Images (2D or 3D, same shape).
    spacing : list[float]
        Pixel/voxel spacing in physical units.
    band_edges : list[float] or None
        Band boundary frequencies in cy/physical-unit. Nyquist is appended
        automatically. Default: [0.0, 0.3, 0.7, 1.2].
    filter_order : int
        Butterworth filter order.
    apodization : str
        Apodization window type.
    noise_corrected : bool
        If True, noise-corrected EV score (Multiband_EV_NC).
        If False, EV-weighted PCC (Multiband_EV_PCC).

    Returns
    -------
    score : float
        EV-weighted aggregate score.
    details : dict
        Per-band and global diagnostics.
    """
    xp = get_array_module(target)

    # Radial Nyquist (inscribed sphere)
    k_nyq = min(1.0 / (2.0 * s) for s in spacing)
    if band_edges is None:
        band_edges = [0.0, 0.3, 0.7, 1.2]
    edges = list(band_edges) + [k_nyq]
    n_bands = len(edges) - 1

    # Mean-center and apodize
    apo_fn = _APODIZATION_FNS[apodization]
    pred = prediction.astype(np.float32) - xp.mean(prediction)
    targ = target.astype(np.float32) - xp.mean(target)
    pred = apo_fn(pred)
    targ = apo_fn(targ)

    # FFT, zero DC
    F_pred = xp.fft.fftn(pred)
    F_targ = xp.fft.fftn(targ)
    # DC index = (0,0,...,0) — set to 0
    F_pred.ravel()[0] = 0.0
    F_targ.ravel()[0] = 0.0

    # Build radial frequency map
    ndim = target.ndim
    freq_components = []
    for i in range(ndim):
        n = target.shape[i]
        freqs = xp.fft.fftfreq(n, d=spacing[i])
        shape = [1] * ndim
        shape[i] = n
        freq_components.append(freqs.reshape(shape))

    k_rad = xp.zeros(target.shape, dtype=np.float32)
    for fc in freq_components:
        k_rad = k_rad + fc.astype(np.float32) ** 2
    k_rad = xp.sqrt(k_rad)

    # Bandpass decomposition
    bp_pred_list = []
    bp_targ_list = []
    for j in range(n_bands):
        k_lo, k_hi = edges[j], edges[j + 1]
        # LP_hi - LP_lo
        if k_lo <= 0:
            H = _butterworth_lp(asnumpy(k_rad), k_hi, filter_order)
        else:
            H_hi = _butterworth_lp(asnumpy(k_rad), k_hi, filter_order)
            H_lo = _butterworth_lp(asnumpy(k_rad), k_lo, filter_order)
            H = H_hi - H_lo
        H = xp.asarray(H) if xp is not np else H
        bp_pred = xp.real(xp.fft.ifftn(F_pred * H))
        bp_targ = xp.real(xp.fft.ifftn(F_targ * H))
        bp_pred_list.append(asnumpy(bp_pred).astype(np.float32))
        bp_targ_list.append(asnumpy(bp_targ).astype(np.float32))

    # σ² estimation: B3 (highest band) first, then per-band for B1/B2
    sigma2 = np.zeros(n_bands)
    n_keep = np.zeros(n_bands, dtype=int)

    # Highest band (B3 or last band) — always noise-dominated
    sigma2[-1], n_keep[-1] = _trimmed_mad_sigma2(bp_targ_list[-1])

    # Mid bands: per-band trimmed MAD
    for j in range(1, n_bands - 1):
        sigma2[j], n_keep[j] = _trimmed_mad_sigma2(bp_targ_list[j])

    # B0: use B3 anchor (structure dominates B0, MAD unreliable)
    sigma2[0] = sigma2[-1]
    n_keep[0] = n_keep[-1]

    # Fit affine 'a' on B0+B1 (or just B0 if only 1 band)
    n_fit = min(2, n_bands)
    x_fit = np.concatenate([bp_pred_list[j].ravel() for j in range(n_fit)])
    y_fit = np.concatenate([bp_targ_list[j].ravel() for j in range(n_fit)])
    x_fit = x_fit - x_fit.mean()
    y_fit = y_fit - y_fit.mean()
    xx = float(np.dot(x_fit, x_fit))
    if xx > 1e-30:
        a = float(np.dot(x_fit, y_fit)) / xx
    else:
        a = 1.0
    a = max(a, 0.0)  # clamp non-negative

    # Per-band scores
    band_details: dict[str, object] = {}
    ev_values = np.zeros(n_bands)
    scores = np.zeros(n_bands)

    for j in range(n_bands):
        bp_t = bp_targ_list[j]
        bp_p = bp_pred_list[j]
        v_j = float(np.var(bp_t))
        ev_j = max(v_j - sigma2[j], 0.0)
        e_pred_j = float(np.mean(bp_p**2))
        e_pred_norm_j = e_pred_j / (sigma2[j] + 1e-30)

        ev_values[j] = ev_j

        if ev_j > 0:
            if noise_corrected:
                residual = bp_t - a * bp_p
                m_j = float(np.mean(residual**2))
                err_j = max(m_j - sigma2[j], 0.0)
                s_j = float(np.clip(1.0 - err_j / ev_j, -1.0, 1.0))
            else:
                # PCC for this band
                bp_t_flat = bp_t.ravel()
                bp_p_flat = bp_p.ravel()
                bp_t_c = bp_t_flat - bp_t_flat.mean()
                bp_p_c = bp_p_flat - bp_p_flat.mean()
                denom = np.sqrt(float(np.dot(bp_t_c, bp_t_c)) * float(np.dot(bp_p_c, bp_p_c)))
                s_j = float(np.dot(bp_t_c, bp_p_c)) / denom if denom > 1e-12 else 0.0
                m_j = 0.0
                err_j = 0.0
        else:
            s_j = 0.0
            m_j = 0.0
            err_j = 0.0

        scores[j] = s_j
        band_label = f"B{j}_{edges[j]:.1f}-{edges[j + 1]:.1f}"
        band_details[band_label] = {
            "EV": ev_j,
            "score": s_j,
            "sigma2": sigma2[j],
            "mse": m_j,
            "var": v_j,
            "E_pred": e_pred_j,
            "E_pred_norm": e_pred_norm_j,
            "n_keep": int(n_keep[j]),
        }

    # Aggregate: EV-weighted
    ev_total = float(np.sum(ev_values))
    if ev_total > 0:
        score = float(np.sum(ev_values * scores)) / ev_total
    else:
        score = 0.0

    band_details["a"] = a
    band_details["EV_total"] = ev_total

    return score, band_details


def compute_timepoint_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: list[float],
    fsc_kwargs: dict,
    dcr_kwargs: dict,
    spectral_pcc_kwargs: dict | None = None,
    bandlimited_kwargs: dict | None = None,
    optics: dict | None = None,
    ref_noise_floor: float | None = None,
) -> dict[str, float]:
    """Compute pixel and resolution metrics for a single timepoint.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth volume (Z, Y, X).
    pred : np.ndarray
        Predicted volume (Z, Y, X).
    spacing : list[float]
        Voxel spacing [z, y, x] in physical units.
    fsc_kwargs : dict
        Keyword arguments for ``fsc_resolution``.
    dcr_kwargs : dict
        Keyword arguments for ``dcr_resolution``.
    spectral_pcc_kwargs : dict or None
        Keyword arguments for ``spectral_pcc``. None to skip.
    bandlimited_kwargs : dict or None
        Keyword arguments for ``band_limited_pcc`` / ``band_limited_ssim``.
        None to skip.
    optics : dict or None
        Microscope optics for OTF-based cutoff. Keys:
        ``numerical_aperture``, ``wavelength_emission``, ``modality``.
        None to skip OTF-based bandlimited metrics.

    Returns
    -------
    dict[str, float]
        Flat dict with keys PCC, PSNR, SSIM, resolution metrics,
        and bandlimited variants (DCR, FSC, OTF suffixed).
    """
    gt_f = ascupy(gt.astype(np.float32))
    pred_f = ascupy(pred.astype(np.float32))

    # Handle zero-padded registration artifacts in GT
    gt_filled, pred_filled, mask, data_range, zero_frac = _prepare_masked_inputs(gt_f, pred_f)

    # Pixel metrics: use original arrays + mask to exclude zero regions
    # Note: SSIM with 3D mask fails in cucim's morphology.erosion, so skip mask for SSIM
    metrics: dict[str, float] = {
        "PCC": corr_coef(gt_f, pred_f, mask=mask),
        "PSNR": psnr(gt_f, pred_f, data_range=data_range, mask=mask),
        "SSIM": ssim(gt_f, pred_f, data_range=data_range),
        "zero_frac": zero_frac,
    }

    # FFT metrics: use mean-filled arrays (zeros become spectrally invisible)
    fsc = fsc_resolution(pred_filled, gt_filled, spacing=spacing, **fsc_kwargs)
    metrics["FSC_XY"] = fsc["xy"]
    metrics["FSC_Z"] = fsc["z"]

    fsc_gt = fsc_resolution(gt_filled, spacing=spacing, **fsc_kwargs)
    metrics["FSC_GT_XY"] = fsc_gt["xy"]
    metrics["FSC_GT_Z"] = fsc_gt["z"]

    dcr = dcr_resolution(pred_filled, spacing=spacing, **dcr_kwargs)
    metrics["DCR_XY"] = dcr["xy"]
    metrics["DCR_Z"] = dcr["z"]

    # Pre-compute OTF cutoff for use by both spectral PCC and bandlimited metrics
    otf_cut = None
    if optics is not None:
        otf_cut = otf_cutoff(
            optics["numerical_aperture"],
            optics["wavelength_emission"],
            modality=optics.get("modality", "widefield"),
        )

    if spectral_pcc_kwargs is not None:
        # Filter out frcw_* keys that spectral_pcc doesn't accept
        spcc_kw = {k: v for k, v in spectral_pcc_kwargs.items() if not k.startswith("frcw_")}
        metrics["Spectral_PCC"] = float(_spectral_pcc(pred_filled, gt_filled, spacing=spacing, **spcc_kw))
        if otf_cut is not None:
            metrics["Spectral_PCC_OTF"] = float(
                _spectral_pcc(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=otf_cut,
                    **spcc_kw,
                )
            )
        # Fixed noise floor variant (anchored to t=0)
        if ref_noise_floor is not None:
            fixed_kw = {
                k: v for k, v in spectral_pcc_kwargs.items() if k in ("bin_delta", "cutoff", "apodization", "nbins_low")
            }
            metrics["Spectral_PCC_Fixed"] = float(
                _spectral_pcc_fixed_noise(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    noise_floor=ref_noise_floor,
                    **fixed_kw,
                )
            )
        # Per-timepoint noise floor (shared by Wiener, SNR², and log-SNR)
        shared_kw = {
            k: v for k, v in spectral_pcc_kwargs.items() if k in ("bin_delta", "cutoff", "apodization", "nbins_low")
        }
        bin_delta_tp = shared_kw.get("bin_delta", 1.0)
        tail_frac_tp = spectral_pcc_kwargs.get("tail_fraction", 0.2)
        radii_tp, power_tp = radial_power_spectrum(gt_filled, spacing=spacing, bin_delta=bin_delta_tp)
        nf_tp = estimate_noise_floor(radii_tp, power_tp, tail_fraction=tail_frac_tp)

        # k90 diagnostic: frequency below which 90% of weight mass lives
        w_bins_diag = spectral_weights(radii_tp, power_tp, nf_tp, cutoff=shared_kw.get("cutoff"))
        _nbl_diag = min(shared_kw.get("nbins_low", 0), len(w_bins_diag))
        if _nbl_diag > 0:
            w_bins_diag[:_nbl_diag] = 0.0
        edges_diag, _ = radial_edges(gt_filled.shape, bin_delta=bin_delta_tp, spacing=spacing)
        edges_dev = to_same_device(edges_diag, gt_filled)
        bid_diag = radial_bin_id(gt_filled.shape, edges_dev, spacing=spacing)
        bid_np = asnumpy(bid_diag)
        counts_per_bin = np.bincount(bid_np[bid_np >= 0], minlength=len(w_bins_diag))
        mass = w_bins_diag * counts_per_bin[: len(w_bins_diag)]
        total_mass = mass.sum()
        if total_mass > 0:
            cum_mass = np.cumsum(mass) / total_mass
            k_nyq = min(1.0 / (2.0 * s) for s in spacing)
            k90_idx = int(np.searchsorted(cum_mass, 0.9))
            k90_idx = min(k90_idx, len(radii_tp) - 1)
            metrics["k90"] = float(radii_tp[k90_idx]) / k_nyq
        else:
            metrics["k90"] = 0.0

        metrics["Spectral_PCC_Wiener"] = float(
            _spectral_pcc_fixed_noise(
                pred_filled,
                gt_filled,
                spacing=spacing,
                noise_floor=nf_tp,
                weighting="wiener",
                **shared_kw,
            )
        )
        metrics["Spectral_PCC_SNR2"] = float(
            _spectral_pcc_fixed_noise(
                pred_filled,
                gt_filled,
                spacing=spacing,
                noise_floor=nf_tp,
                weighting="snr_squared",
                **shared_kw,
            )
        )
        metrics["Spectral_PCC_LogSNR"] = float(
            _spectral_pcc_fixed_noise(
                pred_filled,
                gt_filled,
                spacing=spacing,
                noise_floor=nf_tp,
                weighting="log_snr",
                **shared_kw,
            )
        )

        # Multi-band explainable variance metrics
        ev_nc, _ = multiband_ev_score(
            pred_filled,
            gt_filled,
            spacing=spacing,
            noise_corrected=True,
        )
        metrics["Multiband_EV_NC"] = ev_nc

        ev_pcc, _ = multiband_ev_score(
            pred_filled,
            gt_filled,
            spacing=spacing,
            noise_corrected=False,
        )
        metrics["Multiband_EV_PCC"] = ev_pcc
    if bandlimited_kwargs is not None:
        bl_kw = dict(bandlimited_kwargs)
        ssim_extra = {}
        for key in ("win_size", "data_range"):
            if key in bl_kw:
                ssim_extra[key] = bl_kw.pop(key)

        # Filter kwargs without 'method' for explicit-cutoff calls
        otf_kw = {k: v for k, v in bl_kw.items() if k != "method"}

        # DCR-based cutoff (XY) — reuse pre-computed DCR resolution
        dcr_xy_cut = 1.0 / dcr["xy"] if dcr["xy"] > 0 else None
        if dcr_xy_cut is not None:
            metrics["BL_PCC_DCR_XY"] = float(
                band_limited_pcc(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=dcr_xy_cut,
                    **otf_kw,
                )
            )
            metrics["BL_SSIM_DCR_XY"] = float(
                band_limited_ssim(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=dcr_xy_cut,
                    **otf_kw,
                    **ssim_extra,
                )
            )

        # DCR_Z-based cutoff (Z resolution)
        if dcr["z"] > 0:
            dcr_z_cut = 1.0 / dcr["z"]
            metrics["BL_PCC_DCR_Z"] = float(
                band_limited_pcc(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=dcr_z_cut,
                    **otf_kw,
                )
            )
            metrics["BL_SSIM_DCR_Z"] = float(
                band_limited_ssim(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=dcr_z_cut,
                    **otf_kw,
                    **ssim_extra,
                )
            )

        # FSC-based cutoff (XY) — reuse pre-computed FSC resolution
        fsc_xy_cut = 1.0 / fsc["xy"] if fsc.get("xy") and fsc["xy"] > 0 else None
        if fsc_xy_cut is not None:
            metrics["BL_PCC_FSC_XY"] = float(
                band_limited_pcc(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=fsc_xy_cut,
                    **otf_kw,
                )
            )
            metrics["BL_SSIM_FSC_XY"] = float(
                band_limited_ssim(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=fsc_xy_cut,
                    **otf_kw,
                    **ssim_extra,
                )
            )

        # FSC_Z-based cutoff (Z resolution)
        if fsc.get("z") and fsc["z"] > 0:
            fsc_z_cut = 1.0 / fsc["z"]
            metrics["BL_PCC_FSC_Z"] = float(
                band_limited_pcc(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=fsc_z_cut,
                    **otf_kw,
                )
            )
            metrics["BL_SSIM_FSC_Z"] = float(
                band_limited_ssim(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=fsc_z_cut,
                    **otf_kw,
                    **ssim_extra,
                )
            )

        # OTF-based cutoff (pre-computed above, bypasses estimate_cutoff)
        if otf_cut is not None:
            metrics["BL_PCC_OTF"] = float(
                band_limited_pcc(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=otf_cut,
                    **otf_kw,
                )
            )
            metrics["BL_SSIM_OTF"] = float(
                band_limited_ssim(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=otf_cut,
                    **otf_kw,
                    **ssim_extra,
                )
            )

    return metrics


def compute_timepoint_metrics_2d(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: list[float],
    dcr_kwargs: dict,
    spectral_pcc_kwargs: dict | None = None,
    bandlimited_kwargs: dict | None = None,
    optics: dict | None = None,
    ref_noise_floor: float | None = None,
    frozen_frcw_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute 2D pixel and resolution metrics for a single YX slice.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth slice (Y, X).
    pred : np.ndarray
        Predicted slice (Y, X).
    spacing : list[float]
        Pixel spacing [y, x] in physical units.
    dcr_kwargs : dict
        Keyword arguments for ``dcr_resolution``.
    spectral_pcc_kwargs : dict or None
        Keyword arguments for ``spectral_pcc``. None to skip.
    bandlimited_kwargs : dict or None
        Keyword arguments for ``band_limited_pcc`` / ``band_limited_ssim``.
        None to skip.
    optics : dict or None
        Microscope optics for OTF-based cutoff. None to skip.
    frozen_frcw_weights : np.ndarray or None
        Pre-computed FRC weights (from early-window median) for the
        frozen FRCW variant. None to skip.

    Returns
    -------
    dict[str, float]
        Flat dict with ``_2D`` suffixed keys.
    """
    gt_f = ascupy(gt.astype(np.float32))
    pred_f = ascupy(pred.astype(np.float32))

    gt_filled, pred_filled, mask, data_range, _ = _prepare_masked_inputs(gt_f, pred_f)

    metrics: dict[str, float] = {
        "PCC_2D": corr_coef(gt_f, pred_f, mask=mask),
        "PSNR_2D": psnr(gt_f, pred_f, data_range=data_range, mask=mask),
        "SSIM_2D": ssim(gt_f, pred_f, data_range=data_range),
    }

    dcr_val = dcr_resolution(pred_filled, spacing=spacing, **dcr_kwargs)
    metrics["DCR_2D"] = float(dcr_val)

    if spectral_pcc_kwargs is not None:
        # Filter out frcw_* keys that spectral_pcc doesn't accept
        spcc_kw = {k: v for k, v in spectral_pcc_kwargs.items() if not k.startswith("frcw_")}
        metrics["Spectral_PCC_2D"] = float(_spectral_pcc(pred_filled, gt_filled, spacing=spacing, **spcc_kw))
        metrics["Spectral_PCC_Smooth_2D"] = float(
            _spectral_pcc(
                pred_filled,
                gt_filled,
                spacing=spacing,
                smooth=True,
                **spcc_kw,
            )
        )
        # FRCW variant (FRC-as-weight spectral PCC)
        frcw_kw = {k: v for k, v in spectral_pcc_kwargs.items() if k in ("bin_delta", "apodization")}
        metrics["Spectral_PCC_FRCW_2D"] = float(_spectral_pcc_frcw(pred_filled, gt_filled, spacing=spacing, **frcw_kw))
        # Frozen FRCW variant (weights from early-window median)
        if frozen_frcw_weights is not None:
            metrics["Spectral_PCC_FRCW_Frozen_2D"] = float(
                _spectral_pcc_frcw(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    frozen_weights=frozen_frcw_weights,
                    **frcw_kw,
                )
            )

        # Fixed noise floor variant (anchored to t=0)
        if ref_noise_floor is not None:
            fixed_kw_2d = {
                k: v for k, v in spectral_pcc_kwargs.items() if k in ("bin_delta", "cutoff", "apodization", "nbins_low")
            }
            metrics["Spectral_PCC_Fixed_2D"] = float(
                _spectral_pcc_fixed_noise(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    noise_floor=ref_noise_floor,
                    **fixed_kw_2d,
                )
            )
        # Per-timepoint noise floor for Wiener and SNR-adaptive 2D variants
        shared_kw_2d = {
            k: v for k, v in spectral_pcc_kwargs.items() if k in ("bin_delta", "cutoff", "apodization", "nbins_low")
        }
        bd_2d = shared_kw_2d.get("bin_delta", 1.0)
        tf_2d = spectral_pcc_kwargs.get("tail_fraction", 0.2)
        radii_2d, power_2d = radial_power_spectrum(gt_filled, spacing=spacing, bin_delta=bd_2d)
        nf_2d = estimate_noise_floor(radii_2d, power_2d, tail_fraction=tf_2d)

        # k90 diagnostic (2D)
        w_bins_2d = spectral_weights(radii_2d, power_2d, nf_2d, cutoff=shared_kw_2d.get("cutoff"))
        _nbl_2d = min(shared_kw_2d.get("nbins_low", 0), len(w_bins_2d))
        if _nbl_2d > 0:
            w_bins_2d[:_nbl_2d] = 0.0
        edges_2d, _ = radial_edges(gt_filled.shape, bin_delta=bd_2d, spacing=spacing)
        edges_2d_dev = to_same_device(edges_2d, gt_filled)
        bid_2d = radial_bin_id(gt_filled.shape, edges_2d_dev, spacing=spacing)
        bid_2d_np = asnumpy(bid_2d)
        counts_2d = np.bincount(bid_2d_np[bid_2d_np >= 0], minlength=len(w_bins_2d))
        mass_2d = w_bins_2d * counts_2d[: len(w_bins_2d)]
        total_mass_2d = mass_2d.sum()
        if total_mass_2d > 0:
            cum_mass_2d = np.cumsum(mass_2d) / total_mass_2d
            k_nyq_2d = min(1.0 / (2.0 * s) for s in spacing)
            k90_idx_2d = min(int(np.searchsorted(cum_mass_2d, 0.9)), len(radii_2d) - 1)
            metrics["k90_2D"] = float(radii_2d[k90_idx_2d]) / k_nyq_2d
        else:
            metrics["k90_2D"] = 0.0

        metrics["Spectral_PCC_Wiener_2D"] = float(
            _spectral_pcc_fixed_noise(
                pred_filled,
                gt_filled,
                spacing=spacing,
                noise_floor=nf_2d,
                weighting="wiener",
                **shared_kw_2d,
            )
        )
        metrics["Spectral_PCC_SNR2_2D"] = float(
            _spectral_pcc_fixed_noise(
                pred_filled,
                gt_filled,
                spacing=spacing,
                noise_floor=nf_2d,
                weighting="snr_squared",
                **shared_kw_2d,
            )
        )
        metrics["Spectral_PCC_LogSNR_2D"] = float(
            _spectral_pcc_fixed_noise(
                pred_filled,
                gt_filled,
                spacing=spacing,
                noise_floor=nf_2d,
                weighting="log_snr",
                **shared_kw_2d,
            )
        )
        # Multi-band EV 2D
        ev_nc_2d, _ = multiband_ev_score(
            pred_filled,
            gt_filled,
            spacing=spacing,
            noise_corrected=True,
        )
        metrics["Multiband_EV_NC_2D"] = ev_nc_2d
        ev_pcc_2d, _ = multiband_ev_score(
            pred_filled,
            gt_filled,
            spacing=spacing,
            noise_corrected=False,
        )
        metrics["Multiband_EV_PCC_2D"] = ev_pcc_2d

    if bandlimited_kwargs is not None:
        bl_kw = dict(bandlimited_kwargs)
        ssim_extra = {}
        for key in ("win_size", "data_range"):
            if key in bl_kw:
                ssim_extra[key] = bl_kw.pop(key)

        otf_kw = {k: v for k, v in bl_kw.items() if k != "method"}

        # DCR-based cutoff — reuse pre-computed DCR_2D resolution
        dcr_2d_cut = 1.0 / dcr_val if dcr_val > 0 else None
        if dcr_2d_cut is not None:
            metrics["BL_PCC_DCR_2D"] = float(
                band_limited_pcc(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=dcr_2d_cut,
                    **otf_kw,
                )
            )
            metrics["BL_SSIM_DCR_2D"] = float(
                band_limited_ssim(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=dcr_2d_cut,
                    **otf_kw,
                    **ssim_extra,
                )
            )

        # OTF-based cutoff
        if optics is not None:
            otf_cut = otf_cutoff(
                optics["numerical_aperture"],
                optics["wavelength_emission"],
                modality=optics.get("modality", "widefield"),
            )
            metrics["BL_PCC_OTF_2D"] = float(
                band_limited_pcc(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=otf_cut,
                    **otf_kw,
                )
            )
            metrics["BL_SSIM_OTF_2D"] = float(
                band_limited_ssim(
                    pred_filled,
                    gt_filled,
                    spacing=spacing,
                    cutoff=otf_cut,
                    **otf_kw,
                    **ssim_extra,
                )
            )

    return metrics


def evaluate_position(
    pos_name: str,
    pos_gt,
    pos_pred,
    gt_ch_idx: int,
    pred_ch_idx: int,
    spacing: list[float],
    cfg: DictConfig,
) -> pd.DataFrame:
    """Evaluate all timepoints for a single position.

    Parameters
    ----------
    pos_name : str
        Position name for logging.
    pos_gt : Position
        iohub Position object for ground truth.
    pos_pred : Position
        iohub Position object for predictions.
    gt_ch_idx : int
        Channel index for ground truth.
    pred_ch_idx : int
        Channel index for predictions.
    spacing : list[float]
        Voxel spacing [z, y, x].
    cfg : DictConfig
        Hydra config with fsc/dcr kwargs.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timepoint, PCC, PSNR, SSIM, FSC_XY, FSC_Z,
        DCR_XY, DCR_Z.
    """
    fsc_kwargs = dict(cfg.fsc)
    dcr_kwargs = dict(cfg.dcr)
    spectral_pcc_kwargs = dict(cfg.spectral_pcc) if cfg.get("spectral_pcc") else None
    bandlimited_kwargs = dict(cfg.bandlimited) if cfg.get("bandlimited") else None
    optics_kwargs = dict(cfg.optics) if cfg.get("optics") else None

    n_timepoints = pos_gt.data.shape[0]
    rows = []

    # Estimate noise floor from t=0 GT (high SNR) and reuse for all timepoints
    ref_noise_floor = None
    if spectral_pcc_kwargs is not None:
        gt_t0 = np.asarray(pos_gt.data[0, gt_ch_idx])
        ref_noise_floor = estimate_gt_noise_floor(gt_t0, spacing, spectral_pcc_kwargs)
        log.info("  Reference noise floor (t=0): %.4f", ref_noise_floor)

    # Compute frozen FRCW weights from first K=5 frames (median)
    frozen_frcw = None
    if spectral_pcc_kwargs is not None:
        from scipy.ndimage import median_filter

        K = min(5, n_timepoints)
        mid_z_ref = pos_gt.data.shape[2] // 2
        frcw_per_frame = []
        frcw_kw_frozen = {k: v for k, v in spectral_pcc_kwargs.items() if k in ("bin_delta",)}
        nbins_low = spectral_pcc_kwargs.get("frcw_nbins_low", 3)
        smooth_window = spectral_pcc_kwargs.get("frcw_smooth_window", 5)
        for t_ref in range(K):
            gt_t = np.asarray(pos_gt.data[t_ref, gt_ch_idx, mid_z_ref]).astype(np.float32)
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

    for t in range(n_timepoints):
        log.info("  timepoint %d / %d", t + 1, n_timepoints)
        gt_vol = np.asarray(pos_gt.data[t, gt_ch_idx])
        pred_vol = np.asarray(pos_pred.data[t, pred_ch_idx])

        m = compute_timepoint_metrics(
            gt_vol,
            pred_vol,
            spacing,
            fsc_kwargs,
            dcr_kwargs,
            spectral_pcc_kwargs,
            bandlimited_kwargs,
            optics_kwargs,
            ref_noise_floor,
        )

        # 2D metrics from mid-Z slice
        mid_z = gt_vol.shape[0] // 2
        spacing_2d = spacing[1:]  # [y, x]
        m_2d = compute_timepoint_metrics_2d(
            gt_vol[mid_z],
            pred_vol[mid_z],
            spacing_2d,
            dcr_kwargs,
            spectral_pcc_kwargs,
            bandlimited_kwargs,
            optics_kwargs,
            ref_noise_floor,
            frozen_frcw_weights=frozen_frcw,
        )
        m.update(m_2d)

        # DCR A₀ reliability (GT mid-Z slice only, no prediction)
        a0, r0 = compute_gt_reliability(gt_vol[mid_z], spacing_2d, dcr_kwargs)
        m["DCR_A0"] = a0
        m["DCR_r0"] = r0

        m["timepoint"] = t
        rows.append(m)

    df = pd.DataFrame(rows)

    # Compute DCR_A0 reliability weights (per position)
    if "DCR_A0" in df.columns:
        a0_vals = df["DCR_A0"].values
        k_ref = 5  # frames for reference levels
        a_good = float(np.median(a0_vals[:k_ref]))
        a_bad = float(np.median(a0_vals[-k_ref:]))
        eps = 1e-6
        if a_good <= 0:
            df["DCR_w"] = 0.0  # unscorable position
        elif (a_good - a_bad) < eps:
            df["DCR_w"] = 1.0  # no bleaching
        else:
            w = np.clip((a0_vals - a_bad) / (a_good - a_bad), 0.0, 1.0)
            w = np.where(np.isfinite(a0_vals), w, 0.0)
            df["DCR_w"] = w

    cols = ["timepoint"] + [c for c in df.columns if c != "timepoint"]
    return df[cols]


def plot_metrics(
    df: pd.DataFrame,
    pos_name: str,
    output_dir: Path,
    slices: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Plot metrics vs timepoint with optional GT/pred image panels.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics DataFrame with a 'timepoint' column.
    pos_name : str
        Position name (used in title and filename).
    output_dir : Path
        Directory where the plot PNG is saved.
    slices : list of (label, gt_xy, pred_xy) or None
        Optional mid-Z XY slices at selected timepoints. Each tuple
        contains a label string (e.g. "t=0"), a GT 2D array, and a
        pred 2D array. Displayed as image panels above the metric plots.
    """
    all_metrics = [
        "PCC",
        "PSNR",
        "SSIM",
        "Spectral_PCC",
        "Spectral_PCC_OTF",
        "Spectral_PCC_Fixed",
        "Spectral_PCC_Wiener",
        "Spectral_PCC_SNR2",
        "Spectral_PCC_LogSNR",
        "Multiband_EV_NC",
        "Multiband_EV_PCC",
        "BL_PCC_DCR_XY",
        "BL_SSIM_DCR_XY",
        "BL_PCC_DCR_Z",
        "BL_SSIM_DCR_Z",
        "BL_PCC_FSC_XY",
        "BL_SSIM_FSC_XY",
        "BL_PCC_FSC_Z",
        "BL_SSIM_FSC_Z",
        "BL_PCC_OTF",
        "BL_SSIM_OTF",
        "FSC_XY",
        "FSC_Z",
        "FSC_GT_XY",
        "FSC_GT_Z",
        "DCR_XY",
        "DCR_Z",
        "DCR_A0",
        "DCR_r0",
        "DCR_w",
        "PCC_2D",
        "PSNR_2D",
        "SSIM_2D",
        "Spectral_PCC_2D",
        "Spectral_PCC_Smooth_2D",
        "Spectral_PCC_FRCW_2D",
        "Spectral_PCC_FRCW_Frozen_2D",
        "Spectral_PCC_Fixed_2D",
        "Spectral_PCC_Wiener_2D",
        "Spectral_PCC_SNR2_2D",
        "Spectral_PCC_LogSNR_2D",
        "Multiband_EV_NC_2D",
        "Multiband_EV_PCC_2D",
        "DCR_2D",
        "BL_PCC_DCR_2D",
        "BL_SSIM_DCR_2D",
        "BL_PCC_OTF_2D",
        "BL_SSIM_OTF_2D",
        "zero_frac",
    ]
    metrics = [m for m in all_metrics if m in df.columns]
    n = len(metrics)
    ncols = 3
    metric_rows = (n + ncols - 1) // ncols
    img_rows = 2 if slices else 0
    total_rows = img_rows + metric_rows

    fig = plt.figure(figsize=(4 * ncols, 3 * total_rows))
    gs = GridSpec(
        total_rows,
        ncols,
        figure=fig,
        height_ratios=[1] * img_rows + [1] * metric_rows,
    )

    # Image panels (top 2 rows)
    if slices:
        n_slices = min(len(slices), ncols)
        for col in range(n_slices):
            label, gt_xy, pred_xy = slices[col]
            # GT row
            ax_gt = fig.add_subplot(gs[0, col])
            ax_gt.imshow(gt_xy, cmap="gray")
            ax_gt.set_title(f"GT {label}", fontsize=9)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            # Pred row
            ax_pred = fig.add_subplot(gs[1, col])
            ax_pred.imshow(pred_xy, cmap="gray")
            ax_pred.set_title(f"Pred {label}", fontsize=9)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

    # Metric line charts
    t_vals = df["timepoint"].values
    for i, name in enumerate(metrics):
        row = img_rows + i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        vals = df[name].values
        ax.plot(t_vals, vals, marker="o", markersize=2, linewidth=1)
        # Linear fit overlay + stats in title
        mask = np.isfinite(vals)
        if mask.sum() > 1:
            slope, intercept = np.polyfit(t_vals[mask], vals[mask], 1)
            ax.plot(
                t_vals,
                slope * t_vals + intercept,
                color="red",
                linewidth=1,
                linestyle="--",
            )
            y0 = intercept
            yT = slope * t_vals[-1] + intercept
            drop = (y0 - yT) / y0 * 100 if y0 > 0 else 0
            cv = np.std(vals[mask]) / np.mean(vals[mask]) * 100
            ax.set_title(f"{name}\ndrop={drop:.1f}%  CV={cv:.1f}%", fontsize=9)
        else:
            ax.set_title(name, fontsize=9)
        ax.set_xlabel("Timepoint")
        ax.grid(True, alpha=0.3)

    fig.suptitle(pos_name, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics.png", dpi=150)
    plt.close(fig)


def resolve_spacing(pos, cfg: DictConfig) -> list[float]:
    """Read voxel spacing from zarr metadata, falling back to config.

    Parameters
    ----------
    pos : Position
        iohub Position object.
    cfg : DictConfig
        Config with ``spacing`` fallback value.

    Returns
    -------
    list[float]
        Spacing as [z, y, x].
    """
    try:
        scale = pos.scale
        z_idx = pos.get_axis_index("z")
        y_idx = pos.get_axis_index("y")
        x_idx = pos.get_axis_index("x")
        spacing = [scale[z_idx], scale[y_idx], scale[x_idx]]
        if all(s == 1.0 for s in spacing):
            log.warning("Zarr scale is all 1.0, using config spacing: %s", list(cfg.spacing))
            return list(cfg.spacing)
        log.info("Using zarr metadata spacing: %s", spacing)
        return spacing
    except Exception:
        log.warning("Could not read spacing from zarr, using config: %s", list(cfg.spacing))
        return list(cfg.spacing)


def resolve_channel_index(pos, channel_name: str) -> int:
    """Resolve a channel name to its index in the position.

    Parameters
    ----------
    pos : Position
        iohub Position object.
    channel_name : str
        Channel name to look up.

    Returns
    -------
    int
        Channel index.

    Raises
    ------
    ValueError
        If the channel name is not found.
    """
    names = pos.channel_names
    for i, name in enumerate(names):
        if name.lower() == channel_name.lower():
            return i
    raise ValueError(f"Channel '{channel_name}' not found. Available: {names}")


def compute(cfg: DictConfig) -> None:
    """Compute metrics and save CSVs + mid-Z slices."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from contextlib import ExitStack

    allowed_positions = set(cfg.positions) if cfg.get("positions") else None
    two_zarr = cfg.pred_zarr is not None

    with ExitStack() as stack:
        input_store = stack.enter_context(open_ome_zarr(cfg.input_zarr, mode="r"))
        pred_store = stack.enter_context(open_ome_zarr(cfg.pred_zarr, mode="r")) if two_zarr else input_store

        for pos_name, pos_gt in input_store.positions():
            if allowed_positions is not None and pos_name not in allowed_positions:
                log.debug("Skipping position: %s", pos_name)
                continue
            log.info("Processing position: %s", pos_name)

            pos_pred = pred_store[pos_name] if two_zarr else pos_gt
            gt_channel = cfg.gt_channel or cfg.channel
            pred_channel = cfg.pred_channel or cfg.channel
            gt_ch_idx = resolve_channel_index(pos_gt, gt_channel)
            pred_ch_idx = resolve_channel_index(pos_pred, pred_channel)

            spacing = resolve_spacing(pos_gt, cfg)

            df = evaluate_position(pos_name, pos_gt, pos_pred, gt_ch_idx, pred_ch_idx, spacing, cfg)

            pos_dir = output_dir / pos_name
            pos_dir.mkdir(parents=True, exist_ok=True)

            csv_path = pos_dir / "metrics.csv"
            df.to_csv(csv_path, index=False)
            log.info("  Saved %s", csv_path)

            # Extract and save mid-Z XY slices for later plotting
            n_t = pos_gt.data.shape[0]
            n_z = pos_gt.data.shape[2]
            mid_z = n_z // 2
            t_indices = [0, n_t // 2, n_t - 1]
            labels, gt_slices, pred_slices = [], [], []
            for t_idx in t_indices:
                labels.append(f"t={t_idx}")
                gt_slices.append(np.asarray(pos_gt.data[t_idx, gt_ch_idx, mid_z]))
                pred_slices.append(np.asarray(pos_pred.data[t_idx, pred_ch_idx, mid_z]))

            np.savez(
                pos_dir / "slices.npz",
                labels=labels,
                gt=gt_slices,
                pred=pred_slices,
            )
            log.info("  Saved %s/slices.npz", pos_dir)

    log.info("Compute done.")


def plot(cfg: DictConfig) -> None:
    """Generate plots from saved CSVs and slices."""
    output_dir = Path(cfg.output_dir)

    for csv_path in sorted(output_dir.rglob("metrics.csv")):
        pos_dir = csv_path.parent
        pos_name = str(pos_dir.relative_to(output_dir))

        allowed_positions = set(cfg.positions) if cfg.get("positions") else None
        if allowed_positions is not None and pos_name not in allowed_positions:
            continue

        df = pd.read_csv(csv_path)

        slices = None
        slices_path = pos_dir / "slices.npz"
        if slices_path.exists():
            data = np.load(slices_path, allow_pickle=True)
            slices = list(zip(data["labels"], data["gt"], data["pred"]))

        plot_metrics(df, pos_name, pos_dir, slices=slices)
        log.info("  Saved %s/metrics.png", pos_dir)

    log.info("Plot done.")


@hydra.main(
    version_base="1.2",
    config_path="../_configs/spectral_pcc",
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    """Evaluate per-position time-series metrics from OME-Zarr stores."""
    mode = cfg.get("mode", "all")
    if mode in ("compute", "all"):
        compute(cfg)
    if mode in ("plot", "all"):
        plot(cfg)


if __name__ == "__main__":
    main()
