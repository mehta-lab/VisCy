"""Radial distribution and concentric ring uniformity features."""

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.signal import find_peaks

__all__ = ["radial_distribution_features", "concentric_uniformity_features"]

_N_RADIAL_BINS = 8
_N_SECTORS = 8


def _radial_profile(
    image: np.ndarray, cy: float, cx: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity and fraction of total intensity per concentric ring.

    Uses all pixels in the patch, centred on (cy, cx).
    """
    h, w = image.shape
    ys, xs = np.mgrid[:h, :w]
    ys_flat = ys.ravel()
    xs_flat = xs.ravel()
    pixels_flat = image.ravel()
    radii = np.hypot(ys_flat - cy, xs_flat - cx)
    edges = np.linspace(0, radii.max() + 1e-6, n_bins + 1)
    total = pixels_flat.sum() + 1e-10
    bin_means = np.zeros(n_bins)
    bin_fracs = np.zeros(n_bins)
    for i in range(n_bins):
        idx = (radii >= edges[i]) & (radii < edges[i + 1])
        ring_pixels = pixels_flat[idx]
        if ring_pixels.size > 0:
            bin_means[i] = ring_pixels.mean()
            bin_fracs[i] = ring_pixels.sum() / total
    return bin_means, bin_fracs


def radial_distribution_features(
    image: np.ndarray, nuclear_mask: np.ndarray, n_bins: int = _N_RADIAL_BINS
) -> dict[str, float]:
    """Problem 1: Radial distribution of fluorescence signal.

    Parameters
    ----------
    image : np.ndarray, shape (Y, X)
        Single-channel fluorescence patch.
    nuclear_mask : np.ndarray, shape (Y, X), bool or int
        Binary nuclear mask — used only to locate the nuclear centroid.
    n_bins : int
        Number of concentric radial bins.

    Returns
    -------
    dict[str, float]
        ``radial_frac_bin{i}`` — fraction of total intensity in each ring.
        ``radial_frac_cv`` — CV across bins (high = signal concentrated in few rings).
        ``radial_slope`` — slope of a linear fit to mean intensity vs radius, negated so
        that positive values indicate centre-bright signal (intensity decreases outward)
        and negative values indicate edge-bright / boundary signal (intensity increases
        outward). Normalised by the mean intensity so it is scale-invariant.
        ``com_offset_norm`` — intensity centre-of-mass offset from nuclear centroid,
        normalised by equivalent circle radius (high = signal on one side).
        ``angular_cv`` — CV of mean intensity across 8 angular sectors
        (high = signal concentrated in one angular direction).
    """
    mask_bool = nuclear_mask.astype(bool)
    cy, cx = center_of_mass(mask_bool)

    bin_means, bin_fracs = _radial_profile(image, cy, cx, n_bins)

    out: dict[str, float] = {}
    for i in range(n_bins):
        out[f"radial_frac_bin{i}"] = float(bin_fracs[i])
    out["radial_frac_cv"] = float(bin_fracs.std() / (bin_fracs.mean() + 1e-10))

    # Slope of linear fit to mean intensity vs bin index; negated so positive = centre-bright
    bin_indices = np.arange(n_bins, dtype=float)
    mean_intensity = bin_means.mean() + 1e-10
    slope = float(np.polyfit(bin_indices, bin_means, 1)[0])
    out["radial_slope"] = float(-slope / mean_intensity)

    # Intensity CoM (full patch) offset from nuclear centroid
    h, w = image.shape
    ys_g, xs_g = np.mgrid[:h, :w]
    total = image.sum() + 1e-10
    iy = float((image * ys_g).sum() / total)
    ix = float((image * xs_g).sum() / total)
    com_offset = np.hypot(iy - cy, ix - cx)
    eq_radius = np.sqrt(mask_bool.sum() / np.pi) + 1e-10
    out["com_offset_norm"] = float(com_offset / eq_radius)

    # Angular asymmetry across _N_SECTORS sectors (all patch pixels)
    ys_flat = ys_g.ravel()
    xs_flat = xs_g.ravel()
    pixels_flat = image.ravel()
    angles = np.arctan2(ys_flat - cy, xs_flat - cx)
    edges_a = np.linspace(-np.pi, np.pi, _N_SECTORS + 1)
    sector_means = np.zeros(_N_SECTORS)
    for s in range(_N_SECTORS):
        idx = (angles >= edges_a[s]) & (angles < edges_a[s + 1])
        sector_pixels = pixels_flat[idx]
        sector_means[s] = sector_pixels.mean() if sector_pixels.size > 0 else 0.0
    out["angular_cv"] = float(sector_means.std() / (sector_means.mean() + 1e-10))

    return out


def concentric_uniformity_features(
    image: np.ndarray, nuclear_mask: np.ndarray, n_bins: int = 16
) -> dict[str, float]:
    """Problem 3: Uniformity of concentric ring pattern (ER-like structures).

    Parameters
    ----------
    image : np.ndarray, shape (Y, X)
        Single-channel fluorescence patch.
    nuclear_mask : np.ndarray, shape (Y, X), bool or int
        Binary nuclear mask — used only to locate the nuclear centroid.
    n_bins : int
        Number of radial bins for the profile (higher = finer resolution).

    Returns
    -------
    dict[str, float]
        ``radial_profile_cv`` — CV of the radial intensity profile
        (low = uniform ring brightness).
        ``radial_dominant_freq`` — index of dominant FFT frequency in the profile
        (1 = one bright ring, 2 = two rings, etc.).
        ``radial_spectral_cv`` — CV of FFT amplitudes (low = one dominant frequency).
        ``radial_autocorr_lag1`` — lag-1 autocorrelation of profile
        (high = slowly varying / smooth rings).
        ``peak_spacing_cv`` — CV of distances between intensity peaks in the profile
        (low = evenly spaced rings). NaN if fewer than 2 peaks found.
    """
    mask_bool = nuclear_mask.astype(bool)
    cy, cx = center_of_mass(mask_bool)
    bin_means, _ = _radial_profile(image, cy, cx, n_bins)
    out: dict[str, float] = {}

    out["radial_profile_cv"] = float(bin_means.std() / (bin_means.mean() + 1e-10))

    profile = bin_means - bin_means.mean()
    fft_amps = np.abs(np.fft.rfft(profile))[1:]  # skip DC
    if fft_amps.sum() > 1e-10:
        out["radial_dominant_freq"] = float(np.argmax(fft_amps) + 1)
        out["radial_spectral_cv"] = float(fft_amps.std() / (fft_amps.mean() + 1e-10))
    else:
        out["radial_dominant_freq"] = 0.0
        out["radial_spectral_cv"] = 0.0

    if len(bin_means) > 2 and bin_means.std() > 1e-10:
        out["radial_autocorr_lag1"] = float(
            np.corrcoef(bin_means[:-1], bin_means[1:])[0, 1]
        )
    else:
        out["radial_autocorr_lag1"] = 0.0

    peaks, _ = find_peaks(bin_means)
    if len(peaks) >= 2:
        spacings = np.diff(peaks).astype(float)
        out["peak_spacing_cv"] = float(spacings.std() / (spacings.mean() + 1e-10))
    else:
        out["peak_spacing_cv"] = float("nan")

    return out
