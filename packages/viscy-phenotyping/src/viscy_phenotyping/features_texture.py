"""Signal homogeneity and texture features."""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

__all__ = ["texture_features"]

_GLCM_DISTANCES = [1, 3]
_GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_GLCM_LEVELS = 64
_LBP_RADIUS = 2
_LBP_POINTS = 8 * _LBP_RADIUS


def _scale_to_uint(image: np.ndarray, levels: int) -> np.ndarray:
    """Scale image to [0, levels-1] using the full image range."""
    lo, hi = image.min(), image.max()
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - lo) / (hi - lo) * (levels - 1)).clip(0, levels - 1).astype(np.uint8)


def texture_features(image: np.ndarray) -> dict[str, float]:
    """Problem 2: Homogeneity of fluorescence signal.

    Parameters
    ----------
    image : np.ndarray, shape (Y, X)
        Single-channel fluorescence patch.

    Returns
    -------
    dict[str, float]
        ``intensity_mean`` — mean intensity of all patch pixels.
        ``intensity_median`` — median intensity of all patch pixels.
        ``intensity_cv`` — coefficient of variation of all patch pixels.
        ``intensity_entropy`` — Shannon entropy of the intensity histogram.
        ``glcm_{prop}_mean/std`` — Haralick GLCM features averaged/spread over
        distances and angles: contrast, dissimilarity, homogeneity, energy,
        correlation, ASM.
        ``lbp_entropy`` — entropy of the Local Binary Pattern histogram.
        ``lbp_energy`` — energy of the LBP histogram.
    """
    out: dict[str, float] = {}
    pixels = image.ravel()

    out["intensity_mean"] = float(pixels.mean())
    out["intensity_median"] = float(np.median(pixels))
    out["intensity_cv"] = float(pixels.std() / (pixels.mean() + 1e-10))
    hist, _ = np.histogram(pixels, bins=64)
    out["intensity_entropy"] = float(scipy_entropy(hist + 1e-10))

    scaled = _scale_to_uint(image, _GLCM_LEVELS)
    glcm = graycomatrix(
        scaled,
        distances=_GLCM_DISTANCES,
        angles=_GLCM_ANGLES,
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        vals = graycoprops(glcm, prop).ravel()
        out[f"glcm_{prop}_mean"] = float(vals.mean())
        out[f"glcm_{prop}_std"] = float(vals.std())

    lbp = local_binary_pattern(_scale_to_uint(image, 256), _LBP_POINTS, _LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=_LBP_POINTS + 2, density=True)
    out["lbp_entropy"] = float(scipy_entropy(lbp_hist + 1e-10))
    out["lbp_energy"] = float((lbp_hist**2).sum())

    return out
