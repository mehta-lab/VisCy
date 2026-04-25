"""Signal gradient and sharpness features."""

import numpy as np
from scipy.ndimage import laplace
from scipy.stats import entropy as scipy_entropy
from skimage.filters import sobel

__all__ = ["gradient_features"]


def gradient_features(image: np.ndarray, nuclear_mask: np.ndarray) -> dict[str, float]:
    """Problem 7: Gradient changes in the fluorescence signal.

    Parameters
    ----------
    image : np.ndarray, shape (Y, X)
        Single-channel fluorescence patch.
    nuclear_mask : np.ndarray, shape (Y, X), bool or int
        Binary nuclear mask — used only for ``nucleus_to_cytoplasm_ratio``.

    Returns
    -------
    dict[str, float]
        ``gradient_mean`` — mean Sobel gradient magnitude over the full patch.
        ``gradient_std`` — std of gradient magnitude.
        ``gradient_p95`` — 95th percentile of gradient magnitude (sharpest edges).
        ``laplacian_variance`` — variance of the discrete Laplacian over the full patch
        (high = sharp, well-defined signal boundaries).
        ``gradient_entropy`` — Shannon entropy of the gradient magnitude histogram.
        ``nucleus_mean_intensity`` — mean intensity of pixels inside the nuclear mask.
        Directly reflects the brightness of nuclear signal independent of background.
        ``cytoplasm_mean_intensity`` — mean intensity of all pixels outside the nuclear mask.
        ``nucleus_to_cytoplasm_ratio`` — mean intensity inside the nuclear mask divided by the
        mean intensity of all pixels outside the nuclear mask. High = bright nuclear signal
        against a dark background; values < 1 = cytoplasmic signal brighter than nuclear.
    """
    out: dict[str, float] = {}

    grad = sobel(image)
    grad_flat = grad.ravel()
    out["gradient_mean"] = float(grad_flat.mean())
    out["gradient_std"] = float(grad_flat.std())
    out["gradient_p95"] = float(np.percentile(grad_flat, 95))

    out["laplacian_variance"] = float(laplace(image).var())

    hist, _ = np.histogram(grad_flat, bins=64)
    out["gradient_entropy"] = float(scipy_entropy(hist + 1e-10))

    mask_bool = nuclear_mask.astype(bool)
    nucleus_pixels = image[mask_bool]
    if nucleus_pixels.size > 0:
        background_pixels = image[~mask_bool]
        mean_nucleus = float(nucleus_pixels.mean())
        mean_bg = float(background_pixels.mean()) if background_pixels.size > 0 else 1e-10
        out["nucleus_mean_intensity"] = mean_nucleus
        out["cytoplasm_mean_intensity"] = mean_bg
        out["nucleus_to_cytoplasm_ratio"] = float(mean_nucleus / (mean_bg + 1e-10))
    else:
        out["nucleus_mean_intensity"] = 0.0
        out["cytoplasm_mean_intensity"] = float(image.mean())
        out["nucleus_to_cytoplasm_ratio"] = 1.0

    return out
