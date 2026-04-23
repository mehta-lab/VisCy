"""Nuclear shape and circularity features."""

import numpy as np
from skimage.measure import find_contours, regionprops
from skimage.morphology import convex_hull_image

__all__ = ["shape_features"]

_N_FOURIER_DESCRIPTORS = 6


def shape_features(mask: np.ndarray) -> dict[str, float]:
    """Problem 6: Circularity of the nuclear shape.

    Operates on the binary nuclear mask only (no intensity image needed).
    Call once per cell without a channel prefix.

    Parameters
    ----------
    mask : np.ndarray, shape (Y, X), bool or int
        Binary nuclear mask.

    Returns
    -------
    dict[str, float]
        ``circularity`` — 4π·area/perimeter² (1.0 = perfect circle).
        ``convexity`` — convex-hull perimeter / object perimeter
        (1.0 = fully convex; < 1 = lobes / indentations).
        ``radial_std_norm`` — std of boundary radii normalised by mean radius
        (high = irregular / multilobed boundary).
        ``fsd_{1..6}`` — Fourier shape descriptor amplitudes normalised by the
        first harmonic (low-order = global shape; high-order = fine lobes).
    """
    out: dict[str, float] = {}
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return out

    props = regionprops(mask_bool.astype(np.uint8))[0]
    area = float(props.area)
    perimeter = float(props.perimeter)
    cy, cx = props.centroid

    out["circularity"] = float(4.0 * np.pi * area / (perimeter**2 + 1e-10))

    hull = convex_hull_image(mask_bool)
    hull_perimeter = float(regionprops(hull.astype(np.uint8))[0].perimeter)
    out["convexity"] = float(hull_perimeter / (perimeter + 1e-10))

    contours = find_contours(mask_bool.astype(float), 0.5)
    if contours:
        contour = max(contours, key=len)
        radii = np.hypot(contour[:, 0] - cy, contour[:, 1] - cx)
        out["radial_std_norm"] = float(radii.std() / (radii.mean() + 1e-10))

        boundary = (contour[:, 1] - cx) + 1j * (contour[:, 0] - cy)
        fsd = np.abs(np.fft.fft(boundary))
        norm = fsd[1] + 1e-10
        for k in range(1, _N_FOURIER_DESCRIPTORS + 1):
            out[f"fsd_{k}"] = float(fsd[k] / norm if k < len(fsd) else 0.0)
    else:
        out["radial_std_norm"] = 0.0
        for k in range(1, _N_FOURIER_DESCRIPTORS + 1):
            out[f"fsd_{k}"] = 0.0

    return out
