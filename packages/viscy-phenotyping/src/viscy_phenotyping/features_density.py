"""Signal packing density features."""

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening

__all__ = ["density_features"]

_N_GRANULARITY_SCALES = 8


def density_features(image: np.ndarray) -> dict[str, float]:
    """Problem 4: Signal packing density.

    Parameters
    ----------
    image : np.ndarray, shape (Y, X)
        Single-channel fluorescence patch.

    Returns
    -------
    dict[str, float]
        ``binary_area_fraction`` — fraction of patch pixels above Otsu threshold.
        ``spot_count`` — number of connected components in thresholded signal.
        ``spot_mean_area`` — mean area of those components.
        ``spot_max_area`` — largest component area.
        ``spot_density`` — spot count per patch pixel.
        ``granularity_{r}`` — fraction of signal removed by morphological opening
        with disk of radius r (r = 1..8); high value at small r = fine-grained/dense signal.
    """
    out: dict[str, float] = {}
    pixels = image.ravel()

    try:
        thresh = threshold_otsu(pixels)
    except ValueError:
        thresh = pixels.mean()
    binary = image > thresh
    out["binary_area_fraction"] = float(binary.sum() / image.size)

    labeled = label(binary)
    props = regionprops(labeled)
    if props:
        areas = np.array([p.area for p in props])
        out["spot_count"] = float(len(props))
        out["spot_mean_area"] = float(areas.mean())
        out["spot_max_area"] = float(areas.max())
        out["spot_density"] = float(len(props) / image.size)
    else:
        out.update(spot_count=0.0, spot_mean_area=0.0, spot_max_area=0.0, spot_density=0.0)

    # Granularity spectrum
    lo, hi = pixels.min(), pixels.max()
    img_uint8 = (
        ((image - lo) / (hi - lo + 1e-10) * 255).clip(0, 255).astype(np.uint8)
        if hi > lo
        else np.zeros_like(image, dtype=np.uint8)
    )
    baseline = float(img_uint8.sum()) + 1e-10
    prev_sum = baseline
    for r in range(1, _N_GRANULARITY_SCALES + 1):
        opened = opening(img_uint8, disk(r))
        curr_sum = float(opened.sum())
        out[f"granularity_{r}"] = float((prev_sum - curr_sum) / baseline)
        prev_sum = curr_sum

    return out
