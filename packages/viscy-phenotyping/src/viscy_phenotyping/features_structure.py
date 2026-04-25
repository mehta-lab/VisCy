"""Edge density and strand / filament continuity features."""

import numpy as np
from scipy.ndimage import convolve
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.measure import euler_number, label, regionprops
from skimage.morphology import skeletonize

__all__ = ["structure_features"]

_NEIGHBOR_KERNEL = np.ones((3, 3), dtype=np.int32)


def structure_features(
    image: np.ndarray, canny_sigma: float = 1.0
) -> dict[str, float]:
    """Problem 5: Edge count and strand/filament continuity.

    Parameters
    ----------
    image : np.ndarray, shape (Y, X)
        Single-channel fluorescence patch.
    canny_sigma : float
        Gaussian smoothing sigma for Canny edge detection.

    Returns
    -------
    dict[str, float]
        ``edge_density`` — fraction of patch pixels classified as edges by Canny.
        ``n_connected_components`` — number of connected components in thresholded signal.
        ``cc_mean_area``, ``cc_max_area`` — mean / max component area.
        ``signal_euler_number`` — Euler number of binary signal (objects minus holes).
        ``skeleton_length`` — total number of skeleton pixels.
        ``skeleton_branch_points`` — junction pixels in the skeleton (high = complex network).
        ``skeleton_endpoints`` — terminal pixels (high = many broken strand ends).
        ``skeleton_mean_segment_length`` — proxy for strand continuity
        (high = few breaks / long strands).
    """
    out: dict[str, float] = {}
    pixels = image.ravel()

    edges = canny(image, sigma=canny_sigma)
    out["edge_density"] = float(edges.sum() / image.size)

    try:
        thresh = threshold_otsu(pixels)
    except ValueError:
        thresh = pixels.mean()
    binary = image > thresh

    labeled = label(binary)
    props = regionprops(labeled)
    out["n_connected_components"] = float(len(props))
    if props:
        areas = np.array([p.area for p in props])
        out["cc_mean_area"] = float(areas.mean())
        out["cc_max_area"] = float(areas.max())
    else:
        out.update(cc_mean_area=0.0, cc_max_area=0.0)

    out["signal_euler_number"] = float(euler_number(binary))

    if binary.any():
        skel = skeletonize(binary)
        skel_length = int(skel.sum())
        neighbor_count = (
            convolve(skel.astype(np.int32), _NEIGHBOR_KERNEL, mode="constant")
            - skel.astype(np.int32)
        )
        branch_pts = int((skel & (neighbor_count > 2)).sum())
        endpoints = int((skel & (neighbor_count == 1)).sum())
        out["skeleton_length"] = float(skel_length)
        out["skeleton_branch_points"] = float(branch_pts)
        out["skeleton_endpoints"] = float(endpoints)
        out["skeleton_mean_segment_length"] = float(
            skel_length / (branch_pts + endpoints / 2.0 + 1.0)
        )
    else:
        out.update(
            skeleton_length=0.0,
            skeleton_branch_points=0.0,
            skeleton_endpoints=0.0,
            skeleton_mean_segment_length=0.0,
        )

    return out
