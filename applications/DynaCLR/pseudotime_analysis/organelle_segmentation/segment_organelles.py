import logging

import numpy as np
from numpy.typing import ArrayLike
from skimage import measure, morphology
from skimage.exposure import equalize_adapthist
from skimage.filters import frangi, threshold_otsu, threshold_triangle

_logger = logging.getLogger("viscy")


def segment_zyx(
    input_zyx: ArrayLike,
    clahe_kernel_size=None,
    clahe_clip_limit=0.01,
    sigma_range=(0.5, 3.0),
    sigma_steps=5,
    auto_optimize_sigma=True,
    frangi_alpha=0.5,
    frangi_beta=0.5,
    frangi_gamma=None,
    threshold_method="otsu",
    min_object_size=10,
    apply_morphology=True,
):
    """
    Segment mitochondria from a 2D or 3D input_zyx using CLAHE preprocessing,
    Frangi filtering, and connected component labeling.

    Based on:
    Lefebvre, A.E.Y.T., Sturm, G., Lin, TY. et al.
    Nellie (2025) https://doi.org/10.1038/s41592-025-02612-7

    Parameters
    ----------
    input_zyx : ndarray
        Input image with shape (Z, Y, X) for 3D.
        If 2D, uses 2D Frangi filter. If 3D with Z=1, squeezes to 2D.
    clahe_kernel_size : int or None
        Kernel size for CLAHE (Contrast Limited Adaptive Histogram Equalization).
        If None, automatically set to max(input_zyx.shape) // 8.
    clahe_clip_limit : float
        Clipping limit for CLAHE, normalized between 0 and 1 (default: 0.01).
    sigma_range : tuple of float
        Range of sigma values to test for Frangi filter (min_sigma, max_sigma).
        Represents the scale of structures to detect.
    sigma_steps : int
        Number of sigma values to test in the range.
    auto_optimize_sigma : bool
        If True, automatically finds optimal sigma by maximizing vesselness response.
        If False, uses all sigmas in range for multi-scale filtering.
    frangi_alpha : float
        Frangi filter sensitivity to plate-j    like structures (2D) or blob-like (3D).
    frangi_beta : float
        Frangi filter sensitivity to blob-like structures (2D) or tube-like (3D).
    frangi_gamma : float or None
        Frangi filter sensitivity to background noise. If None, auto-computed.
    threshold_method : str
        Thresholding method: 'otsu', 'triangle', 'percentile', 'manual_X'.
    min_object_size : int
        Minimum object size in pixels for connected components.
    apply_morphology : bool
        If True, applies morphological closing to connect nearby structures.

    Returns
    -------
    labels : ndarray
        Instance segmentation labels with same dimensionality as input.
    vesselness : ndarray
        Filtered vesselness response with same dimensionality as input.
    optimal_sigma : float or None
        The optimal sigma value if auto_optimize_sigma=True, else None.
    """

    assert input_zyx.ndim == 3
    Z, Y, X = input_zyx.shape[-3:]

    if clahe_kernel_size is None:
        clahe_kernel_size = max(Z, Y, X) // 8

    # Apply CLAHE for contrast enhancement
    enhanced_zyx = equalize_adapthist(
        input_zyx,
        kernel_size=clahe_kernel_size,
        clip_limit=clahe_clip_limit,
    )

    # Generate sigma values
    sigmas = np.linspace(sigma_range[0], sigma_range[1], sigma_steps)

    # Auto-optimize sigma or use multi-scale
    if auto_optimize_sigma:
        optimal_sigma, vesselness = _find_optimal_sigma(
            enhanced_zyx, sigmas, frangi_alpha, frangi_beta, frangi_gamma
        )
    else:
        optimal_sigma = None
        vesselness = _multiscale_frangi(
            enhanced_zyx, sigmas, frangi_alpha, frangi_beta, frangi_gamma
        )

    # Threshold the vesselness response
    if threshold_method == "otsu":
        threshold = threshold_otsu(vesselness)
        _logger.debug(f"Otsu threshold: {threshold:.4f}")
    elif threshold_method == "triangle":
        threshold = threshold_triangle(vesselness)
        _logger.debug(f"Triangle threshold: {threshold:.4f}")
    elif threshold_method == "nellie_min":
        threshold_otsu_val = threshold_otsu(vesselness)
        threshold_triangle_val = threshold_triangle(vesselness)
        threshold = min(threshold_otsu_val, threshold_triangle_val)
        _logger.debug(
            f"Nellie-min threshold: otsu={threshold_otsu_val:.4f}, triangle={threshold_triangle_val:.4f}, using min={threshold:.4f}"
        )
    elif threshold_method == "nellie_max":
        threshold_otsu_val = threshold_otsu(vesselness)
        threshold_triangle_val = threshold_triangle(vesselness)
        threshold = max(threshold_otsu_val, threshold_triangle_val)
        _logger.debug(
            f"Nellie-max threshold: otsu={threshold_otsu_val:.4f}, triangle={threshold_triangle_val:.4f}, using max={threshold:.4f}"
        )
    elif threshold_method == "percentile":
        # Use percentile-based threshold (good for sparse features)
        threshold = np.percentile(vesselness[vesselness > 0], 95)  # Keep top 5%
        _logger.debug(f"Percentile (95th) threshold: {threshold:.4f}")
    elif threshold_method.startswith("manual_"):
        # Manual threshold: "manual_0.05" means threshold at 0.05
        threshold = float(threshold_method.split("_")[1])
        _logger.debug(f"Manual threshold: {threshold:.4f}")
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")

    binary_mask = vesselness > threshold

    _logger.debug(
        f"    Selected {binary_mask.sum()} / {binary_mask.size} pixels ({100 * binary_mask.sum() / binary_mask.size:.2f}%)"
    )

    # Apply morphological operations
    if apply_morphology:
        binary_mask = morphology.binary_closing(
            binary_mask, footprint=morphology.ball(1)
        )
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=64)

    # Label connected components
    labels = measure.label(binary_mask, connectivity=2)

    # Remove small objects
    labels = morphology.remove_small_objects(labels, min_size=min_object_size)

    if Z == 1:
        labels = labels[np.newaxis, ...]
        vesselness = vesselness[np.newaxis, ...]

    return labels, vesselness, optimal_sigma


def _find_optimal_sigma(input_zyx, sigmas, alpha, beta, gamma):
    """
    Find the optimal sigma that maximizes the vesselness response.

    Parameters
    ----------
    input_zyx : ndarray
         3D input_zyx (Z, Y, X).
    sigmas : array-like
        Sigma values to test.
    alpha, beta, gamma : float
        Frangi filter parameters.

    Returns
    -------
    optimal_sigma : float
        The sigma with the highest mean vesselness response.
    vesselness : ndarray
        The vesselness response using optimal sigma.
    """
    best_sigma = sigmas[0]
    best_score = -np.inf
    best_vesselness = None

    if input_zyx.shape[0] == 1:
        input_zyx = input_zyx[0]

    for sigma in sigmas:
        vessel = frangi(
            input_zyx,
            sigmas=[sigma],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=False,
        )

        # Score is the mean of top 10% vesselness values
        score = np.mean(
            np.partition(vessel.ravel(), -int(0.1 * vessel.size))[
                -int(0.1 * vessel.size) :
            ]
        )

        if score > best_score:
            best_score = score
            best_sigma = sigma
            best_vesselness = vessel

    if input_zyx.shape[0] == 1:
        best_vesselness = best_vesselness[np.newaxis, ...]

    return best_sigma, best_vesselness


def _multiscale_frangi(
    input_zyx, sigmas: ArrayLike, alpha: float, beta: float, gamma: float
):
    """
    Apply Frangi filter at multiple scales and return the maximum response.

    Parameters
    ----------
    input_zyx : ndarray
        3D input_zyx (Z, Y, X).
    sigmas : array-like
        Sigma values for multi-scale filtering.
    alpha, beta, gamma : float
        Frangi filter parameters.

    Returns
    -------
    vesselness : ndarray
        Maximum vesselness response across all scales.
    """
    if input_zyx.shape[0] == 1:
        input_zyx = input_zyx[0]
    vesselness = frangi(
        input_zyx,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=False,
    )
    if input_zyx.shape[0] == 1:
        vesselness = vesselness[np.newaxis, ...]
    return vesselness


def calculate_nellie_sigmas(
    min_radius_um, max_radius_um, pixel_size_um, num_sigma=5, min_step_size_px=0.2
):
    """
    Calculate sigma values following Nellie's approach.

    Parameters
    ----------
    min_radius_um : float
        Minimum structure radius in micrometers (e.g., 0.2 for diffraction limit)
    max_radius_um : float
        Maximum structure radius in micrometers (e.g., 1.0 for thick tubules)
    pixel_size_um : float
        Pixel size in micrometers
    num_sigma : int
        Target number of sigma values
    min_step_size_px : float
        Minimum step size between sigmas in pixels

    Returns
    -------
    tuple : (sigma_min, sigma_max)
        Sigma range in pixels
    """
    min_radius_px = min_radius_um / pixel_size_um
    max_radius_px = max_radius_um / pixel_size_um

    # Nellie uses radius/2 to radius/3 as sigma
    sigma_1 = min_radius_px / 2
    sigma_2 = max_radius_px / 3
    sigma_min = min(sigma_1, sigma_2)
    sigma_max = max(sigma_1, sigma_2)

    # Calculate step size with minimum constraint
    sigma_step_calculated = (sigma_max - sigma_min) / num_sigma
    sigma_step = max(min_step_size_px, sigma_step_calculated)

    sigmas = list(np.arange(sigma_min, sigma_max + sigma_step, sigma_step))

    _logger.debug(f"  Nellie-style sigmas: {sigma_min:.3f} to {sigma_max:.3f} pixels")
    _logger.debug(
        f"  Radius range: {min_radius_um:.3f}-{max_radius_um:.3f} µm = {min_radius_px:.2f}-{max_radius_px:.2f} pixels"
    )
    _logger.debug(f"  Sigma values: {[f'{s:.2f}' for s in sigmas]}")

    return (sigma_min, sigma_max)
