"""Image normalization related functions."""

import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skimage.exposure import equalize_adapthist


def zscore(
    input_image: ArrayLike, im_mean: float | None = None, im_std: float | None = None
) -> NDArray[Any]:
    """Perform z-score normalization.

    Adds epsilon in denominator for robustness.

    Parameters
    ----------
    input_image : np.array
        Input image for intensity normalization.
    im_mean : float, optional
        Image mean, by default None.
    im_std : float, optional
        Image std, by default None.

    Returns
    -------
    np.array
        Z-score normalized image.
    """
    if not im_mean:
        im_mean = np.nanmean(input_image)
    if not im_std:
        im_std = np.nanstd(input_image)
    norm_img = (input_image - im_mean) / (im_std + sys.float_info.epsilon)
    return norm_img


def unzscore(
    im_norm: ArrayLike, zscore_median: float, zscore_iqr: float
) -> NDArray[Any]:
    """Revert z-score normalization applied during preprocessing.

    Necessary before computing SSIM.

    Parameters
    ----------
    im_norm : array_like
        Normalized image for un-zscore.
    zscore_median : float
        Image median.
    zscore_iqr : float
        Image interquartile range.

    Returns
    -------
    array_like
        Image at its original scale.
    """
    im = im_norm * (zscore_iqr + sys.float_info.epsilon) + zscore_median
    return im


def hist_clipping(
    input_image: ArrayLike,
    min_percentile: int | float = 2,
    max_percentile: int | float = 98,
) -> NDArray[Any]:
    """Clip and rescale histogram from min to max intensity percentiles.

    rescale_intensity with input check.

    Parameters
    ----------
    input_image : np.array
        Input image for intensity normalization.
    min_percentile : int or float, optional
        Min intensity percentile, by default 2.
    max_percentile : int or float, optional
        Max intensity percentile, by default 98.

    Returns
    -------
    np.array
        Intensity clipped and rescaled image.
    """
    assert (min_percentile < max_percentile) and max_percentile <= 100
    pmin, pmax = np.percentile(input_image, (min_percentile, max_percentile))
    hist_clipped_image = np.clip(input_image, pmin, pmax)
    return hist_clipped_image


def hist_adapteq_2D(
    input_image: NDArray[Any],
    kernel_size: int | list[int] | tuple[int, ...] | None = None,
    clip_limit: float | None = None,
) -> NDArray[Any]:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on 2D images.

    skimage.exposure.equalize_adapthist works only for 2D. Extend to 3D or use
    openCV? Not ideal, as it enhances noise in homogeneous areas.

    Parameters
    ----------
    input_image : np.array
        Input image for intensity normalization.
    kernel_size : int or list, optional
        Neighbourhood to be used for histogram equalization. If None, use default
        of 1/8th image size, by default None.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast, ~ max percent of voxels in any histogram bin, if > this limit,
        the voxel intensities are redistributed). If None, default=0.01,
        by default None.

    Returns
    -------
    np.array
        Adaptive histogram equalized image.
    """
    nrows, ncols = input_image.shape
    if kernel_size is not None:
        if isinstance(kernel_size, int):
            assert kernel_size < min(nrows, ncols)
        elif isinstance(kernel_size, (list, tuple)):
            assert len(kernel_size) == len(input_image.shape)
        else:
            raise ValueError("kernel size invalid: not an int / list / tuple")

    if clip_limit is not None:
        assert 0 <= clip_limit <= 1, f"Clip limit {clip_limit} is out of range [0, 1]"

    adapt_eq_image = equalize_adapthist(
        input_image, kernel_size=kernel_size, clip_limit=clip_limit
    )
    return adapt_eq_image
