"""Image normalization related functions."""

import sys

import numpy as np
from skimage.exposure import equalize_adapthist


def zscore(input_image, im_mean=None, im_std=None):
    """Perform z-score normalization.

    Parameters
    ----------
    input_image : np.ndarray
        Input image for intensity normalization.
    im_mean : float or None, optional
        Image mean.
    im_std : float or None, optional
        Image std.

    Returns
    -------
    np.ndarray
        Z-score normalized image.
    """
    if not im_mean:
        im_mean = np.nanmean(input_image)
    if not im_std:
        im_std = np.nanstd(input_image)
    norm_img = (input_image - im_mean) / (im_std + sys.float_info.epsilon)
    return norm_img


def unzscore(im_norm, zscore_median, zscore_iqr):
    """Revert z-score normalization applied during preprocessing.

    Parameters
    ----------
    im_norm : np.ndarray
        Normalized image.
    zscore_median : float
        Image median.
    zscore_iqr : float
        Image interquartile range.

    Returns
    -------
    np.ndarray
        Image at its original scale.
    """
    im = im_norm * (zscore_iqr + sys.float_info.epsilon) + zscore_median
    return im


def hist_clipping(input_image, min_percentile=2, max_percentile=98):
    """Clip and rescale histogram from min to max intensity percentiles.

    Parameters
    ----------
    input_image : np.ndarray
        Input image for intensity normalization.
    min_percentile : int or float
        Min intensity percentile.
    max_percentile : int or float
        Max intensity percentile.

    Returns
    -------
    np.ndarray
        Intensity-clipped and rescaled image.
    """
    assert (min_percentile < max_percentile) and max_percentile <= 100
    pmin, pmax = np.percentile(input_image, (min_percentile, max_percentile))
    hist_clipped_image = np.clip(input_image, pmin, pmax)
    return hist_clipped_image


def hist_adapteq_2D(input_image, kernel_size=None, clip_limit=None):
    """CLAHE on 2D images.

    Parameters
    ----------
    input_image : np.ndarray
        Input image for intensity normalization.
    kernel_size : int or list or None, optional
        Neighbourhood for histogram equalization.
    clip_limit : float or None, optional
        Clipping limit, normalized between 0 and 1.

    Returns
    -------
    np.ndarray
        Adaptive-histogram equalized image.
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
