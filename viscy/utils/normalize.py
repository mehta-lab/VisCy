"""Image normalization related functions"""

import sys

import numpy as np
from skimage.exposure import equalize_adapthist


def zscore(input_image, im_mean=None, im_std=None):
    """
    Performs z-score normalization. Adds epsilon in denominator for robustness

    :param np.array input_image: input image for intensity normalization
    :param float/None im_mean: Image mean
    :param float/None im_std: Image std
    :return np.array norm_img: z score normalized image
    """
    if not im_mean:
        im_mean = np.nanmean(input_image)
    if not im_std:
        im_std = np.nanstd(input_image)
    norm_img = (input_image - im_mean) / (im_std + sys.float_info.epsilon)
    return norm_img


def unzscore(im_norm, zscore_median, zscore_iqr):
    """
    Revert z-score normalization applied during preprocessing. Necessary
    before computing SSIM

    :param im_norm: Normalized image for un-zscore
    :param zscore_median: Image median
    :param zscore_iqr: Image interquartile range
    :return im: image at its original scale
    """
    im = im_norm * (zscore_iqr + sys.float_info.epsilon) + zscore_median
    return im


def hist_clipping(input_image, min_percentile=2, max_percentile=98):
    """Clips and rescales histogram from min to max intensity percentiles

    rescale_intensity with input check

    :param np.array input_image: input image for intensity normalization
    :param int/float min_percentile: min intensity percentile
    :param int/flaot max_percentile: max intensity percentile
    :return: np.float, intensity clipped and rescaled image
    """

    assert (min_percentile < max_percentile) and max_percentile <= 100
    pmin, pmax = np.percentile(input_image, (min_percentile, max_percentile))
    hist_clipped_image = np.clip(input_image, pmin, pmax)
    return hist_clipped_image


def hist_adapteq_2D(input_image, kernel_size=None, clip_limit=None):
    """CLAHE on 2D images

    skimage.exposure.equalize_adapthist works only for 2D. Extend to 3D or use
    openCV? Not ideal, as it enhances noise in homogeneous areas

    :param np.array input_image: input image for intensity normalization
    :param int/list kernel_size: Neighbourhood to be used for histogram
     equalization. If none, use default of 1/8th image size.
    :param float clip_limit: Clipping limit, normalized between 0 and 1
     (higher values give more contrast, ~ max percent of voxels in any
     histogram bin, if > this limit, the voxel intensities are redistributed).
     if None, default=0.01
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
        assert 0 <= clip_limit <= 1, "Clip limit {} is out of range [0, 1]".format(
            clip_limit
        )

    adapt_eq_image = equalize_adapthist(
        input_image, kernel_size=kernel_size, clip_limit=clip_limit
    )
    return adapt_eq_image
