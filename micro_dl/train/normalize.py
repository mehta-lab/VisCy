"""Image normalization related functions"""
import numpy as np
from skimage.exposure import rescale_intensity, equalize_adapthist


def zscore(input_image):
    """Performs z-score normalization

    :param input_image: input image for intensity normalization
    :return: z score normalized image
    """

    norm_img = (input_image - np.mean(input_image)) / np.std(input_image)
    return norm_img


def hist_clipping(input_image, min_percentile=5, max_percentile=95):
    """Clips and rescales histogram from min to max intensity percentiles

    rescale_intensity with input check

    :param np.array input_image: input image for intensity normalization
    :param int/float min_percentile: min intensity percentile
    :param int/flaot max_percentile: max intensity percentile
    :return: np.float, intensity clipped and rescaled image
    """

    assert (min_percentile < max_percentile) and max_percentile <= 100
    pmin, pmax = np.percentile(input_image, (min_percentile, max_percentile))
    hist_clipped_image = rescale_intensity(input_image, in_range=(pmin, pmax))
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
            raise ValueError('kernel size invalid: not an int / list / tuple')

    if clip_limit is not None:
        assert clip_limit >= 0 and clip_limit <= 1

    adapt_eq_image = equalize_adapthist(
        input_image, kernel_size=kernel_size, clip_limit=clip_limit
    )
    return adapt_eq_image
