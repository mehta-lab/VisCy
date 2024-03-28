"""Utility functions for processing images"""

import itertools
import sys

import numpy as np

import viscy.utils.normalize as normalize


def im_bit_convert(im, bit=16, norm=False, limit=[]):
    im = im.astype(
        np.float32, copy=False
    )  # convert to float32 without making a copy to save memory
    if norm:
        if not limit:
            # scale each image individually based on its min and max
            limit = [np.nanmin(im[:]), np.nanmax(im[:])]
        im = (
            (im - limit[0])
            / (limit[1] - limit[0] + sys.float_info.epsilon)
            * (2**bit - 1)
        )
    im = np.clip(im, 0, 2**bit - 1)  # clip the values to avoid wrap-around by np.astype
    if bit == 8:
        im = im.astype(np.uint8, copy=False)  # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False)  # convert to 16 bit
    return im


def im_adjust(img, tol=1, bit=8):
    """
    Stretches contrast of the image and converts to 'bit'-bit.
    Useful for weight-maps in masking
    """
    limit = np.percentile(img, [tol, 100 - tol])
    im_adjusted = im_bit_convert(img, bit=bit, norm=True, limit=limit.tolist())
    return im_adjusted


def grid_sample_pixel_values(im, grid_spacing):
    """Sample pixel values in the input image at the grid. Any incomplete
    grids (remainders of modulus operation) will be ignored.

    :param np.array im: 2D image
    :param int grid_spacing: spacing of the grid
    :return int row_ids: row indices of the grids
    :return int col_ids: column indices of the grids
    :return np.array sample_values: sampled pixel values
    """

    im_shape = im.shape
    assert grid_spacing < im_shape[0], "grid spacing larger than image height"
    assert grid_spacing < im_shape[1], "grid spacing larger than image width"
    # leave out the grid points on the edges
    sample_coords = np.array(
        list(
            itertools.product(
                np.arange(grid_spacing, im_shape[0], grid_spacing),
                np.arange(grid_spacing, im_shape[1], grid_spacing),
            )
        )
    )
    row_ids = sample_coords[:, 0]
    col_ids = sample_coords[:, 1]
    sample_values = im[row_ids, col_ids]
    return row_ids, col_ids, sample_values


def preprocess_image(
    im,
    hist_clip_limits=None,
    is_mask=False,
    normalize_im=None,
    zscore_mean=None,
    zscore_std=None,
):
    """
    Do histogram clipping, z score normalization, and potentially binarization.

    :param np.array im: Image (stack)
    :param tuple hist_clip_limits: Percentile histogram clipping limits
    :param bool is_mask: True if mask
    :param str/None normalize_im: Normalization, if any
    :param float/None zscore_mean: Data mean
    :param float/None zscore_std: Data std
    """
    # remove singular dimension for 3D images
    if len(im.shape) > 3:
        im = np.squeeze(im)
    if not is_mask:
        if hist_clip_limits is not None:
            im = normalize.hist_clipping(im, hist_clip_limits[0], hist_clip_limits[1])
        if normalize_im is not None:
            im = normalize.zscore(
                im,
                im_mean=zscore_mean,
                im_std=zscore_std,
            )
    else:
        if im.dtype != bool:
            im = im > 0
    return im
