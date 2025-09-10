"""Utility functions for processing images."""

import itertools
import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

import viscy.utils.normalize as normalize


def im_bit_convert(
    im: ArrayLike, bit: int = 16, norm: bool = False, limit: list[float] = []
) -> NDArray[Any]:
    """Convert image to specified bit depth with optional normalization.

    FIXME: Verify parameter types and exact behavior for edge cases.

    Parameters
    ----------
    im : array_like
        Input image to convert.
    bit : int, optional
        Target bit depth (8 or 16), by default 16.
    norm : bool, optional
        Whether to normalize image to [0, 2^bit-1] range, by default False.
    limit : list, optional
        Min/max values for normalization. If empty, uses image min/max,
        by default [].

    Returns
    -------
    np.array
        Image converted to specified bit depth.
    """
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


def im_adjust(img: ArrayLike, tol: int | float = 1, bit: int = 8) -> NDArray[Any]:
    """Stretch contrast of the image and convert to specified bit depth.

    Useful for weight-maps in masking.

    Parameters
    ----------
    img : array_like
        Input image to adjust.
    tol : int or float, optional
        Tolerance percentile for contrast stretching, by default 1.
    bit : int, optional
        Target bit depth, by default 8.

    Returns
    -------
    np.array
        Contrast-adjusted image in specified bit depth.
    """
    limit = np.percentile(img, [tol, 100 - tol])
    im_adjusted = im_bit_convert(img, bit=bit, norm=True, limit=limit.tolist())
    return im_adjusted


def grid_sample_pixel_values(
    im: NDArray[Any], grid_spacing: int
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Sample pixel values in the input image at grid points.

    Any incomplete grids (remainders of modulus operation) will be ignored.

    Parameters
    ----------
    im : np.array
        2D image to sample from.
    grid_spacing : int
        Spacing of the grid points.

    Returns
    -------
    row_ids : np.array
        Row indices of the grid points.
    col_ids : np.array
        Column indices of the grid points.
    sample_values : np.array
        Sampled pixel values at grid points.
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
    im: ArrayLike,
    hist_clip_limits: tuple[float, float] | None = None,
    is_mask: bool = False,
    normalize_im: str | None = None,
    zscore_mean: float | None = None,
    zscore_std: float | None = None,
) -> NDArray[Any]:
    """Preprocess image with histogram clipping, z-score normalization, and binarization.

    Performs histogram clipping, z-score normalization, and potentially binarization
    depending on the input parameters.

    Parameters
    ----------
    im : np.array
        Input image or image stack.
    hist_clip_limits : tuple, optional
        Percentile histogram clipping limits (min_percentile, max_percentile),
        by default None.
    is_mask : bool, optional
        True if input is a mask (will be binarized), by default False.
    normalize_im : str, optional
        Normalization method to apply, by default None.
    zscore_mean : float, optional
        Precomputed mean for z-score normalization, by default None.
    zscore_std : float, optional
        Precomputed standard deviation for z-score normalization, by default None.

    Returns
    -------
    np.array
        Preprocessed image.
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
