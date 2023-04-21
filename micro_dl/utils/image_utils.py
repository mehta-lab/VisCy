"""Utility functions for processing images"""

import cv2
import itertools
import math
import numpy as np
import os
import pandas as pd
import sys
from scipy.ndimage.interpolation import zoom
from skimage.transform import resize
import iohub.ngff as ngff

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.normalize as normalize


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
    im = np.clip(
        im, 0, 2**bit - 1
    )  # clip the values to avoid wrap-around by np.astype
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


def resize_image(input_image, output_shape):
    """Resize image to a specified shape

    :param np.ndarray input_image: image to be resized
    :param tuple/np.array output_shape: desired shape of the output image
    :return: np.array, resized image
    """

    msg = "the output shape does not match the image dimension"
    assert len(output_shape) == len(input_image.shape), msg
    assert input_image.dtype != "bool"

    resized_image = resize(input_image, output_shape)
    return resized_image


def rescale_image(im, scale_factor):
    """
    Rescales a 2D image equally in x and y given a scale factor.
    Uses bilinear interpolation (the OpenCV default).

    :param np.array im: 2D image
    :param float scale_factor:
    :return np.array: 2D image resized by scale factor
    """

    assert scale_factor > 0, "Scale factor must be > 0, not {}".format(scale_factor)

    im_shape = im.shape
    assert len(im_shape) == 2, "OpenCV only works with 2D images"
    dsize = (
        int(round(im_shape[1] * scale_factor)),
        int(round(im_shape[0] * scale_factor)),
    )

    return cv2.resize(im, dsize=dsize)


def rescale_nd_image(input_volume, scale_factor):
    """Rescale a nd array, mainly used for 3D volume

    For non-int dims, the values are rounded off to closest int. 0.5 is iffy,
    when downsampling the value gets floored and upsampling it gets rounded to
    next int

    :param np.array input_volume: 3D stack
    :param float/list scale_factor: if scale_factor is a float, scale all
     dimensions by this. Else scale_factor has to be specified for each
     dimension in a list or tuple
    :return np.array res_volume: rescaled volume
    """

    assert (
        not input_volume.dtype == "bool"
    ), "input image is binary, not ideal for spline interpolation"

    if not isinstance(scale_factor, float):
        assert len(input_volume.shape) == len(
            scale_factor
        ), "Missing scale factor:" "scale_factor:{} != input_volume:{}".format(
            len(scale_factor), len(input_volume.shape)
        )

    res_image = zoom(input_volume, scale_factor)
    return res_image


def crop2base(im, base=2):
    """
    Crop image to nearest smaller factor of the base (usually 2), assumes xyz
    format, will work for zyx too but the x_shape, y_shape and z_shape will be
    z_shape, y_shape and x_shape respectively

    :param nd.array im: Image
    :param int base: Base to use, typically 2
    :param bool crop_z: crop along z dim, only for UNet3D
    :return nd.array im: Cropped image
    :raises AssertionError: if base is less than zero
    """
    assert base > 0, "Base needs to be greater than zero, not {}".format(base)
    im_shape = im.shape

    x_shape = base ** int(math.log(im_shape[0], base))
    y_shape = base ** int(math.log(im_shape[1], base))
    if x_shape < im_shape[0]:
        # Approximate center crop
        start_idx = (im_shape[0] - x_shape) // 2
        im = im[start_idx : start_idx + x_shape, ...]
    if y_shape < im_shape[1]:
        # Approximate center crop
        start_idx = (im_shape[1] - y_shape) // 2
        im = im[:, start_idx : start_idx + y_shape, ...]
    return im


def center_crop_to_shape(input_image, output_shape, image_format="zyx"):
    """Center crop the image to a given shape

    :param np.array input_image: input image to be cropped
    :param list output_shape: desired crop shape
    :param str image_format: Image format; zyx or xyz
    :return np.array center_block: Center of input image with output shape
    """

    input_shape = np.array(input_image.shape)
    singleton_dims = np.where(input_shape == 1)[0]
    input_image = np.squeeze(input_image)
    modified_shape = output_shape.copy()
    if len(input_image.shape) == len(output_shape) + 1:
        # This means we're dealing with multichannel 2D
        if image_format == "zyx":
            modified_shape.insert(0, input_image.shape[0])
        else:
            modified_shape.append(input_image.shape[-1])
    assert np.all(
        np.array(modified_shape) <= np.array(input_image.shape)
    ), "output shape is larger than image shape, use resize or rescale"

    start_0 = (input_image.shape[0] - modified_shape[0]) // 2
    start_1 = (input_image.shape[1] - modified_shape[1]) // 2
    if len(input_image.shape) > 2:
        start_2 = (input_image.shape[2] - modified_shape[2]) // 2
        center_block = input_image[
            start_0 : start_0 + modified_shape[0],
            start_1 : start_1 + modified_shape[1],
            start_2 : start_2 + modified_shape[2],
        ]
    else:
        center_block = input_image[
            start_0 : start_0 + modified_shape[0], start_1 : start_1 + modified_shape[1]
        ]
    for idx in singleton_dims:
        center_block = np.expand_dims(center_block, axis=idx)
    return center_block


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


def read_image(file_path):
    """
    Read 2D grayscale image from file.
    Checks file extension for npy and load array if true. Otherwise
    reads regular image using OpenCV (png, tif, jpg, see OpenCV for supported
    files) of any bit depth.

    :param str file_path: Full path to image
    :return array im: 2D image
    :raise IOError if image can't be opened
    """
    if file_path[-3:] == "npy":
        im = np.load(file_path)
    else:
        im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        if im is None:
            raise IOError('Image "{}" cannot be found.'.format(file_path))
    return im


def read_image_from_row(meta_row, dir_name=None):
    """
    Read 2D grayscale image from file.
    Checks file extension for npy and load array if true. Otherwise
    reads regular image using OpenCV (png, tif, jpg, see OpenCV for supported
    files) of any bit depth.

    :param pd.DataFrame meta_row: Row in metadata
    :param str/None dir_name: Directory containing images (none if using frames meta dir_name)
    :return array im: 2D image
    :raise IOError if image can't be opened
    """
    if isinstance(meta_row, (pd.DataFrame, pd.Series)):
        meta_row = meta_row.squeeze()
    if dir_name is None:
        dir_name = meta_row["dir_name"]
    file_path = os.path.join(dir_name, meta_row["file_name"])
    if file_path[-3:] == "npy":
        im = np.load(file_path)
    elif "zarr" in file_path[-5:]:
        plate = ngff.open_ome_zarr(store_path=file_path, mode='r+')
        # zarr_reader = io_utils.ZarrReader(file_path)
        im = plate.data
        # im = zarr_reader.get_image(
        #     p=meta_row["pos_idx"],
        #     t=meta_row["time_idx"],
        #     c=meta_row["channel_idx"],
        #     z=meta_row["slice_idx"],
        # )
    else:
        # Assumes files are tiff or png
        im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        if im is None:
            raise IOError('Image "{}" cannot be found.'.format(file_path))
    return im


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


def read_imstack_from_meta(
    frames_meta_sub,
    dir_name=None,
    hist_clip_limits=None,
    is_mask=False,
    normalize_im=None,
    zscore_mean=None,
    zscore_std=None,
):
    """
    Read images (>1) from metadata rows and assembles a stack.
    If images are masks, make sure they're boolean by setting >0 to True

    :param pd.DataFrame frames_meta_sub: Selected subvolume to be read
    :param str/None dir_name: Directory path (none if using dir in frames_meta)
    :param tuple hist_clip_limits: Percentile limits for histogram clipping
    :param bool is_mask: Indicator for if files contain masks
    :param bool/None normalize_im: Whether to zscore normalize im stack
    :param float zscore_mean: mean for z-scoring the image
    :param float zscore_std: std for z-scoring the image
    """
    im_stack = []
    meta_shape = frames_meta_sub.shape
    nbr_images = meta_shape[0] if len(meta_shape) > 1 else 1
    if nbr_images > 1:
        for idx in range(meta_shape[0]):
            meta_row = frames_meta_sub.iloc[idx]
            im = read_image_from_row(meta_row, dir_name)
            im_stack.append(im)
    else:
        # In case of series
        im = read_image_from_row(frames_meta_sub, dir_name)
        im_stack = [im]

    input_image = np.stack(im_stack, axis=-1)
    # Norm, hist clip, binarize for mask
    input_image = preprocess_image(
        input_image,
        hist_clip_limits,
        is_mask,
        normalize_im,
        zscore_mean,
        zscore_std,
    )
    return input_image


def read_imstack(
    input_fnames,
    hist_clip_limits=None,
    is_mask=False,
    normalize_im=None,
    zscore_mean=None,
    zscore_std=None,
):
    """
    Read the images in the fnames and assembles a stack.
    If images are masks, make sure they're boolean by setting >0 to True

    :param tuple/list input_fnames: Paths to input files
    :param tuple hist_clip_limits: limits for histogram clipping
    :param bool is_mask: Indicator for if files contain masks
    :param bool/None normalize_im: Whether to zscore normalize im stack
    :param float zscore_mean: mean for z-scoring the image
    :param float zscore_std: std for z-scoring the image
    """
    im_stack = []
    input_image = np.stack(im_stack, axis=-1)
    # Norm, hist clip, binarize for mask
    input_image = preprocess_image(
        input_image,
        hist_clip_limits,
        is_mask,
        normalize_im,
        zscore_mean,
        zscore_std,
    )
    return input_image


def preprocess_imstack(
    frames_metadata,
    depth,
    time_idx,
    channel_idx,
    slice_idx,
    pos_idx,
    dir_name=None,
    hist_clip_limits=None,
    normalize_im="stack",
):
    """
    Preprocess image given by indices:
    histogram clipping and z-score normalization is performed.

    :param pd.DataFrame frames_metadata: DF with meta info for all images
    :param int depth: num of slices in stack if 2.5D or depth for 3D
    :param int time_idx: Time index
    :param int channel_idx: Channel index
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param str/None dir_name: Image directory (none if using the frames_meta dir_name)
    :param list hist_clip_limits: Limits for histogram clipping (size 2)
    :param str or None normalize_im: options to z-score the image
    :return np.array im: 3D preprocessed image
    """
    assert normalize_im in [
        "stack",
        "dataset",
        "volume",
        "slice",
        None,
    ], "'normalize_im' can only be 'stack', 'dataset', 'volume', 'slice', or None"

    metadata_ids, _ = aux_utils.validate_metadata_indices(
        frames_metadata=frames_metadata,
        slice_ids=-1,
        uniform_structure=True,
    )
    margin = 0 if depth == 1 else depth // 2
    im_stack = []
    for z in range(slice_idx - margin, slice_idx + margin + 1):
        meta_idx = aux_utils.get_meta_idx(
            frames_metadata,
            time_idx,
            channel_idx,
            z,
            pos_idx,
        )
        meta_row = frames_metadata.loc[meta_idx]
        im = read_image_from_row(meta_row, dir_name)

        zscore_median = None
        zscore_iqr = None
        if normalize_im in ["dataset", "volume", "slice"]:
            if "zscore_median" in frames_metadata:
                zscore_median = frames_metadata.loc[meta_idx, "zscore_median"]
            if "zscore_iqr" in frames_metadata:
                zscore_iqr = frames_metadata.loc[meta_idx, "zscore_iqr"]
        if normalize_im is not None:
            im = normalize.zscore(
                im,
                im_mean=zscore_median,
                im_std=zscore_iqr,
            )
        im_stack.append(im)

    if len(im.shape) == 3:
        # each channel is tiled independently and stacked later in dataset cls
        im_stack = im
        assert depth == 1, "more than one 3D volume gets read"
    else:
        # Stack images in same channel
        im_stack = np.stack(im_stack, axis=2)
    # normalize
    if hist_clip_limits is not None:
        im_stack = normalize.hist_clipping(
            im_stack,
            hist_clip_limits[0],
            hist_clip_limits[1],
        )

    return im_stack
