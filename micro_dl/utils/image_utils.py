"""Utility functions for processing images"""

import cv2
import itertools
import math
import numpy as np
import os
from scipy.ndimage.interpolation import zoom
from skimage.transform import resize

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.normalize as normalize


def resize_image(input_image, output_shape):
    """Resize image to a specified shape

    :param np.ndarray input_image: image to be resized
    :param tuple/np.array output_shape: desired shape of the output image
    :return: np.array, resized image
    """

    msg = 'the output shape does not match the image dimension'
    assert len(output_shape) == len(input_image.shape), msg
    assert input_image.dtype is not 'bool'

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

    assert scale_factor > 0,\
        'Scale factor must be > 0, not {}'.format(scale_factor)

    im_shape = im.shape
    assert len(im_shape) == 2, "OpenCV only works with 2D images"
    dsize = (int(round(im_shape[1] * scale_factor)),
             int(round(im_shape[0] * scale_factor)))

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

    assert not input_volume.dtype == 'bool', \
        'input image is binary, not ideal for spline interpolation'

    if not isinstance(scale_factor, float):
        assert len(input_volume.shape) == len(scale_factor), \
            'Missing scale factor:' \
            'scale_factor:{} != input_volume:{}'.format(
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
        im = im[start_idx:start_idx + x_shape, ...]
    if y_shape < im_shape[1]:
        # Approximate center crop
        start_idx = (im_shape[1] - y_shape) // 2
        im = im[:, start_idx:start_idx + y_shape, ...]
    return im


def resize_mask(input_image, target_size):
    """Resample label/bool images"""
    raise NotImplementedError


def apply_flat_field_correction(input_image, **kwargs):
    """Apply flat field correction.

    :param np.array input_image: image to be corrected
    Kwargs:
        flat_field_image (np.float): flat_field_image for correction
        flat_field_dir (str): dir with split images from stack (or individual
         sample images
        channel_idx (int): input image channel index
    :return: np.array (float) corrected image
    """

    input_image = input_image.astype('float')
    if 'flat_field_image' in kwargs:
        corrected_image = input_image / kwargs['flat_field_image']
    else:
        msg = 'flat_field_dir and channel_id are required to fetch flat field image'
        assert all(k in kwargs for k in ('flat_field_dir', 'channel_idx')), msg
        flat_field_image = np.load(
            os.path.join(
                kwargs['flat_field_dir'],
                'flat-field_channel-{}.npy'.format(kwargs['channel_idx']),
            )
        )
        corrected_image = input_image / flat_field_image
    return corrected_image


def fit_polynomial_surface_2D(sample_coords,
                              sample_values,
                              im_shape,
                              order=2,
                              normalize=True):
    """
    Given coordinates and corresponding values, this function will fit a
    2D polynomial of given order, then create a surface of given shape.

    :param np.array sample_coords: 2D sample coords (nbr of points, 2)
    :param np.array sample_values: Corresponding intensity values (nbr points,)
    :param tuple im_shape:         Shape of desired output surface (height, width)
    :param int order:              Order of polynomial (default 2)
    :param bool normalize:         Normalize surface by dividing by its mean
                                   for flatfield correction (default True)

    :return np.array poly_surface: 2D surface of shape im_shape
    """
    assert (order + 1) ** 2 <= len(sample_values), \
        "Can't fit a higher degree polynomial than there are sampled values"
    # Number of coefficients in determined by order + 1 squared
    orders = np.arange(order + 1)
    variable_matrix = np.zeros((sample_coords.shape[0], (order + 1) ** 2))
    variable_iterator = itertools.product(orders, orders)
    for idx, (m, n) in enumerate(variable_iterator):
        variable_matrix[:, idx] = sample_coords[:, 0] ** n * sample_coords[:, 1] ** m
    # Least squares fit of the points to the polynomial
    coeffs, _, _, _ = np.linalg.lstsq(variable_matrix, sample_values, rcond=-1)
    # Create a grid of image (x, y) coordinates
    x_mesh, y_mesh = np.meshgrid(np.linspace(0, im_shape[1] - 1, im_shape[1]),
                                 np.linspace(0, im_shape[0] - 1, im_shape[0]))
    # Reconstruct the surface from the coefficients
    poly_surface = np.zeros(im_shape, np.float)
    variable_iterator = itertools.product(orders, orders)
    for coeff, (m, n) in zip(coeffs, variable_iterator):
        poly_surface += coeff * x_mesh ** m * y_mesh ** n

    if normalize:
        poly_surface /= np.mean(poly_surface)
    return poly_surface


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
    if file_path[-3:] == 'npy':
        im = np.load(file_path)
    else:
        try:
            im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        except IOError as e:
            raise e
    return im


def read_imstack(input_fnames,
                 flat_field_fname=None,
                 hist_clip_limits=None,
                 is_mask=False,
                 normalize_im=True):
    """
    Read the images in the fnames and assembles a stack.
    If images are masks, make sure they're boolean by setting >0 to True

    :param tuple input_fnames: tuple of input fnames with full path
    :param str flat_field_fname: fname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :param bool is_mask: Indicator for if files contain masks
    :param bool normalize_im: Whether to zscore normalize im stack
    :return np.array: input stack flat_field correct and z-scored if regular
        images, booleans if they're masks
    """
    im_stack = []
    for idx, fname in enumerate(input_fnames):
        im = read_image(fname)
        if flat_field_fname is not None:
            # multiple flat field images are passed in case of mask generation
            if isinstance(flat_field_fname, (list, tuple)):
                flat_field_image = np.load(flat_field_fname[idx])
            else:
                flat_field_image = np.load(flat_field_fname)
            if not is_mask and not normalize_im:
                im = apply_flat_field_correction(
                    im,
                    flat_field_image=flat_field_image,
                )
        im_stack.append(im)

    input_image = np.stack(im_stack, axis=-1)
    # remove singular dimension for 3D images
    if len(input_image.shape) > 3:
        input_image = np.squeeze(input_image)
    if not is_mask:
        if hist_clip_limits is not None:
            input_image = normalize.hist_clipping(
                input_image,
                hist_clip_limits[0],
                hist_clip_limits[1]
            )
        if normalize_im:
            input_image = normalize.zscore(input_image)
    else:
        if input_image.dtype != bool:
            input_image = input_image > 0
    return input_image


def preprocess_imstack(frames_metadata,
                       input_dir,
                       depth,
                       time_idx,
                       channel_idx,
                       slice_idx,
                       pos_idx,
                       flat_field_im=None,
                       hist_clip_limits=None,
                       normalize_im=False):
    """
    Preprocess image given by indices: flatfield correction, histogram
    clipping and z-score normalization is performed.

    :param pd.DataFrame frames_metadata: DF with meta info for all images
    :param str input_dir: dir containing input images
    :param int depth: num of slices in stack if 2.5D or depth for 3D
    :param int time_idx: Time index
    :param int channel_idx: Channel index
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param np.array flat_field_im: Flat field image for channel
    :param list hist_clip_limits: Limits for histogram clipping (size 2)
    :param bool normalize_im: indicator to normalize image based on z-score or not
    :return np.array im: 3D preprocessed image
    """
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
        file_path = os.path.join(
            input_dir,
            frames_metadata.loc[meta_idx, "file_name"],
        )
        im = read_image(file_path)
        # Only flatfield correct images that will be normalized
        if flat_field_im is not None and not normalize_im:
            im = apply_flat_field_correction(
                im,
                flat_field_image=flat_field_im,
            )
        im_stack.append(im)

    if len(im.shape) == 3:
        # each channel is tiled independently and stacked later in dataset cls
        im_stack = im
        assert depth == 1, 'more than one 3D volume gets read'
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
    if normalize_im:
        im_stack = normalize.zscore(im_stack)
    return im_stack


def center_crop_to_shape(input_image, output_shape):
    """Center crop the image to a given shape

    :param np.array input_image: input image to be cropped
    :param list output_shape: desired crop shape
    """

    input_shape = np.array(input_image.shape)
    singleton_dims = np.where(input_shape == 1)[0]
    input_image = np.squeeze(input_image)
    assert np.all(np.array(output_shape) <= np.array(input_image.shape)), \
        'output shape is larger than image shape, use resize or rescale'

    start_0 = (input_image.shape[0] - output_shape[0]) // 2
    start_1 = (input_image.shape[1] - output_shape[1]) // 2
    start_2 = (input_image.shape[2] - output_shape[2]) // 2
    center_block = input_image[start_0: start_0 + output_shape[0],
                   start_1: start_1 + output_shape[1],
                   start_2: start_2 + output_shape[2]]
    for idx in singleton_dims:
        center_block = np.expand_dims(center_block, axis=idx)
    return center_block
