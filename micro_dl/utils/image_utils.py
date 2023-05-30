"""Utility functions for processing images"""
import cv2
import itertools
import math
import numpy as np
import os
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk, ball, binary_opening
from skimage.transform import resize


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


def crop2base(im, base=2):
    """
    Crop image to nearest smaller factor of the base (usually 2)

    :param nd.array im: Image
    :param int base: Base to use, typically 2
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


def create_mask(input_image, str_elem_size=3):
    """Create a binary mask using morphological operations

    Opening removes small objects in the foreground.

    :param np.array input_image: generate masks from this image
    :param int str_elem_size: size of the structuring element. typically 3, 5
    :return: mask of input_image, np.array
    """

    if np.min(input_image) == np.max(input_image):
        thr = np.unique(input_image)
    else:
        thr = threshold_otsu(input_image, nbins=512)
    if len(input_image.shape) == 2:
        str_elem = disk(str_elem_size)
    else:
        str_elem = ball(str_elem_size)
    # remove small objects in mask
    thr_image = binary_opening(input_image > thr, str_elem)
    mask = binary_fill_holes(thr_image)
    return mask


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
            print(e)
            raise
    return im
