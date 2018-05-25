"""Utility functions for processing images"""
import glob
import itertools
import numpy as np
from skimage.transform import resize


def resize_image(input_image, output_shape):
    """Resize image to a specified shape

    :param np.ndarray input_image: image to be resized
    :param tuple/np.array output_shape: desired shape of the output image
    """

    msg = 'the output shape does not match the image dimension'
    assert len(output_shape) == len(input_image.shape), msg
    assert input_image.dtype is not 'bool'

    resized_image = resize(input_image, output_shape)
    return resized_image


def resize_mask(input_image, target_size):
    """Resample label/bool images"""
    raise NotImplementedError


def sample_block_medians(im, block_size=32):
    """
    Subdivide a 2D image in smaller blocks of size block_size and
    compute the median intensity value for each block. Any incomplete
    blocks (remainders of modulo operation) will be ignored.

    :param np.array im:             2D image
    :param int block_size:          Size of blocks image will be divided into

    :return np.array sample_coords: Image coordinates for block centers (float)
    :return np.array sample_values: Median intensity values for blocks (float)
    """
    im_shape = im.shape
    assert block_size < im_shape[0], "Block size larger than image height"
    assert block_size < im_shape[1], "Block size larger than image width"

    blocks_height = im_shape[0] // block_size
    blocks_width = im_shape[1] // block_size
    sample_coords = np.zeros((blocks_height * blocks_width, 2), dtype=np.float64)
    sample_values = np.zeros((blocks_height * blocks_width, ), dtype=np.float64)
    for x in range(blocks_height):
        for y in range(blocks_width):
            idx = y * blocks_height + x
            sample_coords[idx, :] = [x * block_size + (block_size - 1) / 2,
                                     y * block_size + (block_size - 1) / 2]
            sample_values[idx] = np.median(im[x * block_size:(x + 1) * block_size,
                                              y * block_size:(y + 1) * block_size])
    return sample_coords, sample_values


def fit_polynomial_surface(sample_coords,
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


def get_flatfield(im, block_size=32, order=2, normalize=True):
    """
    Combine sampling and polynomial surface fit for flatfield estimation.
    To flatfield correct an image, divide it by flatfield

    :param np.array im:            2D image
    :param int block_size:         Size of blocks image will be divided into
    :param int order:              Order of polynomial (default 2)
    :param bool normalize:         Normalize surface by dividing by its mean
                                   for flatfield correction (default True)

    :return np.array flatfield:    Flatfield image
    """
    coords, values = sample_block_medians(im, block_size=block_size)
    flatfield = fit_polynomial_surface(coords,
                                       values,
                                       im.shape,
                                       order=order,
                                       normalize=normalize)
    # Flatfields can't contain zeros or negative values
    if flatfield.min() <= 0:
        raise ValueError("The generated flatfield was not strictly positive.")
    return flatfield
