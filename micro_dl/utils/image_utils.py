"""Utility functions for processing images"""
import itertools
import numpy as np
import os
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


def resize_mask(input_image, target_size):
    """Resample label/bool images"""
    raise NotImplementedError


def apply_flat_field_correction(input_image, **kwargs):
    """Apply flat field correction.

    :param np.array input_image: image to be corrected
    Kwargs:
        flat_field_image (np.float): flat_field_image for correction
        image_dir (str): dir with split images from stack (or individual
         sample images
        channel_id (int): input image channel
    :return: np.array (float) corrected image
    """

    input_image = input_image.astype('float')
    if 'flat_field_image' in kwargs:
        corrected_image = input_image / kwargs['flat_field_image']
    else:
        msg = 'image_dir and channel_id are required to fetch flat field image'
        assert all(k in kwargs for k in ('image_dir', 'channel_id')), msg
        flat_field_image = np.load(os.path.join(
            kwargs['image_dir'], 'flat_field_images',
            'flat-field_channel-{}.npy'.format(kwargs['channel_id'])
        ))
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


def tile_image(input_image, tile_size, step_size,
               isotropic=False, return_index=False):
    """Crops the image from given crop and step size.

    :param np.array input_image: input image to be tiled
    :param list/tuple/np array tile_size: size of the blocks to be cropped
     from the image
    :param list/tuple/np array step_size: size of the window shift. In case of
     no overlap, the step size is tile_size. If overlap, step_size < tile_size
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :param bool return_index: indicator for returning crop indices
    :return: a list with tuples of cropped image id of the format
     rrmin-rmax_ccmin-cmax_slslmin-slmax and cropped image
     if return_index=True: return a list with tuples of crop indices
    """

    check_1 = len(tile_size) == len(step_size)
    check_2 = np.all(step_size <= tile_size)
    check_3 = np.all(tile_size) > 0
    assert check_1 and check_2 and check_3

    n_rows = input_image.shape[0]
    n_cols = input_image.shape[1]

    n_dim = len(input_image.shape)
    if n_dim == 3:
        n_slices = input_image.shape[2]

    if isotropic:
        isotropic_shape = [tile_size[0], ] * len(tile_size)
        isotropic_cond = not(list(tile_size) == isotropic_shape)
    else:
        isotropic_cond = isotropic

    cropped_image_list = []
    cropping_index = []
    for row in range(0, n_rows - tile_size[0] + 1, step_size[0]):
        for col in range(0, n_cols - tile_size[1] + 1, step_size[1]):
            img_id = 'r{}-{}_c{}-{}'.format(row, row + tile_size[0],
                                            col, col + tile_size[1])
            if n_dim == 3:
                for sl in range(0, n_slices - tile_size[2] + 1, step_size[2]):
                    cur_index = (row, row + tile_size[0],
                                 col, col + tile_size[1],
                                 sl, sl + tile_size[2])
                    img_id = '{}_sl{}-{}'.format(img_id, sl, sl + tile_size[2])
                    cropped_img = input_image[row: row + tile_size[0],
                                              col: col + tile_size[1],
                                              sl: sl + tile_size[2]]
                    if isotropic_cond:
                        cropped_img = resize_image(cropped_img,
                                                   isotropic_shape)
                    cropped_image_list.append((img_id, cropped_img))
                    cropping_index.append(cur_index)
            else:
                cur_index = (row, row + tile_size[0], col, col + tile_size[1])
                cropped_img = input_image[row: row + tile_size[0],
                                          col: col + tile_size[1]]
                cropped_image_list.append((img_id, cropped_img))
                cropping_index.append(cur_index)
    if return_index:
        return cropped_image_list, cropping_index
    return cropped_image_list


def crop_at_indices(input_image, crop_indices, isotropic=False):
    """Crop image into tiles at given indices

    :param np.array input_image: input image for cropping
    :param list crop_indices: list of indices for cropping
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :return: a list with tuples of cropped image id of the format
     rrmin-rmax_ccmin-cmax_slslmin-slmax and cropped image
    """

    n_dim = len(input_image.shape)
    cropped_img_list = []
    for cur_idx in crop_indices:
        img_id = 'r{}-{}_c{}-{}'.format(cur_idx[0], cur_idx[1],
                                        cur_idx[2], cur_idx[3])
        if n_dim == 3:
            img_id = '{}_sl{}-{}'.format(img_id, cur_idx[4], cur_idx[5])
            cropped_img = input_image[cur_idx[0]: cur_idx[1],
                                      cur_idx[2]: cur_idx[3],
                                      cur_idx[4]: cur_idx[5]]
            if isotropic:
                img_shape = cropped_img.shape
                isotropic_shape = [img_shape[0], ] * len(img_shape)
                cropped_img = resize_image(cropped_img, isotropic_shape)
        else:
            cropped_img = input_image[cur_idx[0]: cur_idx[1],
                                      cur_idx[2]: cur_idx[3]]
        cropped_img_list.append((img_id, cropped_img))
    return cropped_img_list
