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


def get_crop_indices(mask_ip_dir, min_fraction, mask_op_dir, tile_size,
                     step_size, isotropic=False):
    """Get crop indices and save mask for tiles with roi_vf >= min_fraction

    Crops and saves mask to mask_output_dir

    :param str mask_ip_dir: directory containing individual sample masks
    :param float min_fraction: minimum volume fraction of the ROI to retain
    :param str mask_op_dir: directory to save the cropped masks
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :return: dict with fname as keys and list of indices as values
    """

    if isotropic:
        isotropic_shape = [tile_size[0], ] * len(tile_size)
        isotropic_cond = list(tile_size) == isotropic_shape
    else:
        isotropic_cond = isotropic
    masks_in_dir = glob.glob(os.path.join(mask_ip_dir, '*.npy'))
    index_dict = {}

    for mask_idx, mask_fname in enumerate(masks_in_dir):
        _, fname = os.path.split(mask_fname)
        mask = np.load(mask_fname)
        size_x = mask.shape[0]
        size_y = mask.shape[1]
        n_dim = len(mask.shape)
        if n_dim == 3:
            size_z = mask.shape[2]

        sample_num = int(fname.split('_')[1][1:])
        cur_index_list = []
        for x in range(0, size_x - tile_size[0] + 1, step_size[0]):
            for y in range(0, size_y - tile_size[1] + 1, step_size[1]):
                img_id = 'n{}_x{}-{}_y{}-{}'.format(
                    sample_num, x, x + tile_size[0], y, y + tile_size[1]
                )
                if n_dim == 3:
                    for z in range(0, size_z - tile_size[2] + 1, step_size[2]):
                        cropped_mask = mask[x: x + tile_size[0],
                                            y: y + tile_size[1],
                                            z: z + tile_size[2]]
                        roi_vf = np.sum(cropped_mask) / np.prod(cropped_mask.shape)
                        if roi_vf >= min_fraction:
                            cur_index = [x, x + tile_size[0], y,
                                         y + tile_size[1], z, z + tile_size[2]]
                            cur_index_list.append(cur_index)
                            if isotropic_cond is not None:
                                cropped_mask = resize_mask(cropped_mask,
                                                           isotropic_shape)
                            img_id = '{}_z{}-{}.npy'.format(
                                img_id, z, z + tile_size[2]
                            )
                            cropped_mask_fname = os.path.join(mask_op_dir,
                                                              img_id)
                            np.save(cropped_mask_fname, cropped_mask,
                                    allow_pickle=True, fix_imports=True)
                else:
                    cropped_mask = mask[x: x + tile_size[0], y: y + tile_size[1]]
                    roi_vf = np.sum(cropped_mask) / np.prod(cropped_mask.shape)
                    if roi_vf >= min_fraction:
                        cur_index = [x, x + tile_size[0], y, y + tile_size[1]]
                        cur_index_list.append(cur_index)
                        img_id = '{}.npy'.format(img_id)
                        cropped_mask_fname = os.path.join(mask_op_dir, img_id)
                        np.save(cropped_mask_fname, cropped_mask,
                                allow_pickle=True, fix_imports=True)
        index_dict[fname] = cur_index_list
    return index_dict


def crop_at_indices(input_image, crop_indices, isotropic=False):
    """Crop image into tiles at given indices

    :param np.array input_image: input image in 3d
    :param list crop_indices: list of indices for cropping
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :return: a list with tuples of cropped image id of the format
     xxmin-xmz_yymin-ymax_zzmin-zmax and cropped image
    """

    n_dim = len(input_image.shape)
    cropped_img_list = []
    for cur_idx in crop_indices:
        img_id = 'x{}-{}_y{}-{}'.format(cur_idx[0], cur_idx[1],
                                        cur_idx[2], cur_idx[3])
        if n_dim == 3:
            img_id = '{}_z{}-{}'.format(img_id, cur_idx[4], cur_idx[5])
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
