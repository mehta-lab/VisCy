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


def preprocess_imstack(frames_metadata,
                       input_dir,
                       depth,
                       time_idx,
                       channel_idx,
                       slice_idx,
                       pos_idx,
                       flat_field_im=None,
                       hist_clip_limits=None):
    """
    Preprocess image given by indices: flatfield correction, histogram
    clipping and z-score normalization is performed.

    :param int time_idx: Time index
    :param int channel_idx: Channel index
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param np.array flat_field_im: Flat field image for channel
    :param list hist_clip_limits: Limits for histogram clipping (size 2)
    :return np.array im: 2D preprocessed image
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
        if flat_field_im is not None:
            im = apply_flat_field_correction(
                im,
                flat_field_image=flat_field_im,
            )
        im_stack.append(im)
    # Stack images
    im_stack = np.stack(im_stack, axis=2)
    # normalize
    if hist_clip_limits is not None:
        im_stack = normalize.hist_clipping(
            im_stack,
            hist_clip_limits[0],
            hist_clip_limits[1],
        )
    return normalize.zscore(im_stack)


def tile_image(input_image,
               tile_size,
               step_size,
               isotropic=False,
               return_index=False,
               min_fraction=None):
    """
     Tiles the image based on given tile and step size.

    :param np.array input_image: input image to be tiled
    :param list/tuple/np array tile_size: size of the blocks to be tiled
     from the image
    :param list/tuple/np array step_size: size of the window shift. In case of
     no overlap, the step size is tile_size. If overlap, step_size < tile_size
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :param bool return_index: indicator for returning tile indices
    :param float min_fraction: Minimum fraction of foreground in mask for
    including tile
    :return: a list with tuples of tiled image id of the format
     rrmin-rmax_ccmin-cmax_slslmin-slmax and tiled image
     if return_index=True: return a list with tuples of crop indices
    """

    def check_in_range(cur_value, max_value, tile_size):
        """Get the start index for edge tiles

        :param int cur_value: cur index in row / col / slice
        :param int max_value: n_rows, n_cols or n_slices
        :param int tile_size: size of tile along one dimension (row, col,
         slice)
        :return: int start_value - adjusted start_index to fit the edge tile
        """
        cur_length = max_value - cur_value
        miss_length = tile_size - cur_length
        start_value = cur_value - miss_length
        return start_value

    def use_tile(cropped_img, min_fraction):
        """
        Determine if tile should be used given minimum image foreground fraction
        :param np.array cropped_img: Image tile
        :param float min_fraction: Minimum fraction of image being foreground
        :return bool use_tile: Indicator if tile should be used
        """
        use_tile = True
        if min_fraction is not None:
            mask_fraction = np.mean(cropped_img)
            if mask_fraction < min_fraction:
                use_tile = False
        return use_tile

    # Add to tile size and step size in case of 3D images
    im_shape = input_image.shape
    if len(im_shape) == 3:
        if len(tile_size) == 2:
            tile_size.append(im_shape[2])
        # Step size in z is assumed to be the same as depth
        if len(step_size) == 2:
            step_size.append(im_shape[2])

    check_1 = len(tile_size) == len(step_size)
    check_2 = np.all(step_size <= tile_size)
    check_3 = np.all(tile_size) > 0
    assert check_1 and check_2 and check_3,\
        "Tiling not valid with tile size {} and step {}".format(
            tile_size, step_size)

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
    for row in range(0, n_rows - tile_size[0] + step_size[0], step_size[0]):
        if row + tile_size[0] > n_rows:
            row = check_in_range(row, n_rows, tile_size[0])
        for col in range(0, n_cols - tile_size[1] + step_size[1], step_size[1]):
            if col + tile_size[1] > n_cols:
                col = check_in_range(col, n_cols, tile_size[1])
            img_id = 'r{}-{}_c{}-{}'.format(row, row + tile_size[0],
                                            col, col + tile_size[1])
            if n_dim == 3:
                for sl in range(0, n_slices, step_size[2]):
                    if sl + step_size[2] > n_slices:
                        sl = check_in_range(sl, n_slices, tile_size[2])

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
                    if use_tile(cropped_img, min_fraction):
                        cropped_image_list.append((img_id, cropped_img))
                        cropping_index.append(cur_index)
            else:
                cur_index = (row, row + tile_size[0], col, col + tile_size[1])
                cropped_img = input_image[row: row + tile_size[0],
                                          col: col + tile_size[1]]
                if use_tile(cropped_img, min_fraction):
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
    im_depth = input_image.shape[2]
    for cur_idx in crop_indices:
        img_id = 'r{}-{}_c{}-{}'.format(cur_idx[0], cur_idx[1],
                                        cur_idx[2], cur_idx[3])

        cropped_img = input_image[cur_idx[0]: cur_idx[1],
                      cur_idx[2]: cur_idx[3], ...]
        if n_dim == 3 and len(cur_idx) == 6:
            img_id = '{}_sl{}-{}'.format(img_id, 0, im_depth)

            if isotropic:
                img_shape = cropped_img.shape
                isotropic_shape = [img_shape[0], ] * len(img_shape)
                cropped_img = resize_image(cropped_img, isotropic_shape)

        cropped_img_list.append((img_id, cropped_img))
    return cropped_img_list


def create_mask(input_image, str_elem_size=3):
    """Create a binary mask using morphological operations

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
