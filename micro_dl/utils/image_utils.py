"""Utility functions for processing images"""
import numpy as np
from skimage.transform import resize


def crop_image(input_image, tile_size, step_size, isotropic=False):
    """Creates 3D blocks from the image from given crop and overlap size.

    :param np.array input_image: input image in 3d
    :param list/tuple/np array tile_size: size of the blocks to be cropped
     from the image
    :param list/tuple/np array step_size: size of the window shift. In case of
     no overlap, the step size is tile_size. If overlap, step_size < tile_size
    :return: a list with tuples of cropped image id of the format
     xxmin-xmz_yymin-ymax_zzmin-zmax and cropped image
    """

    assert len(tile_size) == len(step_size)
    assert np.all(tile_size) > 0
    size_x = input_image.shape[0]
    size_y = input_image.shape[1]
    step_size_x = step_size[0]
    step_size_y = step_size[1]

    cropped_image_list = []
    if len(input_image.shape) == 3:
        size_z = input_image.shape[2]
        step_size_z = step_size[2]
        for z in range(0, size_z - step_size_z + 1, step_size_z):
            for y in range(0, size_y - step_size_y + 1, step_size_y):
                for x in range(0, size_x - step_size_x + 1, step_size_x):
                    img_id = 'x{}-{}_y{}-{}_z{}-{}'.format(
                        x, x + tile_size[0], y, y + tile_size[1],
                        z, z + tile_size[2]
                    )
                    cropped_img = input_image[x : x + tile_size[0],
                                              y : y + tile_size[1],
                                              z : z + tile_size[2]]
                    if isotropic:
                        isotropic_shape = [tile_size[0], ] * len(tile_size)
                        cond = list(tile_size) == isotropic_shape
                        if not cond:
                            cropped_img = resample_image(cropped_img,
                                                         isotropic_shape)
                    # tiled_img = np.rollaxis(tiled_img, 2, 0)
                    cropped_image_list.append((img_id, cropped_img))
    else:
        for y in range(0, size_y - step_size_y + 1, step_size_y):
            for x in range(0, size_x - step_size_x + 1, step_size_x):
                img_id = 'x{}-{}_y{}-{}'.format(x, x + tile_size[0],
                                                y, y + tile_size[1])
                cropped_img = input_image[x : x + tile_size[0],
                                          y : y + tile_size[1]]
                cropped_image_list.append((img_id, cropped_img))
    return cropped_image_list


def normalize_zscore(input_image):
    """Performs z-score normalization

    :param input_image:
    :return:
    """

    norm_img = (input_image - np.mean(input_image)) / np.std(input_image)
    return norm_img


def resample_image(input_image, output_shape):
    """Resize image to a specified shape

    :param np.ndarray input_image: image to be resized
    :param tuple/np.array output_shape: desired shape of the output image
    """

    msg = 'the output shape does not match the image dimension'
    assert len(output_shape) == len(input_image.shape), msg
    assert input_image.dtype is not 'bool'

    resized_image = resize(input_image, output_shape)
    return resized_image


def resample_label(input_image, target_size):
    """Resample label/bool images"""
    raise NotImplementedError
