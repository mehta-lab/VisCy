"""Utility functions for processing images"""
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


