"""Estimate flat field images"""

from abc import ABCMeta, abstractmethod
import numpy as np
import os
import pandas as pd

from micro_dl.utils.aux_utils import get_row_idx
from micro_dl.utils.image_utils import fit_polynomial_surface_2D


class FlatFieldEstimator(metaclass=ABCMeta):
    """Estimates flat field correction image"""

    def __init__(self, image_dir):
        """Init.

        :param str image_dir: base dir with individual/split images from stack
        """

        self.image_dir = image_dir

    @abstractmethod
    def sample_block_medians(self, im, block_size=32):
        """Subdivide a 2D image in smaller blocks of size block_size and
        compute the median intensity value for each block. Any incomplete
        blocks (remainders of modulo operation) will be ignored.
        """

        raise NotImplementedError

    @abstractmethod
    def get_flatfield(self, im, block_size=32, order=2, normalize=True):
        """
        Combine sampling and polynomial surface fit for flatfield estimation.
        To flatfield correct an image, divide it by flatfield.
        """

        raise NotImplementedError

    def estimate_flat_field(self, focal_plane_idx=None):
        """Estimates flat field correction image.

        :param int focal_plane_idx: for 2D acquisitions with multiple images
         along focal plane/axis, this specifies the plane to use
        """

        meta_fname = os.path.join(self.image_dir, 'split_images_info.csv')
        try:
            volume_metadata = pd.read_csv(meta_fname)
        except IOError as e:
            e.args += 'cannot read split image info'
            raise

        all_channels = volume_metadata['channel_num'].unique()
        flat_field_dir = os.path.join(self.image_dir, 'flat_field_images')
        os.makedirs(flat_field_dir, exist_ok=True)
        tp_idx = 0  # flat_field constant over time
        for channel_idx in all_channels:
            row_idx = get_row_idx(volume_metadata, tp_idx,
                                  channel_idx, focal_plane_idx)
            channel_metadata = volume_metadata[row_idx]
            for idx, row in channel_metadata.iterrows():
                sample_fname = row['fname']
                cur_image = np.load(sample_fname)
                n_dim = len(cur_image.shape)
                if n_dim == 3:
                    cur_image = np.mean(cur_image, axis=2)
                if idx == 0:
                    summed_image = cur_image.astype('float64')
                else:
                    summed_image += cur_image
            mean_image = summed_image / len(row_idx)
            # TODO (Jenny): it currently samples median values from a mean
            # images, not very statistically meaningful but easier than
            # computing median of image stack
            flatfield = self.get_flatfield(mean_image)
            fname = 'flat-field_channel-{}.npy'.format(channel_idx)
            cur_fname = os.path.join(flat_field_dir, fname)
            np.save(cur_fname, flatfield, allow_pickle=True, fix_imports=True)


class FlatFieldEstimator2D(FlatFieldEstimator):
    """Estimate flat field from 2D mean images"""

    def sample_block_medians(self, im, block_size=32):
        """Subdivide a 2D image in smaller blocks of size block_size and
        compute the median intensity value for each block. Any incomplete
        blocks (remainders of modulo operation) will be ignored.

        :param np.array im:         2D image
        :param int block_size:      Size of blocks image will be divided into

        :return np.array(float) sample_coords: Image coordinates for block
                                               centers
        :return np.array(float) sample_values: Median intensity values for
                                               blocks
        """

        im_shape = im.shape
        assert block_size < im_shape[0], "Block size larger than image height"
        assert block_size < im_shape[1], "Block size larger than image width"

        nbr_blocks_x = im_shape[0] // block_size
        nbr_blocks_y = im_shape[1] // block_size
        sample_coords = np.zeros((nbr_blocks_x * nbr_blocks_y, 2),
                                 dtype=np.float64)
        sample_values = np.zeros((nbr_blocks_x * nbr_blocks_y, ),
                                 dtype=np.float64)
        for x in range(nbr_blocks_x):
            for y in range(nbr_blocks_y):
                idx = y * nbr_blocks_x + x
                sample_coords[idx, :] = [x * block_size + (block_size - 1) / 2,
                                         y * block_size + (block_size - 1) / 2]
                sample_values[idx] = np.median(
                    im[x * block_size:(x + 1) * block_size,
                       y * block_size:(y + 1) * block_size]
                )
        return sample_coords, sample_values

    def get_flatfield(self, im, block_size=32, order=2, normalize=True):
        """
        Combine sampling and polynomial surface fit for flatfield estimation.
        To flatfield correct an image, divide it by flatfield.

        :param np.array im:        2D image
        :param int block_size:     Size of blocks image will be divided into
        :param int order:          Order of polynomial (default 2)
        :param bool normalize:     Normalize surface by dividing by its mean
                                   for flatfield correction (default True)

        :return np.array flatfield:    Flatfield image
        """

        coords, values = self.sample_block_medians(im, block_size=block_size)
        flatfield = fit_polynomial_surface_2D(coords, values, im.shape,
                                              order=order, normalize=normalize)
        # Flatfields can't contain zeros or negative values
        if flatfield.min() <= 0:
            raise ValueError(
                "The generated flatfield was not strictly positive."
            )
        return flatfield
