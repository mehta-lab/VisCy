"""Estimate flat field images"""

import numpy as np
import os

import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.image_utils import fit_polynomial_surface_2D, read_image


class FlatFieldEstimator2D:
    """Estimates flat field image"""

    def __init__(self, input_dir, output_dir, slice_ids):
        """
        Flatfield images are estimated once per channel for 2D data

        :param str input_dir: Directory with 2D image frames from dataset
        :param str output_dir: Base output directory
        :param int/list slice_ids: Z slice indices for flatfield correction
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        # Create flat_field_dir as a subdirectory of output_dir
        self.flat_field_dir = os.path.join(self.output_dir,
                                           'flat_field_images')
        os.makedirs(self.flat_field_dir, exist_ok=True)
        self.slice_ids = slice_ids
        self.frames_metadata = aux_utils.read_meta(self.input_dir)
        metadata_ids, _ = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            channel_ids=-1,
            slice_ids=slice_ids,
        )
        self.channels_ids = metadata_ids['channel_ids']
        self.slice_ids = metadata_ids['slice_ids']

    def get_flat_field_dir(self):
        """
        Return flatfield directory
        :return str flat_field_dir: Flatfield directory
        """
        return self.flat_field_dir

    def estimate_flat_field(self):
        """
        Estimates flat field correction image.
        """
        # flat_field constant over time, so time_idx=0. And use only first
        # slice if multiple are present
        for channel_idx in self.channels_ids:
            row_idx = aux_utils.get_row_idx(
                frames_metadata=self.frames_metadata,
                time_idx=0,
                channel_idx=channel_idx,
                slice_idx=self.slice_ids[0],
            )
            channel_metadata = self.frames_metadata[row_idx]
            for idx, row in channel_metadata.iterrows():
                file_path = os.path.join(self.input_dir, row['file_name'])
                im = read_image(file_path)
                if len(im.shape) == 3:
                    im = np.mean(im, axis=2)
                if idx == 0:
                    summed_image = im.astype('float64')
                else:
                    summed_image += im
            mean_image = summed_image / len(row_idx)
            # TODO (Jenny): it currently samples median values from a mean
            # images, not very statistically meaningful but easier than
            # computing median of image stack
            flatfield = self.get_flatfield(mean_image)
            fname = 'flat-field_channel-{}.npy'.format(channel_idx)
            cur_fname = os.path.join(self.flat_field_dir, fname)
            np.save(cur_fname, flatfield, allow_pickle=True, fix_imports=True)

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

        coords, values = self.sample_block_medians(
            im=im,
            block_size=block_size,
        )
        flatfield = fit_polynomial_surface_2D(
            sample_coords=coords,
            sample_values=values,
            im_shape=im.shape,
            order=order,
            normalize=normalize,
        )
        # Flatfields can't contain zeros or negative values
        if flatfield.min() <= 0:
            raise ValueError(
                "The generated flatfield was not strictly positive."
            )
        return flatfield
