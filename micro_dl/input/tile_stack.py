"""Tile images for training"""

import cv2
import numpy as np
import os
import pandas as pd
import pickle

from micro_dl.input.gen_crop_masks import MaskProcessor
from micro_dl.utils.aux_utils import get_row_idx, validate_tp_channel
from micro_dl.utils.normalize import hist_clipping, zscore
from micro_dl.utils.aux_utils import save_tile_meta
import micro_dl.utils.image_utils as image_utils


class ImageStackTiler:
    """Tiles all images images in a stack"""

    def __init__(self, base_output_dir, tile_size, step_size,
                 timepoint_ids=-1, tile_channels=-1, correct_flat_field=False,
                 isotropic=False, meta_path=None):
        """Init

        Isotropic here refers to the same dimension/shape along row, col, slice
        and not really isotropic resolution in mm.

        :param str base_output_dir: base folder for storing the individual
         and tiled images
        :param list/tuple/np array tile_size: size of the blocks to be cropped
         from the image
        :param list/tuple/np array step_size: size of the window shift. In case
         of no overlap, the step size is tile_size. If overlap, step_size <
         tile_size
        :param list/int timepoint_ids: timepoints to consider
        :param list/int tile_channels: tile images in the given channels.
         default=-1, tile all channels
        :param bool correct_flat_field: bool indicator for correcting for flat
         field
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param meta_path: If none, assume metadata csv is in base_output_dir
            + split_images/ and is named split_images_info.csv
        :return: a list with tuples - (cropped image id of the format
         rrmin-rmax_ccmin-cmax_slslmin-slmax, cropped image)
        """

        self.base_output_dir = base_output_dir
        if meta_path is None:
            volume_metadata = pd.read_csv(os.path.join(
                self.base_output_dir, 'split_images', 'split_images_info.csv'
            ))
        else:
            volume_metadata = pd.read_csv(meta_path)
        self.volume_metadata = volume_metadata

        tp_channel_ids = validate_tp_channel(volume_metadata,
                                             timepoint_ids=timepoint_ids,
                                             channel_ids=tile_channels)

        self.tile_channels = tp_channel_ids['channels']
        self.timepoint_ids = tp_channel_ids['timepoints']

        self.tile_size = tile_size
        self.step_size = step_size

        str_tile_size = '-'.join([str(val) for val in tile_size])
        str_step_size = '-'.join([str(val) for val in step_size])
        tiled_dir_name = 'image_tile_{}_step_{}'.format(str_tile_size,
                                                        str_step_size)
        tiled_dir = os.path.join(base_output_dir, tiled_dir_name)

        self.tiled_dir = tiled_dir
        self.isotropic = isotropic
        self.correct_flat_field = correct_flat_field

    @staticmethod
    def _save_tiled_images(cropped_image_info, meta_row,
                           channel_dir, cropped_meta):
        """Save cropped images for individual/sample image

        :param list cropped_image_info: a list with tuples (cropped image id
         of the format rrmin-rmax_ccmin-cmax_slslmin-slmax and cropped image)
         for the current image
        :param pd.DataFrame(row) meta_row: row of metadata from split images
        :param str channel_dir: dir to save cropped images
        :param list cropped_meta: list of tuples with (cropped image id of the
        format rrmin-rmax_ccmin-cmax_slslmin-slmax, cropped image) for all
        images in current channel
        """

        for id_img_tuple in cropped_image_info:
            rcsl_idx = id_img_tuple[0]
            img_fname = 'n{}_{}'.format(meta_row['sample_num'], rcsl_idx)
            cropped_img = id_img_tuple[1]
            cropped_img_fname = os.path.join(
                channel_dir, '{}.npy'.format(img_fname)
            )
            np.save(cropped_img_fname, cropped_img,
                    allow_pickle=True, fix_imports=True)
            cropped_meta.append(
                (meta_row['timepoint'], meta_row['channel_num'],
                 meta_row['sample_num'], meta_row['slice_num'],
                 cropped_img_fname)
            )

    def _tile_channel(self, tile_function, channel_dir, channel_metadata,
                      flat_field_image, hist_clip_limits, metadata,
                      crop_indices=None):
        """Tiles and saves tiles for one channel

        :param str channel_dir: dir for saving tiled images
        :param pd.DataFrame channel_metadata: DF with meta info for the current
         channel
        :param np.array flat_field_image: flat_filed image for this channel
        :param list hist_clip_limits: lower and upper hist clipping limits
        :param list metadata: list of tuples with tiled info (cropped image id
        of the format rrmin-rmax_ccmin-cmax_slslmin-slmax, cropped image)
        :param dict of lists crop_indices: dict with key as fname and values
         are list of crop indices
        """

        for _, row in channel_metadata.iterrows():
            sample_fname = row['fname']
            # Read npy or image
            if sample_fname[-3:] == 'npy':
                cur_image = np.load(sample_fname)
            else:
                cur_image = cv2.imread(sample_fname, cv2.IMREAD_ANYDEPTH)

            if self.correct_flat_field:
                cur_image = image_utils.apply_flat_field_correction(
                    cur_image, flat_field_image=flat_field_image
                )
            # normalize
            if hist_clip_limits is not None:
                cur_image = hist_clipping(cur_image,
                                          hist_clip_limits[0],
                                          hist_clip_limits[1])
            cur_image = zscore(cur_image)
            if tile_function == image_utils.tile_image:
                cropped_image_data = tile_function(
                    input_image=cur_image, tile_size=self.tile_size,
                    step_size=self.step_size, isotropic=self.isotropic
                )
            elif tile_function == image_utils.crop_at_indices:
                assert crop_indices is not None
                _, fname = os.path.split(sample_fname)
                cropped_image_data = tile_function(
                    input_image=cur_image,
                    crop_indices=crop_indices[fname],
                    isotropic=self.isotropic
                )
            else:
                raise ValueError('tile function invalid')
            self._save_tiled_images(cropped_image_data, row,
                                    channel_dir, metadata)

    def tile_stack(self, focal_plane_idx=None, hist_clip_limits=None):
        """Tiles images in the specified channels.

        Saves a csv with columns ['timepoint', 'channel_num', 'sample_num',
        'slice_num', 'fname_0',  'fname_1',... ] for all the tiles

        :param list hist_clip_limits: lower and upper percentiles used for
         histogram clipping.
        :param int focal_plane_idx: Index of which focal plane acquisition to
         use (2D).
        """

        os.makedirs(self.tiled_dir, exist_ok=True)
        for tp_idx in self.timepoint_ids:
            tp_dir = os.path.join(self.tiled_dir,
                                  'timepoint_{}'.format(tp_idx))
            os.makedirs(tp_dir, exist_ok=True)
            for channel in self.tile_channels:
                row_idx = get_row_idx(
                    self.volume_metadata, tp_idx, channel, focal_plane_idx
                )
                channel_metadata = self.volume_metadata[row_idx]
                channel_dir = os.path.join(tp_dir,
                                           'channel_{}'.format(channel))
                os.makedirs(channel_dir, exist_ok=True)
                if self.correct_flat_field:
                    flat_field_image = np.load(
                        os.path.join(
                            self.base_output_dir,
                            'split_images',
                            'flat_field_images',
                            'flat-field_channel-{}.npy'.format(channel)
                        )
                    )
                else:
                    flat_field_image = None
                metadata = []
                self._tile_channel(image_utils.tile_image,
                                   channel_dir, channel_metadata,
                                   flat_field_image, hist_clip_limits,
                                   metadata)
                save_tile_meta(metadata, channel, self.tiled_dir)

    def tile_stack_with_vf_constraint(self, mask_channels, min_fraction,
                                      save_cropped_masks=False,
                                      isotropic=False,
                                      focal_plane_idx=None,
                                      hist_clip_limits=None):
        """Crop and retain tiles that have minimum foreground

        Minimum foreground is defined as the percent of non-zero pixels/ volume
        fraction in a mask which is a thresholded sum of flurophore image.

        :param int/list mask_channels: generate mask from the sum of these
         (flurophore) channels
        :param float min_fraction: threshold for using a cropped image for
         training. minimum volume fraction / percent occupied in cropped image
        :param list hist_clip_limits: lower and upper percentiles used for
         histogram clipping.
        :param bool save_cropped_masks: bool indicator for saving cropped masks
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param int focal_plane_idx: Index of which focal plane acquisition to
         use
        """

        if isinstance(mask_channels, int):
            mask_channels = [mask_channels]
        mask_dir_name = '-'.join(map(str, mask_channels))
        mask_dir_name = 'mask_{}_vf-{}'.format(mask_dir_name, min_fraction)

        tiled_dir = '{}_vf-{}'.format(self.tiled_dir, min_fraction)

        os.makedirs(tiled_dir, exist_ok=True)

        for tp_idx in self.timepoint_ids:
            tp_dir = os.path.join(tiled_dir,
                                  'timepoint_{}'.format(tp_idx))
            os.makedirs(tp_dir, exist_ok=True)

            crop_indices_fname = os.path.join(
                self.base_output_dir, 'split_images',
                'timepoint_{}'.format(tp_idx),
                '{}.pkl'.format(mask_dir_name)
            )

            if not os.path.exists(crop_indices_fname):
                mask_gen_obj = MaskProcessor(
                    os.path.join(self.base_output_dir, 'split_images'),
                    mask_channels,
                    tp_idx
                )
                if save_cropped_masks:
                    cropped_mask_dir = os.path.join(tp_dir, mask_dir_name)
                    os.makedirs(cropped_mask_dir, exist_ok=True)
                else:
                    cropped_mask_dir = None
                mask_gen_obj.get_crop_indices(min_fraction, self.tile_size,
                                              self.step_size, cropped_mask_dir,
                                              save_cropped_masks, isotropic)

            with open(crop_indices_fname, 'rb') as f:
                crop_indices_dict = pickle.load(f)

            for channel in self.tile_channels:
                row_idx = get_row_idx(
                    self.volume_metadata, tp_idx, channel, focal_plane_idx
                )
                channel_metadata = self.volume_metadata[row_idx]
                channel_dir = os.path.join(tp_dir,
                                           'channel_{}'.format(channel))
                os.makedirs(channel_dir, exist_ok=True)

                flat_field_image = np.load(
                    os.path.join(self.base_output_dir, 'split_images',
                                 'flat_field_images',
                                 'flat-field_channel-{}.npy'.format(channel)
                                 )
                )

                metadata = []
                self._tile_channel(image_utils.crop_at_indices,
                                   channel_dir, channel_metadata,
                                   flat_field_image, hist_clip_limits,
                                   metadata, crop_indices_dict)
                save_tile_meta(metadata, channel, tiled_dir)
