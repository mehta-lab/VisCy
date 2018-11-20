"""Tile images for training"""

import numpy as np
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils


class ImageTiler:
    """Tiles all images images in a dataset"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 tile_dict,
                 tile_size=[256, 256],
                 step_size=[64, 64],
                 depths=1,
                 mask_depth=1,
                 time_ids=-1,
                 channel_ids=-1,
                 slice_ids=-1,
                 pos_ids=-1,
                 hist_clip_limits=None,
                 flat_field_dir=None,
                 isotropic=False,
                 data_format='channels_first'):
        """
        Normalizes images using z-score, then tiles them.
        Isotropic here refers to the same dimension/shape along row, col, slice
        and not really isotropic resolution in mm.

        :param str input_dir: Directory with frames to be tiled
        :param str output_dir: Base output directory
        :param list tile_size: size of the blocks to be cropped
         from the image
        :param list step_size: size of the window shift. In case
         of no overlap, the step size is tile_size. If overlap, step_size <
         tile_size
        :param int/list depths: The z depth for generating stack training data
            Default 1 assumes 2D data for all channels to be tiled.
            For cases where input and target shapes are not the same (e.g. stack
             to 2D) you should specify depths for each channel in tile.channels.
        :param int mask_depth: Depth for mask channel
        :param list/int time_ids: Tile given timepoint indices
        :param list/int tile_channels: Tile images in the given channel indices
         default=-1, tile all channels
        :param int slice_ids: Index of which focal plane acquisition to
         use (for 2D). default=-1 for the whole z-stack
        :param int pos_ids: Position (FOV) indices to use
        :param list hist_clip_limits: lower and upper percentiles used for
         histogram clipping.
        :param str flat_field_dir: Flatfield directory. None if no flatfield
            correction
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param str data_format: Channels first or last
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        if 'depths' in tile_dict:
            depths = tile_dict['depths']
        if 'mask_depth' in tile_dict:
            mask_depth = tile_dict['mask_depth']
        if 'tile_size' in tile_dict:
            tile_size = tile_dict['tile_size']
        if 'step_size' in tile_dict:
            step_size = tile_dict['step_size']
        if 'isotropic' in tile_dict:
            isotropic = tile_dict['isotropic']
        if 'channels' in tile_dict:
            channel_ids = tile_dict['channels']
        if 'positions' in tile_dict:
            pos_ids = tile_dict['positions']
        if 'hist_clip_limits' in tile_dict:
            hist_clip_limits = tile_dict['hist_clip_limits']
        if 'data_format' in tile_dict:
            data_format = tile_dict['data_format']
            assert data_format in {'channels_first', 'channels_last'},\
                "Data format must be channels_first or channels_last"
        self.depths = depths
        self.mask_depth = mask_depth
        self.tile_size = tile_size
        self.step_size = step_size
        self.isotropic = isotropic
        self.hist_clip_limits = hist_clip_limits
        self.data_format = data_format

        self.str_tile_step = 'tiles_{}_step_{}'.format(
            '-'.join([str(val) for val in tile_size]),
            '-'.join([str(val) for val in step_size]),
        )
        self.tile_dir = os.path.join(
            output_dir,
            self.str_tile_step,
        )
        # If tile dir already exist, things could get messy because we don't
        # have any checks in place for how to add to existing tiles
        try:
            os.makedirs(self.tile_dir, exist_ok=False)
        except FileExistsError as e:
            print("You're trying to write to existing dir. ", e)
            raise

        self.tile_mask_dir = None
        self.flat_field_dir = flat_field_dir
        self.frames_metadata = aux_utils.read_meta(self.input_dir)
        # Get metadata indices
        metadata_ids = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
        )
        self.channel_ids = metadata_ids['channel_ids']
        self.time_ids = metadata_ids['time_ids']
        self.slice_ids = metadata_ids['slice_ids']
        self.pos_ids = metadata_ids['pos_ids']
        # If more than one depth is specified, they must match channel ids
        if isinstance(self.depths, list):
            assert len(self.depths) == len(self.channel_ids),\
             "depths ({}) and channels ({}) length mismatch".format(
                len(self.depths), len(self.channel_ids)
            )
            # Get max of all specified depths
            max_depth = max(max(self.depths), self.mask_depth)
            # Convert channels + depths to dict for lookup
            self.channel_depth = dict(zip(self.channel_ids, self.depths))
        else:
            max_depth = max(self.depths, self.mask_depth)
            self.channel_depth = dict(zip(
                self.channel_ids,
                [self.depths] * len(self.channel_ids)),
            )
        # Adjust slice margins
        self.slice_ids = aux_utils.adjust_slice_margins(
            slice_ids=self.slice_ids,
            depth=max_depth,
        )

    def get_tile_dir(self):
        """
        Return directory containing tiles
        :return str tile_dir: Directory with tiles
        """
        return self.tile_dir

    def get_tile_mask_dir(self):
        """
        Return directory containing tiles of mask
        :return str tile_mask_dir: Directory with tiled mask
        """
        return self.tile_mask_dir

    def _write_tiled_data(self,
                          tiled_data,
                          save_dir,
                          time_idx=None,
                          channel_idx=None,
                          slice_idx=None,
                          pos_idx=None,
                          tile_indices=None,
                          tiled_metadata=None,
                          ):
        """
        Loops through tuple and writes all tile image data. Adds row to metadata
        dataframe as well if that is present.

        :param list of tuples tiled_data: Tile name and np.array
        :param str save_dir: Directory where tiles will be written
        :param int time_idx: Time index
        :param int channel_idx: Channel index
        :param int slice_idx: Slice (z) index
        :param int pos_idx: Position (FOV) index
        :param list of tuples tile_indices: Tile indices
        :param dataframe tiled_metadata: Dataframe containing metadata for all
         tiles
        :return dataframe tiled_metadata: Metadata with rows added to it
        """
        for i, data_tuple in enumerate(tiled_data):
            rcsl_idx = data_tuple[0]
            file_name = aux_utils.get_im_name(
                time_idx=time_idx,
                channel_idx=channel_idx,
                slice_idx=slice_idx,
                pos_idx=pos_idx,
                extra_field=rcsl_idx,
            )
            tile = data_tuple[1]
            # Check and potentially flip dimensions for 3D data
            if self.data_format == 'channels_first' and len(tile.shape) > 2:
                tile = np.transpose(tile, (2, 0, 1))
            np.save(os.path.join(save_dir, file_name),
                    tile,
                    allow_pickle=True,
                    fix_imports=True)
            tile_idx = tile_indices[i]
            if tiled_metadata is not None:
                tiled_metadata = tiled_metadata.append(
                    {"channel_idx": channel_idx,
                     "slice_idx": slice_idx,
                     "time_idx": time_idx,
                     "file_name": file_name,
                     "pos_idx": pos_idx,
                     "row_start": tile_idx[0],
                     "col_start": tile_idx[2],
                     },
                    ignore_index=True,
                )
        return tiled_metadata

    def _get_flat_field(self, channel_idx):
        """
        Get flat field image for a given channel index

        :param int channel_idx: Channel index
        :return np.array flat_field_im: flat field image for channel
        """
        flat_field_im = None
        if self.flat_field_dir is not None:
            flat_field_im = np.load(
                os.path.join(
                    self.flat_field_dir,
                    'flat-field_channel-{}.npy'.format(channel_idx),
                )
            )
        return flat_field_im

    def _get_dataframe(self):
        """
        Creates an empty dataframe with metadata column names for tiles. It's
        the same names as for frames, but with channel_name removed and with
        the addition of row_start and col_start.
        TODO: Should I also save row_end and col_end while I'm at it?
        Might be useful if we want to recreate tiles from a previous preprocessing
        with mask run... Or just retrieve tile_size from preprocessing_info...
        This is one of the functions that will have to be adapted once tested on
        3D data.

        :return dataframe tiled_metadata
        """
        return pd.DataFrame(columns=[
            "channel_idx",
            "slice_idx",
            "time_idx",
            "file_name",
            "pos_idx",
            "row_start",
            "col_start"])

    def tile_stack(self):
        """
        Tiles images in the specified channels.

        Saves a csv with columns
        ['time_idx', 'channel_idx', 'pos_idx','slice_idx', 'file_name']
        for all the tiles
        """
        tiled_metadata = self._get_dataframe()
        tile_indices = None
        for channel_idx in self.channel_ids:
            # Perform flatfield correction if flatfield dir is specified
            flat_field_im = self._get_flat_field(channel_idx=channel_idx)

            for slice_idx in self.slice_ids:
                for time_idx in self.time_ids:
                    for pos_idx in self.pos_ids:
                        im = image_utils.preprocess_imstack(
                            frames_metadata=self.frames_metadata,
                            input_dir=self.input_dir,
                            depth=self.channel_depth[channel_idx],
                            time_idx=time_idx,
                            channel_idx=channel_idx,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                            flat_field_im=flat_field_im,
                            hist_clip_limits=self.hist_clip_limits,
                        )
                        # Now to the actual tiling
                        if tile_indices is None:
                            tiled_image_data, tile_indices =\
                                image_utils.tile_image(
                                    input_image=im,
                                    tile_size=self.tile_size,
                                    step_size=self.step_size,
                                    isotropic=self.isotropic,
                                    return_index=True,
                                )
                        else:
                            tiled_image_data = image_utils.crop_at_indices(
                                input_image=im,
                                crop_indices=tile_indices,
                                isotropic=self.isotropic,
                            )
                        tiled_metadata = self._write_tiled_data(
                            tiled_image_data,
                            save_dir=self.tile_dir,
                            time_idx=time_idx,
                            channel_idx=channel_idx,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                            tile_indices=tile_indices,
                            tiled_metadata=tiled_metadata,
                        )
        # Finally, save all the metadata
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=",",
        )

    def _get_mask(self, time_idx, mask_channel, slice_idx, pos_idx, mask_dir):
        """
        Load a mask image or an image stack, depending on depth

        :param int time_idx: Time index
        :param str mask_channel: Channel index for mask
        :param int slice_idx: Slice (z) index
        :param int pos_idx: Position index
        :param str mask_dir: Directory containing masks
        :return np.array im_stack: Mask image/stack
        """
        margin = self.mask_depth // 2
        im_stack = []
        for z in range(slice_idx - margin, slice_idx + margin + 1):
            file_name = aux_utils.get_im_name(
                time_idx=time_idx,
                channel_idx=mask_channel,
                slice_idx=z,
                pos_idx=pos_idx,
            )
            file_path = os.path.join(
                mask_dir,
                file_name,
            )
            im_stack.append(image_utils.read_image(file_path))
        # Stack images
        return np.stack(im_stack, axis=2)

    def tile_mask_stack(self,
                        mask_dir=None,
                        save_tiled_masks=None,
                        mask_channel=None,
                        min_fraction=None,
                        isotropic=False):
        """
        Tiles images in the specified channels assuming there are masks
        already created in mask_dir. Only tiles above a certain fraction
        of foreground in mask tile will be saved and added to metadata.

        Saves a csv with columns ['time_idx', 'channel_idx', 'pos_idx',
        'slice_idx', 'file_name'] for all the tiles

        :param str mask_dir: Directory containing masks
        :param str save_tiled_masks: How/if to save mask tiles. If None, don't
            save masks.
            If 'as_channel', save masked tiles as a channel given
            by mask_channel in tile_dir.
            If 'as_masks', create a new tile_mask_dir and save them there
        :param str mask_channel: Channel number assigned to mask
        :param float min_fraction: Minimum fraction of foreground in tiled masks
        :param bool isotropic: Indicator of isotropy
        """
        if save_tiled_masks == 'as_masks':
            self.tile_mask_dir = os.path.join(
                self.output_dir,
                'mask_' + '-'.join(map(str, self.channel_ids)) +
                self.str_tile_step,
            )
            os.makedirs(self.tile_mask_dir, exist_ok=True)
        elif save_tiled_masks == 'as_channel':
            self.tile_mask_dir = self.tile_dir

        tiled_metadata = self._get_dataframe()
        mask_metadata = self._get_dataframe()
        # Load flatfield images if flatfield dir is specified
        flat_field_im = None
        if self.flat_field_dir is not None:
            flat_field_ims = []
            for channel_idx in self.channel_ids:
                flat_field_ims.append(self._get_flat_field(channel_idx))

        for slice_idx in self.slice_ids:
            for time_idx in self.time_ids:
                for pos_idx in np.unique(self.frames_metadata["pos_idx"]):
                    # Since masks are generated across channels, we only need
                    # load them once across channels
                    mask_image = self._get_mask(
                        time_idx=time_idx,
                        mask_channel=mask_channel,
                        slice_idx=slice_idx,
                        pos_idx=pos_idx,
                        mask_dir=mask_dir)
                    tiled_mask_data, tile_indices = image_utils.tile_image(
                        input_image=mask_image,
                        min_fraction=min_fraction,
                        tile_size=self.tile_size,
                        step_size=self.step_size,
                        isotropic=isotropic,
                        return_index=True,
                    )
                    # Loop through all the mask tiles, write tiled masks
                    mask_metadata = self._write_tiled_data(
                        tiled_data=tiled_mask_data,
                        save_dir=self.tile_mask_dir,
                        time_idx=time_idx,
                        channel_idx=mask_channel,
                        slice_idx=slice_idx,
                        pos_idx=pos_idx,
                        tile_indices=tile_indices,
                        tiled_metadata=mask_metadata,
                    )
                    # Loop through all channels and tile from indices
                    for i, channel_idx in enumerate(self.channel_ids):

                        if self.flat_field_dir is not None:
                            flat_field_im = flat_field_ims[i]

                        im = image_utils.preprocess_imstack(
                            frames_metadata=self.frames_metadata,
                            input_dir=self.input_dir,
                            depth=self.channel_depth[channel_idx],
                            time_idx=time_idx,
                            channel_idx=channel_idx,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                            flat_field_im=flat_field_im,
                            hist_clip_limits=self.hist_clip_limits,
                        )
                        # Now to the actual tiling of data
                        tiled_image_data = image_utils.crop_at_indices(
                            input_image=im,
                            crop_indices=tile_indices,
                            isotropic=self.isotropic,
                        )
                        # Loop through all the tiles, write and add to metadata
                        tiled_metadata = self._write_tiled_data(
                            tiled_data=tiled_image_data,
                            save_dir=self.tile_dir,
                            time_idx=time_idx,
                            channel_idx=channel_idx,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                            tile_indices=tile_indices,
                            tiled_metadata=tiled_metadata,
                        )

        # Finally, save all the metadata
        if self.tile_mask_dir == self.tile_dir:
            tiled_metadata = tiled_metadata.append(
                mask_metadata,
                ignore_index=True,
            )
        else:
            mask_metadata.to_csv(
                os.path.join(self.tile_mask_dir, "frames_meta.csv"),
                sep=",",
            )
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=",",
        )
