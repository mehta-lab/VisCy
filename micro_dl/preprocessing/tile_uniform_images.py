import numpy as np
import os
import pandas as pd

import micro_dl.utils.tile_utils as tile_utils
from micro_dl.utils import aux_utils as aux_utils
from micro_dl.utils.mp_utils import mp_tile_save, mp_crop_save


class ImageTilerUniform:
    """Tiles all images images in a dataset"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 tile_dict,
                 tile_size=[256, 256],
                 step_size=[64, 64],
                 depths=1,
                 time_ids=-1,
                 channel_ids=-1,
                 slice_ids=-1,
                 pos_ids=-1,
                 hist_clip_limits=None,
                 flat_field_dir=None,
                 isotropic=False,
                 image_format='zyx',
                 num_workers=4,
                 int2str_len=3):
        """
        Normalizes images using z-score, then tiles them.
        Isotropic here refers to the same dimension/shape along row, col, slice
        and not really isotropic resolution in mm.
        If tile_dir already exist, it will check which channels are already
        tiled, get indices from them and tile from indices only on the channels
        not already present.

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
        :param str image_format: zyx (preferred) or yxz
        :param int num_workers: number of workers for multiprocessing
        :param int int2str_len: number of characters for each idx to be used
         in file names
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        if 'depths' in tile_dict:
            depths = tile_dict['depths']
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
        if 'image_format' in tile_dict:
            image_format = tile_dict['image_format']
            assert image_format in {'zyx', 'yxz'}, \
                'Data format must be zyx or yxz'
        self.depths = depths
        self.tile_size = tile_size
        self.step_size = step_size
        self.isotropic = isotropic
        self.hist_clip_limits = hist_clip_limits
        self.image_format = image_format
        self.num_workers = num_workers

        self.str_tile_step = 'tiles_{}_step_{}'.format(
            '-'.join([str(val) for val in tile_size]),
            '-'.join([str(val) for val in step_size]),
        )
        self.tile_dir = os.path.join(
            output_dir,
            self.str_tile_step,
        )

        # If tile dir already exist, only tile channels not already present
        self.tiles_exist = False
        # If tile dir already exist, things could get messy because we don't
        # have any checks in place for how to add to existing tiles
        try:
            os.makedirs(self.tile_dir, exist_ok=False)
            # make dir for saving indiv meta per image, could be used for
            # tracking job success / fail
            os.makedirs(os.path.join(self.tile_dir, 'meta_dir'),
                        exist_ok=False)
        except FileExistsError as e:
            print("Tile dir exists. Only add untiled channels.")
            self.tiles_exist = True

        # make dir for saving indiv meta per image, could be used for
        # tracking job success / fail
        os.makedirs(os.path.join(self.tile_dir, 'meta_dir'),
                    exist_ok=True)

        self.flat_field_dir = flat_field_dir
        self.frames_metadata = aux_utils.read_meta(self.input_dir)
        # Get metadata indices
        metadata_ids, _ = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
            uniform_structure=True
        )
        self.channel_ids = metadata_ids['channel_ids']
        self.time_ids = metadata_ids['time_ids']
        self.slice_ids = metadata_ids['slice_ids']
        self.pos_ids = metadata_ids['pos_ids']

        # If more than one depth is specified, length must match channel ids
        if isinstance(self.depths, list):
            assert len(self.depths) == len(self.channel_ids),\
             "depths ({}) and channels ({}) length mismatch".format(
                len(self.depths), len(self.channel_ids)
            )
            # Get max of all specified depths
            max_depth = max(self.depths)
            # Convert channels + depths to dict for lookup
            self.channel_depth = dict(zip(self.channel_ids, self.depths))
        else:
            # If depth is scalar, make depth the same for all channels
            max_depth = self.depths
            self.channel_depth = dict(zip(
                self.channel_ids,
                [self.depths] * len(self.channel_ids)),
            )

        # Adjust slice margins
        self.slice_ids = aux_utils.adjust_slice_margins(
            slice_ids=self.slice_ids,
            depth=max_depth,
        )
        self.int2str_len = int2str_len

    def get_tile_dir(self):
        """
        Return directory containing tiles
        :return str tile_dir: Directory with tiles
        """
        return self.tile_dir

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

    def _get_tile_indices(self, tiled_meta,
                          time_idx,
                          channel_idx,
                          pos_idx,
                          slice_idx):
        """Get the tile indices from saved meta data

        :param pd.DataFrame tiled_meta: DF with image level meta info
        :param int time_idx: time index for current image
        :param int channel_idx: channel index for current image
        :param int pos_idx: position / sample index for current image
        :param int slice_idx: slice index of current image
        :return list tile_indices: list of tile indices
        """

        # Get tile indices from one channel only
        c = tiled_meta['channel_idx'] == channel_idx
        z = tiled_meta['slice_idx'] == slice_idx
        p = tiled_meta['pos_idx'] == pos_idx
        t = tiled_meta['time_idx'] == time_idx
        channel_meta = tiled_meta[c & z & p & t]
        # Get tile_indices
        tile_indices = pd.concat([
            channel_meta['row_start'],
            channel_meta['row_start'].add(self.tile_size[0]),
            channel_meta['col_start'],
            channel_meta['col_start'].add(self.tile_size[1]),
        ], axis=1)
        # Match list format similar to tile_image
        tile_indices = tile_indices.values.tolist()
        return tile_indices

    def _get_tiled_data(self):
        """
        If tile directory already exists, check which channels have been
        processed and only tile new channels.

        :return dataframe tiled_meta: Metadata with previously tiled channels
        :return list of lists tile_indices: Nbr tiles x 4 indices with row
        start + stop and column start + stop indices
        """
        if self.tiles_exist:
            tiled_meta = aux_utils.read_meta(self.tile_dir)
            # Find untiled channels
            tiled_channels = np.unique(tiled_meta['channel_idx'])
            new_channels = list(set(self.channel_ids) -
                                set(tiled_channels))
            if len(new_channels) == 0:
                print('All channels in config have already been tiled')
                return
            self.channel_ids = new_channels
            tile_indices = self._get_tile_indices(
                tiled_meta=tiled_meta,
                time_idx=self.time_ids[0],
                channel_idx=tiled_channels[0],
                pos_idx=self.pos_ids[0],
                slice_idx=self.slice_ids[0]
            )
        else:
            tiled_meta = self._get_dataframe()
            tile_indices = None
        return tiled_meta, tile_indices

    def _get_input_fnames(self,
                          time_idx,
                          channel_idx,
                          slice_idx,
                          pos_idx,
                          mask_dir=None):
        """Get input_fnames

        :param int time_idx: Time index
        :param int channel_idx: Channel index
        :param int slice_idx: Slice (z) index
        :param int pos_idx: Position (FOV) index
        :param str mask_dir: Directory containing masks
        :return: list of input fnames
        """

        if mask_dir is None:
            depth = self.channel_depth[channel_idx]
        else:
            depth = self.mask_depth
        margin = 0 if depth == 1 else depth // 2
        im_fnames = []
        for z in range(slice_idx - margin, slice_idx + margin + 1):
            if mask_dir is not None:
                file_name = aux_utils.get_im_name(time_idx=time_idx,
                                                  channel_idx=channel_idx,
                                                  slice_idx=z,
                                                  pos_idx=pos_idx,
                                                  int2str_len=self.int2str_len)
                file_path = os.path.join(mask_dir,
                                         file_name)
            else:

                meta_idx = aux_utils.get_meta_idx(self.frames_metadata,
                                                  time_idx,
                                                  channel_idx,
                                                  z,
                                                  pos_idx)
                file_path = os.path.join(
                    self.input_dir,
                    self.frames_metadata.loc[meta_idx, 'file_name'],
                )
            # check if file_path exists
            im_fnames.append(file_path)
        return im_fnames

    def get_crop_tile_args(self,
                           channel_idx,
                           time_idx,
                           slice_idx,
                           pos_idx,
                           task_type,
                           tile_indices=None,
                           mask_dir=None,
                           min_fraction=None):
        """Gather arguments for cropping or tiling

        :param int channel_idx: channel index for current image
        :param int time_idx: time index for current image
        :param int slice_idx: slice index for current image
        :param int pos_idx: position / sample index for current image
        :param str task_type: crop or tile
        :param list tile_indices: list of tile indices
        :param str mask_dir: dir containing image level masks
        :param float min_fraction: min foreground volume fraction for use tile
        :return list cur_args: tuple of arguments for tiling
                list tile_indices: tile indices for current image
        """

        input_fnames = self._get_input_fnames(
            time_idx=time_idx,
            channel_idx=channel_idx,
            slice_idx=slice_idx,
            pos_idx=pos_idx,
            mask_dir=mask_dir
        )

        # no flat field correction for mask
        flat_field_fname = None
        if mask_dir is None:
            if self.flat_field_dir is not None:
                flat_field_fname = os.path.join(
                        self.flat_field_dir,
                        'flat-field_channel-{}.npy'.format(channel_idx)
                    )
        # no hist_clipping for mask as mask is bool
        hist_clip_limits = None
        if mask_dir is None:
            if self.hist_clip_limits is not None:
                hist_clip_limits = tuple(
                    self.hist_clip_limits
                )

        if task_type == 'crop':
            cur_args = (tuple(input_fnames),
                        flat_field_fname,
                        hist_clip_limits,
                        time_idx,
                        channel_idx,
                        pos_idx,
                        slice_idx,
                        tuple(tile_indices),
                        self.image_format,
                        self.isotropic,
                        self.tile_dir,
                        self.int2str_len)
        elif task_type == 'tile':
            cur_args = (tuple(input_fnames),
                        flat_field_fname,
                        hist_clip_limits,
                        time_idx,
                        channel_idx,
                        pos_idx,
                        slice_idx,
                        self.tile_size,
                        self.step_size,
                        min_fraction,
                        self.image_format,
                        self.isotropic,
                        self.tile_dir,
                        self.int2str_len)
        return cur_args

    def tile_stack(self):
        """
        Tiles images in the specified channels.

        https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

        Saves a csv with columns
        ['time_idx', 'channel_idx', 'pos_idx','slice_idx', 'file_name']
        for all the tiles
        """
        # Get or create tiled metadata and tile indices
        prev_tiled_metadata, tile_indices = self._get_tiled_data()
        tiled_meta0 = None
        fn_args = []
        for channel_idx in self.channel_ids:
            # Perform flatfield correction if flatfield dir is specified
            flat_field_im = self._get_flat_field(channel_idx=channel_idx)
            for slice_idx in self.slice_ids:
                for time_idx in self.time_ids:
                    for pos_idx in self.pos_ids:
                        if tile_indices is None:
                            # tile and save first image
                            # get meta data and tile_indices
                            im = tile_utils.preprocess_imstack(
                                frames_metadata=self.frames_metadata,
                                input_dir=self.input_dir,
                                depth=self.channel_depth[channel_idx],
                                time_idx=time_idx,
                                channel_idx=channel_idx,
                                slice_idx=slice_idx,
                                pos_idx=pos_idx,
                                flat_field_im=flat_field_im,
                                hist_clip_limits=self.hist_clip_limits
                            )
                            save_dict = {'time_idx': time_idx,
                                         'channel_idx': channel_idx,
                                         'pos_idx': pos_idx,
                                         'slice_idx': slice_idx,
                                         'save_dir': self.tile_dir,
                                         'image_format': self.image_format,
                                         'int2str_len': self.int2str_len}
                            tiled_meta0, tile_indices = \
                                tile_utils.tile_image(
                                    input_image=im,
                                    tile_size=self.tile_size,
                                    step_size=self.step_size,
                                    isotropic=self.isotropic,
                                    return_index=True,
                                    save_dict=save_dict
                                )
                        else:
                            cur_args = self.get_crop_tile_args(
                                channel_idx,
                                time_idx,
                                slice_idx,
                                pos_idx,
                                task_type='crop',
                                tile_indices=tile_indices
                            )
                            fn_args.append(cur_args)
        tiled_meta_df_list = mp_crop_save(fn_args,
                                          workers=self.num_workers)
        if tiled_meta0 is not None:
            tiled_meta_df_list.append(tiled_meta0)
        tiled_metadata = pd.concat(tiled_meta_df_list, ignore_index=True)
        if self.tiles_exist:
            tiled_metadata.reset_index(drop=True, inplace=True)
            prev_tiled_metadata.reset_index(drop=True, inplace=True)
            tiled_metadata = pd.concat([prev_tiled_metadata, tiled_metadata],
                                       ignore_index=True)
        # Finally, save all the metadata
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=",",
        )

    def tile_mask_stack(self,
                        mask_dir,
                        mask_channel,
                        min_fraction,
                        mask_depth=1):
        """
        Tiles images in the specified channels assuming there are masks
        already created in mask_dir. Only tiles above a certain fraction
        of foreground in mask tile will be saved and added to metadata.

        Saves a csv with columns ['time_idx', 'channel_idx', 'pos_idx',
        'slice_idx', 'file_name'] for all the tiles

        :param str mask_dir: Directory containing masks
        :param int mask_channel: Channel number assigned to mask
        :param float min_fraction: Minimum fraction of foreground in tiled masks
        :param int mask_depth: Depth for mask channel
        """

        # mask depth has to match input or ouput channel depth
        assert mask_depth <= max(self.channel_depth.values())
        self.mask_depth = mask_depth

        # tile and save masks
        # if mask channel is already tiled
        if self.tiles_exist and mask_channel in self.channel_ids:
            mask_meta_df = pd.read_csv(
                os.path.join(self.tile_dir, 'frames_meta.csv')
            )
        else:
            mask_fn_args = []
            for slice_idx in self.slice_ids:
                for time_idx in self.time_ids:
                    for pos_idx in self.pos_ids:
                        # Evaluate mask, then channels.The masks will influence
                        # tiling indices, so it's not allowed to add masks to
                        # existing tiled data sets (indices will be retrieved
                        # from existing meta)
                        cur_args = self.get_crop_tile_args(
                            channel_idx=mask_channel,
                            time_idx=time_idx,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                            task_type='tile',
                            mask_dir=mask_dir,
                            min_fraction=min_fraction
                        )
                        mask_fn_args.append(cur_args)

            # tile_image uses min_fraction assuming input_image is a bool
            mask_meta_df_list = mp_tile_save(mask_fn_args,
                                             workers=self.num_workers)
            mask_meta_df = pd.concat(mask_meta_df_list, ignore_index=True)
            # Finally, save all the metadata
            mask_meta_df = mask_meta_df.sort_values(by=['file_name'])
            mask_meta_df.to_csv(os.path.join(self.tile_dir, 'frames_meta.csv'),
                                sep=',')

        # remove mask_channel from self.channel_ids if included
        _ = [self.channel_ids.pop(idx)
             for idx, val in enumerate(self.channel_ids)
             if val == mask_channel]

        fn_args = []
        for slice_idx in self.slice_ids:
            for time_idx in self.time_ids:
                for pos_idx in np.unique(self.frames_metadata["pos_idx"]):
                    # Loop through all channels and tile from indices
                    cur_tile_indices = self._get_tile_indices(
                        tiled_meta=mask_meta_df,
                        time_idx=time_idx,
                        channel_idx=mask_channel,
                        pos_idx=pos_idx,
                        slice_idx=slice_idx
                    )
                    if np.any(cur_tile_indices):
                        for i, channel_idx in enumerate(self.channel_ids):
                            cur_args = self.get_crop_tile_args(
                                channel_idx,
                                time_idx,
                                slice_idx,
                                pos_idx,
                                task_type='crop',
                                tile_indices=cur_tile_indices
                            )
                            fn_args.append(cur_args)
        tiled_meta_df_list = mp_crop_save(fn_args,
                                          workers=self.num_workers)
        tiled_metadata = pd.concat(tiled_meta_df_list, ignore_index=True)
        prev_tiled_metadata = aux_utils.read_meta(self.tile_dir)
        tiled_metadata = pd.concat([prev_tiled_metadata.reset_index(drop=True),
                                    tiled_metadata.reset_index(drop=True)],
                                   axis=0,
                                   ignore_index=True)
        # Finally, save all the metadata
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=',',
        )
