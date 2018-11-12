import copy
import numpy as np
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
from micro_dl.input.tile_images_uni_struct import ImageTilerUniform
from micro_dl.utils.mp_utils import mp_tile_save, mp_crop_at_indices_save


class ImageTilerNonUniform(ImageTilerUniform):
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
        """Init

        Assuming same structure across channels and same number of samples
        across channels. The dataset could have varying number of time points
        and / or varying number of slices / size for each sample / position
        Please ref to init of ImageTilerUniform.
        """

        super().__init__(input_dir,
                         output_dir,
                         tile_dict,
                         tile_size,
                         step_size,
                         depths,
                         time_ids,
                         channel_ids,
                         slice_ids,
                         pos_ids,
                         hist_clip_limits,
                         flat_field_dir,
                         isotropic,
                         image_format,
                         num_workers,
                         int2str_len)
        # Get metadata indices
        metadata_ids, nested_id_dict = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
            uniform_structure=False
        )
        self.nested_id_dict = nested_id_dict
        # self.tile_dir is already created in super(). Check if frames_meta
        # exists in self.tile_dir
        meta_path = os.path.join(self.tile_dir, 'frames_meta.csv')
        assert not os.path.exists(meta_path), 'Tile dir exists. ' \
                                              'cannot add to existing dir'

    def tile_first_channel(self,
                           channel0_ids,
                           channel0_depth,
                           cur_mask_dir=None,
                           min_fraction=None):
        """Tile first channel or mask and use the tile indices for the rest

        Tiles and saves the tiles, meta_df for each image in
        self.tile_dir/meta_dir. The list of meta_df for all images gets saved
        as frames_meta.csv

        :param list channel0_ids: [tp_idx, ch_idx, ch_dict] for first channel
         or mask channel
        :param int channel0_depth: image depth for first channel or mask
        :param str cur_mask_dir: mask dir if tiling mask channel else none
        :param float min_fraction: Min fraction of foreground in tiled masks
        :return pd.DataFrame ch0_meta_df: pd.Dataframe with ids, row_start
         and col_start
        """

        fn_args = []
        for tp_idx, ch_idx, ch_dict in channel0_ids:
            for pos_idx, sl_idx_list in ch_dict.items():
                cur_sl_idx_list = aux_utils.adjust_slice_margins(
                    sl_idx_list, channel0_depth
                )
                for sl_idx in cur_sl_idx_list:
                    cur_args = super().get_args_tile_image(
                        channel_idx=ch_idx,
                        time_idx=tp_idx,
                        slice_idx=sl_idx,
                        pos_idx=pos_idx,
                        mask_dir=cur_mask_dir,
                        min_fraction=min_fraction
                    )
                    fn_args.append(cur_args)
        # tile_image uses min_fraction assuming input_image is a bool
        ch0_meta_df_list = mp_tile_save(fn_args, workers=self.num_workers)
        ch0_meta_df = pd.concat(ch0_meta_df_list, ignore_index=True)
        # Finally, save all the metadata
        ch0_meta_df = ch0_meta_df.sort_values(by=['file_name'])
        ch0_meta_df.to_csv(os.path.join(self.tile_dir, 'frames_meta.csv'),
                           sep=",")
        return ch0_meta_df

    def tile_remaining_channels(self,
                                nested_id_dict,
                                tiled_ch_id,
                                cur_meta_df):
        """Tile remaining channels using tile indices of 1st channel / mask

        :param dict nested_id_dict: nested dict with time, channel, pos and
         slice indices
        :param int tiled_ch_id: self.channel_ids[0] or mask_channel
        :param pd.DataFrame cur_meta_df: DF with meta for the already tiled
         channel
        """

        fn_args = []
        for tp_idx, tp_dict in nested_id_dict.items():
            for ch_idx, ch_dict in tp_dict.items():
                for pos_idx, sl_idx_list in ch_dict.items():
                    cur_sl_idx_list = aux_utils.adjust_slice_margins(
                        sl_idx_list, self.channel_depth[ch_idx]
                    )
                    for sl_idx in cur_sl_idx_list:
                        cur_tile_indices = super()._get_tile_indices(
                            tiled_meta=cur_meta_df,
                            time_idx=tp_idx,
                            channel_idx=tiled_ch_id,
                            pos_idx=pos_idx,
                            slice_idx=sl_idx
                        )
                        if np.any(cur_tile_indices):
                            cur_args = super().get_args_crop_at_indices(
                                cur_tile_indices,
                                ch_idx,
                                tp_idx,
                                sl_idx,
                                pos_idx
                            )
                            fn_args.append(cur_args)

        tiled_meta_df_list = mp_crop_at_indices_save(fn_args,
                                                     workers=self.num_workers)
        tiled_metadata = pd.concat(tiled_meta_df_list, ignore_index=True)

        tiled_metadata = pd.concat([cur_meta_df.reset_index(drop=True),
                                    tiled_metadata.reset_index(drop=True)],
                                   axis=0,
                                   ignore_index=True)
        # Finally, save all the metadata
        tiled_metadata = tiled_metadata.sort_values(by=['file_name'])
        tiled_metadata.to_csv(
            os.path.join(self.tile_dir, "frames_meta.csv"),
            sep=",",
        )

    def tile_stack(self):
        """Tiles images in the specified channels.

        Assuming mask channel is not included in frames_meta in self.input_dir.
        Else this will cause an error as the filename = self.input_dir +
        file_name from frames_meta.csv. Masks are generally stored in a
        different folder.

        Saves a csv with columns
        ['time_idx', 'channel_idx', 'pos_idx','slice_idx', 'file_name']
        for all the tiles
        """

        ch_to_tile = self.channel_ids[0]
        ch_depth = self.channel_depth[0]

        # create a copy of nested_id_dict to remove the entries of the first
        # channel
        nested_id_dict_1 = copy.deepcopy(self.nested_id_dict)

        ch0_ids = []
        for tp_idx, tp_dict in self.nested_id_dict.items():
            for ch_idx, ch_dict in tp_dict.items():
                if ch_idx == ch_to_tile:
                    cur_idx = [tp_idx, ch_idx, ch_dict]
                    ch0_ids.append(cur_idx)
                    del nested_id_dict_1[tp_idx][ch_idx]

        # tile first channel and use the tile indices to tile the rest
        meta_df = self.tile_first_channel(channel0_ids=ch0_ids,
                                          channel0_depth=ch_depth)
        # remove channel 0 from self.channel_ids
        _ = self.channel_ids.pop(0)
        if self.channel_ids:
            self.tile_remaining_channels(nested_id_dict=nested_id_dict_1,
                                         tiled_ch_id=ch_to_tile,
                                         cur_meta_df=meta_df)

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
        :param float min_fraction: Min fraction of foreground in tiled masks
        :param int mask_depth: Depth for mask channel
        """

        # mask depth has to match input or output channel depth
        assert mask_depth <= max(self.channel_depth.values())
        self.mask_depth = mask_depth

        # nested_id_dict had no info on mask channel if channel_ids != -1.
        # Assuming structure is same across channels. Get time, pos and slice
        # indices for ch_idx=0 or mask channel if channel_ids == -1
        mask_ch_in_dict = mask_channel in self.channel_ids
        ch0 = mask_channel if mask_ch_in_dict else self.channel_ids[0]

        # create a copy of nested_id_dict to remove the entries of the first
        # channel
        nested_id_dict_1 = copy.deepcopy(self.nested_id_dict)

        # get t, z, p indices for mask_channel
        ch0_ids = []
        for tp_idx, tp_dict in self.nested_id_dict.items():
            for ch_idx, ch_dict in tp_dict.items():
                if ch_idx == ch0:
                    cur_idx = [tp_idx, mask_channel, ch_dict]
                    ch0_ids.append(cur_idx)
                    if mask_ch_in_dict:
                        del nested_id_dict_1[tp_idx][ch_idx]

        # tile first channel and use the tile indices to tile the rest
        meta_df = self.tile_first_channel(channel0_ids=ch0_ids,
                                          channel0_depth=mask_depth,
                                          cur_mask_dir=mask_dir,
                                          min_fraction=min_fraction)

        nested_dict = nested_id_dict_1 if mask_ch_in_dict \
            else self.nested_id_dict
        # tile the rest
        self.tile_remaining_channels(nested_id_dict=nested_dict,
                                     tiled_ch_id=mask_channel,
                                     cur_meta_df=meta_df)
