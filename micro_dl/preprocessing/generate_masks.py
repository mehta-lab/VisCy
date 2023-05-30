"""Generate masks from sum of flurophore channels"""
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.mp_utils import mp_create_save_mask


class MaskProcessor:
    """Generate masks from channels"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 channel_ids,
                 flat_field_dir=None,
                 time_ids=-1,
                 slice_ids=-1,
                 pos_ids=-1,
                 int2str_len=3,
                 uniform_struct=True,
                 num_workers=4):
        """
        :param str input_dir: Directory with image frames
        :param str output_dir: Base output directory
        :param str flat_field_dir: Directory with flatfield images if
            flatfield correction is applied
        :param int/list channel_ids: generate mask from the sum of these
         (flurophore) channel indices
        :param list/int time_ids: timepoints to consider
        :param int slice_ids: Index of which focal plane (z)
            acquisition to use (default -1 includes all slices)
        :param int pos_ids: Position (FOV) indices to use
        :param int int2str_len: Length of str when converting ints
        :param bool uniform_struct: bool indicator for same structure across
         pos and time points
        :param int num_workers: number of workers for multiprocessing
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.flat_field_dir = flat_field_dir
        self.num_workers = num_workers

        self.frames_metadata = aux_utils.read_meta(self.input_dir)
        # Create a unique mask channel number so masks can be treated
        # as a new channel
        self.mask_channel = int(self.frames_metadata["channel_idx"].max() + 1)
        metadata_ids, nested_id_dict = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
            uniform_structure=uniform_struct
        )
        self.time_ids = metadata_ids['time_ids']
        self.channel_ids = metadata_ids['channel_ids']
        self.slice_ids = metadata_ids['slice_ids']
        self.pos_ids = metadata_ids['pos_ids']
        # Create mask_dir as a subdirectory of output_dir
        self.mask_dir = os.path.join(
            self.output_dir,
            'mask_channels_' + '-'.join(map(str, self.channel_ids)),
        )
        os.makedirs(self.mask_dir, exist_ok=True)

        self.int2str_len = int2str_len
        self.uniform_struct = uniform_struct
        self.nested_id_dict = nested_id_dict

    def get_mask_dir(self):
        """
        Return mask directory
        :return str mask_dir: Directory where masks are stored
        """
        return self.mask_dir

    def get_mask_channel(self):
        """
        Return mask channel
        :return int mask_channel: Assigned channel number for mask
        """
        return self.mask_channel

    def _get_args_read_image(self,
                             time_idx,
                             channel_idx,
                             slice_idx,
                             pos_idx,
                             correct_flat_field):
        """Read image from t, c, pos and ch indices

        :param int time_idx: time points to use for generating mask
        :param list channel_idx: channel ids to use for generating mask
        :param int slice_idx: generate masks for given slice ids
        :param int pos_idx: generate masks for given position / sample ids
        :param bool correct_flat_field: bool indicator to correct for flat
         field
        :return np.array im: image corresponding to the given ids and flat
         field corrected
        """

        input_fnames = []
        for ch_idx in channel_idx:
            frame_idx = aux_utils.get_meta_idx(self.frames_metadata,
                                               time_idx,
                                               ch_idx,
                                               slice_idx,
                                               pos_idx)
            file_path = os.path.join(
                self.input_dir,
                self.frames_metadata.loc[frame_idx, 'file_name'],
            )
            input_fnames.append(file_path)

        flat_field_fname = None
        if correct_flat_field:
            flat_field_fname = os.path.join(
                self.flat_field_dir,
                'flat-field_channel-{}.npy'.format(channel_idx)
            )
        return tuple(input_fnames), flat_field_fname

    def generate_masks(self,
                       correct_flat_field=False,
                       str_elem_radius=5):
        """
        Generate masks from flat-field corrected flurophore images.
        The sum of flurophore channels is thresholded to generate a foreground
        mask.

        :param bool correct_flat_field: bool indicator to correct for flat
         field or not
        :param int str_elem_radius: Radius of structuring element for
         morphological operations
        """

        # Loop through all the indices and create masks
        fn_args = []
        if self.uniform_struct:
            for slice_idx in self.slice_ids:
                for time_idx in self.time_ids:
                    for pos_idx in self.pos_ids:
                        fname_args = self._get_args_read_image(
                            time_idx=time_idx,
                            channel_idx=self.channel_ids,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                            correct_flat_field=correct_flat_field)
                        cur_args = (fname_args[0], fname_args[1],
                                    str_elem_radius,
                                    self.mask_dir,
                                    self.mask_channel,
                                    time_idx,
                                    pos_idx,
                                    slice_idx,
                                    self.int2str_len)
                        fn_args.append(cur_args)
        else:
            for tp_idx, tp_dict in self.nested_id_dict.items():
                mask_channel_dict = tp_dict[self.channel_ids[0]]
                for pos_idx, sl_idx_list in mask_channel_dict.items():
                    for sl_idx in sl_idx_list:
                        fname_args = self._get_args_read_image(
                            time_idx=tp_idx,
                            channel_idx=self.channel_ids,
                            slice_idx=sl_idx,
                            pos_idx=pos_idx,
                            correct_flat_field=correct_flat_field
                        )
                        cur_args = (fname_args[0], fname_args[1],
                                    str_elem_radius,
                                    self.mask_dir,
                                    self.mask_channel,
                                    tp_idx,
                                    pos_idx,
                                    sl_idx,
                                    self.int2str_len)
                        fn_args.append(cur_args)

        mask_meta_list = mp_create_save_mask(fn_args, self.num_workers)
        mask_meta_df = pd.DataFrame.from_dict(mask_meta_list)
        mask_meta_df = mask_meta_df.sort_values(by=['file_name'])
        meta_df = pd.concat([self.frames_metadata, mask_meta_df],
                            axis=0,
                            ignore_index=True)
        meta_df.to_csv(os.path.join(self.input_dir, 'frames_meta.csv'),
                       sep=',')
