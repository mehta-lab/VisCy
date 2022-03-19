"""Generate masks from sum of flurophore channels"""

import numpy as np
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.mp_utils as mp_utils


class ImageResizer:
    """Resize images for given indices"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 scale_factor,
                 channel_ids=-1,
                 time_ids=-1,
                 slice_ids=-1,
                 pos_ids=-1,
                 int2str_len=3,
                 num_workers=4,
                 flat_field_dir=None):
        """
        :param str input_dir: Directory with image frames
        :param str output_dir: Base output directory
        :param float/list scale_factor: Scale factor for resizing frames.
        :param int/list channel_ids: Channel indices to resize
            (default -1 includes all slices)
        :param int/list time_ids: timepoints to use
        :param int/list slice_ids: Index of slice (z) indices to use
        :param int/list pos_ids: Position (FOV) indices to use
        :param int int2str_len: Length of str when converting ints
        :param int num_workers: number of workers for multiprocessing
        :param str flat_field_dir: dir with flat field images
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        if isinstance(scale_factor, list):
            scale_factor = np.array(scale_factor)
        assert np.all(scale_factor > 0), \
            "Scale factor should be positive float, not {}".format(scale_factor)
        self.scale_factor = scale_factor

        self.frames_metadata = aux_utils.read_meta(self.input_dir)
        metadata_ids, _ = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
        )
        self.time_ids = metadata_ids['time_ids']
        self.channel_ids = metadata_ids['channel_ids']
        self.slice_ids = metadata_ids['slice_ids']
        self.pos_ids = metadata_ids['pos_ids']

        # Create resize_dir as a subdirectory of output_dir
        self.resize_dir = os.path.join(
            self.output_dir,
            'resized_images',
        )
        os.makedirs(self.resize_dir, exist_ok=True)

        self.int2str_len = int2str_len
        self.num_workers = num_workers
        self.flat_field_dir = flat_field_dir

    def get_resize_dir(self):
        """
        Return directory with resized images
        :return str resize_dir: Directory where resized images are stored
        """
        return self.resize_dir

    def resize_frames(self):
        """
        Resize frames for given indices.
        """

        assert isinstance(self.scale_factor, (float, int)), \
            'different scale factors provided for x and y'
        mp_args = []
        resized_metadata = aux_utils.make_dataframe()
        # Loop through all the indices and resize images
        for slice_idx in self.slice_ids:
            for time_idx in self.time_ids:
                for pos_idx in self.pos_ids:
                    for channel_idx in self.channel_ids:
                        frame_idx = aux_utils.get_meta_idx(
                            self.frames_metadata,
                            time_idx,
                            channel_idx,
                            slice_idx,
                            pos_idx,
                        )
                        file_name = self.frames_metadata.loc[frame_idx,
                                                             "file_name"]
                        file_path = os.path.join(self.input_dir, file_name)
                        write_path = os.path.join(self.resize_dir, file_name)
                        ff_path = None
                        if self.flat_field_dir is not None:
                            ff_path = os.path.join(
                                self.flat_field_dir,
                                'flat-field_channel-{}.npy'.format(channel_idx)
                            )
                        kwargs = {
                            'file_path': file_path,
                            'write_path': write_path,
                            'scale_factor': self.scale_factor,
                            'ff_path': ff_path
                        }
                        mp_args.append(kwargs)
                        resized_metadata = resized_metadata.append(
                            self.frames_metadata.iloc[frame_idx],
                            ignore_index=True,
                        )
        # Multiprocessing of kwargs
        mp_utils.mp_resize_save(mp_args, self.num_workers)
        resized_metadata = resized_metadata.sort_values(by=['file_name'])
        resized_metadata.to_csv(
            os.path.join(self.resize_dir, "frames_meta.csv"),
            sep=',',
        )

    def resize_volumes(self, num_slices_subvolume=-1):
        """Down or up sample volumes

        Overlap of one slice across subvolumes

        :param int num_slices_subvolume: num of 2D slices to include in each
         volume. if -1, include all slices
        """

        # assuming slice_ids will be continuous
        num_total_slices = len(self.slice_ids)
        if not isinstance(self.scale_factor, float):
            sc_str = '-'.join(self.scale_factor.astype('str'))
        else:
            sc_str = self.scale_factor

        mp_args = []
        if num_slices_subvolume == -1:
            num_slices_subvolume = len(self.slice_ids)
        num_blocks = np.floor(
            num_total_slices / (num_slices_subvolume - 1)
        ).astype('int')
        for time_idx in self.time_ids:
            for pos_idx in self.pos_ids:
                for channel_idx in self.channel_ids:
                    ff_path = None
                    if self.flat_field_dir is not None:
                        ff_path = os.path.join(
                            self.flat_field_dir,
                            'flat-field_channel-{}.npy'.format(channel_idx)
                        )
                    for block_idx in range(num_blocks):
                        idx = self.slice_ids[0] + \
                              block_idx * (num_slices_subvolume - 1)
                        start_idx = np.maximum(self.slice_ids[0], idx)
                        end_idx = start_idx + num_slices_subvolume
                        if end_idx > self.slice_ids[-1]:
                            end_idx = self.slice_ids[-1] + 1
                            start_idx = end_idx - num_slices_subvolume
                        op_fname = aux_utils.get_im_name(
                            time_idx,
                            channel_idx,
                            start_idx,
                            pos_idx,
                            extra_field=sc_str,
                            ext='.npy',
                        )
                        write_fpath = os.path.join(self.resize_dir, op_fname)
                        mp_args.append((time_idx,
                                        pos_idx,
                                        channel_idx,
                                        start_idx,
                                        end_idx,
                                        self.frames_metadata,
                                        write_fpath,
                                        self.scale_factor,
                                        self.input_dir,
                                        ff_path))

        # Multiprocessing of kwargs
        resized_metadata_list = mp_utils.mp_rescale_vol(mp_args, self.num_workers)
        resized_metadata_df = pd.DataFrame.from_dict(resized_metadata_list)
        resized_metadata_df.to_csv(
            os.path.join(self.resize_dir, 'frames_meta.csv'),
            sep=',',
        )

        if num_slices_subvolume == -1:
            slice_ids = self.slice_ids[0]
        else:
            slice_ids = self.slice_ids[0: -1: num_slices_subvolume - 1]

        return slice_ids
