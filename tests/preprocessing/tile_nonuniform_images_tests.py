import copy
import nose.tools
import numpy as np
import os
import pandas as pd
import skimage.io as sk_im_io
from testfixtures import TempDirectory
import unittest
import warnings

import micro_dl.preprocessing.tile_nonuniform_images as tile_images
import micro_dl.utils.aux_utils as aux_utils


class TestImageTilerNonUniform(unittest.TestCase):

    def setUp(self):
        """Set up a dir for tiling with flatfield"""

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        frames_meta = aux_utils.make_dataframe()
        self.im = 127 * np.ones((15, 11), dtype=np.uint8)
        self.im2 = 234 * np.ones((15, 11), dtype=np.uint8)
        self.int2str_len = 3
        self.channel_idx = [1, 2]
        self.pos_idx1 = 7
        self.pos_idx2 = 8

        # write pos1 with 3 time points and 5 slices
        for z in range(5):
            for t in range(3):
                for c in self.channel_idx:
                    im_name = aux_utils.get_im_name(
                        channel_idx=c,
                        slice_idx=z,
                        time_idx=t,
                        pos_idx=self.pos_idx1,
                        ext='.png',
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sk_im_io.imsave(
                            os.path.join(self.temp_path, im_name),
                            self.im,
                        )
                    frames_meta = frames_meta.append(
                        aux_utils.get_ids_from_imname(im_name),
                        ignore_index=True,
                    )
        # write pos2 with 2 time points and 3 slices
        for z in range(3):
            for t in range(2):
                for c in self.channel_idx:
                    im_name = aux_utils.get_im_name(
                        channel_idx=c,
                        slice_idx=z,
                        time_idx=t,
                        pos_idx=self.pos_idx2,
                        ext='.png',
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sk_im_io.imsave(
                            os.path.join(self.temp_path, im_name),
                            self.im,
                        )
                    frames_meta = frames_meta.append(
                        aux_utils.get_ids_from_imname(im_name),
                        ignore_index=True,
                    )

        # Write metadata
        frames_meta.to_csv(os.path.join(self.temp_path, self.meta_name),
                           sep=',',)
        # Instantiate tiler class
        self.output_dir = os.path.join(self.temp_path, 'tile_dir')
        self.tile_dict = {'channels': [1, 2],
                          'tile_size': [5, 5],
                          'step_size': [4, 4],
                          'depths': 3,
                          'image_format': 'zyx', }
        self.tile_inst = tile_images.ImageTilerNonUniform(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            tile_dict=self.tile_dict,
        )

    def tearDown(self):
        """Tear down temporary folder and file structure"""

        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """Test init"""

        nose.tools.assert_equal(self.tile_inst.channel_ids, [1, 2])
        nose.tools.assert_list_equal(list(self.tile_inst.time_ids),
                                     [0, 1, 2])
        nose.tools.assert_equal(self.tile_inst.flat_field_dir, None)
        # Depth is 3 so first and last frame will not be used
        nose.tools.assert_list_equal(list(self.tile_inst.slice_ids), [1, 2, 3])
        np.testing.assert_array_equal(self.tile_inst.pos_ids,
                                      np.asarray([7, 8]))

        # for each tp, ch, pos check if slice_ids are the same
        for tp_idx in range(3):
            for ch_idx in self.channel_idx:
                for pos_idx in [7, 8]:
                    if ch_idx == 1:
                        if pos_idx == 7:
                            sl_ids = [0, 1, 2, 3, 4]
                            np.testing.assert_array_equal(
                                self.tile_inst.nested_id_dict[tp_idx][ch_idx][
                                    pos_idx],
                                np.asarray(sl_ids)
                            )
                        elif pos_idx == 8 and tp_idx < 2:
                            sl_ids = [0, 1, 2]
                            np.testing.assert_array_equal(
                                self.tile_inst.nested_id_dict[tp_idx][ch_idx][
                                    pos_idx],
                                np.asarray(sl_ids)
                            )

    def test_tile_first_channel(self):
        """Test tile_first_channel"""

        ch0_ids = {}
        # get the indices for first channel
        for tp_idx, tp_dict in self.tile_inst.nested_id_dict.items():
            for ch_idx, ch_dict in tp_dict.items():
                if ch_idx == 1:
                    ch0_dict = {ch_idx: ch_dict}
            ch0_ids[tp_idx] = ch0_dict

        # get the expected meta df
        exp_meta = []
        for row in [0, 4, 8, 10]:
            for col in [0, 4, 6]:
                for z in [1, 2, 3]:
                    for t in [0, 1, 2]:
                        fname = aux_utils.get_im_name(
                            channel_idx=1,
                            slice_idx=z,
                            time_idx=t,
                            pos_idx=7,
                        )
                        tile_id = '_r{}-{}_c{}-{}_sl0-3'.format(row, row+5,
                                                               col, col+5)
                        fname = fname.split('.')[0] + tile_id + '.npy'
                        cur_meta = {'channel_idx': 1,
                                    'slice_idx': z,
                                    'time_idx': t,
                                    'file_name': fname,
                                    'pos_idx': 7,
                                    'row_start': row,
                                    'col_start': col}
                        exp_meta.append(cur_meta)
                for t in [0, 1]:
                    fname = aux_utils.get_im_name(
                        channel_idx=1,
                        slice_idx=1,
                        time_idx=t,
                        pos_idx=8,
                    )
                    tile_id = '_r{}-{}_c{}-{}_sl0-3'.format(row, row + 5,
                                                            col, col + 5)
                    fname = fname.split('.')[0] + tile_id + '.npy'
                    cur_meta = {'channel_idx': 1,
                                'slice_idx': 1,
                                'time_idx': t,
                                'file_name': fname,
                                'pos_idx': 8,
                                'row_start': row,
                                'col_start': col}
                    exp_meta.append(cur_meta)
        exp_meta_df = pd.DataFrame.from_dict(exp_meta)
        exp_meta_df = exp_meta_df.sort_values(by=['file_name'])

        ch0_meta_df = self.tile_inst.tile_first_channel(ch0_ids, 3)
        ch0_meta_df = ch0_meta_df.sort_values(by=['file_name'])
        # compare values of the returned and expected dfs
        np.testing.assert_array_equal(exp_meta_df.values, ch0_meta_df.values)

    def test_tile_remaining_channels(self):
        """Test tile_remaining_channels"""

        # tile channel 1
        nested_id_dict_copy = copy.deepcopy(self.tile_inst.nested_id_dict)
        ch0_ids = {}
        for tp_idx, tp_dict in self.tile_inst.nested_id_dict.items():
            for ch_idx, ch_dict in tp_dict.items():
                if ch_idx == 1:
                    ch0_dict = {ch_idx: ch_dict}
                    del nested_id_dict_copy[tp_idx][ch_idx]
            ch0_ids[tp_idx] = ch0_dict

        ch0_meta_df = self.tile_inst.tile_first_channel(ch0_ids, 3)
        # tile channel 2
        self.tile_inst.tile_remaining_channels(nested_id_dict_copy,
                                               tiled_ch_id=1,
                                               cur_meta_df=ch0_meta_df)
        frames_meta = pd.read_csv(os.path.join(self.tile_inst.tile_dir,
                                               'frames_meta.csv'),
                                  sep=',')
        # get the expected meta df which is a concat of the first channel df
        # and the current. it does seem to retain orig index, not sure how to
        # replace index in-place!
        exp_meta = []
        for row in [0, 4, 8, 10]:
            for col in [0, 4, 6]:
                for z in [1, 2, 3]:
                    for t in [0, 1, 2]:
                        for c in self.channel_idx:
                            fname = aux_utils.get_im_name(
                                channel_idx=c,
                                slice_idx=z,
                                time_idx=t,
                                pos_idx=7,
                            )
                            tile_id = '_r{}-{}_c{}-{}_sl0-3'.format(row, row+5,
                                                                    col, col+5)
                            fname = fname.split('.')[0] + tile_id + '.npy'
                            cur_meta = {'channel_idx': c,
                                        'slice_idx': z,
                                        'time_idx': t,
                                        'file_name': fname,
                                        'pos_idx': 7,
                                        'row_start': row,
                                        'col_start': col}
                            exp_meta.append(cur_meta)
                for t in [0, 1]:
                    for c in self.channel_idx:
                        fname = aux_utils.get_im_name(
                            channel_idx=c,
                            slice_idx=1,
                            time_idx=t,
                            pos_idx=8,
                        )
                        tile_id = '_r{}-{}_c{}-{}_sl0-3'.format(row, row + 5,
                                                                col, col + 5)
                        fname = fname.split('.')[0] + tile_id + '.npy'
                        cur_meta = {'channel_idx': c,
                                    'slice_idx': 1,
                                    'time_idx': t,
                                    'file_name': fname,
                                    'pos_idx': 8,
                                    'row_start': row,
                                    'col_start': col}
                        exp_meta.append(cur_meta)
        exp_meta_df = pd.DataFrame.from_dict(exp_meta, )
        frames_meta = frames_meta.sort_values(by=['file_name'])
        nose.tools.assert_equal(len(exp_meta_df), len(frames_meta))

        for i in range(len(frames_meta)):
            act_row = frames_meta.loc[i]
            row_idx = ((exp_meta_df['channel_idx'] == act_row['channel_idx']) &
                       (exp_meta_df['slice_idx'] == act_row['slice_idx']) &
                       (exp_meta_df['time_idx'] == act_row['time_idx']) &
                       (exp_meta_df['pos_idx'] == act_row['pos_idx']) &
                       (exp_meta_df['row_start'] == act_row['row_start']) &
                       (exp_meta_df['col_start'] == act_row['col_start']))
            exp_row = exp_meta_df.loc[row_idx]
            nose.tools.assert_equal(len(exp_row), 1)
            np.testing.assert_array_equal(act_row['file_name'],
                                          exp_row['file_name'])

    def test_tile_mask_stack(self):
        """Test tile_mask_stack"""

        nested_id_dict_1 = copy.deepcopy(self.tile_inst.nested_id_dict)
        # remove entries for pos_idx2 from nested_id_dict
        for tp_idx, tp_dict in self.tile_inst.nested_id_dict.items():
            for ch_idx, ch_dict in tp_dict.items():
                for pos_idx, pos_dict in ch_dict.items():
                    if pos_idx == 8:
                        del nested_id_dict_1[tp_idx][ch_idx][pos_idx]

        self.tile_inst.nested_id_dict = nested_id_dict_1
        # create a mask
        mask_dir = os.path.join(self.temp_path, 'mask_dir')
        os.makedirs(mask_dir, exist_ok=True)
        mask_images = np.zeros((15, 11, 5), dtype='bool')
        mask_images[4:12, 4:9, 2:4] = 1
        frames_meta = pd.read_csv(
            os.path.join(self.tile_inst.input_dir, 'frames_meta.csv'),
            index_col=0,
        )
        # write mask images and add meta to frames_meta. same mask across all
        # timepoints for testing
        for z in range(5):
            for t in range(3):
                cur_im = mask_images[:, :, z]
                im_name = aux_utils.get_im_name(
                    channel_idx=3,
                    slice_idx=z,
                    time_idx=t,
                    pos_idx=self.pos_idx1,
                )
                np.save(os.path.join(mask_dir, im_name), cur_im)
                cur_meta = {'channel_idx': 3,
                            'slice_idx': z,
                            'time_idx': t,
                            'pos_idx': self.pos_idx1,
                            'file_name': im_name}
                frames_meta = frames_meta.append(cur_meta, ignore_index=True)
        # Write metadata
        frames_meta.to_csv(
            os.path.join(mask_dir, 'frames_meta.csv'),
            sep=',',
        )
        self.tile_inst.pos_ids = [7]

        self.tile_inst.tile_mask_stack(mask_dir,
                                       mask_channel=3,
                                       min_fraction=0.5,
                                       mask_depth=3)
        nose.tools.assert_equal(self.tile_inst.mask_depth, 3)

        frames_meta = pd.read_csv(os.path.join(self.tile_inst.tile_dir,
                                               'frames_meta.csv'),
                                  sep=',')
        # only 4 tiles have >= min_fraction. 4 tiles x 3 slices x 3 tps
        nose.tools.assert_equal(len(frames_meta), 36)
        nose.tools.assert_list_equal(
            frames_meta['row_start'].unique().tolist(),
            [4, 8])
        nose.tools.assert_equal(frames_meta['col_start'].unique().tolist(),
                                [4])
        nose.tools.assert_equal(frames_meta['slice_idx'].unique().tolist(),
                                [2, 3])
        self.assertSetEqual(set(frames_meta.channel_idx.tolist()), {1, 2, 3})
        self.assertSetEqual(set(frames_meta.time_idx.tolist()), {0, 1, 2})
        self.assertSetEqual(set(frames_meta.pos_idx.tolist()), {self.pos_idx1})
