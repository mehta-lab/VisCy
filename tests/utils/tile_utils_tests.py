import cv2
import nose.tools
import numpy as np
import numpy.testing
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.utils.tile_utils as tile_utils
import micro_dl.utils.aux_utils as aux_utils


class TestTileUtils(unittest.TestCase):

    def setUp(self):
        """Set up a dictionary with images"""

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        meta_fname = 'frames_meta.csv'
        frames_meta = aux_utils.make_dataframe()

        x = np.linspace(-4, 4, 32)
        y = x.copy()
        z = np.linspace(-3, 3, 8)
        xx, yy, zz = np.meshgrid(x, y, z)
        sph = (xx ** 2 + yy ** 2 + zz ** 2)
        sph = (sph <= 8) * (8 - sph)
        sph = (sph / sph.max()) * 255
        sph = sph.astype('uint8')
        self.sph = sph

        self.input_image = self.sph[:, :, 3:6]
        self.tile_size = [16, 16]
        self.step_size = [8, 8]

        self.channel_idx = 1
        self.time_idx = 0
        self.pos_idx = 1
        self.int2str_len = 3
        self.crop_indices = [
            (0, 16, 8, 24, 0, 3),
            (8, 24, 0, 16, 0, 3),
            (8, 24, 8, 24, 0, 3),
            (8, 24, 16, 32, 0, 3),
            (16, 32, 8, 24, 0, 3),
        ]

        for z in range(sph.shape[2]):
            im_name = aux_utils.get_im_name(
                channel_idx=1,
                slice_idx=z,
                time_idx=self.time_idx,
                pos_idx=self.pos_idx,
            )
            meta_row = aux_utils.parse_idx_from_name(
                im_name=im_name,
                dir_name=self.temp_path,
            )
            meta_row['mean'] = np.nanmean(sph[:, :, z])
            meta_row['std'] = np.nanstd(sph[:, :, z])
            cv2.imwrite(os.path.join(self.temp_path, im_name), sph[:, :, z])
            frames_meta = frames_meta.append(
                meta_row,
                ignore_index=True
            )
        self.dataset_mean = frames_meta['mean'].mean()
        self.dataset_std = frames_meta['std'].mean()
        # Write metadata
        frames_meta.to_csv(os.path.join(self.temp_path, meta_fname), sep=',')
        self.frames_meta = frames_meta

        self.sph_fname = os.path.join(
            self.temp_path,
            'im_c001_z000_t000_p001_3d.npy',
        )
        np.save(self.sph_fname, self.sph, allow_pickle=True, fix_imports=True)
        meta_3d = pd.DataFrame.from_dict([{
            'channel_idx': 1,
            'slice_idx': 0,
            'time_idx': 0,
            'channel_name': '3d_test',
            'file_name': 'im_c001_z000_t000_p001_3d.npy',
            'pos_idx': 1,
            'mean': self.dataset_mean,
            'std': self.dataset_std,
        }])
        self.meta_3d = meta_3d

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_tile_image(self):
        """Test tile_image"""
        # returns at tuple of (img_id, tile)
        tiles_list, cropping_index = tile_utils.tile_image(
            input_image=self.input_image,
            tile_size=self.tile_size,
            step_size=self.step_size,
            return_index=True,
        )
        nose.tools.assert_equal(len(tiles_list), 9)
        c = 0
        for row in range(0, 17, 8):
            for col in range(0, 17, 8):
                expected_idx = (
                    row,
                    row + self.tile_size[0],
                    col,
                    col + self.tile_size[1],
                )
                nose.tools.assert_equal(expected_idx, cropping_index[c])
                tile = self.input_image[row:row + self.tile_size[0],
                                   col: col + self.tile_size[1], ...]
                numpy.testing.assert_array_equal(tile, tiles_list[c])
                c += 1

    def test_tile_image_return_index(self):
        # returns tuple_list, cropping_index
        _, tile_index = tile_utils.tile_image(
            self.input_image,
            tile_size=self.tile_size,
            step_size=self.step_size,
            return_index=True,
        )
        exp_tile_index = [(0, 16, 0, 16), (0, 16, 8, 24),
                          (0, 16, 16, 32), (8, 24, 0, 16),
                          (8, 24, 8, 24), (8, 24, 16, 32),
                          (16, 32, 0, 16), (16, 32, 8, 24),
                          (16, 32, 16, 32)]

        numpy.testing.assert_equal(exp_tile_index, tile_index)

    def test_tile_image_save_dict(self):
        # save tiles in place and return meta_df
        tile_dir = os.path.join(self.temp_path, 'tile_dir')
        os.makedirs(tile_dir, exist_ok=True)
        meta_dir = os.path.join(tile_dir, 'meta_dir')
        os.makedirs(meta_dir, exist_ok=True)
        save_dict = {'time_idx': self.time_idx,
                     'channel_idx': self.channel_idx,
                     'slice_idx': 4,
                     'pos_idx': self.pos_idx,
                     'image_format': 'zyx',
                     'int2str_len': 3,
                     'save_dir': tile_dir}
        tile_meta_df = tile_utils.tile_image(
            self.input_image,
            tile_size=self.tile_size,
            step_size=self.step_size,
            save_dict=save_dict,
        )
        tile_meta = []
        for row in range(0, 17, 8):
            for col in range(0, 17, 8):
                id_str = 'r{}-{}_c{}-{}_sl{}-{}'.format(
                    row,
                    row + self.tile_size[0],
                    col,
                    col + self.tile_size[1],
                    0,
                    3,
                )
                cur_fname = aux_utils.get_im_name(
                    time_idx=self.time_idx,
                    channel_idx=self.channel_idx,
                    slice_idx=4,
                    pos_idx=self.pos_idx,
                    int2str_len=3,
                    extra_field=id_str,
                    ext='.npy',
                )
                cur_path = os.path.join(tile_dir, cur_fname)
                nose.tools.assert_equal(os.path.exists(cur_path), True)
                cur_meta = {'channel_idx': self.channel_idx,
                            'slice_idx': 4,
                            'time_idx': self.time_idx,
                            'file_name': cur_fname,
                            'pos_idx': self.pos_idx,
                            'row_start': row,
                            'col_start': col,
                            'dir_name': tile_dir}
                tile_meta.append(cur_meta)
        exp_tile_meta_df = pd.DataFrame.from_dict(tile_meta)
        exp_tile_meta_df = exp_tile_meta_df.sort_values(by=['file_name'])
        pd.testing.assert_frame_equal(tile_meta_df, exp_tile_meta_df)

    def test_tile_image_mask(self):
        # use mask and min_fraction to select tiles to retain
        input_image_bool = self.input_image > 128
        _, tile_index = tile_utils.tile_image(
            input_image_bool,
            tile_size=self.tile_size,
            step_size=self.step_size,
            min_fraction=0.3,
            return_index=True,
        )
        exp_tile_index = [(0, 16, 8, 24),
                          (8, 24, 0, 16), (8, 24, 8, 24),
                          (8, 24, 16, 32),
                          (16, 32, 8, 24)]
        numpy.testing.assert_array_equal(tile_index, exp_tile_index)

    def test_tile_image_3d(self):
        # tile_3d
        input_image = self.sph
        tile_size = [16, 16, 6]
        step_size = [8, 8, 4]
        # returns at tuple of (img_id, tile)
        tiles_list, cropping_index = tile_utils.tile_image(
            input_image,
            tile_size=tile_size,
            step_size=step_size,
            return_index=True,
        )
        nose.tools.assert_equal(len(tiles_list), 18)
        c = 0
        for row in range(0, 17, 8):
            for col in range(0, 17, 8):
                for sl in range(0, 8, 6):
                    if sl == 0:
                        sl_start_end = [0, 6]
                    else:
                        sl_start_end = [2, 8]

                    expected_idx = (
                        row,
                        row + tile_size[0],
                        col,
                        col + tile_size[1],
                        sl_start_end[0],
                        sl_start_end[1],
                    )
                    nose.tools.assert_equal(expected_idx, cropping_index[c])
                    tile = input_image[row:row + tile_size[0],
                                       col: col + tile_size[1],
                                       sl_start_end[0]: sl_start_end[1]]
                    numpy.testing.assert_array_equal(
                        tile,
                        tiles_list[c],
                    )
                    c += 1

    def test_crop_at_indices(self):
        """Test crop_at_indices"""
        input_image = self.sph[:, :, 3:6]
        # return tuple_list
        tiles_list, ids_list = tile_utils.crop_at_indices(
            input_image=input_image,
            crop_indices=self.crop_indices,
        )
        for idx, cur_idx in enumerate(self.crop_indices):
            tile = input_image[cur_idx[0]: cur_idx[1],
                               cur_idx[2]: cur_idx[3],
                               cur_idx[4]: cur_idx[5]]
            id_str = 'r{}-{}_c{}-{}_sl{}-{}'.format(cur_idx[0], cur_idx[1],
                                                    cur_idx[2], cur_idx[3],
                                                    cur_idx[4], cur_idx[5])
            nose.tools.assert_equal(id_str, ids_list[idx])
            numpy.testing.assert_array_equal(tiles_list[idx], tile)

    def test_crop_at_indices_save_dict(self):
        # save tiles in place and return meta_df
        input_image = self.sph[:, :, 3:6]
        tile_dir = os.path.join(self.temp_path, 'tile_dir')
        os.makedirs(tile_dir, exist_ok=True)
        meta_dir = os.path.join(tile_dir, 'meta_dir')
        os.makedirs(meta_dir, exist_ok=True)
        save_dict = {'time_idx': self.time_idx,
                     'channel_idx': self.channel_idx,
                     'slice_idx': 4,
                     'pos_idx': self.pos_idx,
                     'image_format': 'zyx',
                     'int2str_len': 3,
                     'save_dir': tile_dir}

        tile_meta_df = tile_utils.crop_at_indices(
            input_image,
            self.crop_indices,
            save_dict=save_dict,
        )
        exp_tile_meta = []

        for idx, cur_idx in enumerate(self.crop_indices):
            id_str = 'r{}-{}_c{}-{}_sl{}-{}'.format(cur_idx[0], cur_idx[1],
                                                    cur_idx[2], cur_idx[3],
                                                    cur_idx[4], cur_idx[5])
            cur_fname = aux_utils.get_im_name(
                time_idx=self.time_idx,
                channel_idx=self.channel_idx,
                slice_idx=4,
                pos_idx=self.pos_idx,
                int2str_len=3,
                extra_field=id_str,
                ext='.npy',
            )
            cur_path = os.path.join(tile_dir, cur_fname)
            nose.tools.assert_equal(os.path.exists(cur_path), True)
            cur_meta = {'channel_idx': self.channel_idx,
                        'slice_idx': 4,
                        'time_idx': self.time_idx,
                        'file_name': cur_fname,
                        'pos_idx': self.pos_idx,
                        'row_start': cur_idx[0],
                        'col_start': cur_idx[2],
                        'dir_name': tile_dir}
            exp_tile_meta.append(cur_meta)
        exp_tile_meta_df = pd.DataFrame.from_dict(exp_tile_meta)
        exp_tile_meta_df = exp_tile_meta_df.sort_values(by=['file_name'])
        pd.testing.assert_frame_equal(tile_meta_df, exp_tile_meta_df)

    def test_write_tile(self):
        """Test write_tile"""

        tile_dir = os.path.join(self.temp_path, 'tile_dir')
        os.makedirs(tile_dir, exist_ok=True)
        save_dict = {'time_idx': self.time_idx,
                     'channel_idx': self.channel_idx,
                     'slice_idx': 4,
                     'pos_idx': self.pos_idx,
                     'image_format': 'zyx',
                     'int2str_len': 3,
                     'save_dir': tile_dir}

        input_image = self.sph[:, :, 3:6]
        cur_tile = input_image[8: 24, 8: 24, 0: 3]
        tile_name = 'im_c001_z004_t000_p001_r8-24_c8-24_sl0-3.npy'
        op_fname = tile_utils.write_tile(cur_tile, tile_name, save_dict)

        exp_path = os.path.join(tile_dir, tile_name)
        nose.tools.assert_equal(op_fname, exp_path)
        nose.tools.assert_equal(os.path.exists(exp_path), True)

    def test_write_meta(self):
        """Test write_meta"""

        # save tiles in place and return meta_df
        tile_dir = os.path.join(self.temp_path, 'tile_dir')
        os.makedirs(tile_dir, exist_ok=True)
        meta_dir = os.path.join(tile_dir, 'meta_dir')
        os.makedirs(meta_dir, exist_ok=True)
        save_dict = {'time_idx': self.time_idx,
                     'channel_idx': self.channel_idx,
                     'slice_idx': 4,
                     'pos_idx': self.pos_idx,
                     'image_format': 'zyx',
                     'int2str_len': 3,
                     'save_dir': tile_dir}

        tile_meta = [{'channel_idx': self.channel_idx,
                      'slice_idx': 4,
                      'time_idx': self.time_idx,
                      'file_name': 'im_c001_z004_t000_p001_r8-24_c8-24_sl0-3',
                      'pos_idx': self.pos_idx,
                      'row_start': 8,
                      'col_start': 8}]

        tile_meta_df = tile_utils.write_meta(tile_meta, save_dict)

        exp_tile_meta_df = pd.DataFrame.from_dict(tile_meta)
        exp_tile_meta_df = exp_tile_meta_df.sort_values(by=['file_name'])
        pd.testing.assert_frame_equal(tile_meta_df, exp_tile_meta_df)

        # when tile_meta is an empty list
        tile_meta = []
        tile_meta_df = tile_utils.write_meta(tile_meta, save_dict)
        nose.tools.assert_equal(tile_meta_df, None)
