import nose.tools
import numpy as np
import os
import pandas as pd
import skimage.io as sk_im_io
from testfixtures import TempDirectory
import unittest
import warnings

import micro_dl.preprocessing.tile_uniform_images as tile_images
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.normalize as norm_util


class TestImageTilerUniform(unittest.TestCase):

    def setUp(self):
        """Set up a dir for tiling with flatfield"""

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        frames_meta = aux_utils.make_dataframe()
        # Write images
        self.im = 127 * np.ones((15, 11), dtype=np.uint8)
        self.im2 = 234 * np.ones((15, 11), dtype=np.uint8)
        self.channel_idx = 1
        self.time_idx = 5
        self.pos_idx1 = 7
        self.pos_idx2 = 8
        self.int2str_len = 3

        # Write test images with 4 z and 2 pos idx
        for z in range(15, 20):
            im_name = aux_utils.get_im_name(
                channel_idx=self.channel_idx,
                slice_idx=z,
                time_idx=self.time_idx,
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
                aux_utils.parse_idx_from_name(im_name),
                ignore_index=True,
            )

        for z in range(15, 20):
            im_name = aux_utils.get_im_name(
                channel_idx=self.channel_idx,
                slice_idx=z,
                time_idx=self.time_idx,
                pos_idx=self.pos_idx2,
                ext='.png',
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk_im_io.imsave(
                    os.path.join(self.temp_path, im_name),
                    self.im2,
                )
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name),
                ignore_index=True,
            )

        # Write metadata
        frames_meta.to_csv(
            os.path.join(self.temp_path, self.meta_name),
            sep=',',
        )
        # Add flatfield
        self.flat_field_dir = os.path.join(self.temp_path, 'ff_dir')
        self.tempdir.makedir('ff_dir')
        self.ff_im = 4. * np.ones((15, 11))
        self.ff_name = os.path.join(
            self.flat_field_dir,
            'flat-field_channel-1.npy',
        )
        np.save(
            self.ff_name,
            self.ff_im,
            allow_pickle=True,
            fix_imports=True
        )
        # Instantiate tiler class
        self.output_dir = os.path.join(self.temp_path, 'tile_dir')
        self.tile_dict = {'channels': [1],
                          'tile_size': [5, 5],
                          'step_size': [4, 4],
                          'depths': 3,
                          'image_format': 'zyx',
                          'tile_3d': False}
        self.tile_inst = tile_images.ImageTilerUniform(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            tile_dict=self.tile_dict,
            flat_field_dir=self.flat_field_dir,
        )
        exp_fnames = ['im_c001_z015_t005_p007.png',
                      'im_c001_z016_t005_p007.png',
                      'im_c001_z017_t005_p007.png']
        self.exp_fnames = [os.path.join(self.temp_path, fname)
                           for fname in exp_fnames]
        self.exp_tile_indices = [
            [0, 5, 0, 5], [0, 5, 4, 9], [0, 5, 6, 11],
            [10, 15, 0, 5], [10, 15, 4, 9], [10, 15, 6, 11],
            [4, 9, 0, 5], [4, 9, 4, 9], [4, 9, 6, 11],
            [8, 13, 0, 5], [8, 13, 4, 9], [8, 13, 6, 11],
        ]

        # create a mask
        mask_dir = os.path.join(self.temp_path, 'mask_dir')
        os.makedirs(mask_dir, exist_ok=True)
        mask_images = np.zeros((15, 11, 5), dtype='bool')
        mask_images[4:12, 4:9, 2:4] = 1

        # write mask images and add meta to frames_meta
        self.mask_channel = 3
        mask_meta = []
        for z in range(5):
            cur_im = mask_images[:, :, z]
            im_name = aux_utils.get_im_name(
                channel_idx=3,
                slice_idx=z + 15,
                time_idx=self.time_idx,
                pos_idx=self.pos_idx1,
            )
            np.save(os.path.join(mask_dir, im_name), cur_im)
            cur_meta = {'channel_idx': 3,
                        'slice_idx': z + 15,
                        'time_idx': self.time_idx,
                        'pos_idx': self.pos_idx1,
                        'file_name': im_name}
            mask_meta.append(cur_meta)
        mask_meta_df = pd.DataFrame.from_dict(mask_meta)
        mask_meta_df.to_csv(os.path.join(mask_dir, 'frames_meta.csv'), sep=',')
        self.mask_dir = mask_dir

        exp_tile_indices = [[0, 5, 0, 5], [0, 5, 4, 9], [0, 5, 6, 11],
                            [10, 15, 0, 5], [10, 15, 4, 9], [10, 15, 6, 11],
                            [4, 9, 0, 5], [4, 9, 4, 9], [4, 9, 6, 11],
                            [8, 13, 0, 5], [8, 13, 4, 9], [8, 13, 6, 11]]
        self.exp_tile_indices = exp_tile_indices

    def tearDown(self):
        """Tear down temporary folder and file structure"""

        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """Test init"""

        nose.tools.assert_equal(self.tile_inst.depths, 3)
        nose.tools.assert_equal(self.tile_inst.tile_size, [5, 5])
        nose.tools.assert_equal(self.tile_inst.step_size, [4, 4])
        nose.tools.assert_equal(self.tile_inst.hist_clip_limits, None)
        nose.tools.assert_equal(self.tile_inst.image_format, 'zyx')
        nose.tools.assert_equal(self.tile_inst.num_workers, 4)
        nose.tools.assert_equal(self.tile_inst.str_tile_step,
                                'tiles_5-5_step_4-4',)
        nose.tools.assert_equal(self.tile_inst.channel_ids, [self.channel_idx])
        nose.tools.assert_equal(self.tile_inst.time_ids, [self.time_idx])
        nose.tools.assert_equal(self.tile_inst.flat_field_dir,
                                self.flat_field_dir,)
        # Depth is 3 so first and last frame will not be used
        np.testing.assert_array_equal(self.tile_inst.slice_ids,
                                      np.asarray([16, 17, 18]),)
        np.testing.assert_array_equal(self.tile_inst.pos_ids,
                                      np.asarray([7, 8]),)
        # channel_depth should be a dict containing depths for each channel
        self.assertListEqual(list(self.tile_inst.channel_depth),
                             [self.channel_idx],)
        nose.tools.assert_equal(
            self.tile_inst.channel_depth[self.channel_idx],
            3,
        )

    def test_tile_dir(self):
        nose.tools.assert_equal(self.tile_inst.get_tile_dir(),
                                os.path.join(self.output_dir,
                                             "tiles_5-5_step_4-4"))

    def test_get_dataframe(self):
        df = self.tile_inst._get_dataframe()
        self.assertListEqual(
            list(df),
            ['channel_idx',
             'slice_idx',
             'time_idx',
             'file_name',
             'pos_idx',
             'row_start',
             'col_start'])

    def test_get_flat_field(self):
        flat_field_im = self.tile_inst._get_flat_field(channel_idx=1)
        np.testing.assert_array_equal(flat_field_im, self.ff_im)

    def test_get_tile_indices(self):
        """Test get_tiled_indices"""

        self.tile_inst.tile_stack()
        # Read the saved metadata
        tile_dir = self.tile_inst.get_tile_dir()
        tile_meta = pd.read_csv(os.path.join(tile_dir, "frames_meta.csv"))

        tile_indices = self.tile_inst._get_tile_indices(
            tiled_meta=tile_meta,
            time_idx=self.time_idx,
            channel_idx=self.channel_idx,
            pos_idx=7,
            slice_idx=16
        )
        exp_tile_indices = np.asarray(self.exp_tile_indices, dtype='uint8')
        row_ids = list(range(len(exp_tile_indices)))
        for ret_idx in tile_indices:
            row_idx = np.where((exp_tile_indices[:, 0] == ret_idx[0]) &
                               (exp_tile_indices[:, 1] == ret_idx[1]) &
                               (exp_tile_indices[:, 2] == ret_idx[2]) &
                               (exp_tile_indices[:, 3] == ret_idx[3]))
            nose.tools.assert_in(row_idx[0], row_ids)

    def test_get_tiled_data(self):
        """Test get_tiled_indices"""

        # no tiles_exist
        tile_meta, tile_indices = self.tile_inst._get_tiled_data()
        nose.tools.assert_equal(tile_indices, None)
        init_df = pd.DataFrame(columns=['channel_idx',
                                        'slice_idx',
                                        'time_idx',
                                        'file_name',
                                        'pos_idx',
                                        'row_start',
                                        'col_start'])
        pd.testing.assert_frame_equal(tile_meta, init_df)
        # tile exists
        self.tile_inst.tile_stack()
        self.tile_inst.tiles_exist = True
        self.tile_inst.channel_ids = [1, 2]
        tile_meta, _ = self.tile_inst._get_tiled_data()

        exp_tile_meta = []
        for exp_idx in self.exp_tile_indices:
            for z in [16, 17, 18]:
                cur_img_id = 'r{}-{}_c{}-{}_sl{}-{}'.format(
                    exp_idx[0], exp_idx[1], exp_idx[2], exp_idx[3], 0, 3
                )
                pos1_fname = aux_utils.get_im_name(
                    time_idx=self.time_idx,
                    channel_idx=self.channel_idx,
                    slice_idx=z,
                    pos_idx=self.pos_idx1,
                    extra_field=cur_img_id,
                )
                pos1_meta = {'channel_idx': self.channel_idx,
                             'slice_idx': z,
                             'time_idx': self.time_idx,
                             'file_name': pos1_fname,
                             'pos_idx': self.pos_idx1,
                             'row_start': exp_idx[0],
                             'col_start': exp_idx[2]}
                exp_tile_meta.append(pos1_meta)
                pos2_fname = aux_utils.get_im_name(
                    time_idx=self.time_idx,
                    channel_idx=self.channel_idx,
                    slice_idx=z,
                    pos_idx=self.pos_idx2,
                    extra_field=cur_img_id,
                )
                pos2_meta = {'channel_idx': self.channel_idx,
                             'slice_idx': z,
                             'time_idx': self.time_idx,
                             'file_name': pos2_fname,
                             'pos_idx': self.pos_idx2,
                             'row_start': exp_idx[0],
                             'col_start': exp_idx[2]}
                exp_tile_meta.append(pos2_meta)
        exp_tile_meta_df = pd.DataFrame.from_dict(exp_tile_meta)
        exp_tile_meta_df = exp_tile_meta_df.sort_values(by=['file_name'])
        exp_tile_meta_df.reset_index(drop=True, inplace=True)
        tile_meta = tile_meta.sort_values(by=['file_name'])
        tile_meta.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(tile_meta, exp_tile_meta_df)

    def test_get_input_fnames(self):
        """Test get_input_fnames"""

        im_fnames = self.tile_inst._get_input_fnames(
            time_idx=self.time_idx,
            channel_idx=self.channel_idx,
            slice_idx=16,
            pos_idx=self.pos_idx1
        )
        nose.tools.assert_list_equal(self.exp_fnames, im_fnames)

    def test_get_crop_args(self):
        """Test get_crop_tile_args with task_type=crop"""

        cur_args = self.tile_inst.get_crop_tile_args(
            channel_idx=self.channel_idx,
            time_idx=self.time_idx,
            slice_idx=16,
            pos_idx=7,
            task_type='crop',
            tile_indices=self.exp_tile_indices
        )
        nose.tools.assert_list_equal(list(cur_args[0]), self.exp_fnames)
        nose.tools.assert_equal(cur_args[1], self.ff_name)
        nose.tools.assert_equal(cur_args[2], None)
        nose.tools.assert_equal(cur_args[3], self.time_idx)
        nose.tools.assert_equal(cur_args[4], self.channel_idx)
        nose.tools.assert_equal(cur_args[5], 7)
        nose.tools.assert_equal(cur_args[6], 16)
        nose.tools.assert_equal(cur_args[8], 'zyx')
        nose.tools.assert_equal(cur_args[9], self.tile_inst.tile_dir)
        nose.tools.assert_equal(cur_args[10], self.int2str_len)

    def test_tile_stack(self):
        """Test tile_stack"""

        self.tile_inst.tile_stack()
        # Read and validate the saved metadata
        tile_dir = self.tile_inst.get_tile_dir()
        frames_meta = pd.read_csv(os.path.join(tile_dir, 'frames_meta.csv'))

        self.assertSetEqual(set(frames_meta.channel_idx.tolist()), {1})
        self.assertSetEqual(set(frames_meta.slice_idx.tolist()), {16, 17, 18})
        self.assertSetEqual(set(frames_meta.time_idx.tolist()), {5})
        self.assertSetEqual(set(frames_meta.pos_idx.tolist()), {7, 8})
        # 15 rows and step size 4, so it can take 3 full steps and 1 short step
        self.assertSetEqual(set(frames_meta.row_start.tolist()), {0, 4, 8, 10})
        # 11 cols and step size 4, so it can take 2 full steps and 1 short step
        self.assertSetEqual(set(frames_meta.col_start.tolist()), {0, 4, 6})

        # Read and validate tiles
        im_val = np.mean(norm_util.zscore(self.im / self.ff_im))
        im_norm = im_val * np.ones((3, 5, 5))
        im_val = np.mean(norm_util.zscore(self.im2 / self.ff_im))
        im2_norm = im_val * np.ones((3, 5, 5))
        for i, row in frames_meta.iterrows():
            tile = np.load(os.path.join(tile_dir, row.file_name))
            if row.pos_idx == 7:
                np.testing.assert_array_equal(tile, im_norm)
            else:
                np.testing.assert_array_equal(tile, im2_norm)

    def test_get_tile_args(self):
        """Test get_crop_tile_args with task_type=tile"""

        self.tile_inst.mask_depth = 3
        cur_args = self.tile_inst.get_crop_tile_args(
            channel_idx=self.mask_channel,
            time_idx=self.time_idx,
            slice_idx=16,
            pos_idx=7,
            task_type='tile',
            mask_dir=self.mask_dir,
            min_fraction=0.3
        )

        exp_fnames = ['im_c003_z015_t005_p007.npy',
                      'im_c003_z016_t005_p007.npy',
                      'im_c003_z017_t005_p007.npy']
        exp_fnames = [os.path.join(self.mask_dir, fname)
                      for fname in exp_fnames]

        nose.tools.assert_list_equal(list(cur_args[0]), exp_fnames)
        # flat field fname is None
        nose.tools.assert_equal(cur_args[1], None)
        # hist clip limits is None
        nose.tools.assert_equal(cur_args[2], None)
        nose.tools.assert_equal(cur_args[3], self.time_idx)
        nose.tools.assert_equal(cur_args[4], self.mask_channel)
        nose.tools.assert_equal(cur_args[5], 7)
        nose.tools.assert_equal(cur_args[6], 16)
        nose.tools.assert_list_equal(cur_args[7], self.tile_inst.tile_size)
        nose.tools.assert_list_equal(cur_args[8], self.tile_inst.step_size)
        nose.tools.assert_equal(cur_args[9], 0.3)
        nose.tools.assert_equal(cur_args[10], 'zyx')
        nose.tools.assert_equal(cur_args[11], self.tile_inst.tile_dir)
        nose.tools.assert_equal(cur_args[12], self.int2str_len)

        # not a mask channel
        cur_args = self.tile_inst.get_crop_tile_args(
            channel_idx=self.channel_idx,
            time_idx=self.time_idx,
            slice_idx=16,
            pos_idx=7,
            task_type='tile'
        )
        nose.tools.assert_list_equal(list(cur_args[0]), self.exp_fnames)

        exp_ff_fname = os.path.join(
            self.flat_field_dir,
            'flat-field_channel-{}.npy'.format(self.channel_idx),
        )
        nose.tools.assert_equal(cur_args[1], exp_ff_fname)
        nose.tools.assert_equal(cur_args[9], None)

    def test_tile_mask_stack(self):
        """Test tile_mask_stack"""

        self.tile_inst.pos_ids = [7]

        # use the saved masks to tile other channels
        self.tile_inst.tile_mask_stack(
            mask_dir=self.mask_dir,
            mask_channel=3,
            min_fraction=0.5,
            mask_depth=3
        )

        # Read and validate the saved metadata
        tile_dir = self.tile_inst.get_tile_dir()
        frames_meta = pd.read_csv(os.path.join(tile_dir, 'frames_meta.csv'))

        self.assertSetEqual(set(frames_meta.channel_idx.tolist()), {1, 3})
        self.assertSetEqual(set(frames_meta.slice_idx.tolist()), {17, 18})
        self.assertSetEqual(set(frames_meta.time_idx.tolist()), {5})
        self.assertSetEqual(set(frames_meta.pos_idx.tolist()), {self.pos_idx1})

        # with vf >= 0.5, 4 tiles will be saved for mask channel & [1]
        # [4,9,4,9,17], [8,13,4,9,17], [4,9,4,9,18], [8,13,4,9,18]
        nose.tools.assert_equal(len(frames_meta), 8)
        nose.tools.assert_list_equal(frames_meta['row_start'].unique().tolist(),
                                     [4, 8])
        nose.tools.assert_equal(frames_meta['col_start'].unique().tolist(),
                                [4])
