import cv2
import nose.tools
import numpy as np
import os
from testfixtures import TempDirectory
import unittest

import micro_dl.cli.preprocess_script as pp
import micro_dl.utils.aux_utils as aux_utils


class TestPreprocessScript(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with some images to resample
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.image_dir = self.temp_path
        self.output_dir = os.path.join(self.temp_path, 'out_dir')
        self.tempdir.makedir(self.output_dir)
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        self.frames_meta = aux_utils.make_dataframe()
        # Write images
        self.time_idx = 0
        self.pos_ids = [7, 8, 10]
        self.channel_ids = [0, 1, 2, 3]
        self.slice_ids = [0, 1, 2, 3, 4, 5]
        self.im = 1500 * np.ones((30, 20), dtype=np.uint16)
        self.im[10:20, 5:15] = 3000

        for c in self.channel_ids:
            for p in self.pos_ids:
                for z in self.slice_ids:
                    im_name = aux_utils.get_im_name(
                        channel_idx=c,
                        slice_idx=z,
                        time_idx=self.time_idx,
                        pos_idx=p,
                    )
                    cv2.imwrite(
                        os.path.join(self.image_dir, im_name),
                        self.im + c * 100,
                    )
                    self.frames_meta = self.frames_meta.append(
                        aux_utils.parse_idx_from_name(im_name),
                        ignore_index=True,
                    )
        # Write metadata
        self.frames_meta.to_csv(
            os.path.join(self.image_dir, self.meta_name),
            sep=',',
        )
        # Make input masks
        self.input_mask_channel = 111
        self.input_mask_dir = os.path.join(self.temp_path, 'input_mask_dir')
        self.tempdir.makedir(self.input_mask_dir)
        # Must have at least two foreground classes in mask for weight map to work
        mask = np.zeros((30, 20), dtype=np.uint16)
        mask[5:10, 5:15] = 1
        mask[20:25, 5:10] = 2
        mask_meta = aux_utils.make_dataframe()
        for p in self.pos_ids:
            for z in self.slice_ids:
                im_name = aux_utils.get_im_name(
                    channel_idx=self.input_mask_channel,
                    slice_idx=z,
                    time_idx=self.time_idx,
                    pos_idx=p,
                )
                cv2.imwrite(
                    os.path.join(self.input_mask_dir, im_name),
                    mask,
                )
                mask_meta = mask_meta.append(
                    aux_utils.parse_idx_from_name(im_name),
                    ignore_index=True,
                )
        mask_meta.to_csv(
            os.path.join(self.input_mask_dir, self.meta_name),
            sep=',',
        )
        # Create preprocessing config
        self.pp_config = {
            'output_dir': self.output_dir,
            'input_dir': self.image_dir,
            'channel_ids': [0, 1, 3],
            'num_workers': 4,
            'flat_field': {'estimate': True,
                           'block_size': 2,
                           'correct': True},
            'masks': {'channels': [3],
                      'str_elem_radius': 3,
                      'normalize_im': False},
            'tile': {'tile_size': [10, 10],
                     'step_size': [10, 10],
                     'depths': [1, 1, 1],
                     'mask_depth': 1,
                     'image_format': 'zyx',
                     'normalize_channels': [True, True, True]
                     },
        }
        # Create base config, generated party from pp_config in script
        self.base_config = {
            'input_dir': self.image_dir,
            'output_dir': self.output_dir,
            'slice_ids': -1,
            'time_ids': -1,
            'pos_ids': -1,
            'channel_ids': self.pp_config['channel_ids'],
            'uniform_struct': True,
            'int2strlen': 3,
            'num_workers': 4,
            'normalize_channels': [True, True, True]
        }

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_pre_process(self):
        out_config, runtime = pp.pre_process(self.pp_config, self.base_config)
        self.assertIsInstance(runtime, np.float)
        self.assertEqual(
            self.base_config['input_dir'],
            self.image_dir,
        )
        self.assertEqual(
            self.base_config['channel_ids'],
            self.pp_config['channel_ids'],
        )
        self.assertEqual(
            out_config['flat_field']['flat_field_dir'],
            os.path.join(self.output_dir, 'flat_field_images')
        )
        self.assertEqual(
            out_config['masks']['mask_dir'],
            os.path.join(self.output_dir, 'mask_channels_3')
        )
        self.assertEqual(
            out_config['tile']['tile_dir'],
            os.path.join(self.output_dir, 'tiles_10-10_step_10-10'),
        )
        # Make sure new mask channel assignment is correct
        self.assertEqual(out_config['masks']['mask_channel'], 4)
        # Check that masks are generated
        mask_dir = out_config['masks']['mask_dir']
        mask_meta = aux_utils.read_meta(mask_dir)
        mask_names = os.listdir(mask_dir)
        mask_names.pop(mask_names.index('frames_meta.csv'))
        # Validate that all masks are there
        self.assertEqual(
            len(mask_names),
            len(self.slice_ids) * len(self.pos_ids),
        )
        for p in self.pos_ids:
            for z in self.slice_ids:
                im_name = aux_utils.get_im_name(
                    channel_idx=out_config['masks']['mask_channel'],
                    slice_idx=z,
                    time_idx=self.time_idx,
                    pos_idx=p,
                )
                im = cv2.imread(
                    os.path.join(mask_dir, im_name),
                    cv2.IMREAD_ANYDEPTH,
                )
                self.assertTupleEqual(im.shape, (30, 20))
                self.assertTrue(im.dtype == 'uint8')
                self.assertTrue(im_name in mask_names)
                self.assertTrue(im_name in mask_meta['file_name'].tolist())
        # Check flatfield images
        ff_dir = out_config['flat_field']['flat_field_dir']
        ff_names = os.listdir(ff_dir)
        self.assertEqual(len(ff_names), 3)
        for processed_channel in [0, 1, 3]:
            expected_name = "flat-field_channel-{}.npy".format(processed_channel)
            self.assertTrue(expected_name in ff_names)
            im = np.load(os.path.join(ff_dir, expected_name))
            self.assertTrue(im.dtype == np.float64)
            self.assertTupleEqual(im.shape, (30, 20))
        # Check tiles
        tile_dir = out_config['tile']['tile_dir']
        tile_meta = aux_utils.read_meta(tile_dir)
        # 4 processed channels (0, 1, 3, 4), 6 tiles per image
        expected_rows = 4 * 6 * len(self.slice_ids) * len(self.pos_ids)
        self.assertEqual(tile_meta.shape[0], expected_rows)
        # Check indices
        self.assertListEqual(
            tile_meta.channel_idx.unique().tolist(),
            [0, 1, 3, 4],
        )
        self.assertListEqual(
            tile_meta.pos_idx.unique().tolist(),
            self.pos_ids,
        )
        self.assertListEqual(
            tile_meta.slice_idx.unique().tolist(),
            self.slice_ids,
        )
        self.assertListEqual(
            tile_meta.time_idx.unique().tolist(),
            [self.time_idx],
        )
        self.assertListEqual(
            list(tile_meta),
            ['channel_idx',
             'col_start',
             'file_name',
             'pos_idx',
             'row_start',
             'slice_idx',
             'time_idx']
        )
        self.assertListEqual(
            tile_meta.row_start.unique().tolist(),
            [0, 10, 20],
        )
        self.assertListEqual(
            tile_meta.col_start.unique().tolist(),
            [0, 10],
        )
        # Read one tile and check format
        # r = row start/end idx, c = column start/end, sl = slice start/end
        # sl0-1 signifies depth of 1
        im = np.load(os.path.join(
            tile_dir,
            'im_c001_z000_t000_p007_r10-20_c10-20_sl0-1.npy',
        ))
        self.assertTupleEqual(im.shape, (1, 10, 10))
        self.assertTrue(im.dtype == np.float64)

    def test_pre_process_weight_maps(self):
        cur_config = self.pp_config
        # Use preexisiting masks with more than one class, otherwise
        # weight map generation doesn't work
        cur_config['masks'] = {
            'mask_dir': self.input_mask_dir,
            'mask_channel': self.input_mask_channel,
        }
        cur_config['make_weight_map'] = True
        out_config, runtime = pp.pre_process(cur_config, self.base_config)

        # Check weights dir
        self.assertEqual(
            out_config['weights']['weights_dir'],
            os.path.join(self.output_dir, 'mask_channels_111')
        )
        weights_meta = aux_utils.read_meta(out_config['weights']['weights_dir'])
        # Check indices
        self.assertListEqual(
            weights_meta.channel_idx.unique().tolist(),
            [112],
        )
        self.assertListEqual(
            weights_meta.pos_idx.unique().tolist(),
            self.pos_ids,
        )
        self.assertListEqual(
            weights_meta.slice_idx.unique().tolist(),
            self.slice_ids,
        )
        self.assertListEqual(
            weights_meta.time_idx.unique().tolist(),
            [self.time_idx],
        )
        # Load one weights file and check contents
        im = np.load(os.path.join(
            out_config['weights']['weights_dir'],
            'im_c112_z002_t000_p007.npy',
        ))
        self.assertTupleEqual(im.shape, (30, 20))
        self.assertTrue(im.dtype == np.float64)
        # Check tiles
        tile_dir = out_config['tile']['tile_dir']
        tile_meta = aux_utils.read_meta(tile_dir)
        # 5 processed channels (0, 1, 3, 111, 112), 6 tiles per image
        expected_rows = 5 * 6 * len(self.slice_ids) * len(self.pos_ids)
        self.assertEqual(tile_meta.shape[0], expected_rows)
        # Check indices
        self.assertListEqual(
            tile_meta.channel_idx.unique().tolist(),
            [0, 1, 3, 111, 112],
        )
        self.assertListEqual(
            tile_meta.pos_idx.unique().tolist(),
            self.pos_ids,
        )
        self.assertListEqual(
            tile_meta.slice_idx.unique().tolist(),
            self.slice_ids,
        )
        self.assertListEqual(
            tile_meta.time_idx.unique().tolist(),
            [self.time_idx],
        )
        # Load one tile
        im = np.load(os.path.join(
            tile_dir,
            'im_c111_z002_t000_p008_r0-10_c10-20_sl0-1.npy',
        ))
        self.assertTupleEqual(im.shape, (1, 10, 10))
        self.assertTrue(im.dtype == bool)

    def test_pre_process_resize2d(self):
        cur_config = self.pp_config
        cur_config['resize'] = {
            'scale_factor': 2,
            'resize_3d': False,
        }
        cur_config['make_weight_map'] = False
        out_config, runtime = pp.pre_process(cur_config, self.base_config)

        self.assertIsInstance(runtime, np.float)
        self.assertEqual(
            out_config['resize']['resize_dir'],
            os.path.join(self.output_dir, 'resized_images')
        )
        resize_dir = out_config['resize']['resize_dir']
        # Check that all images have been resized
        resize_meta = aux_utils.read_meta(resize_dir)
        # 3 resized channels
        expected_rows = 3 * len(self.slice_ids) * len(self.pos_ids)
        self.assertEqual(resize_meta.shape[0], expected_rows)
        # Load an image and make sure it's twice as big
        im = cv2.imread(
            os.path.join(resize_dir, 'im_c003_z002_t000_p010.png'),
            cv2.IMREAD_ANYDEPTH,
        )
        self.assertTupleEqual(im.shape, (60, 40))
        self.assertTrue(im.dtype, np.uint8)
        # There should now be 2*2 the amount of tiles, same shape
        tile_dir = out_config['tile']['tile_dir']
        tile_meta = aux_utils.read_meta(tile_dir)
        # 4 processed channels (0, 1, 3, 4), 24 tiles per image
        expected_rows = 4 * 24 * len(self.slice_ids) * len(self.pos_ids)
        self.assertEqual(tile_meta.shape[0], expected_rows)
        # Load a tile and assert shape
        im = np.load(os.path.join(
            tile_dir,
            'im_c001_z000_t000_p007_r40-50_c20-30_sl0-1.npy',
        ))
        self.assertTupleEqual(im.shape, (1, 10, 10))
        self.assertTrue(im.dtype == np.float64)

    def test_pre_process_resize3d(self):
        cur_config = self.pp_config
        cur_config['resize'] = {
            'scale_factor': [1, 1.5, 1],
            'resize_3d': True,
        }
        cur_config['tile'] = {
            'tile_size': [10, 10],
            'step_size': [10, 10],
            'depths': [1, 1, 1],
            'mask_depth': 1,
            'image_format': 'zyx',
            'normalize_channels': [True, True, True],
        }
        out_config, runtime = pp.pre_process(cur_config, self.base_config)

        self.assertIsInstance(runtime, np.float)
        self.assertEqual(
            out_config['resize']['resize_dir'],
            os.path.join(self.output_dir, 'resized_images')
        )
        # Load a resized image and assert shape
        im_path = os.path.join(
            out_config['resize']['resize_dir'],
            'im_c000_z000_t000_p007_1.0-1.5-1.0.npy',
        )
        im = np.load(im_path)
        # shape should be 30, 20*1.5, z=6)
        self.assertTupleEqual(im.shape, (30, 30, 6))
        self.assertTrue(im.dtype == np.float64)

        self.assertEqual(
            out_config['masks']['mask_dir'],
            os.path.join(self.output_dir, 'mask_channels_3')
        )
        self.assertEqual(out_config['masks']['mask_channel'], 4)

        self.assertEqual(
            out_config['tile']['tile_dir'],
            os.path.join(self.output_dir, 'tiles_10-10_step_10-10'),
        )
        im_path = os.path.join(
            out_config['tile']['tile_dir'],
            'im_c000_z000_t000_p008_r0-10_c0-10_sl0-6.npy'
        )
        # A tile channels first should have shape (6, 10, 10)
        tile = np.load(im_path)
        self.assertTupleEqual(tile.shape, (6, 10, 10))
        self.assertTrue(tile.dtype == np.float64)

    def test_pre_process_nonisotropic(self):
        base_config = self.base_config
        base_config['uniform_struct'] = False
        out_config, runtime = pp.pre_process(self.pp_config, base_config)

        self.assertIsInstance(runtime, np.float)
        self.assertEqual(
            out_config['masks']['mask_dir'],
            os.path.join(self.output_dir, 'mask_channels_3')
        )
        self.assertEqual(out_config['masks']['mask_channel'], 4)
        self.assertEqual(
            out_config['tile']['tile_dir'],
            os.path.join(self.output_dir, 'tiles_10-10_step_10-10'),
        )

    def test_save_config(self):
        cur_config = self.pp_config
        cur_config['masks']['mask_dir'] = os.path.join(
            self.output_dir, 'mask_channels_3')
        cur_config['tile']['tile_dir'] = os.path.join(
            self.output_dir, 'tiles_10-10_step_10-10')
        pp.save_config(cur_config, 11.1)
        # Load json back up
        saved_info = aux_utils.read_json(
            os.path.join(self.output_dir, 'preprocessing_info.json'),
        )
        self.assertEqual(len(saved_info), 1)
        saved_config = saved_info[0]['config']
        self.assertDictEqual(saved_config, cur_config)
        # Save one more config
        cur_config['input_dir'] = cur_config['tile']['tile_dir']
        pp.save_config(cur_config, 666.66)
        # Load json back up
        saved_info = aux_utils.read_json(
            os.path.join(self.output_dir, 'preprocessing_info.json'),
        )
        self.assertEqual(len(saved_info), 2)
        saved_config = saved_info[1]['config']
        self.assertDictEqual(saved_config, cur_config)
