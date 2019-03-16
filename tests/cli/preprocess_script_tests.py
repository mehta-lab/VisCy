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
        self.output_dir = os.path.join(self.temp_path, 'out_dir')
        self.tempdir.makedir(self.output_dir)
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        self.frames_meta = aux_utils.make_dataframe()
        # Write images
        self.time_idx = 5
        self.pos_idx = 7
        self.im = 1500 * np.ones((30, 20), dtype=np.uint16)
        self.im[10:20, 5:15] = 3000

        for c in range(4):
            for p in range(self.pos_idx, self.pos_idx + 3):
                for z in range(5):
                    im_name = aux_utils.get_im_name(
                        channel_idx=c,
                        slice_idx=z,
                        time_idx=self.time_idx,
                        pos_idx=p,
                        ext='.png',
                    )
                    cv2.imwrite(os.path.join(self.temp_path, im_name),
                                self.im + c * 100)
                    self.frames_meta = self.frames_meta.append(
                        aux_utils.parse_idx_from_name(im_name),
                        ignore_index=True,
                    )
        # Write metadata
        self.frames_meta.to_csv(
            os.path.join(self.temp_path, self.meta_name),
            sep=',',
        )
        self.pp_config = {
            'output_dir': self.output_dir,
            'input_dir': self.temp_path,
            'channel_ids': [0, 1, 3],
            'num_workers': 4,
            'flat_field': {'estimate': True,
                           'block_size': 2,
                           'correct': True},
            'resize': {'scale_factor': 2,
                       'resize_3d': False},
            'masks': {'channels': [3],
                      'str_elem_radius': 3},
            'tile': {'tile_size': [10, 10],
                     'step_size': [10, 10],
                     'depths': [1, 1, 1],
                     'mask_depth': 1,
                     'image_format': 'zyx'},
        }
        self.base_config = {
            'input_dir': self.temp_path,
            'output_dir': self.output_dir,
            'slice_ids': -1,
            'time_ids': -1,
            'pos_ids': -1,
            'channel_ids': self.pp_config['channel_ids'],
            'uniform_struct': True,
            'int2strlen': 3,
            'num_workers': 4,
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
            os.path.join(self.output_dir, 'resized_images')
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
            out_config['resize']['resize_dir'],
            os.path.join(self.output_dir, 'resized_images')
        )
        self.assertEqual(
            out_config['masks']['mask_dir'],
            os.path.join(self.output_dir, 'mask_channels_3')
        )
        self.assertEqual(out_config['masks']['mask_out_channel'], 4)
        self.assertEqual(
            out_config['tile']['tile_dir'],
            os.path.join(self.output_dir, 'tiles_10-10_step_10-10'),
        )

    def test_pre_process_resize3d(self):
        cur_config = self.pp_config
        cur_config['resize']['scale_factor'] = [2, 1, 1]
        cur_config['resize']['resize_3d'] = True
        out_config, runtime = pp.pre_process(cur_config, self.base_config)

        self.assertIsInstance(runtime, np.float)
        self.assertEqual(
            out_config['resize']['resize_dir'],
            os.path.join(self.output_dir, 'resized_images')
        )
        self.assertEqual(
            out_config['masks']['mask_dir'],
            os.path.join(self.output_dir, 'mask_channels_3')
        )
        self.assertEqual(out_config['masks']['mask_out_channel'], 4)
        self.assertEqual(
            out_config['tile']['tile_dir'],
            os.path.join(self.output_dir, 'tiles_10-10_step_10-10'),
        )

    def test_pre_process_nonisotropic(self):
        base_config = self.base_config
        base_config['uniform_struct'] = False
        out_config, runtime = pp.pre_process(self.pp_config, base_config)

        self.assertIsInstance(runtime, np.float)
        self.assertEqual(
            out_config['resize']['resize_dir'],
            os.path.join(self.output_dir, 'resized_images')
        )
        self.assertEqual(
            out_config['masks']['mask_dir'],
            os.path.join(self.output_dir, 'mask_channels_3')
        )
        self.assertEqual(out_config['masks']['mask_out_channel'], 4)
        self.assertEqual(
            out_config['tile']['tile_dir'],
            os.path.join(self.output_dir, 'tiles_10-10_step_10-10'),
        )

    def test_save_config(self):
        cur_config = self.pp_config
        cur_config['resize']['resize_dir'] = os.path.join(
            self.output_dir, 'resized_images')
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
