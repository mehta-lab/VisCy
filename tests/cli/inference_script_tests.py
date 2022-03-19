import cv2
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest
from unittest.mock import patch
import yaml

import micro_dl.cli.inference_script as inference_script
import micro_dl.utils.aux_utils as aux_utils


class TestInferenceScript(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with some images to generate frames_meta.csv for
        """
        self.tempdir = TempDirectory()
        self.temp_dir = self.tempdir.path
        self.model_dir = os.path.join(self.temp_dir, 'model_dir')
        self.pred_dir = os.path.join(self.model_dir, 'predictions')
        self.image_dir = os.path.join(self.temp_dir, 'image_dir')
        self.tempdir.makedir(self.model_dir)
        self.tempdir.makedir(self.pred_dir)
        self.tempdir.makedir(self.image_dir)
        # Write images
        self.time_idx = 5
        self.pos_idx = 7
        self.im = 1500 * np.ones((30, 20), dtype=np.uint16)
        self.ext = '.tif'
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        self.frames_meta = aux_utils.make_dataframe()

        for c in range(3):
            for z in range(5, 10):
                im_name = aux_utils.get_im_name(
                    channel_idx=c,
                    slice_idx=z,
                    time_idx=self.time_idx,
                    pos_idx=self.pos_idx,
                    ext=self.ext,
                )
                cv2.imwrite(os.path.join(self.image_dir, im_name), self.im)
                meta_row = aux_utils.parse_idx_from_name(
                    im_name)
                meta_row['zscore_median'] = 1500
                meta_row['zscore_iqr'] = 1
                self.frames_meta = self.frames_meta.append(
                    meta_row,
                    ignore_index=True,
                )
        # Write metadata
        self.frames_meta.to_csv(
            os.path.join(self.image_dir, self.meta_name),
            sep=',',
        )
        # Write split samples
        split_fname = os.path.join(self.model_dir, 'split_samples.json')
        split_samples = {'test': [7]}
        aux_utils.write_json(split_samples, split_fname)
        # Create preprocessing config
        self.pp_config = {
            'normalize_im': 'dataset'
        }
        processing_info = [{'processing_time': 1,
                            'config':
                                self.pp_config}]
        pp_fname = os.path.join(self.image_dir, 'preprocessing_info.json')
        aux_utils.write_json(processing_info, pp_fname)
        # Write train config in model dir
        self.train_config = {
            'dataset': {
                'data_dir': self.image_dir,
                'input_channels': [0, 1],
                'target_channels': [2],
                'split_by_column': 'pos_idx',
                'model_task': 'regression',
            },
            'network': {
                'class': 'UNet2D',
                'data_format': 'channels_first',
                'depth': 1,
                'width': 10,
                'height': 10,
            },
        }
        self.train_config_name = os.path.join(self.model_dir, 'config.yml')
        with open(self.train_config_name, 'w') as outfile:
            yaml.dump(self.train_config, outfile, default_flow_style=False)
        # Write inference config in pred dir
        self.inference_config = {
            'model_dir': self.model_dir,
            'model_fname': 'dummy_weights.hdf5',
            'image_dir': self.image_dir,
            'data_split': 'test',
            'images': {
                'image_format': 'zyx',
                'image_ext': '.tif',
            },
            'metrics': {
                'metrics': ['mse', 'mae'],
                'metrics_orientations': ['xy', 'xyz'],
            },
        }
        self.infer_config_name = os.path.join(self.pred_dir, 'config_inference_3d.yml')
        with open(self.infer_config_name, 'w') as outfile:
            yaml.dump(self.inference_config, outfile, default_flow_style=False)

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_dir), False)

    def test_parse_args(self):
        with patch('argparse._sys.argv',
                   ['python',
                    '--gpu', '3',
                    '--config', self.infer_config_name]):
            parsed_args = inference_script.parse_args()
            self.assertEqual(parsed_args.gpu, 3)
            self.assertEqual(parsed_args.config, self.infer_config_name)
            self.assertIsNone(parsed_args.gpu_mem_frac)

    @nose.tools.raises(BaseException)
    def test_parse_args_no_config(self):
        with patch('argparse._sys.argv',
                   ['python',
                    '--gpu', 1]):
            inference_script.parse_args()

    @patch('micro_dl.inference.model_inference.load_model')
    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_run_inference(self, mock_predict, mock_model):
        mock_model.return_value = 'dummy_model'
        # Image shape is cropped to the nearest factor of 2
        mock_predict.return_value = np.zeros((1, 16, 16), dtype=np.float32)
        # Run inference
        inference_script.run_inference(
            config_fname=self.infer_config_name,
            gpu_ids=-1,
        )
        # Check 3D metrics
        metrics = pd.read_csv(os.path.join(self.pred_dir, 'metrics_xyz.csv'))
        self.assertTupleEqual(metrics.shape, (1, 3))
        self.assertEqual(metrics.mse[0], 0.)
        self.assertEqual(metrics.mae[0], 0.)
        # Rhe name will use the first indices in stack so z = 5
        self.assertEqual(metrics.pred_name[0], 'im_c002_z005_t005_p007')
        # Check 2D xy metrics
        metrics = pd.read_csv(os.path.join(self.pred_dir, 'metrics_xy.csv'))
        self.assertTupleEqual(metrics.shape, (5, 3))
        for i, test_z in enumerate([5, 6, 7, 8, 9]):
            self.assertEqual(metrics.mse[i], 0.)
            self.assertEqual(metrics.mae[i], 0.)
            self.assertEqual(
                metrics.pred_name[i],
                'im_c002_z00{}_t005_p007_xy0'.format(test_z))

        # Check that all predictions are there
        for test_z in [5, 6, 7, 8, 9]:
            pred_name = os.path.join(
                self.pred_dir,
                'im_c002_z00{}_t005_p007.tif'.format(test_z),
            )
            pred_im = cv2.imread(pred_name, cv2.IMREAD_ANYDEPTH)
            self.assertEqual(pred_im.dtype, np.uint16)
            self.assertTupleEqual(pred_im.shape, (16, 16))
