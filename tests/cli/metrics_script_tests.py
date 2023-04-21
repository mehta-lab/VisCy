import cv2
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest
from unittest.mock import patch
import yaml

import micro_dl.cli.metrics_script as metrics_script
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.normalize as normalize


class TestMetricsScript(unittest.TestCase):

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
        im_add = np.zeros((30, 20), dtype=np.uint16)
        im_add[15:, :] = 10
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
                if c == 2:
                    norm_im = normalize.zscore(self.im + im_add).astype(np.float32)
                    cv2.imwrite(
                        os.path.join(self.pred_dir, im_name),
                        norm_im,
                    )
                self.frames_meta = self.frames_meta.append(
                    aux_utils.parse_idx_from_name(
                        im_name=im_name,
                        dir_name=self.image_dir,
                    ),
                    ignore_index=True,
                )
        # Write metadata
        self.frames_meta.to_csv(
            os.path.join(self.image_dir, self.meta_name),
            sep=',',
        )
        # Write as test metadata in model dir too
        self.frames_meta.to_csv(
            os.path.join(self.model_dir, 'test_metadata.csv'),
            sep=',',
        )
        # Write split samples
        split_idx_fname = os.path.join(self.model_dir, 'split_samples.json')
        split_samples = {'test': [5, 6, 7, 8, 9]}
        aux_utils.write_json(split_samples, split_idx_fname)
        # Write config in model dir
        config = {
            'dataset': {
                'input_channels': [0, 1],
                'target_channels': [2],
                'split_by_column': 'slice_idx',
                'data_dir': self.image_dir
            },
            'network': {}
        }
        config_name = os.path.join(self.model_dir, 'config.yml')
        with open(config_name, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        # Write preprocess config
        pp_config = {
            'normalize_im': 'stack',
        }
        processing_info = [{'processing_time': 5,
                            'config': pp_config}]
        config_name = os.path.join(self.image_dir, 'preprocessing_info.json')
        aux_utils.write_json(processing_info, config_name)

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_dir), False)

    def test_parse_args(self):
        with patch('argparse._sys.argv',
                   ['python',
                    '--model_dir', self.model_dir,
                    '--image_dir', self.image_dir,
                    '--metrics', 'ssim', 'corr',
                    '--orientations', 'xy', 'xz']):
            parsed_args = metrics_script.parse_args()
            self.assertEqual(parsed_args.model_dir, self.model_dir)
            self.assertTrue(parsed_args.test_data)
            self.assertEqual(parsed_args.image_dir, self.image_dir)
            self.assertListEqual(parsed_args.metrics, ['ssim', 'corr'])
            self.assertListEqual(parsed_args.orientations, ['xy', 'xz'])

    @nose.tools.raises(BaseException)
    def test_parse_args_no_input(self):
        with patch('argparse._sys.argv',
                   ['python',
                    '--model_dir', self.model_dir]):
            metrics_script.parse_args()

    def test_compute_metrics(self):
        metrics_script.compute_metrics(
            model_dir=self.model_dir,
            image_dir=self.image_dir,
            metrics_list=['mse', 'mae'],
            orientations_list=['xy', 'xyz'],
            name_parser='parse_idx_from_name',
        )
        metrics_xy = pd.read_csv(os.path.join(self.pred_dir, 'metrics_xy.csv'))
        self.assertTupleEqual(metrics_xy.shape, (5, 3))
        for i, row in metrics_xy.iterrows():
            expected_name = 't5_p7_xy{}'.format(i)
            self.assertEqual(row.pred_name, expected_name)
            # TODO: Double check values below
            # self.assertEqual(row.mse, 1.0)
            # self.assertEqual(row.mae, 1.0)
        # Same for xyz
        metrics_xyz = pd.read_csv(
            os.path.join(self.pred_dir, 'metrics_xyz.csv'),
        )
        self.assertTupleEqual(metrics_xyz.shape, (1, 3))
        # self.assertEqual(metrics_xyz.loc[0, 'mse'], 1.0)
        # self.assertEqual(metrics_xyz.loc[0, 'mae'], 1.0)
        self.assertEqual(metrics_xyz.loc[0, 'pred_name'], 't5_p7')
