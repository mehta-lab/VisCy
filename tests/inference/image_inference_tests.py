import cv2
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest
from unittest.mock import patch

import micro_dl.inference.image_inference as image_inference
import micro_dl.utils.aux_utils as aux_utils


class TestImageInference(unittest.TestCase):

    @patch('micro_dl.inference.model_inference.load_model')
    def setUp(self, mock_model):
        """
        Set up a directory with images
        """
        mock_model.return_value = 'dummy_model'

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.tempdir.makedir('image_dir')
        self.tempdir.makedir('mask_dir')
        self.tempdir.makedir('model_dir')
        self.image_dir = os.path.join(self.temp_path, 'image_dir')
        self.mask_dir = os.path.join(self.temp_path, 'mask_dir')
        self.model_dir = os.path.join(self.temp_path, 'model_dir')
        # Create a temp image dir
        self.im = np.zeros((10, 16), dtype=np.uint8)
        self.frames_meta = aux_utils.make_dataframe()
        self.time_idx = 2
        self.slice_idx = 3
        for p in range(5):
            for c in range(3):
                im_name = aux_utils.get_im_name(
                    time_idx=self.time_idx,
                    channel_idx=c,
                    slice_idx=self.slice_idx,
                    pos_idx=p,
                )
                cv2.imwrite(os.path.join(self.image_dir, im_name), self.im + c * 10)
                self.frames_meta = self.frames_meta.append(
                    aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                    ignore_index=True,
                )
        # Write frames meta to image dir too
        self.frames_meta.to_csv(os.path.join(self.image_dir, 'frames_meta.csv'))
        # Save masks and mask meta
        self.mask_meta = aux_utils.make_dataframe()
        self.mask_channel = 50
        for p in range(5):
            im_name = aux_utils.get_im_name(
                time_idx=self.time_idx,
                channel_idx=self.mask_channel,
                slice_idx=self.slice_idx,
                pos_idx=p,
            )
            cv2.imwrite(os.path.join(self.mask_dir, im_name), self.im + 1)
            self.mask_meta = self.mask_meta.append(
                aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                ignore_index=True,
            )
        # Write frames meta to mask dir too
        self.mask_meta.to_csv(os.path.join(self.mask_dir, 'frames_meta.csv'))
        # Setup model dir
        split_samples = {
            "train": [0, 1],
            "val": [2],
            "test": [3, 4],
        }
        aux_utils.write_json(
            split_samples,
            os.path.join(self.model_dir, 'split_samples.json'),
        )
        # Make configs with fields necessary for 2D segmentation inference
        self.train_config = {
            'network': {
                'class': 'UNet2D',
                'data_format': 'channels_first',
                'depth': 1,
                'width': 10,
                'height': 10},
            'dataset': {
                'split_by_column': 'pos_idx',
                'input_channels': [1, 2],
                'target_channels': [self.mask_channel],
                'model_task': 'segmentation',
            },
        }
        self.inference_config = {
            'model_dir': self.model_dir,
            'model_fname': 'dummy_weights.hdf5',
            'image_dir': self.image_dir,
            'data_split': 'test',
            'images': {
                'image_format': 'zyx',
                'image_ext': '.png',
            },
            'metrics': {
                'metrics': ['mae'],
                'metrics_orientations': ['xy'],
            },
            'masks': {
                'mask_dir': self.mask_dir,
                'mask_type': 'target',
                'mask_channel': 50,
            }
        }
        # Instantiate class
        self.infer_inst = image_inference.ImagePredictor(
            train_config=self.train_config,
            inference_config=self.inference_config,
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        self.assertEqual(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Test init of inference dataset
        """
        # Check proper init
        self.assertEqual(self.infer_inst.model_dir, self.model_dir)
        self.assertEqual(self.infer_inst.image_dir, self.image_dir)
        self.assertEqual(self.infer_inst.data_format, 'channels_first')
        self.assertEqual(self.infer_inst.model, 'dummy_model')
        self.assertEqual(self.infer_inst.image_format, 'zyx')
        self.assertEqual(self.infer_inst.image_ext, '.png')
        self.assertFalse(self.infer_inst.mask_metrics)
        self.assertEqual(self.infer_inst.mask_dir, self.mask_dir)
        self.assertListEqual(self.infer_inst.metrics_orientations, ['xy'])
        self.assertEqual(self.infer_inst.num_overlap, 0)
        self.assertIsNone(self.infer_inst.stitch_inst)
        self.assertIsNone(self.infer_inst.tile_option)
        self.assertIsNone(self.infer_inst.crop_shape)

    def test_get_split_ids(self):
        split_col, infer_ids = self.infer_inst._get_split_ids()
        self.assertEqual(split_col, 'pos_idx')
        self.assertListEqual(infer_ids, [3, 4])

    def test_get_split_ids_no_json(self):
        self.infer_inst.model_dir = self.infer_inst.image_dir
        split_col, infer_ids = self.infer_inst._get_split_ids()
        self.assertEqual(split_col, 'pos_idx')
        self.assertListEqual(infer_ids, [0, 1, 2, 3, 4])

    def test_save_pred_image(self):
        im = np.zeros((1, 10, 15), dtype=np.uint8)
        im[:, 5, :] = 128
        self.infer_inst.save_pred_image(
            predicted_image=im,
            time_idx=10,
            target_channel_idx=20,
            pos_idx=30,
            slice_idx=40,
        )
        pred_name = os.path.join(
            self.model_dir,
            'predictions/im_c020_z040_t010_p030.png',
        )
        im_pred = cv2.imread(pred_name, cv2.IMREAD_ANYDEPTH)
        self.assertEqual(im_pred.dtype, np.uint16)
        self.assertTupleEqual(im_pred.shape, (10, 15))
        # Prediction intensities are maximized to range
        self.assertEqual(im_pred.max(), 65535)
        self.assertEqual(im_pred.min(), 0)

    def test_estimate_metrics_xy(self):
        target = np.ones((10, 15, 5), dtype=np.float64)
        prediction = np.zeros_like(target)
        prediction[:5, :, :] = 1
        pred_names = ['test1', 'test2', 'test3', 'test4', 'test5']
        self.infer_inst.estimate_metrics(target, prediction, pred_names, None)
        metrics = self.infer_inst.df_xy
        self.assertTupleEqual(metrics.shape, (5, 2))
        self.assertListEqual(list(metrics), ['mae', 'pred_name'])
        self.assertEqual(metrics.mae.mean(), 0.5)

    def test_estimate_metrics_xy_one_name(self):
        target = np.ones((10, 15, 5), dtype=np.float64)
        prediction = np.zeros_like(target)
        prediction[:5, :, :] = 1
        self.infer_inst.estimate_metrics(target, prediction, ['test_name'], None)
        metrics = self.infer_inst.df_xy
        self.assertTupleEqual(metrics.shape, (5, 2))
        self.assertListEqual(list(metrics), ['mae', 'pred_name'])
        self.assertEqual(metrics.mae.mean(), 0.5)

    def test_estimate_metrics_xyz(self):
        target = np.ones((10, 15, 5), dtype=np.float64)
        prediction = np.zeros_like(target)
        prediction[:5, :, :] = 1
        self.infer_inst.metrics_orientations = ['xyz']
        self.infer_inst.estimate_metrics(target, prediction, ['test_name'], None)
        metrics = self.infer_inst.df_xyz
        self.assertTupleEqual(metrics.shape, (1, 2))
        self.assertListEqual(list(metrics), ['mae', 'pred_name'])
        self.assertEqual(metrics.mae[0], 0.5)
        self.assertEqual(metrics.pred_name[0], 'test_name')

    def test_estimate_metrics_xz(self):
        target = np.ones((10, 15, 5), dtype=np.float64)
        prediction = np.zeros_like(target)
        prediction[:5, :, :] = 1
        self.infer_inst.metrics_orientations = ['xz']
        self.infer_inst.estimate_metrics(target, prediction, ['test_name'], None)
        metrics = self.infer_inst.df_xz
        self.assertTupleEqual(metrics.shape, (10, 2))
        self.assertListEqual(list(metrics), ['mae', 'pred_name'])
        self.assertEqual(metrics.mae[0], 0.0)
        self.assertEqual(metrics.mae[5], 1.0)
        self.assertEqual(metrics.pred_name[9], 'test_name_xz9')

    def test_estimate_metrics_yz(self):
        target = np.ones((10, 15, 5), dtype=np.float64)
        prediction = np.zeros_like(target)
        prediction[:5, :, :] = 1
        self.infer_inst.metrics_orientations = ['yz']
        self.infer_inst.estimate_metrics(target, prediction, ['test_name'], None)
        metrics = self.infer_inst.df_yz
        self.assertTupleEqual(metrics.shape, (15, 2))
        self.assertListEqual(list(metrics), ['mae', 'pred_name'])
        self.assertEqual(metrics.mae[0], 0.5)
        self.assertEqual(metrics.pred_name[14], 'test_name_yz14')

    def test_get_mask(self):
        meta_row = dict.fromkeys(aux_utils.DF_NAMES)
        meta_row['channel_idx'] = self.mask_channel
        meta_row['time_idx'] = self.time_idx
        meta_row['slice_idx'] = self.slice_idx
        meta_row['pos_idx'] = 2
        mask = self.infer_inst.get_mask(meta_row)
        self.assertTupleEqual(mask.shape, (8, 16))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.max(), 1)
        self.assertEqual(mask.min(), 1)

    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_predict_2d(self, mock_predict):
        mock_predict.return_value = 1. + np.ones((1, 8, 16), dtype=np.float64)
        # Predict row 0 from inference dataset iterator
        pred_im, target_im, mask_im = self.infer_inst.predict_2d([0])
        self.assertTupleEqual(pred_im.shape, (8, 16, 1))
        self.assertEqual(pred_im.dtype, np.float64)
        self.assertEqual(pred_im.max(), 2.0)
        # Read saved prediction too
        pred_name = os.path.join(
            self.model_dir,
            'predictions/im_c050_z003_t002_p003.png',
        )
        im_pred = cv2.imread(pred_name, cv2.IMREAD_ANYDEPTH)
        self.assertEqual(im_pred.dtype, np.uint16)
        self.assertTupleEqual(im_pred.shape, (8, 16))
        # Check target and no mask
        self.assertTupleEqual(target_im.shape, (8, 16, 1))
        self.assertEqual(target_im.dtype, np.float64)
        self.assertEqual(target_im.max(), 1.)
        self.assertListEqual(mask_im, [])

    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_run_prediction(self, mock_predict):
        mock_predict.return_value = 1. + np.ones((1, 8, 16), dtype=np.float64)
        # Run prediction. Should create a metrics_xy.csv in pred dir
        self.infer_inst.run_prediction()
        metrics = pd.read_csv(os.path.join(self.model_dir, 'predictions/metrics_xy.csv'))
        self.assertTupleEqual(metrics.shape, (2, 2))
        # MAE should be 1.
        self.assertEqual(metrics.mae.mean(), 1.0)
        # There should be two rows, one per test index
        self.assertEqual(metrics.pred_name[0], 'im_c050_z003_t002_p003_xy0')
        self.assertEqual(metrics.pred_name[1], 'im_c050_z003_t002_p004_xy0')
        # There should be 2 predictions saved in pred dir
        for pos in range(3, 5):
            pred_name = os.path.join(
                self.model_dir,
                'predictions/im_c050_z003_t002_p00{}.png'.format(pos),
            )
            im_pred = cv2.imread(pred_name, cv2.IMREAD_ANYDEPTH)
            self.assertEqual(im_pred.dtype, np.uint16)
            self.assertTupleEqual(im_pred.shape, (8, 16))


class TestImageInference2p5D(unittest.TestCase):

    @patch('micro_dl.inference.model_inference.load_model')
    def setUp(self, mock_model):
        """
        Set up a directory with images
        """
        mock_model.return_value = 'dummy_model'

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.tempdir.makedir('image_dir')
        self.tempdir.makedir('mask_dir')
        self.tempdir.makedir('model_dir')
        self.image_dir = os.path.join(self.temp_path, 'image_dir')
        self.mask_dir = os.path.join(self.temp_path, 'mask_dir')
        self.model_dir = os.path.join(self.temp_path, 'model_dir')
        # Create a temp image dir
        self.im = np.zeros((10, 16), dtype=np.uint8)
        self.frames_meta = aux_utils.make_dataframe()
        self.time_idx = 2
        for p in range(5):
            for c in range(3):
                for z in range(6):
                    im_name = aux_utils.get_im_name(
                        time_idx=self.time_idx,
                        channel_idx=c,
                        slice_idx=z,
                        pos_idx=p,
                    )
                    cv2.imwrite(os.path.join(self.image_dir, im_name), self.im + c * 10)
                    self.frames_meta = self.frames_meta.append(
                        aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                        ignore_index=True,
                    )
        # Write frames meta to image dir too
        self.frames_meta.to_csv(os.path.join(self.image_dir, 'frames_meta.csv'))
        # Save masks and mask meta
        self.mask_meta = aux_utils.make_dataframe()
        self.mask_channel = 50
        for p in range(5):
            for z in range(6):
                im_name = aux_utils.get_im_name(
                    time_idx=self.time_idx,
                    channel_idx=self.mask_channel,
                    slice_idx=z,
                    pos_idx=p,
                )
                cv2.imwrite(os.path.join(self.mask_dir, im_name), self.im + 1)
                self.mask_meta = self.mask_meta.append(
                    aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                    ignore_index=True,
                )
        # Write frames meta to mask dir too
        self.mask_meta.to_csv(os.path.join(self.mask_dir, 'frames_meta.csv'))
        # Setup model dir
        split_samples = {
            "train": [0, 1],
            "val": [2],
            "test": [3, 4],
        }
        aux_utils.write_json(
            split_samples,
            os.path.join(self.model_dir, 'split_samples.json'),
        )
        # Make configs with fields necessary for 2.5D segmentation inference
        self.train_config = {
            'network': {
                'class': 'UNetStackTo2D',
                'data_format': 'channels_first',
                'depth': 5,
                'width': 10,
                'height': 10},
            'dataset': {
                'split_by_column': 'pos_idx',
                'input_channels': [1],
                'target_channels': [self.mask_channel],
                'model_task': 'segmentation',
            },
        }
        self.inference_config = {
            'model_dir': self.model_dir,
            'model_fname': 'dummy_weights.hdf5',
            'image_dir': self.image_dir,
            'data_split': 'test',
            'images': {
                'image_format': 'zyx',
                'image_ext': '.png',
            },
            'metrics': {
                'metrics': ['dice'],
                'metrics_orientations': ['xy'],
            },
            'masks': {
                'mask_dir': self.mask_dir,
                'mask_type': 'target',
                'mask_channel': 50,
            }
        }
        # Instantiate class
        self.infer_inst = image_inference.ImagePredictor(
            train_config=self.train_config,
            inference_config=self.inference_config,
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        self.assertEqual(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Test init of inference dataset
        """
        # Check proper init
        self.assertEqual(self.infer_inst.model_dir, self.model_dir)
        self.assertEqual(self.infer_inst.image_dir, self.image_dir)
        self.assertEqual(self.infer_inst.data_format, 'channels_first')
        self.assertEqual(self.infer_inst.model, 'dummy_model')
        self.assertEqual(self.infer_inst.image_format, 'zyx')
        self.assertEqual(self.infer_inst.image_ext, '.png')
        self.assertFalse(self.infer_inst.mask_metrics)
        self.assertEqual(self.infer_inst.mask_dir, self.mask_dir)
        self.assertListEqual(self.infer_inst.metrics_orientations, ['xy'])
        self.assertEqual(self.infer_inst.num_overlap, 0)
        self.assertIsNone(self.infer_inst.stitch_inst)
        self.assertIsNone(self.infer_inst.tile_option)
        self.assertIsNone(self.infer_inst.crop_shape)

    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_predict_2d(self, mock_predict):
        mock_predict.return_value = 1. + np.ones((1, 1, 1, 8, 16), dtype=np.float64)
        # Predict row 0 from inference dataset iterator
        pred_im, target_im, mask_im = self.infer_inst.predict_2d([0])
        self.assertTupleEqual(pred_im.shape, (8, 16, 1))
        self.assertEqual(pred_im.dtype, np.float64)
        self.assertEqual(pred_im.max(), 2.0)
        # Read saved prediction, z=2 for first slice with depth=5
        pred_name = os.path.join(
            self.model_dir,
            'predictions/im_c050_z002_t002_p003.png',
        )
        im_pred = cv2.imread(pred_name, cv2.IMREAD_ANYDEPTH)
        self.assertEqual(im_pred.dtype, np.uint16)
        self.assertTupleEqual(im_pred.shape, (8, 16))
        # Check target and no mask
        self.assertTupleEqual(target_im.shape, (8, 16, 1))
        self.assertEqual(target_im.dtype, np.float64)
        self.assertEqual(target_im.max(), 1.)
        self.assertListEqual(mask_im, [])

    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_run_prediction(self, mock_predict):
        mock_predict.return_value = 1. + np.ones((1, 1, 1, 8, 16), dtype=np.float64)
        # Run prediction. Should create a metrics_xy.csv in pred dir
        self.infer_inst.run_prediction()
        metrics = pd.read_csv(os.path.join(self.model_dir, 'predictions/metrics_xy.csv'))
        self.assertTupleEqual(metrics.shape, (4, 2))
        # Dice should be 1.
        self.assertEqual(metrics.dice.mean(), 1.0)
        # There should be four rows, one per test index pos=3,4
        # depth=5 means center slices z=2,3 will be evaluated
        self.assertEqual(metrics.pred_name[0], 'im_c050_z002_t002_p003_xy0')
        self.assertEqual(metrics.pred_name[1], 'im_c050_z003_t002_p003_xy0')
        self.assertEqual(metrics.pred_name[2], 'im_c050_z002_t002_p004_xy0')
        self.assertEqual(metrics.pred_name[3], 'im_c050_z003_t002_p004_xy0')
        # There should be 4 predictions saved in pred dir
        for p in [3, 4]:
            for z in [2, 3]:
                pred_name = os.path.join(
                    self.model_dir,
                    'predictions/im_c050_z00{}_t002_p00{}.png'.format(z, p),
                )
            im_pred = cv2.imread(pred_name, cv2.IMREAD_ANYDEPTH)
            self.assertEqual(im_pred.dtype, np.uint16)
            self.assertTupleEqual(im_pred.shape, (8, 16))


class TestImageInference3D(unittest.TestCase):

    @patch('micro_dl.inference.model_inference.load_model')
    def setUp(self, mock_model):
        """
        Set up a directory with 3D images
        """
        mock_model.return_value = 'dummy_model'

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.tempdir.makedir('image_dir')
        self.tempdir.makedir('mask_dir')
        self.tempdir.makedir('model_dir')
        self.image_dir = os.path.join(self.temp_path, 'image_dir')
        self.mask_dir = os.path.join(self.temp_path, 'mask_dir')
        self.model_dir = os.path.join(self.temp_path, 'model_dir')
        # Create a temp image dir
        self.im = np.zeros((10, 10, 8), dtype=np.uint8)
        self.frames_meta = aux_utils.make_dataframe()
        self.time_idx = 2
        self.slice_idx = 0
        for p in range(5):
            for c in range(3):
                    im_name = aux_utils.get_im_name(
                        time_idx=self.time_idx,
                        channel_idx=c,
                        slice_idx=self.slice_idx,
                        pos_idx=p,
                        ext='.npy',
                    )
                    np.save(os.path.join(self.image_dir, im_name),
                            self.im + c * 10,
                            allow_pickle=True,
                            fix_imports=True)
                    self.frames_meta = self.frames_meta.append(
                        aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                        ignore_index=True,
                    )
        # Write frames meta to image dir too
        self.frames_meta.to_csv(os.path.join(self.image_dir, 'frames_meta.csv'))
        # Save masks and mask meta
        self.mask_meta = aux_utils.make_dataframe()
        self.mask_channel = 50
        # Mask half the image
        mask = np.zeros_like(self.im)
        mask[:5, ...] = 1
        for p in range(5):
            im_name = aux_utils.get_im_name(
                time_idx=self.time_idx,
                channel_idx=self.mask_channel,
                slice_idx=self.slice_idx,
                pos_idx=p,
                ext='.npy',
            )
            np.save(os.path.join(self.mask_dir, im_name), mask)
            self.mask_meta = self.mask_meta.append(
                aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                ignore_index=True,
            )
        # Write frames meta to mask dir too
        self.mask_meta.to_csv(os.path.join(self.mask_dir, 'frames_meta.csv'))
        # Setup model dir
        split_samples = {
            "train": [0, 1],
            "val": [2],
            "test": [3, 4],
        }
        aux_utils.write_json(
            split_samples,
            os.path.join(self.model_dir, 'split_samples.json'),
        )
        # Make configs with fields necessary for 2.5D segmentation inference
        self.train_config = {
            'network': {
                'class': 'UNet3D',
                'data_format': 'channels_first',
                'num_filters_per_block': [8, 16],
                'depth': 5,
                'width': 5,
                'height': 5},
            'dataset': {
                'split_by_column': 'pos_idx',
                'input_channels': [1],
                'target_channels': [2],
                'model_task': 'regression',
            },
        }
        self.inference_config = {
            'model_dir': self.model_dir,
            'model_fname': 'dummy_weights.hdf5',
            'image_dir': self.image_dir,
            'data_split': 'test',
            'images': {
                'image_format': 'zyx',
                'image_ext': '.png',
            },
            'metrics': {
                'metrics': ['mse'],
                'metrics_orientations': ['xyz'],
            },
            'masks': {
                'mask_dir': self.mask_dir,
                'mask_type': 'metrics',
                'mask_channel': 50,
            },
            'inference_3d': {
                'tile_shape': [5, 5, 5],
                'num_overlap': [1, 1, 1],
                'overlap_operation': 'mean',
            },
        }
        # Instantiate class
        self.infer_inst = image_inference.ImagePredictor(
            train_config=self.train_config,
            inference_config=self.inference_config,
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        self.assertEqual(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Test init of inference dataset
        """
        # Check proper init
        self.assertEqual(self.infer_inst.model_dir, self.model_dir)
        self.assertEqual(self.infer_inst.image_dir, self.image_dir)
        self.assertEqual(self.infer_inst.data_format, 'channels_first')
        self.assertEqual(self.infer_inst.model, 'dummy_model')
        self.assertEqual(self.infer_inst.image_format, 'zyx')
        self.assertEqual(self.infer_inst.image_ext, '.npy')
        self.assertTrue(self.infer_inst.mask_metrics)
        self.assertEqual(self.infer_inst.mask_dir, self.mask_dir)
        self.assertListEqual(self.infer_inst.metrics_orientations, ['xyz'])
        self.assertIsNone(self.infer_inst.crop_shape)
        self.assertEqual(self.infer_inst.z_dim, 2)
        self.assertEqual(self.infer_inst.tile_option, 'tile_xyz')
        self.assertListEqual(self.infer_inst.num_overlap, [1, 1, 1])

    def test_assign_3d_inference(self):
        # Test other settings
        self.infer_inst.params_3d = {
            'num_slices': 5,
            'num_overlap': 1,
            'overlap_operation': 'mean',
        }
        self.infer_inst._assign_3d_inference()
        self.assertEqual(self.infer_inst.z_dim, 2)
        self.assertEqual(self.infer_inst.tile_option, 'tile_z')
        self.assertEqual(self.infer_inst.num_overlap, 1)

    @nose.tools.raises(AssertionError)
    def test_assign_3d_inference_few_slices(self):
        # Test other settings
        self.infer_inst.params_3d = {
            'num_slices': 3,
            'num_overlap': 1,
            'overlap_operation': 'mean',
        }
        self.infer_inst._assign_3d_inference()

    @nose.tools.raises(AssertionError)
    def test_assign_3d_inference_not_3d(self):
        self.infer_inst.params_3d = {
            'num_slices': 5,
            'num_overlap': 1,
            'overlap_operation': 'mean',
        }
        self.infer_inst.config['network']['class'] = 'UNet2D'
        self.infer_inst._assign_3d_inference()

    def test_assign_3d_inference_on_center(self):
        self.infer_inst.params_3d = {
            'inf_shape': [5, 5, 5],
            'num_overlap': 1,
            'overlap_operation': 'mean',
        }
        self.infer_inst._assign_3d_inference()
        self.assertEqual(self.infer_inst.tile_option, 'infer_on_center')
        self.assertEqual(self.infer_inst.num_overlap, 0)

    def test_get_sub_block_z(self):
        # 3D image for prediction should have channel and batch dim
        im = np.zeros((1, 2, 8, 10, 10), dtype=np.float64)
        block = self.infer_inst._get_sub_block_z(
            input_image=im,
            start_z_idx=2,
            end_z_idx=5,
        )
        self.assertTupleEqual(block.shape, (1, 2, 3, 10, 10))

    def test_get_sub_block_z_channels_last(self):
        self.infer_inst.data_format = 'channels_last'
        im = np.zeros((1, 8, 10, 10, 2), dtype=np.float64)
        block = self.infer_inst._get_sub_block_z(
            input_image=im,
            start_z_idx=2,
            end_z_idx=5,
        )
        self.assertTupleEqual(block.shape, (1, 3, 10, 10, 2))

    def test_get_sub_block_z_xyz(self):
        self.infer_inst.image_format = 'xyz'
        im = np.zeros((1, 2, 10, 10, 8), dtype=np.float64)
        block = self.infer_inst._get_sub_block_z(
            input_image=im,
            start_z_idx=2,
            end_z_idx=5,
        )
        self.assertTupleEqual(block.shape, (1, 2, 10, 10, 3))

    def test_get_sub_block_z_xyz_channels_last(self):
        self.infer_inst.image_format = 'xyz'
        self.infer_inst.data_format = 'channels_last'
        im = np.zeros((1, 10, 10, 8, 2), dtype=np.float64)
        block = self.infer_inst._get_sub_block_z(
            input_image=im,
            start_z_idx=2,
            end_z_idx=5,
        )
        self.assertTupleEqual(block.shape, (1, 10, 10, 3, 2))

    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_predict_sub_block_z(self, mock_predict):
        mock_predict.return_value = np.zeros((1, 1, 5, 10, 10), dtype=np.float64)
        self.infer_inst.params_3d = {
            'num_slices': 5,
            'num_overlap': 1,
            'overlap_operation': 'mean',
        }
        self.infer_inst.num_overlap = 1
        im = np.zeros((1, 1, 8, 10, 10), dtype=np.float64)
        pred_ims, start_end_idx = self.infer_inst._predict_sub_block_z(
            input_image=im,
        )
        # z = 5 slices and 1 overlap, two prediction of im can fit
        self.assertEqual(len(pred_ims), 2)
        self.assertEqual(len(start_end_idx), 2)
        # Z locations (out of 0-8)
        self.assertTupleEqual(start_end_idx[0], (0, 5))
        self.assertTupleEqual(start_end_idx[1], (3, 8))

    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_predict_sub_block_xyz(self, mock_predict):
        mock_predict.return_value = np.zeros((1, 1, 5, 5, 5), dtype=np.float64)
        self.infer_inst.num_overlap = 1
        im = np.zeros((1, 1, 8, 10, 10), dtype=np.float64)
        pred_ims = self.infer_inst._predict_sub_block_xyz(
            input_image=im,
            crop_indices=[(0, 5, 0, 5, 0, 5), (3, 8, 0, 5, 0, 5)],
        )
        self.assertEqual(len(pred_ims), 2)

    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_predict_sub_block_xyz_channels_last(self, mock_predict):
        mock_predict.return_value = np.zeros((1, 1, 5, 5, 5), dtype=np.float64)
        self.infer_inst.num_overlap = 1
        im = np.zeros((1, 8, 10, 10, 1), dtype=np.float64)
        pred_ims = self.infer_inst._predict_sub_block_xyz(
            input_image=im,
            crop_indices=[(3, 8, 0, 5, 0, 5)],
        )
        self.assertEqual(len(pred_ims), 1)

    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_predict_3d(self, mock_predict):
        mock_predict.return_value = np.zeros((1, 1, 5, 5, 5), dtype=np.float64)
        # Predict row 0 from inference dataset iterator
        pred_im, target_im, mask_im = self.infer_inst.predict_3d([0])
        self.assertTupleEqual(pred_im.shape, (8, 8, 8))
        self.assertEqual(pred_im.dtype, np.float64)
        self.assertTupleEqual(target_im.shape, (8, 8, 8))
        self.assertEqual(target_im.dtype, np.float64)
        self.assertTupleEqual(mask_im.shape, (8, 8, 8))
        self.assertEqual(mask_im.dtype, np.uint8)
        # Read saved prediction, z=0 target channel=2
        pred_name = os.path.join(
            self.model_dir,
            'predictions/im_c002_z000_t002_p003.npy',
        )
        im_pred = np.load(pred_name)
        self.assertEqual(im_pred.dtype, np.float64)
        self.assertTupleEqual(im_pred.shape, (8, 8, 8))

    @patch('micro_dl.inference.model_inference.predict_large_image')
    def test_run_prediction(self, mock_predict):
        mock_predict.return_value = np.zeros((1, 1, 5, 5, 5), dtype=np.float64)
        # Run prediction. Should create a metrics_xy.csv in pred dir
        self.infer_inst.run_prediction()
        metrics = pd.read_csv(os.path.join(self.model_dir, 'predictions/metrics_xyz.csv'))
        self.assertTupleEqual(metrics.shape, (2, 4))
        # MSE should be 0.
        self.assertEqual(metrics.mse.mean(), 0.0)
        self.assertEqual(metrics.mse_masked.mean(), 0.0)
        # Mask volume fraction should be 0.5
        self.assertEqual(metrics.vol_frac.mean(), 0.5)
        # # There should be four rows, one per test index pos=3,4
        self.assertEqual(metrics.pred_name[0], 'im_c002_z000_t002_p003')
        self.assertEqual(metrics.pred_name[1], 'im_c002_z000_t002_p004')
        # # There should be 2 predictions saved in pred dir, target c=2
        for p in [3, 4]:
            pred_name = os.path.join(
                self.model_dir,
                'predictions/im_c002_z000_t002_p00{}.npy'.format(p),
            )
            im_pred = np.load(pred_name)
            self.assertEqual(im_pred.dtype, np.float64)
            self.assertTupleEqual(im_pred.shape, (8, 8, 8))
