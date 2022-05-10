import cv2
import numpy as np
import os
from testfixtures import TempDirectory
import unittest

import micro_dl.input.inference_dataset as inference_dataset
import micro_dl.utils.aux_utils as aux_utils


class TestInferenceDataSet(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with images
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.tempdir.makedir('image_dir')
        self.tempdir.makedir('model_dir')
        self.tempdir.makedir('mask_dir')
        self.image_dir = os.path.join(self.temp_path, 'image_dir')
        self.model_dir = os.path.join(self.temp_path, 'model_dir')
        self.mask_dir = os.path.join(self.temp_path, 'mask_dir')
        # Create a temp image dir
        im = np.zeros((10, 16), dtype=np.uint8)
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
                cv2.imwrite(os.path.join(self.image_dir, im_name), im + c * 10)
                meta_row = aux_utils.parse_idx_from_name(
                    im_name)
                meta_row['zscore_median'] = 10
                meta_row['zscore_iqr'] = 2
                self.frames_meta = self.frames_meta.append(
                    meta_row,
                    ignore_index=True,
                )
        # Write frames meta to image dir too
        self.frames_meta.to_csv(os.path.join(self.image_dir, 'frames_meta.csv'))
        # Save masks and mask meta
        self.mask_meta = aux_utils.make_dataframe()
        self.mask_channel = 50
        for p in range(5):
            im_name = aux_utils.get_im_name(
                time_idx=2,
                channel_idx=self.mask_channel,
                slice_idx=3,
                pos_idx=p,
            )
            cv2.imwrite(os.path.join(self.mask_dir, im_name), im + 1)
            self.mask_meta = self.mask_meta.append(
                aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                ignore_index=True,
            )
        # Write frames meta to image dir too
        self.mask_meta.to_csv(os.path.join(self.mask_dir, 'frames_meta.csv'))
        # Select inference split of dataset
        self.split_col_ids = ('pos_idx', [1, 3])
        # Make configs with fields necessary for inference dataset
        self.inference_config = {
            'model_dir': 'model_dir',
            'model_fname': 'dummy_weights.hdf5',
            'image_dir': 'image_dir',
            'data_split': 'test',
            'images': {
                'image_format': 'zyx',
                'image_ext': '.npy',
            },
        }
        dataset_config = {
            'input_channels': [2],
            'target_channels': [self.mask_channel],
            'model_task': 'segmentation',
        }
        self.network_config = {
            'class': 'UNet2D',
            'depth': 1,
            'data_format': 'channels_first',
        }
        self.preprocess_config = {
            'normalize_im': 'dataset'
        }
        # Instantiate class
        self.data_inst = inference_dataset.InferenceDataSet(
            image_dir=self.image_dir,
            inference_config=self.inference_config,
            dataset_config=dataset_config,
            network_config=self.network_config,
            split_col_ids=self.split_col_ids,
            preprocess_config=self.preprocess_config,
            mask_dir=self.mask_dir,
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
        self.assertEqual(self.data_inst.image_dir, self.image_dir)
        self.assertEqual(self.data_inst.target_dir, self.mask_dir)
        self.assertIsNone(self.data_inst.flat_field_dir)
        self.assertEqual(self.data_inst.image_format, 'zyx')
        self.assertEqual(self.data_inst.model_task, 'segmentation')
        self.assertEqual(self.data_inst.depth, 1)
        self.assertTrue(self.data_inst.squeeze)
        self.assertFalse(self.data_inst.im_3d)
        self.assertEqual(self.data_inst.data_format, 'channels_first')
        self.assertListEqual(self.data_inst.input_channels, [2])
        self.assertListEqual(self.data_inst.target_channels, [self.mask_channel])
        # Two inference samples (pos idx 1 and 3)
        self.assertEqual(self.data_inst.num_samples, 2)
        self.assertListEqual(
            self.data_inst.frames_meta.pos_idx.unique().tolist(),
            [1, 3],
        )
        # Image channels = 0, 1, 2 and target channel = 50
        self.assertListEqual(
            self.data_inst.frames_meta.channel_idx.unique().tolist(),
            [0, 1, 2, 50])

    def test_adjust_slice_indices(self):
        # First create new frames meta with more slices
        temp_meta = aux_utils.make_dataframe()
        for s in range(10):
            im_name = aux_utils.get_im_name(
                time_idx=2,
                channel_idx=4,
                slice_idx=s,
                pos_idx=6,
            )
            temp_meta = temp_meta.append(
                aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                ignore_index=True,
            )
        self.data_inst.inf_frames_meta = temp_meta
        self.data_inst.depth = 5
        # This should remove first and last two slices
        self.data_inst.adjust_slice_indices()
        # Original slice ids are 0-9 so after removing margins should be 2-7
        self.assertListEqual(
            self.data_inst.inf_frames_meta.slice_idx.unique().tolist(),
            [2, 3, 4, 5, 6, 7])

    def test_get_iteration_meta(self):
        iteration_meta = self.data_inst.get_iteration_meta()
        # This contains metadata for first target channel only
        self.assertTupleEqual(iteration_meta.shape, (2, 9))
        self.assertListEqual(
            iteration_meta.channel_idx.unique().tolist(),
            [self.mask_channel],
        )
        # Contains only test indices pos 1 and 3
        self.assertListEqual(
            iteration_meta.pos_idx.unique().tolist(),
            [1, 3],
        )

    def test__len__(self):
        num_samples = self.data_inst.__len__()
        self.assertEqual(num_samples, 2)

    def test_get_image(self):
        meta_row = dict.fromkeys(aux_utils.DF_NAMES)
        meta_row['channel_idx'] = 2
        meta_row['time_idx'] = self.time_idx
        meta_row['slice_idx'] = self.slice_idx
        meta_row['pos_idx'] = 3
        im_stack = self.data_inst._get_image(
            input_dir=self.image_dir,
            cur_row=meta_row,
            channel_ids=[2],
            depth=1,
            normalize_im=self.preprocess_config['normalize_im']
        )
        # Image shapes are cropped to nearest factor of two, channels first
        self.assertTupleEqual(im_stack.shape, (1, 8, 16))
        # Channel 2 has constant values of 20
        self.assertEqual(im_stack.max(), 5)
        self.assertEqual(im_stack.min(), 5)

    def test__getitem__(self):
        # There are 2 test indices (pos 1 and 3)
        input_stack, target_stack = self.data_inst.__getitem__(1)
        # Cropped to factor of 2, add batch dim
        self.assertTupleEqual(input_stack.shape, (1, 1, 8, 16))
        self.assertTupleEqual(target_stack.shape, (1, 1, 8, 16))
        # input stack should be normalized, not target
        self.assertEqual(input_stack.max(), 5)
        self.assertEqual(target_stack.max(), 1)
        self.assertEqual(input_stack.dtype, np.float32)
        self.assertEqual(target_stack.dtype, np.float32)

    def test__getitem__regression(self):
        dataset_config = {
            'input_channels': [0, 1],
            'target_channels': [2],
            'model_task': 'regression',
        }
        # Instantiate class
        data_inst = inference_dataset.InferenceDataSet(
            image_dir=self.image_dir,
            inference_config=self.inference_config,
            dataset_config=dataset_config,
            network_config=self.network_config,
            preprocess_config=self.preprocess_config,
            split_col_ids=self.split_col_ids,
        )
        # There are 2 test indices (pos 1 and 3)
        input_stack, target_stack = data_inst.__getitem__(0)
        # Cropped to factor of 2, add batch dim
        self.assertTupleEqual(input_stack.shape, (1, 2, 8, 16))
        self.assertTupleEqual(target_stack.shape, (1, 1, 8, 16))
        # input stack should be normalized, not target
        self.assertEqual(input_stack.max(), 0)
        self.assertEqual(target_stack.max(), 20)
        self.assertEqual(input_stack.dtype, np.float32)
        self.assertEqual(target_stack.dtype, np.float32)


class TestInferenceDataSet2p5D(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with images
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.tempdir.makedir('image_dir')
        self.tempdir.makedir('model_dir')
        self.tempdir.makedir('mask_dir')
        self.image_dir = os.path.join(self.temp_path, 'image_dir')
        self.model_dir = os.path.join(self.temp_path, 'model_dir')
        self.mask_dir = os.path.join(self.temp_path, 'mask_dir')
        # Create a temp image dir
        im = np.zeros((10, 16), dtype=np.uint8)
        self.frames_meta = aux_utils.make_dataframe()
        self.time_idx = 2
        for p in range(5):
            for z in range(4):
                for c in range(3):
                    im_name = aux_utils.get_im_name(
                        time_idx=self.time_idx,
                        channel_idx=c,
                        slice_idx=z,
                        pos_idx=p,
                    )
                    cv2.imwrite(os.path.join(self.image_dir, im_name), im + c * 10)
                    meta_row = aux_utils.parse_idx_from_name(
                        im_name)
                    meta_row['zscore_median'] = 10
                    meta_row['zscore_iqr'] = 2
                    self.frames_meta = self.frames_meta.append(
                        meta_row,
                        ignore_index=True,
                    )
        # Write frames meta to image dir too
        self.frames_meta.to_csv(os.path.join(self.image_dir, 'frames_meta.csv'))
        # Save masks and mask meta
        self.mask_meta = aux_utils.make_dataframe()
        self.mask_channel = 50
        for p in range(5):
            for z in range(4):
                im_name = aux_utils.get_im_name(
                    time_idx=2,
                    channel_idx=self.mask_channel,
                    slice_idx=z,
                    pos_idx=p,
                )
                cv2.imwrite(os.path.join(self.mask_dir, im_name), im + 1)
                self.mask_meta = self.mask_meta.append(
                    aux_utils.parse_idx_from_name(im_name, aux_utils.DF_NAMES),
                    ignore_index=True,
            )
        # Write frames meta to image dir too
        self.mask_meta.to_csv(os.path.join(self.mask_dir, 'frames_meta.csv'))
        # Select inference split of dataset
        self.split_col_ids = ('pos_idx', [1, 3])
        # Make configs with fields necessary for inference dataset
        self.inference_config = {
            'model_dir': 'model_dir',
            'model_fname': 'dummy_weights.hdf5',
            'image_dir': 'image_dir',
            'data_split': 'test',
            'images': {
                'image_format': 'zyx',
                'image_ext': '.npy',
            },
        }
        dataset_config = {
            'input_channels': [2],
            'target_channels': [self.mask_channel],
            'model_task': 'segmentation',
        }
        self.network_config = {
            'class': 'UNetStackTo2D',
            'depth': 3,
            'data_format': 'channels_first',
        }
        self.preprocess_config = {
            'normalize_im': 'dataset'
        }
        # Instantiate class
        self.data_inst = inference_dataset.InferenceDataSet(
            image_dir=self.image_dir,
            inference_config=self.inference_config,
            dataset_config=dataset_config,
            network_config=self.network_config,
            preprocess_config=self.preprocess_config,
            split_col_ids=self.split_col_ids,
            mask_dir=self.mask_dir,
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
        self.assertEqual(self.data_inst.image_dir, self.image_dir)
        self.assertEqual(self.data_inst.target_dir, self.mask_dir)
        self.assertIsNone(self.data_inst.flat_field_dir)
        self.assertEqual(self.data_inst.image_format, 'zyx')
        self.assertEqual(self.data_inst.model_task, 'segmentation')
        self.assertEqual(self.data_inst.depth, 3)
        self.assertFalse(self.data_inst.squeeze)
        self.assertFalse(self.data_inst.im_3d)
        self.assertEqual(self.data_inst.data_format, 'channels_first')
        self.assertListEqual(self.data_inst.input_channels, [2])
        self.assertListEqual(self.data_inst.target_channels, [self.mask_channel])
        # Two inference samples (pos idx 1 and 3), two slices (1, 2) each = 4
        self.assertEqual(self.data_inst.num_samples, 4)
        self.assertListEqual(
            self.data_inst.frames_meta.pos_idx.unique().tolist(),
            [1, 3],
        )
        # Image channels = 0, 1, 2 and target channel = 50
        self.assertListEqual(
            self.data_inst.frames_meta.channel_idx.unique().tolist(),
            [0, 1, 2, 50])

    def test_get_iteration_meta(self):
        iteration_meta = self.data_inst.get_iteration_meta()
        # This contains metadata for first target channel only z=1,2, p=1,3
        self.assertTupleEqual(iteration_meta.shape, (4, 9))
        self.assertListEqual(
            iteration_meta.channel_idx.unique().tolist(),
            [self.mask_channel],
        )
        # Contains only test indices pos 1 and 3
        self.assertListEqual(
            iteration_meta.pos_idx.unique().tolist(),
            [1, 3],
        )
        # Contains two slices, 1 and 2
        self.assertListEqual(
            iteration_meta.slice_idx.unique().tolist(),
            [1, 2],
        )

    def test__len__(self):
        num_samples = self.data_inst.__len__()
        self.assertEqual(num_samples, 4)

    def test_get_image(self):
        meta_row = dict.fromkeys(aux_utils.DF_NAMES)
        meta_row['channel_idx'] = 2
        meta_row['time_idx'] = self.time_idx
        meta_row['slice_idx'] = 1
        meta_row['pos_idx'] = 3
        im_stack = self.data_inst._get_image(
            input_dir=self.image_dir,
            cur_row=meta_row,
            channel_ids=[2],
            depth=3,
            normalize_im=self.preprocess_config['normalize_im'],
        )
        # Image shapes are cropped to nearest factor of two, channels first
        self.assertTupleEqual(im_stack.shape, (1, 3, 8, 16))
        # Channel 2 has constant values of 20
        self.assertEqual(im_stack.max(), 5)
        self.assertEqual(im_stack.min(), 5)

    def test__getitem__(self):
        # There are 2 test indices (pos 1 and 3)
        input_stack, target_stack = self.data_inst.__getitem__(1)
        # Cropped to factor of 2, add batch dim
        self.assertTupleEqual(input_stack.shape, (1, 1, 3, 8, 16))
        self.assertTupleEqual(target_stack.shape, (1, 1, 1, 8, 16))
        # input stack should be normalized, not target
        self.assertEqual(input_stack.max(), 5)
        self.assertEqual(target_stack.max(), 1)
        self.assertEqual(input_stack.dtype, np.float32)
        self.assertEqual(target_stack.dtype, np.float32)

    def test__getitem__regression(self):
        dataset_config = {
            'input_channels': [0, 1],
            'target_channels': [2],
            'model_task': 'regression',
        }
        # Instantiate class
        data_inst = inference_dataset.InferenceDataSet(
            image_dir=self.image_dir,
            inference_config=self.inference_config,
            dataset_config=dataset_config,
            network_config=self.network_config,
            preprocess_config=self.preprocess_config,
            split_col_ids=self.split_col_ids,
        )
        # There are 2 test indices (pos 1 and 3)
        input_stack, target_stack = data_inst.__getitem__(0)
        # Cropped to factor of 2, add batch dim
        self.assertTupleEqual(input_stack.shape, (1, 2, 3, 8, 16))
        self.assertTupleEqual(target_stack.shape, (1, 1, 1, 8, 16))
        # input stack should be normalized, not target
        self.assertEqual(input_stack.max(), 0)
        self.assertEqual(target_stack.max(), 20)
        self.assertEqual(input_stack.dtype, np.float32)
        self.assertEqual(target_stack.dtype, np.float32)
