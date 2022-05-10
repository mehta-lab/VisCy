import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.input.dataset_with_mask as dataset_mask


class TestDataSetWithMask(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with input and target tiles, with masks
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.input_fnames = pd.Series([
            'in1.npy',
            'in2.npy',
            'in3.npy',
            'in4.npy',
        ])
        self.target_fnames = pd.Series([
            'out1.npy',
            'out2.npy',
            'out3.npy',
            'out4.npy',
        ])
        self.mask_fnames = pd.Series([
            'mask1.npy',
            'mask2.npy',
            'mask3.npy',
            'mask4.npy',
        ])
        self.batch_size = 2
        # Create images
        self.im = np.zeros((1, 5, 7), dtype=np.float32)
        self.im[0, :, :5] = np.diag([1, 2, 3, 4, 5])
        self.im_target = np.zeros((1, 5, 7), dtype=np.bool)
        self.im_target[0, :4, :4] = 1
        self.im_mask = np.zeros((1, 5, 7), dtype=np.uint8)
        self.im_mask[0, :2, :4] = 1
        self.im_mask[0, 3:, :4] = 2
        # Batch size is 2, input images of shape (1, 5, 7)
        self.batch_shape = (2, 1, 5, 7)
        for i, (in_name, out_name, mask_name) in enumerate(
                zip(self.input_fnames,
                    self.target_fnames,
                    self.mask_fnames)):
            np.save(os.path.join(self.temp_path, in_name), self.im + i)
            np.save(os.path.join(self.temp_path, out_name), self.im_target + i)
            np.save(os.path.join(self.temp_path, mask_name), self.im_mask)
        dataset_config = {
            'augmentations': {
                'noise_std': 0,
            },
            'random_seed': 42,
            'normalize': False,
            'squeeze': True,
            'model_task': 'segmentation',
            'label_weights': [1, 2],
        }
        # Instantiate class
        self.data_inst = dataset_mask.DataSetWithMask(
            tile_dir=self.temp_path,
            input_fnames=self.input_fnames,
            target_fnames=self.target_fnames,
            mask_fnames=self.mask_fnames,
            dataset_config=dataset_config,
            batch_size=self.batch_size,
            image_format='zyx',
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Test dataset init assignments
        """
        nose.tools.assert_equal(self.data_inst.tile_dir, self.temp_path)
        self.assertListEqual(
            self.input_fnames.tolist(),
            self.data_inst.input_fnames.tolist(),
        )
        self.assertListEqual(
            self.target_fnames.tolist(),
            self.data_inst.target_fnames.tolist(),
        )
        self.assertListEqual(
            self.mask_fnames.tolist(),
            self.data_inst.mask_fnames.tolist(),
        )
        self.assertEqual(self.data_inst.batch_size, self.batch_size)
        self.assertEqual(self.data_inst.num_samples, 4)
        self.assertEqual(self.data_inst.num_epoch_samples, 4)
        self.assertTrue(self.data_inst.shuffle)
        self.assertEqual(self.data_inst.num_samples, len(self.input_fnames))
        self.assertTrue(self.data_inst.augmentations)
        self.assertEqual(self.data_inst.model_task, 'segmentation')
        self.assertEqual(self.data_inst.random_seed, 42)
        self.assertFalse(self.data_inst.normalize)
        self.assertListEqual(self.data_inst.label_weights, [1, 2])

    def test__getitem__(self):
        im_in, im_target = self.data_inst.__getitem__(0)
        self.assertTupleEqual(im_in.shape, self.batch_shape)
        # Mask is added to target
        self.assertTupleEqual(im_target.shape, (2, 2, 5, 7))
        # With a fixed random seed, augmentations and shuffles are the same
        augmentations = [2, 4]
        shuf_ids = [1, 3]
        for i in range(2):
            # only compare self.im
            im_test = np.squeeze(im_in[i, ...])
            im_expected = np.squeeze(self.data_inst._augment_image(
                self.im + shuf_ids[i],
                augmentations[i],
            ))
            np.testing.assert_array_equal(im_test, im_expected)

    def test__getitem__batch3(self):
        self.data_inst.batch_size = 3
        im_in, im_target = self.data_inst.__getitem__(1)
        # Batch size =3, total samples 4, so item 1 will only have one tile
        self.assertTupleEqual(im_in.shape, (1, 1, 5, 7))
        # Mask is added to target
        self.assertTupleEqual(im_target.shape, (1, 2, 5, 7))
