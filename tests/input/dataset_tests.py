import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.input.dataset as dataset


class TestBaseDataSet(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with input and target tiles
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
        self.batch_size = 2
        # Normally, tiles would have the same shape in x, y but this helps
        # us test augmentations
        self.im = np.zeros((5, 7, 3))
        self.im[:, :5, 0] = np.diag([1, 2, 3, 4, 5])
        self.im[:4, :4, 1] = 1

        self.im_target = np.zeros((5, 7, 3))
        self.im_target[:4, :4, 0] = 1
        # Batch size is 2, input images of shape (5, 7, 3)
        # stack adds singleton dimension
        self.batch_shape = (2, 1, 5, 7, 3)
        for i, (in_name, out_name) in enumerate(zip(self.input_fnames,
                                                    self.target_fnames)):
            np.save(os.path.join(self.temp_path, in_name), self.im + i)
            np.save(os.path.join(self.temp_path, out_name), self.im_target + i)
        dataset_config = {
            'augmentations': {
                'noise_std': 0,
            },
            'random_seed': 42,
            'normalize': False,
        }
        # Instantiate class
        self.data_inst = dataset.BaseDataSet(
            tile_dir=self.temp_path,
            input_fnames=self.input_fnames,
            target_fnames=self.target_fnames,
            dataset_config=dataset_config,
            batch_size=self.batch_size,
            image_format='xyz',
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
        self.assertEqual(self.data_inst.batch_size, self.batch_size)
        self.assertEqual(self.data_inst.num_samples, 4)
        self.assertEqual(self.data_inst.num_epoch_samples, 4)
        self.assertTrue(self.data_inst.shuffle)
        self.assertEqual(self.data_inst.num_samples, len(self.input_fnames))
        self.assertTrue(self.data_inst.augmentations)
        self.assertTupleEqual(self.data_inst.zoom_range, (1, 1))
        self.assertEqual(self.data_inst.rotate_range, 0)
        self.assertEqual(self.data_inst.mean_jitter, 0)
        self.assertEqual(self.data_inst.std_jitter, 0)
        self.assertEqual(self.data_inst.noise_std, 0)
        self.assertTupleEqual(self.data_inst.blur_range, (0, 0))
        self.assertEqual(self.data_inst.shear_range, 0)
        self.assertEqual(self.data_inst.model_task, 'regression')
        self.assertEqual(self.data_inst.random_seed, 42)
        self.assertFalse(self.data_inst.normalize)

    def test_init_settings(self):
        dataset_config = {
            'random_seed': 42,
            'normalize': True,
            'model_task': 'segmentation',
            'shuffle': False,
            'train_fraction': .5,
            'squeeze': True,
        }
        # Instantiate class
        data_inst = dataset.BaseDataSet(
            tile_dir=self.temp_path,
            input_fnames=self.input_fnames,
            target_fnames=self.target_fnames,
            dataset_config=dataset_config,
            batch_size=self.batch_size,
            image_format='zyx',
        )
        self.assertEqual(data_inst.tile_dir, self.temp_path)
        self.assertListEqual(
            self.input_fnames.tolist(),
            data_inst.input_fnames.tolist(),
        )
        self.assertListEqual(
            self.target_fnames.tolist(),
            data_inst.target_fnames.tolist(),
        )
        self.assertEqual(data_inst.batch_size, self.batch_size)
        self.assertEqual(data_inst.num_samples, 4)
        self.assertEqual(data_inst.num_epoch_samples, 2)
        # Must shuffle if using a train fraction
        self.assertTrue(data_inst.shuffle)
        self.assertFalse(data_inst.augmentations)
        self.assertEqual(data_inst.model_task, 'segmentation')
        self.assertEqual(data_inst.random_seed, 42)
        self.assertTrue(data_inst.normalize)

    def test__len__(self):
        nbr_batches = self.data_inst.__len__()
        expected_batches = len(self.input_fnames) / self.batch_size
        self.assertEqual(nbr_batches, expected_batches)

    def test_get_steps_per_epoch(self):
        steps = self.data_inst.get_steps_per_epoch()
        self.assertEqual(steps, 2)

    def test_augment_image_asis(self):
        trans_im = self.data_inst._augment_image(self.im, 0)
        np.testing.assert_array_equal(trans_im, self.im)

    def test_augment_image_lr(self):
        trans_im = self.data_inst._augment_image(self.im, 1)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.fliplr(self.im[..., i]),
            )

    def test_augment_image_ud(self):
        trans_im = self.data_inst._augment_image(self.im, 2)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.flipud(self.im[..., i]),
            )

    def test_augment_image_rot90(self):
        trans_im = self.data_inst._augment_image(self.im, 3)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.rot90(self.im[..., i], k=1),
            )

    def test_augment_image_rot180(self):
        trans_im = self.data_inst._augment_image(self.im, 4)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.rot90(self.im[..., i], k=2),
            )

    def test_augment_image_rot270(self):
        trans_im = self.data_inst._augment_image(self.im, 5)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[..., i],
                np.rot90(self.im[..., i], k=3),
            )

    @nose.tools.raises(ValueError)
    def test_augment_image_6(self):
        self.data_inst._augment_image(self.im, 6)

    @nose.tools.raises(ValueError)
    def test_augment_image_m1(self):
        self.data_inst._augment_image(self.im, -1)

    def test_augment_image_lr_zyx(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.image_format = 'zyx'
        trans_im = self.data_inst._augment_image(im_test, 1)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.fliplr(im_test[i, ...]),
            )

    def test_augment_image_ud_zyx(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.image_format = 'zyx'
        trans_im = self.data_inst._augment_image(im_test, 2)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.flipud(im_test[i, ...]),
            )

    def test_augment_image_rot90_channels_first(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.image_format = 'zyx'
        trans_im = self.data_inst._augment_image(im_test, 3)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.rot90(im_test[i, ...], k=1),
            )

    def test_augment_image_rot180_zyx(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.image_format = 'zyx'
        trans_im = self.data_inst._augment_image(im_test, 4)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.rot90(im_test[i, ...], k=2),
            )

    def test_augment_image_rot270_zyx(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.image_format = 'zyx'
        trans_im = self.data_inst._augment_image(im_test, 5)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.rot90(im_test[i, ...], k=3),
            )

    def test_get_volume(self):
        image_volume = self.data_inst._get_volume(
            self.input_fnames.tolist(),
            normalize=False,
        )
        # There are 4 input images of shape (5, 7, 3)
        self.assertTupleEqual(image_volume.shape, (4, 5, 7, 3))
        # Check image content (normalize is false)
        for i in range(4):
            im_test = np.squeeze(image_volume[i, ...])
            np.testing.assert_array_equal(im_test, self.im + i)

    def test__getitem__(self):
        im_in, im_target = self.data_inst.__getitem__(0)
        self.assertTupleEqual(im_in.shape, self.batch_shape)
        self.assertTupleEqual(im_target.shape, self.batch_shape)
        # With a fixed random seed, augmentations and shuffles stay the same
        augmentations = [2, 2]
        shuf_ids = [1, 3]
        for i in range(2):
            # only compare self.im
            im_test = np.squeeze(im_in[i, ...])
            im_expected = self.data_inst._augment_image(
                self.im + shuf_ids[i],
                augmentations[i],
            )
            np.testing.assert_array_equal(im_test, im_expected)

    def test__getitem__normalized(self):
        self.data_inst.normalize = True
        im_in, im_target = self.data_inst.__getitem__(0)
        self.assertTupleEqual(im_in.shape, self.batch_shape)
        self.assertTupleEqual(im_target.shape, self.batch_shape)
        # Just test normalization this time
        for i in range(2):
            im_test = np.squeeze(im_in[i, ...])
            nose.tools.assert_almost_equal(im_test.mean(), 0, 1)
            nose.tools.assert_almost_equal(im_test.std(), 1, 2)
            im_test = np.squeeze(im_target[i, ...])
            nose.tools.assert_almost_equal(im_test.mean(), 0, 2)
            nose.tools.assert_almost_equal(im_test.std(), 1, 2)

    def test_on_epoch_end(self):
        row_idx = self.data_inst.row_idx.copy()
        self.data_inst.on_epoch_end()
        new_idx = self.data_inst.row_idx
        self.assertFalse((row_idx == new_idx).all())
