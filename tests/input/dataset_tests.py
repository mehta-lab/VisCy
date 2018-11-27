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
        Set up a directory for tiling with flatfield, no mask
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.input_fnames = pd.Series(['in1.npy',
                                  'in2.npy',
                                  'in3.npy',
                                  'in4.npy'])
        self.target_fnames = pd.Series(['out1.npy',
                                   'out2.npy',
                                   'out3.npy',
                                   'out4.npy'])
        self.batch_size = 2
        # Normally, tiles would have the same shape in x, y but this helps
        # us test augmentations
        self.im = np.zeros((5, 7, 3))
        self.im[:, :5, 0] = np.diag([1, 2, 3, 4, 5])
        self.im[:4, :4, 1] = 1
        # BaseDataSet works for either data format (channels_first or last)
        # To prove it, targets are channels first
        # This wouldn't work for a model, so it assumes you have matched your
        # preprocessing and traning configs with the right data format!
        self.im_target = np.zeros((3, 5, 7))
        self.im_target[0, :4, :4] = 1
        for i, (in_name, out_name) in enumerate(zip(self.input_fnames,
                                                    self.target_fnames)):
            np.save(os.path.join(self.temp_path, in_name), self.im + i)
            np.save(os.path.join(self.temp_path, out_name), self.im_target + i)
        dataset_config = {
            'augmentations': True,
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
            data_format='channels_last',
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Test image tiler on frames temporary dir
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
        nose.tools.assert_equal(self.data_inst.batch_size, self.batch_size)
        nose.tools.assert_true(self.data_inst.shuffle)
        nose.tools.assert_equal(
            self.data_inst.num_samples,
            len(self.input_fnames),
        )
        nose.tools.assert_true(self.data_inst.augmentations)
        nose.tools.assert_equal(self.data_inst.model_task, 'regression')
        nose.tools.assert_equal(self.data_inst.random_seed, 42)
        nose.tools.assert_false(self.data_inst.normalize)

    def test__len__(self):
        nbr_batches = self.data_inst.__len__()
        expected_batches = len(self.input_fnames) / self.batch_size
        nose.tools.assert_equal(nbr_batches, expected_batches)

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

    def test_augment_image_lr_channels_first(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.data_format = 'channels_first'
        trans_im = self.data_inst._augment_image(im_test, 1)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.fliplr(im_test[i, ...]),
            )

    def test_augment_image_ud_channels_first(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.data_format = 'channels_first'
        trans_im = self.data_inst._augment_image(im_test, 2)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.flipud(im_test[i, ...]),
            )

    def test_augment_image_rot90_channels_first(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.data_format = 'channels_first'
        trans_im = self.data_inst._augment_image(im_test, 3)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.rot90(im_test[i, ...], k=1),
            )

    def test_augment_image_rot180_channels_first(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.data_format = 'channels_first'
        trans_im = self.data_inst._augment_image(im_test, 4)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.rot90(im_test[i, ...], k=2),
            )

    def test_augment_image_rot270_channels_first(self):
        im_test = np.transpose(self.im, [2, 0, 1])
        self.data_inst.data_format = 'channels_first'
        trans_im = self.data_inst._augment_image(im_test, 5)
        for i in range(2):
            np.testing.assert_array_equal(
                trans_im[i, ...],
                np.rot90(im_test[i, ...], k=3),
            )

    def test_get_volume(self):
        image_volume = self.data_inst._get_volume(self.input_fnames, normalize=False)
        # There are 4 input images of shape (5, 7, 3)
        self.assertTupleEqual(image_volume.shape, (4, 5, 7, 3))
        # Check image content (normalize is false)
        for i in range(4):
            im_test = np.squeeze(image_volume[i, ...])
            np.testing.assert_array_equal(im_test, self.im + i)

    def test__getitem__(self):
        im_in, im_target = self.data_inst.__getitem__(0)
        # Batch size is 2, input images of shape (5, 7, 3)
        # stack adds singleton dimension
        self.assertTupleEqual(im_in.shape, (2, 1, 5, 7, 3))
        self.assertTupleEqual(im_target.shape, (2, 1, 3, 5, 7))
        # With a fixed random seed, augmentations and shuffles are the same
        augmentations = [2, 4]
        shuf_ids = [1, 3]
        for i in range(2):
            # only compare self.im
            im_test = np.squeeze(im_in[i, ...])
            print(i, im_test[0, ...])
            im_expected = self.data_inst._augment_image(
                self.im + shuf_ids[i],
                augmentations[i],
            )
            print(im_expected[0, ...])
            np.testing.assert_array_equal(im_test, im_expected)

    def test__getitem__normalized(self):
        self.data_inst.normalize = True
        im_in, im_target = self.data_inst.__getitem__(0)
        # Batch size is 2, input images of shape (5, 7, 3)
        # stack adds singleton dimension
        self.assertTupleEqual(im_in.shape, (2, 1, 5, 7, 3))
        self.assertTupleEqual(im_target.shape, (2, 1, 3, 5, 7))
        # Just test normalization this time
        for i in range(2):
            im_test = np.squeeze(im_in[i, ...])
            nose.tools.assert_almost_equal(im_test.mean(), 0, 2)
            nose.tools.assert_almost_equal(im_test.std(), 1, 2)
            im_test = np.squeeze(im_target[i, ...])
            nose.tools.assert_almost_equal(im_test.mean(), 0, 2)
            nose.tools.assert_almost_equal(im_test.std(), 1, 2)

    def test_on_epoch_end(self):
        row_idx = self.data_inst.row_idx
        # Random seed 42 results in same order as before...
        # Random seed 1 swaps all axes
        np.random.seed(1)
        self.data_inst.on_epoch_end()
        new_idx = self.data_inst.row_idx
        self.assertFalse((row_idx == new_idx).all())
