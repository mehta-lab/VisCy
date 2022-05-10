import cv2
import itertools
import nose.tools
import numpy as np
import os
from testfixtures import TempDirectory
import unittest

import micro_dl.preprocessing.estimate_flat_field as flat_field
import micro_dl.utils.aux_utils as aux_utils


class TestEstimateFlatField(unittest.TestCase):

    def setUp(self):
        """
        Set up directories with input images for flatfield correction
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
        self.pos_ids = [7, 8]
        self.channel_ids = [2, 3]
        self.slice_ids = [0, 1, 2]
        self.im = 1500 * np.ones((20, 15), dtype=np.uint16)
        self.im[10:, 10:] = 3000

        for c in self.channel_ids:
            for p in self.pos_ids:
                for z in self.slice_ids:
                    im_name = aux_utils.get_im_name(
                        channel_idx=c,
                        slice_idx=z,
                        time_idx=self.time_idx,
                        pos_idx=p,
                    )
                    im = self.im + c * 100
                    cv2.imwrite(os.path.join(self.temp_path, im_name),
                                im)
                    meta_row = aux_utils.parse_idx_from_name(im_name)
                    meta_row['mean'] = np.nanmean(im)
                    meta_row['std'] = np.nanstd(im)
                    self.frames_meta = self.frames_meta.append(
                        meta_row,
                        ignore_index=True,
                    )
        # Write metadata
        self.frames_meta.to_csv(
            os.path.join(self.image_dir, self.meta_name),
            sep=',',
        )
        self.flat_field_dir = os.path.join(
            self.output_dir,
            'flat_field_images',
        )
        # Create flatfield class instance
        self.flatfield_inst = flat_field.FlatFieldEstimator2D(
            input_dir=self.image_dir,
            output_dir=self.output_dir,
            channel_ids=self.channel_ids,
            slice_ids=self.slice_ids,
            block_size=5,
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Check that an instance was created correctly
        """
        self.assertEqual(self.flatfield_inst.input_dir, self.image_dir)
        self.assertEqual(self.flatfield_inst.output_dir, self.output_dir)
        self.assertEqual(
            self.flatfield_inst.flat_field_dir,
            self.flat_field_dir,
        )
        self.assertListEqual(self.flatfield_inst.slice_ids, self.slice_ids)
        self.assertListEqual(self.flatfield_inst.channels_ids, self.channel_ids)
        self.assertEqual(self.flatfield_inst.block_size, 5)

    def test_get_flat_field_dir(self):
        ff_dir = self.flatfield_inst.get_flat_field_dir()
        self.assertEqual(self.flat_field_dir, ff_dir)

    def test_estimate_flat_field(self):
        self.flatfield_inst.estimate_flat_field()
        flatfields = os.listdir(self.flat_field_dir)
        # Make sure list is sorted
        flatfields.sort()
        for i, c in enumerate(self.channel_ids):
            file_name = 'flat-field_channel-{}.npy'.format(c)
            self.assertEqual(flatfields[i], file_name)
            ff = np.load(os.path.join(self.flat_field_dir, file_name))
            self.assertLessEqual(ff.max(), 5.)
            self.assertLessEqual(0.1, ff.min())
            self.assertTupleEqual(ff.shape, self.im.shape)

    def test_sample_block_medians(self):
        coords, vals = self.flatfield_inst.sample_block_medians(
            im=self.im,
        )
        # Image shape is 20 x 15, so center coordinates will be:
        xc = [2, 7, 12, 17]
        yc = [2, 7, 12]
        coord_iterator = itertools.product(yc, xc)
        # Check that generated center coords are correct
        for i, (y, x) in enumerate(coord_iterator):
            self.assertEqual(x, coords[i, 0])
            self.assertEqual(y, coords[i, 1])
        # Check that values are correct
        # all should be 1500 except the last 2
        expected_vals = [1500] * 10 + [3000] * 2
        self.assertListEqual(list(vals), expected_vals)

    @nose.tools.raises(AssertionError)
    def test_sample_wrong_size_block_medians(self):
        self.flatfield_inst.block_size = 15
        coords, vals = self.flatfield_inst.sample_block_medians(
            im=self.im,
        )

    def test_get_flatfield(self):
        test_im = np.zeros((30, 20), np.uint8) + 100
        test_im[:, 10:] = 200
        flatfield = self.flatfield_inst.get_flatfield(test_im)
        self.assertTupleEqual(flatfield.shape, (30, 20))
        self.assertLessEqual(flatfield.max(), 2)
        self.assertLessEqual(0.1, flatfield.min())

    def test_get_flatfield_no_norm(self):
        test_im = np.zeros((30, 20), np.uint8) + 100
        test_im[:, 10:] = 200
        flatfield = self.flatfield_inst.get_flatfield(
            im=test_im,
            normalize=False,
        )
        self.assertTupleEqual(flatfield.shape, (30, 20))
        self.assertLessEqual(flatfield.max(), 250)
        self.assertLessEqual(50, flatfield.min())

    @nose.tools.raises(AssertionError)
    def test_get_flatfield_small_im(self):
        test_im = np.zeros((10, 15), np.uint8) + 100
        flatfield = self.flatfield_inst.get_flatfield(test_im)

    @nose.tools.raises(ValueError)
    def test_get_flatfield_neg_values(self):
        test_im = np.zeros((30, 20), np.int)
        test_im[15:, 5:] = -100
        flatfield = self.flatfield_inst.get_flatfield(test_im)
