import cv2
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.meta_utils as meta_utils


class TestMetaUtils(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with some images to resample
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.input_dir = os.path.join(self.temp_path, 'input_dir')
        self.tempdir.makedir('input_dir')
        self.ff_dir = os.path.join(self.temp_path, 'ff_dir')
        self.tempdir.makedir('ff_dir')
        self.mask_dir = os.path.join(self.temp_path, 'mask_dir')
        self.tempdir.makedir('mask_dir')
        self.slice_idx = 1
        self.time_idx = 2
        self.im = np.zeros((10, 20), np.uint8) + 5
        self.mask = np.zeros((10, 20), np.uint8)
        self.mask[:, 10:] = 1
        ff_im = np.ones((10, 20), np.float) * 2
        # Mask meta file
        self.csv_name = 'mask_image_matchup.csv'
        self.input_meta = aux_utils.make_dataframe()
        # Make input meta
        for c in range(3):
            ff_path = os.path.join(
                self.ff_dir,
                'flat-field_channel-{}.npy'.format(c)
            )
            np.save(ff_path, ff_im, allow_pickle=True, fix_imports=True)
            for p in range(5):
                im_name = aux_utils.get_im_name(
                    channel_idx=c,
                    slice_idx=self.slice_idx,
                    time_idx=self.time_idx,
                    pos_idx=p,
                )
                cv2.imwrite(
                    os.path.join(self.input_dir, im_name),
                    self.im + p * 10,
                )
                cv2.imwrite(
                    os.path.join(self.mask_dir, im_name),
                    self.mask,
                )
                meta_row = aux_utils.parse_idx_from_name(im_name)
                meta_row['dir_name'] = self.input_dir
                self.input_meta = self.input_meta.append(
                    meta_row,
                    ignore_index=True,
                )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_frames_meta_generator(self):
        frames_meta = meta_utils.frames_meta_generator(
            input_dir=self.input_dir,
            name_parser='parse_idx_from_name',
        )
        for idx, row in frames_meta.iterrows():
            input_row = self.input_meta.iloc[idx]
            nose.tools.assert_equal(input_row['file_name'], row['file_name'])
            nose.tools.assert_equal(input_row['slice_idx'], row['slice_idx'])
            nose.tools.assert_equal(input_row['time_idx'], row['time_idx'])
            nose.tools.assert_equal(input_row['channel_idx'], row['channel_idx'])
            nose.tools.assert_equal(input_row['pos_idx'], row['pos_idx'])

    def test_ints_meta_generator(self):
        # Write metadata
        self.input_meta.to_csv(
            os.path.join(self.input_dir, 'frames_meta.csv'),
            sep=',',
        )
        meta_utils.ints_meta_generator(
            input_dir=self.input_dir,
            block_size=5,
        )
        intensity_meta = pd.read_csv(
            os.path.join(self.input_dir, 'intensity_meta.csv'),
        )
        # There's 15 images and each image should be sampled 3 times
        # at col = 5, 10, 15 and row = 5
        self.assertEqual(intensity_meta.shape[0], 45)
        # Check one image
        meta_im = intensity_meta.loc[
            intensity_meta['file_name'] == 'im_c000_z001_t002_p000.png',
        ]
        for i, col_idx in enumerate([5, 10, 15]):
            self.assertEqual(meta_im.loc[i, 'col_idx'], col_idx)
            self.assertEqual(meta_im.loc[i, 'row_idx'], 5)
            self.assertEqual(meta_im.loc[i, 'intensity'], 5)

    def test_ints_meta_generator_flatfield(self):
        # Write metadata
        self.input_meta.to_csv(
            os.path.join(self.input_dir, 'frames_meta.csv'),
            sep=',',
        )
        meta_utils.ints_meta_generator(
            input_dir=self.input_dir,
            block_size=5,
            flat_field_dir=self.ff_dir,
        )
        intensity_meta = pd.read_csv(
            os.path.join(self.input_dir, 'intensity_meta.csv'),
        )
        # There's 15 images and each image should be sampled 3 times
        self.assertEqual(intensity_meta.shape[0], 45)
        # Check one image
        meta_im = intensity_meta.loc[
            intensity_meta['file_name'] == 'im_c000_z001_t002_p000.png',
        ]
        for i, col_idx in enumerate([5, 10, 15]):
            self.assertEqual(meta_im.loc[i, 'col_idx'], col_idx)
            self.assertEqual(meta_im.loc[i, 'row_idx'], 5)
            self.assertEqual(meta_im.loc[i, 'intensity'], 2.5)

    def test_mask_meta_generator(self):
        self.input_meta.to_csv(
            os.path.join(self.mask_dir, 'frames_meta.csv'),
            sep=',',
        )
        mask_meta = meta_utils.mask_meta_generator(
            input_dir=self.mask_dir,
        )
        self.assertEqual(mask_meta.shape[0], 15)
        expected_cols = [
            'channel_idx',
            'pos_idx',
            'slice_idx',
            'time_idx',
            'channel_name',
            'dir_name',
            'file_name',
            'fg_frac',
        ]
        self.assertListEqual(list(mask_meta), expected_cols)
        # Foreground fraction should be 0.5
        for i in range(15):
            self.assertEqual(mask_meta.loc[i, 'fg_frac'], .5)

    def test_compute_zscore_params(self):
        self.input_meta.to_csv(
            os.path.join(self.input_dir, 'frames_meta.csv'),
            sep=',',
        )
        meta_utils.ints_meta_generator(
            input_dir=self.input_dir,
            block_size=5,
        )
        intensity_meta = pd.read_csv(
            os.path.join(self.input_dir, 'intensity_meta.csv'),
        )
        self.input_meta.to_csv(
            os.path.join(self.mask_dir, 'frames_meta.csv'),
            sep=',',
        )
        mask_meta = meta_utils.mask_meta_generator(
            input_dir=self.mask_dir,
        )
        cols_to_merge = intensity_meta.columns[intensity_meta.columns != 'fg_frac']
        intensity_meta = pd.merge(
            intensity_meta[cols_to_merge],
            mask_meta[['pos_idx', 'time_idx', 'slice_idx', 'fg_frac']],
            how='left',
            on=['pos_idx', 'time_idx', 'slice_idx'],
        )
        frames_meta, ints_meta = meta_utils.compute_zscore_params(
            frames_meta=self.input_meta,
            ints_meta=intensity_meta,
            input_dir=self.input_dir,
            normalize_im='volume',
            min_fraction=.4,
        )
        # Check medians and iqr values
        for i, row in frames_meta.iterrows():
            self.assertEqual(row['zscore_iqr'], 0)
            # Added 10 for each p when saving images
            self.assertEqual(row['zscore_median'], 5 + row['pos_idx'] * 10)
