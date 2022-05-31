import argparse
import cv2
import itertools
import natsort
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest
from unittest.mock import patch

import micro_dl.cli.generate_meta as generate_meta
import micro_dl.utils.meta_utils as meta_utils


class TestGenerateMeta(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory with some images to generate frames_meta.csv for
        """
        self.tempdir = TempDirectory()
        self.temp_dir = self.tempdir.path
        self.idx_dir = os.path.join(self.temp_dir, 'idx_dir')
        self.sms_dir = os.path.join(self.temp_dir, 'sms_dir')
        self.tempdir.makedir(self.idx_dir)
        self.tempdir.makedir(self.sms_dir)
        # Write images
        self.time_idx = 5
        self.pos_idx = 7
        self.im = 1500 * np.ones((30, 20), dtype=np.uint16)
        self.channel_names = ['phase', 'brightfield', 'some_other_c', '666']

        for i, c in enumerate(self.channel_names):
            for z in range(5, 10):
                im_name = 'im_c00{}_z00{}_t005_p007.png'.format(i, z)
                cv2.imwrite(os.path.join(self.idx_dir, im_name), self.im)
                im_name = 'img_{}_t005_p007_z00{}.tif'.format(c, z)
                cv2.imwrite(os.path.join(self.sms_dir, im_name), self.im)

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_dir), False)

    def test_parse_args(self):
        with patch('argparse._sys.argv',
                   ['python',
                    '-i', '/testdir',
                    '--order', 'pzct',
                    '--name_parser', 'parse_sms_name']):
            parsed_args = generate_meta.parse_args()
            self.assertEqual(parsed_args.input, '/testdir')
            self.assertEqual(parsed_args.order, 'pzct')
            self.assertEqual(parsed_args.name_parser, 'parse_sms_name')

    @nose.tools.raises(BaseException)
    def test_parse_args_no_input(self):
        with patch('argparse._sys.argv',
                   ['python',
                    '--order', 'pzct',
                    '--name_parser', 'parse_sms_name']):
            generate_meta.parse_args()

    def test_generate_meta_idx(self):
        args = argparse.Namespace(
            input=self.idx_dir,
            order='cztp',
            name_parser='parse_idx_from_name',
            num_workers=4,
            normalize_im='stack',
        )
        generate_meta.main(args)
        frames_meta = pd.read_csv(os.path.join(self.idx_dir, 'frames_meta.csv'))
        iterator = itertools.product(range(len(self.channel_names)), range(5, 10))
        for i, (c, z) in enumerate(iterator):
            row = frames_meta.iloc[i]
            self.assertEqual(row.channel_idx, c)
            self.assertEqual(row.slice_idx, z)
            self.assertEqual(row.time_idx, self.time_idx)
            self.assertEqual(row.pos_idx, self.pos_idx)

    def test_generate_meta_sms(self):
        args = argparse.Namespace(
            input=self.sms_dir,
            name_parser='parse_sms_name',
            order="cztp",
            normalize_im='stack',
        )
        generate_meta.main(args)
        frames_meta = pd.read_csv(os.path.join(self.sms_dir, 'frames_meta.csv'))
        # This function sorts channel names
        sorted_names = natsort.natsorted(self.channel_names)
        iterator = itertools.product(range(len(self.channel_names)), range(5, 10))
        for i, (c, z) in enumerate(iterator):
            row = frames_meta.iloc[i]
            self.assertEqual(row.channel_idx, c)
            self.assertEqual(row.channel_name, sorted_names[c])
            self.assertEqual(row.slice_idx, z)
            self.assertEqual(row.time_idx, self.time_idx)
            self.assertEqual(row.pos_idx, self.pos_idx)

    @nose.tools.raises(AttributeError)
    def test_generate_meta_wrong_function(self):
        args = argparse.Namespace(
            input=self.sms_dir,
            name_parser='nonexisting_function',
        )
        generate_meta.main(args)

    def test_generate_intensity_meta(self):
        args = argparse.Namespace(
            input=self.sms_dir,
            name_parser='parse_sms_name',
            order="cztp",
            normalize_im='dataset',
            num_workers=4,
            block_size=10,
        )
        generate_meta.main(args)
        # Check intensity data
        ints_meta = pd.read_csv(
            os.path.join(self.sms_dir, 'intensity_meta.csv'),
        )
        expected_cols = ['channel_idx',
                         'pos_idx',
                         'slice_idx',
                         'time_idx',
                         'channel_name',
                         'dir_name',
                         'file_name',
                         'row_idx',
                         'col_idx',
                         'intensity']
        self.assertListEqual(list(ints_meta)[1:], expected_cols)
        # With block size 10 and image size 30x20 there should be 2 points
        # per image and there's 20 images
        self.assertEqual(ints_meta.shape[0], 40)
        # Check values in metadata, every other idx should be (10,10) and (20, 10)
        for idx in range(0, 40, 2):
            row_even = ints_meta.iloc[idx]
            self.assertEqual(row_even['row_idx'], 10)
            self.assertEqual(row_even['col_idx'], 10)
            row_uneven = ints_meta.iloc[idx + 1]
            self.assertEqual(row_uneven['row_idx'], 20)
            self.assertEqual(row_uneven['col_idx'], 10)
