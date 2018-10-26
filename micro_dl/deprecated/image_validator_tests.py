import cv2
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.input.image_validator as image_validator


class TestImageValidator(unittest.TestCase):

    def setUp(self):
        """
        Set up a folder structure containing one timepoint (0)
        one channel (1) and two images in channel subfolder
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.tempdir.makedir('timepoint_0')
        self.tempdir.makedir('timepoint_0/channel_1')
        # Write images as bytes
        im = np.zeros((15, 12), dtype=np.uint16)
        res, im_encoded = cv2.imencode('.png', im)
        im_encoded = im_encoded.tostring()
        self.tempdir.write('timepoint_0/channel_1/im_0.png', im_encoded)
        self.tempdir.write('timepoint_0/channel_1/im_1.png', im_encoded)
        self.tempdir.listdir(recursive=True)

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_image_validator(self):
        """
        Test image validator on temporary folder structure
        """
        meta_name = 'image_volumes_info.csv'
        validator = image_validator.ImageValidator(
            input_dir=self.tempdir.path,
            meta_name=meta_name,
            verbose=10,
        )
        validator.folder_validator()
        # Check written metadata
        metadata = pd.read_csv(os.path.join(self.tempdir.path, meta_name))
        # Metadata should have 8 fields + 1 index and we have two files
        nose.tools.assert_equal(metadata.shape, (2, 9))
        # Metadata should contain the following column names
        expected_names = ['Unnamed: 0',
            'timepoint',
            'channel_num',
            'sample_num',
            'slice_num',
            'fname',
            'size_x_microns',
            'size_y_microns',
            'size_z_microns']
        nose.tools.assert_equal(list(metadata), expected_names)
        # Both files should have timepoint 0
        nose.tools.assert_equal(np.all(metadata['timepoint'] == 0), True)
        # Both file should have channel 1
        nose.tools.assert_equal(np.all(metadata['channel_num'] == 1), True)
        # The two image numbers should be sorted by index
        nose.tools.assert_equal(metadata['sample_num'][0], 0)
        nose.tools.assert_equal(metadata['sample_num'][1], 1)


