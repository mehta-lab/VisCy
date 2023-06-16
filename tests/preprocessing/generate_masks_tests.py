import os
import unittest
import warnings

import nose.tools
import numpy as np
import numpy.testing
import pandas as pd
import skimage.io as sk_im_io
from testfixtures import TempDirectory

from viscy.preprocessing.generate_masks import MaskProcessor
from viscy.utils import aux_utils as aux_utils


class TestMaskProcessor(unittest.TestCase):
    def setUp(self):
        """Set up a directory for mask generation,"""

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.meta_fname = "frames_meta.csv"
        frames_meta = aux_utils.make_dataframe()

        # create an image with bimodal hist
        x = np.linspace(-4, 4, 32)
        y = x.copy()
        z = np.linspace(-3, 3, 8)
        xx, yy, zz = np.meshgrid(x, y, z)
        sph = xx**2 + yy**2 + zz**2
        fg = (sph <= 8) * (8 - sph)
        fg[fg > 1e-8] = (fg[fg > 1e-8] / np.max(fg)) * 127 + 128
        fg = np.around(fg).astype("uint8")
        bg = np.around((sph > 8) * sph).astype("uint8")
        object1 = fg + bg

        # create an image with a rect
        rec = np.zeros(sph.shape)
        rec[3:30, 14:18, 3:6] = 120
        rec[14:18, 3:30, 3:6] = 120

        self.sph_object = object1
        self.rec_object = rec

        self.channel_ids = [1, 2]
        self.time_ids = 0
        self.pos_ids = 1
        self.int2str_len = 3

        for z in range(sph.shape[2]):
            im_name = aux_utils.get_im_name(
                time_idx=self.time_ids,
                channel_idx=1,
                slice_idx=z,
                pos_idx=self.pos_ids,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk_im_io.imsave(
                    os.path.join(self.temp_path, im_name),
                    object1[:, :, z].astype("uint8"),
                )
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name=im_name, dir_name=self.temp_path),
                ignore_index=True,
            )
        for z in range(rec.shape[2]):
            im_name = aux_utils.get_im_name(
                time_idx=self.time_ids,
                channel_idx=2,
                slice_idx=z,
                pos_idx=self.pos_ids,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk_im_io.imsave(
                    os.path.join(self.temp_path, im_name),
                    rec[:, :, z].astype("uint8"),
                )
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name=im_name, dir_name=self.temp_path),
                ignore_index=True,
            )
        # Write metadata
        frames_meta.to_csv(os.path.join(self.temp_path, self.meta_fname), sep=",")

        self.output_dir = os.path.join(self.temp_path, "mask_dir")
        self.mask_gen_inst = MaskProcessor(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            channel_ids=self.channel_ids,
        )

    def tearDown(self):
        """Tear down temporary folder and file structure"""
        TempDirectory.cleanup_all()
        self.assertFalse(os.path.isdir(self.temp_path))

    def test_init(self):
        """Test init"""
        self.assertEqual(self.mask_gen_inst.input_dir, self.temp_path)
        self.assertEqual(self.mask_gen_inst.output_dir, self.output_dir)
        nose.tools.assert_equal(self.mask_gen_inst.mask_channel, 3)
        nose.tools.assert_equal(
            self.mask_gen_inst.mask_dir,
            os.path.join(self.output_dir, "mask_channels_1-2"),
        )
        self.assertListEqual(self.channel_ids, self.channel_ids)
        nose.tools.assert_equal(self.mask_gen_inst.nested_id_dict, None)

    def test_get_mask_dir(self):
        """Test get_mask_dir"""
        mask_dir = os.path.join(self.output_dir, "mask_channels_1-2")
        nose.tools.assert_equal(self.mask_gen_inst.get_mask_dir(), mask_dir)

    def test_get_mask_channel(self):
        """Test get_mask_channel"""
        nose.tools.assert_equal(self.mask_gen_inst.get_mask_channel(), 3)

    def test_generate_masks_uni(self):
        """Test generate masks"""
        self.mask_gen_inst.generate_masks(str_elem_radius=1)
        frames_meta = pd.read_csv(
            os.path.join(self.mask_gen_inst.get_mask_dir(), "frames_meta.csv"),
            index_col=0,
        )
        # 8 slices and 3 channels
        exp_len = 8
        nose.tools.assert_equal(len(frames_meta), exp_len)
        for idx in range(exp_len):
            nose.tools.assert_equal(
                "im_c003_z00{}_t000_p001.npy".format(idx),
                frames_meta.iloc[idx]["file_name"],
            )

    def test_generate_masks_nonuni(self):
        """Test generate_masks with non-uniform structure"""
        rec = self.rec_object[:, :, 3:6]
        channel_ids = 0
        time_ids = 0
        pos_ids = [1, 2]
        frames_meta = aux_utils.make_dataframe()

        for z in range(self.sph_object.shape[2]):
            im_name = aux_utils.get_im_name(
                time_idx=time_ids,
                channel_idx=channel_ids,
                slice_idx=z,
                pos_idx=pos_ids[0],
            )
            sk_im_io.imsave(
                os.path.join(self.temp_path, im_name),
                self.sph_object[:, :, z].astype("uint8"),
            )
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name=im_name, dir_name=self.temp_path),
                ignore_index=True,
            )
        for z in range(rec.shape[2]):
            im_name = aux_utils.get_im_name(
                time_idx=time_ids,
                channel_idx=channel_ids,
                slice_idx=z,
                pos_idx=pos_ids[1],
            )
            sk_im_io.imsave(
                os.path.join(self.temp_path, im_name), rec[:, :, z].astype("uint8")
            )
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name=im_name, dir_name=self.temp_path),
                ignore_index=True,
            )
        # Write metadata
        frames_meta.to_csv(os.path.join(self.temp_path, self.meta_fname), sep=",")

        self.output_dir = os.path.join(self.temp_path, "mask_dir")
        mask_gen_inst = MaskProcessor(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            channel_ids=channel_ids,
            uniform_struct=False,
        )
        exp_nested_id_dict = {0: {0: {1: [0, 1, 2, 3, 4, 5, 6, 7], 2: [0, 1, 2]}}}
        numpy.testing.assert_array_equal(
            mask_gen_inst.nested_id_dict[0][0][1], exp_nested_id_dict[0][0][1]
        )
        numpy.testing.assert_array_equal(
            mask_gen_inst.nested_id_dict[0][0][2], exp_nested_id_dict[0][0][2]
        )

        mask_gen_inst.generate_masks(str_elem_radius=1)

        frames_meta = pd.read_csv(
            os.path.join(mask_gen_inst.get_mask_dir(), "frames_meta.csv"),
            index_col=0,
        )
        # pos1: 8 slices, pos2: 3 slices
        exp_len = 8 + 3
        nose.tools.assert_equal(len(frames_meta), exp_len)
        mask_fnames = frames_meta["file_name"].tolist()
        exp_mask_fnames = [
            "im_c001_z000_t000_p001.npy",
            "im_c001_z000_t000_p002.npy",
            "im_c001_z001_t000_p001.npy",
            "im_c001_z001_t000_p002.npy",
            "im_c001_z002_t000_p001.npy",
            "im_c001_z002_t000_p002.npy",
            "im_c001_z003_t000_p001.npy",
            "im_c001_z004_t000_p001.npy",
            "im_c001_z005_t000_p001.npy",
            "im_c001_z006_t000_p001.npy",
            "im_c001_z007_t000_p001.npy",
        ]
        nose.tools.assert_list_equal(mask_fnames, exp_mask_fnames)
