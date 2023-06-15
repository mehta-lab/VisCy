import os
import unittest
import warnings

import numpy as np
import numpy.testing
import skimage.io as sk_im_io
from testfixtures import TempDirectory

import viscy.utils.aux_utils as aux_utils
import viscy.utils.image_utils as image_utils
import viscy.utils.mp_utils as mp_utils
from viscy.utils.masks import create_otsu_mask


class TestMpUtilsBaseClass(unittest.TestCase):
    def get_sphere(self, shape=(32, 32, 8)):
        # create an image with bimodal hist
        x = np.linspace(-4, 4, shape[0])
        y = x.copy()
        z = np.linspace(-3, 3, shape[2])
        xx, yy, zz = np.meshgrid(x, y, z)
        sph = xx**2 + yy**2 + zz**2
        fg = (sph <= shape[2]) * (shape[2] - sph)
        fg[fg > 1e-8] = (fg[fg > 1e-8] / np.max(fg)) * 127 + 128
        fg = np.around(fg).astype("uint8")
        bg = np.around((sph > shape[2]) * sph).astype("uint8")
        sph = fg + bg
        return sph

    def get_rect(self, shape=(32, 32, 8)):
        rect = np.zeros(shape)
        rect[3:30, 14:18, 3:6] = 120
        rect[14:18, 3:30, 3:6] = 120
        return rect

    def get_name(self, ch_idx, sl_idx, time_idx, pos_idx):
        im_name = (
            "im_c"
            + str(ch_idx).zfill(self.int2str_len)
            + "_z"
            + str(sl_idx).zfill(self.int2str_len)
            + "_t"
            + str(time_idx).zfill(self.int2str_len)
            + "_p"
            + str(pos_idx).zfill(self.int2str_len)
            + ".png"
        )
        return im_name

    def setUp(self):
        """Set up a directory for mask generation"""

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.meta_fname = "frames_meta.csv"
        self.frames_meta = aux_utils.make_dataframe()
        self.channel_ids = [1, 2]
        self.time_ids = 0
        self.pos_ids = 1
        self.int2str_len = 3

    def write_data_in_meta_csv(self, array, frames_meta, ch_idx):
        for z in range(array.shape[2]):
            im_name = self.get_name(ch_idx, z, self.time_ids, self.pos_ids)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk_im_io.imsave(
                    os.path.join(self.temp_path, im_name),
                    array[:, :, z].astype("uint8"),
                )
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name=im_name, dir_name=self.temp_path),
                ignore_index=True,
            )
        return frames_meta

    def tearDown(self):
        """Tear down temporary folder and file structure"""
        TempDirectory.cleanup_all()
        self.assertFalse(os.path.isdir(self.temp_path))


class TestMpUtilsOtsu(TestMpUtilsBaseClass):
    def setUp(self):
        super().setUp()

    def write_mask_data(self):
        self.sph_object = self.get_sphere()
        self.rect_object = self.get_rect()

        frames_meta = self.write_data_in_meta_csv(self.sph_object, self.frames_meta, 1)
        frames_meta = self.write_data_in_meta_csv(self.rect_object, frames_meta, 2)
        self.frames_meta = frames_meta
        # Write metadata
        self.frames_meta.to_csv(os.path.join(self.temp_path, self.meta_fname), sep=",")
        self.output_dir = os.path.join(self.temp_path, "mask_dir")
        os.makedirs(self.output_dir, exist_ok=True)

    def test_create_save_mask_otsu(self):
        """test create_save_mask otsu"""
        self.write_mask_data()
        for sl_idx in range(8):
            channels_meta_sub = aux_utils.get_sub_meta(
                frames_metadata=self.frames_meta,
                time_ids=self.time_ids,
                channel_ids=self.channel_ids,
                slice_ids=sl_idx,
                pos_ids=self.pos_ids,
            )
            cur_meta = mp_utils.create_save_mask(
                channels_meta_sub=channels_meta_sub,
                str_elem_radius=1,
                mask_dir=self.output_dir,
                mask_channel_idx=3,
                int2str_len=3,
                mask_type="otsu",
                mask_ext=".png",
            )
            fname = aux_utils.get_im_name(
                time_idx=self.time_ids,
                channel_idx=3,
                slice_idx=sl_idx,
                pos_idx=self.pos_ids,
            )
            self.assertEqual(cur_meta["channel_idx"], 3)
            self.assertEqual(cur_meta["slice_idx"], sl_idx)
            self.assertEqual(cur_meta["time_idx"], self.time_ids)
            self.assertEqual(cur_meta["pos_idx"], self.pos_ids)
            self.assertEqual(cur_meta["file_name"], fname)
            # Check that mask file has been written
            op_fname = os.path.join(self.output_dir, fname)
            self.assertTrue(os.path.exists(op_fname))
            # Read mask iamge
            mask_image = image_utils.read_image(op_fname)
            if mask_image.dtype != bool:
                mask_image = mask_image > 0
            input_image = (
                self.sph_object[:, :, sl_idx],
                self.rect_object[:, :, sl_idx],
            )
            mask_stack = np.stack(
                [
                    create_otsu_mask(input_image[0], str_elem_size=1),
                    create_otsu_mask(input_image[1], str_elem_size=1),
                ]
            )
            mask_exp = np.any(mask_stack, axis=0)
            numpy.testing.assert_array_equal(
                mask_image,
                mask_exp,
            )
