import nose.tools
import numpy as np
import numpy.testing
import os
import pandas as pd
import skimage.io as sk_im_io
from skimage import draw
from testfixtures import TempDirectory
import unittest
import warnings

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.mp_utils as mp_utils
import micro_dl.utils.image_utils as image_utils

from micro_dl.utils.masks import create_otsu_mask


class TestMpUtilsBaseClass(unittest.TestCase):

    def get_sphere(self, shape=(32, 32, 8)):
        # create an image with bimodal hist
        x = np.linspace(-4, 4, shape[0])
        y = x.copy()
        z = np.linspace(-3, 3, shape[2])
        xx, yy, zz = np.meshgrid(x, y, z)
        sph = (xx ** 2 + yy ** 2 + zz ** 2)
        fg = (sph <= shape[2]) * (shape[2] - sph)
        fg[fg > 1e-8] = (fg[fg > 1e-8] / np.max(fg)) * 127 + 128
        fg = np.around(fg).astype('uint8')
        bg = np.around((sph > shape[2]) * sph).astype('uint8')
        sph = fg + bg
        return sph

    def get_rect(self, shape=(32, 32, 8)):
        rect = np.zeros(shape)
        rect[3:30, 14:18, 3:6] = 120
        rect[14:18, 3:30, 3:6] = 120
        return rect

    def get_name(self, ch_idx, sl_idx, time_idx, pos_idx):
        im_name = 'im_c' + str(ch_idx).zfill(self.int2str_len) + \
                  '_z' + str(sl_idx).zfill(self.int2str_len) + \
                  '_t' + str(time_idx).zfill(self.int2str_len) + \
                  '_p' + str(pos_idx).zfill(self.int2str_len) + ".png"
        return im_name

    def setUp(self):
        """Set up a directory for mask generation, no flatfield"""

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.meta_fname = 'frames_meta.csv'
        self.df_columns = [
            'channel_idx',
            'slice_idx',
            'time_idx',
            'channel_name',
            'file_name',
            'pos_idx']
        self.frames_meta = pd.DataFrame(columns=self.df_columns)
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
                    array[:, :, z].astype('uint8'),
                )
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name, self.df_columns),
                ignore_index=True
            )
        return frames_meta

    def tearDown(self):
        """Tear down temporary folder and file structure"""

        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)


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
        self.frames_meta.to_csv(os.path.join(self.temp_path, self.meta_fname), sep=',')
        self.output_dir = os.path.join(self.temp_path, 'mask_dir')
        os.makedirs(self.output_dir, exist_ok=True)

    def test_create_save_mask_otsu(self):
        """test create_save_mask otsu"""
        self.write_mask_data()
        for sl_idx in range(8):
            input_fnames = ['im_c001_z00{}_t000_p001.png'.format(sl_idx),
                            'im_c002_z00{}_t000_p001.png'.format(sl_idx)]
            input_fnames = [os.path.join(self.temp_path, fname)
                            for fname in input_fnames]
            cur_meta = mp_utils.create_save_mask(
                tuple(input_fnames),
                None,
                str_elem_radius=1,
                mask_dir=self.output_dir,
                mask_channel_idx=3,
                time_idx=self.time_ids,
                pos_idx=self.pos_ids,
                slice_idx=sl_idx,
                int2str_len=3,
                mask_type='otsu',
                mask_ext='.png'
            )
            fname = aux_utils.get_im_name(
                time_idx=self.time_ids,
                channel_idx=3,
                slice_idx=sl_idx,
                pos_idx=self.pos_ids,
            )
            exp_meta = {'channel_idx': 3,
                        'slice_idx': sl_idx,
                        'time_idx': 0,
                        'pos_idx': 1,
                        'file_name': fname}
            nose.tools.assert_dict_equal(cur_meta, exp_meta)

            op_fname = os.path.join(self.output_dir, fname)
            nose.tools.assert_equal(os.path.exists(op_fname),
                                    True)

            mask_image = image_utils.read_image(op_fname)
            if mask_image.dtype != bool:
                mask_image = mask_image > 0
            input_image = (self.sph_object[:, :, sl_idx],
                           self.rect_object[:, :, sl_idx])
            mask_stack = np.stack([create_otsu_mask(input_image[0], str_elem_size=1),
                                  create_otsu_mask(input_image[1], str_elem_size=1)])
            mask_exp = np.any(mask_stack, axis=0)
            numpy.testing.assert_array_equal(
                mask_image, mask_exp
            )

    def test_rescale_vol_and_save(self):
        """test rescale_vol_and_save"""
        self.write_mask_data()
        for ch_idx in self.channel_ids:
            op_fname = os.path.join(
                self.temp_path,
                'im_c{}_z0_t0_p0_sc4.1-1.0-1.0.npy'.format(ch_idx)
            )
            ff_path = None
            mp_utils.rescale_vol_and_save(
                self.time_ids,
                self.pos_ids,
                ch_idx,
                0, 8,
                self.frames_meta,
                op_fname,
                [4.1, 1.0, 1.0],
                self.temp_path,
                ff_path,
            )
            resc_vol = np.load(op_fname)
            nose.tools.assert_tuple_equal(resc_vol.shape,
                                          (131, 32, 8))
            # Used to be (33, 32, 32)


class TestMpUtilsBorderWeightMap(TestMpUtilsBaseClass):

    def setUp(self):
        super().setUp()

    def get_touching_circles(self, shape=(64, 64)):
        # Creating a test image with 3 circles, 2 close to each other and one far away
        self.radius = 10
        self.params = [(20, 16, self.radius), (44, 16, self.radius), (47, 47, self.radius)]
        mask = np.zeros(shape, dtype=np.uint8)
        for i, (cx, cy, radius) in enumerate(self.params):
            rr, cc = draw.circle(cx, cy, radius)
            mask[rr, cc] = i + 1
        mask = mask[:, :, np.newaxis]
        return mask

    def write_mask_data(self):
        self.touching_circles_object = self.get_touching_circles()

        frames_meta = self.write_data_in_meta_csv(self.touching_circles_object, self.frames_meta, 1)
        # Write metadata
        frames_meta.to_csv(os.path.join(self.temp_path, self.meta_fname), sep=',')
        self.frames_meta = frames_meta
        self.output_dir = os.path.join(self.temp_path, 'mask_dir')
        os.makedirs(self.output_dir, exist_ok=True)

    def test_create_save_mask_border_map(self):
        """test create_save_mask border weight map"""
        self.write_mask_data()
        for sl_idx in range(1):
            input_fnames = ['im_c001_z00{}_t000_p001.png'.format(sl_idx)]
            input_fnames = [os.path.join(self.temp_path, fname)
                            for fname in input_fnames]
            cur_meta = mp_utils.create_save_mask(
                tuple(input_fnames),
                None,
                str_elem_radius=1,
                mask_dir=self.output_dir,
                mask_channel_idx=2,
                time_idx=self.time_ids,
                pos_idx=self.pos_ids,
                slice_idx=sl_idx,
                int2str_len=3,
                mask_type='borders_weight_loss_map',
                mask_ext='.png'
            )
            fname = aux_utils.get_im_name(
                time_idx=self.time_ids,
                channel_idx=2,
                slice_idx=sl_idx,
                pos_idx=self.pos_ids,
            )
            exp_meta = {'channel_idx': 2,
                        'slice_idx': sl_idx,
                        'time_idx': 0,
                        'pos_idx': 1,
                        'file_name': fname}
            nose.tools.assert_dict_equal(cur_meta, exp_meta)

            op_fname = os.path.join(self.output_dir, fname)
            nose.tools.assert_equal(os.path.exists(op_fname),
                                    True)
            weight_map = image_utils.read_image(op_fname)
            max_weight_map = np.max(weight_map)
            # weight map between 20, 16 and 44, 16 should be maximum
            # as there is more weight when two objects boundaries overlap
            y_coord = self.params[0][1]
            for x_coord in range(self.params[0][0] + self.radius, self.params[1][0] - self.radius):
                distance_near_intersection = weight_map[x_coord, y_coord]
                nose.tools.assert_equal(max_weight_map, distance_near_intersection)
