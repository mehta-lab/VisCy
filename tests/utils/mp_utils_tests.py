import nose.tools
import numpy as np
import numpy.testing
import os
import pandas as pd
import skimage.io as sk_im_io
from testfixtures import TempDirectory
import unittest
import warnings

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.mp_utils as mp_utils
from micro_dl.utils.masks import create_otsu_mask


class TestMpUtils(unittest.TestCase):

    def setUp(self):
        """Set up a directory for mask generation, no flatfield"""

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        self.meta_fname = 'frames_meta.csv'
        df_columns = ['channel_idx',
                      'slice_idx',
                      'time_idx',
                      'channel_name',
                      'file_name',
                      'pos_idx']
        frames_meta = pd.DataFrame(columns=df_columns)

        # create an image with bimodal hist
        x = np.linspace(-4, 4, 32)
        y = x.copy()
        z = np.linspace(-3, 3, 8)
        xx, yy, zz = np.meshgrid(x, y, z)
        sph = (xx ** 2 + yy ** 2 + zz ** 2)
        fg = (sph <= 8) * (8 - sph)
        fg[fg > 1e-8] = (fg[fg > 1e-8] / np.max(fg)) * 127 + 128
        fg = np.around(fg).astype('uint8')
        bg = np.around((sph > 8) * sph).astype('uint8')
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

        def _get_name(ch_idx, sl_idx, time_idx, pos_idx):
            im_name = 'im_c' + str(ch_idx).zfill(self.int2str_len) + \
                      '_z' + str(sl_idx).zfill(self.int2str_len) + \
                      '_t' + str(time_idx).zfill(self.int2str_len) + \
                      '_p' + str(pos_idx).zfill(self.int2str_len) + ".png"
            return im_name

        for z in range(sph.shape[2]):
            im_name = _get_name(1, z, self.time_ids, self.pos_ids)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk_im_io.imsave(
                    os.path.join(self.temp_path, im_name),
                    object1[:, :, z].astype('uint8'),
                )
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name, df_columns),
                ignore_index=True
            )
        for z in range(rec.shape[2]):
            im_name = _get_name(2, z, self.time_ids, self.pos_ids)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk_im_io.imsave(
                    os.path.join(self.temp_path, im_name),
                    rec[:, :, z].astype('uint8'),
                )
            frames_meta = frames_meta.append(
                aux_utils.parse_idx_from_name(im_name, df_columns),
                ignore_index=True
            )
        # Write metadata
        frames_meta.to_csv(os.path.join(self.temp_path, self.meta_fname),
                           sep=',')
        self.frames_meta = frames_meta
        self.output_dir = os.path.join(self.temp_path, 'mask_dir')
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Tear down temporary folder and file structure"""

        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_create_save_mask(self):
        """test create_save_mask"""

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
            )
            fname = aux_utils.get_im_name(time_idx=self.time_ids,
                                          channel_idx=3,
                                          slice_idx=sl_idx,
                                          pos_idx=self.pos_ids)
            exp_meta = {'channel_idx': 3,
                        'slice_idx': sl_idx,
                        'time_idx': 0,
                        'pos_idx': 1,
                        'file_name': fname}
            nose.tools.assert_dict_equal(cur_meta, exp_meta)

            op_fname = os.path.join(self.output_dir, fname)
            nose.tools.assert_equal(os.path.exists(op_fname),
                                    True)
            mask_image = np.load(op_fname)
            input_image = (self.sph_object[:, :, sl_idx] +
                           self.rec_object[:, :, sl_idx])
            numpy.testing.assert_array_equal(
                mask_image,
                create_otsu_mask(input_image, str_elem_size=1)
            )

    def test_rescale_vol_and_save(self):
        """test rescale_vol_and_save"""

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
