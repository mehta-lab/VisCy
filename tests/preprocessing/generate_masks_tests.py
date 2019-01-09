import nose.tools
import numpy as np
import numpy.testing
import os
import pandas as pd
import skimage.io as sk_im_io
from testfixtures import TempDirectory
import unittest

from micro_dl.utils import aux_utils as aux_utils
from micro_dl.preprocessing.generate_masks import MaskProcessor


class TestMaskProcessor(unittest.TestCase):

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
            sk_im_io.imsave(os.path.join(self.temp_path, im_name),
                            object1[:, :, z].astype('uint8'))
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_columns),
                ignore_index=True
            )
        for z in range(rec.shape[2]):
            im_name = _get_name(2, z, self.time_ids, self.pos_ids)
            sk_im_io.imsave(os.path.join(self.temp_path, im_name),
                            rec[:, :, z].astype('uint8'))
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_columns),
                ignore_index=True
            )
        # Write metadata
        frames_meta.to_csv(os.path.join(self.temp_path, self.meta_fname),
                           sep=',')

        self.output_dir = os.path.join(self.temp_path, 'mask_dir')
        self.mask_gen_inst = MaskProcessor(input_dir=self.temp_path,
                                           output_dir=self.output_dir,
                                           channel_ids=self.channel_ids)

    def tearDown(self):
        """Tear down temporary folder and file structure"""

        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """Test init"""

        nose.tools.assert_equal(self.mask_gen_inst.mask_channel, 3)
        nose.tools.assert_equal(self.mask_gen_inst.channel_ids, [1, 2])
        nose.tools.assert_equal(self.mask_gen_inst.time_ids, 0)
        nose.tools.assert_equal(self.mask_gen_inst.pos_ids, 1)
        numpy.testing.assert_array_equal(self.mask_gen_inst.slice_ids,
                                         [0, 1, 2, 3, 4, 5, 6, 7])
        nose.tools.assert_equal(
            self.mask_gen_inst.mask_dir,
            os.path.join(self.output_dir, 'mask_channels_1-2')
        )
        nose.tools.assert_equal(self.mask_gen_inst.nested_id_dict, None)

    def test_get_mask_dir(self):
        """Test get_mask_dir"""

        mask_dir = os.path.join(self.output_dir, 'mask_channels_1-2')
        nose.tools.assert_equal(self.mask_gen_inst.get_mask_dir(),
                                mask_dir)

    def test_get_mask_channel(self):
        """Test get_mask_channel"""

        nose.tools.assert_equal(self.mask_gen_inst.get_mask_channel(), 3)

    def test_get_args_read_image(self):
        """Test _get_args_read_image"""

        ip_fnames, ff_fname = self.mask_gen_inst._get_args_read_image(
            time_idx=self.time_ids,
            channel_idx=self.channel_ids,
            slice_idx=5,
            pos_idx=self.pos_ids,
            correct_flat_field=None
        )
        exp_fnames = ['im_c001_z005_t000_p001.png',
                      'im_c002_z005_t000_p001.png']
        for idx, fname in enumerate(exp_fnames):
            nose.tools.assert_equal(ip_fnames[idx],
                                    os.path.join(self.temp_path, fname))
        nose.tools.assert_equal(ff_fname, None)

    def test_generate_masks_uni(self):
        """Test generate masks"""

        self.mask_gen_inst.generate_masks(str_elem_radius=1)
        frames_meta = pd.read_csv(
            os.path.join(self.temp_path, 'frames_meta.csv'),
            index_col=0
        )
        # 8 slices and 3 channels
        exp_len = 24
        nose.tools.assert_equal(len(frames_meta), exp_len)
        for idx in range(8):
            nose.tools.assert_equal('im_c003_z00{}_t000_p001.npy'.format(idx),
                                    frames_meta.iloc[16 + idx]['file_name'])

    def test_generate_masks_nonuni(self):
        """Test generate_masks with non-uniform structure"""

        rec = self.rec_object[:, :, 3:6]
        channel_ids = 0
        time_ids = 0
        pos_ids = [1, 2]

        df_columns = ['channel_idx',
                      'slice_idx',
                      'time_idx',
                      'channel_name',
                      'file_name',
                      'pos_idx']
        frames_meta = pd.DataFrame(columns=df_columns)

        def _get_name(ch_idx, sl_idx, time_idx, pos_idx):
            im_name = 'im_c' + str(ch_idx).zfill(self.int2str_len) + \
                      '_z' + str(sl_idx).zfill(self.int2str_len) + \
                      '_t' + str(time_idx).zfill(self.int2str_len) + \
                      '_p' + str(pos_idx).zfill(self.int2str_len) + ".png"
            return im_name

        for z in range(self.sph_object.shape[2]):
            im_name = _get_name(channel_ids, z, time_ids, pos_ids[0])
            sk_im_io.imsave(os.path.join(self.temp_path, im_name),
                            self.sph_object[:, :, z].astype('uint8'))
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_columns),
                ignore_index=True
            )
        for z in range(rec.shape[2]):
            im_name = _get_name(channel_ids, z, time_ids, pos_ids[1])
            sk_im_io.imsave(os.path.join(self.temp_path, im_name),
                            rec[:, :, z].astype('uint8'))
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_columns),
                ignore_index=True
            )
        # Write metadata
        frames_meta.to_csv(os.path.join(self.temp_path, self.meta_fname),
                           sep=',')

        self.output_dir = os.path.join(self.temp_path, 'mask_dir')
        mask_gen_inst = MaskProcessor(input_dir=self.temp_path,
                                      output_dir=self.output_dir,
                                      channel_ids=channel_ids,
                                      uniform_struct=False)
        exp_nested_id_dict = {0: {0: {1: [0, 1, 2, 3, 4, 5, 6, 7],
                                      2: [0, 1, 2]}}}
        numpy.testing.assert_array_equal(mask_gen_inst.nested_id_dict[0][0][1],
                                         exp_nested_id_dict[0][0][1])
        numpy.testing.assert_array_equal(mask_gen_inst.nested_id_dict[0][0][2],
                                         exp_nested_id_dict[0][0][2])

        mask_gen_inst.generate_masks(str_elem_radius=1)

        frames_meta = pd.read_csv(
            os.path.join(self.temp_path, 'frames_meta.csv'),
            index_col=0
        )
        # pos1: 8 slices, pos2: 3 slices
        exp_len = 22
        nose.tools.assert_equal(len(frames_meta), exp_len)
        mask_fnames = frames_meta['file_name'].tolist()[11:22]
        exp_mask_fnames = [
            'im_c001_z000_t000_p001.npy', 'im_c001_z000_t000_p002.npy',
            'im_c001_z001_t000_p001.npy', 'im_c001_z001_t000_p002.npy',
            'im_c001_z002_t000_p001.npy', 'im_c001_z002_t000_p002.npy',
            'im_c001_z003_t000_p001.npy', 'im_c001_z004_t000_p001.npy',
            'im_c001_z005_t000_p001.npy', 'im_c001_z006_t000_p001.npy',
            'im_c001_z007_t000_p001.npy']
        nose.tools.assert_list_equal(mask_fnames, exp_mask_fnames)
