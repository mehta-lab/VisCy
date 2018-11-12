import nose.tools
import numpy as np
import numpy.testing
import os
import pandas as pd
import skimage.io as sk_im_io
from testfixtures import TempDirectory
import unittest

from micro_dl.utils import aux_utils as aux_utils
from micro_dl.utils.image_utils import create_mask
from micro_dl.input.generate_masks import MaskProcessor


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

    def test_read_image(self):
        """Test read_image"""

        for sl_idx in self.mask_gen_inst.slice_ids:
            cur_im = self.mask_gen_inst._read_image(
                time_idx=self.mask_gen_inst.time_ids[0],
                channel_idx=self.mask_gen_inst.channel_ids[0],
                slice_idx=sl_idx,
                pos_idx=self.mask_gen_inst.pos_ids[0],
                correct_flat_field=False
            )
            numpy.testing.assert_array_equal(cur_im,
                                             self.sph_object[:, :, sl_idx])

        for sl_idx in self.mask_gen_inst.slice_ids:
            cur_im = self.mask_gen_inst._read_image(
                time_idx=self.mask_gen_inst.time_ids[0],
                channel_idx=self.mask_gen_inst.channel_ids[1],
                slice_idx=sl_idx,
                pos_idx=self.mask_gen_inst.pos_ids[0],
                correct_flat_field=False
            )
            numpy.testing.assert_array_equal(cur_im,
                                             self.rec_object[:, :, sl_idx])

    def test_create_save_mask(self):
        """Test create_save_mask"""

        input_image = self.sph_object[:, :, 5] + self.rec_object[:, :, 5]
        cur_meta = self.mask_gen_inst._create_save_mask(
            input_image=input_image,
            str_elem_radius=1,
            time_idx=self.time_ids,
            pos_idx=self.pos_ids,
            slice_idx=5
        )
        fname = aux_utils.get_im_name(time_idx=self.time_ids,
                                      channel_idx=3,
                                      slice_idx=5,
                                      pos_idx=self.pos_ids)
        op_fname = os.path.join(self.output_dir, 'mask_channels_1-2', fname)
        nose.tools.assert_equal(os.path.exists(op_fname),
                                True)
        mask_image = np.load(op_fname)
        numpy.testing.assert_array_equal(
            mask_image,
            create_mask(input_image, str_elem_size=1)
        )
        exp_meta = {'channel_idx': 3,
                    'slice_idx': 5,
                    'time_idx': 0,
                    'pos_idx': 1,
                    'file_name': fname}
        nose.tools.assert_dict_equal(cur_meta, exp_meta)

    def test_generate_masks_uni(self):
        """Test generate_masks with uniform structure"""

        self.mask_gen_inst.generate_masks(str_elem_radius=1)
        frames_meta = pd.read_csv(
            os.path.join(self.temp_path, 'frames_meta.csv'),
            index_col=0
        )
        exp_len = 24
        nose.tools.assert_equal(len(frames_meta), exp_len)

        for slice_idx in self.mask_gen_inst.slice_ids:
            im = self.sph_object[:, :, slice_idx] + \
                 self.rec_object[:, :, slice_idx]
            exp_mask = create_mask(im, str_elem_size=1)
            fname = aux_utils.get_im_name(time_idx=self.time_ids,
                                          channel_idx=3,
                                          slice_idx=slice_idx,
                                          pos_idx=self.pos_ids)
            cur_mask = np.load(os.path.join(self.mask_gen_inst.get_mask_dir(),
                                            fname))
            numpy.testing.assert_array_equal(cur_mask, exp_mask)

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
        exp_len = 22
        nose.tools.assert_equal(len(frames_meta), exp_len)

        for pos_idx in pos_ids:
            cur_object = self.sph_object if pos_idx == 1 else rec
            for slice_idx in mask_gen_inst.nested_id_dict[0][0][pos_idx]:
                im = cur_object[:, :, slice_idx]
                exp_mask = create_mask(im, str_elem_size=1)
                fname = aux_utils.get_im_name(time_idx=time_ids,
                                              channel_idx=1,
                                              slice_idx=slice_idx,
                                              pos_idx=pos_idx)
                cur_mask = np.load(os.path.join(mask_gen_inst.get_mask_dir(),
                                                fname))
                numpy.testing.assert_array_equal(cur_mask, exp_mask)
