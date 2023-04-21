import cv2
import nose.tools
import numpy as np
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

# Create a test image and its corresponding coordinates and values
# Create a test image with a bright block to the right
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils
import micro_dl.utils.io_utils as io_utils
import micro_dl.utils.normalize as normalize
from tests.utils.masks_utils_tests import uni_thr_tst_image

test_im = np.zeros((10, 15), np.uint16) + 100
test_im[:, 9:] = 200
x, y = np.meshgrid(np.linspace(1, 7, 3), np.linspace(1, 13, 5))
test_coords = np.vstack((x.flatten(), y.flatten())).T
test_values = np.zeros((15,), dtype=np.float64) + 100.
test_values[9:] = 200.


def test_upscale_image():
    im_out = image_utils.rescale_image(test_im, 2)
    im_shape = im_out.shape
    test_shape = test_im.shape
    nose.tools.assert_equal(im_shape[0], test_shape[0] * 2)
    nose.tools.assert_equal(im_shape[1], test_shape[1] * 2)
    nose.tools.assert_equal(im_out[0, 0], test_im[0, 0])
    nose.tools.assert_equal(im_out[-1, -1], test_im[-1, -1])


def test_downscale_image():
    im_out = image_utils.rescale_image(test_im, 0.5)
    im_shape = im_out.shape
    test_shape = test_im.shape
    nose.tools.assert_equal(im_shape[0], round(test_shape[0] * .5))
    nose.tools.assert_equal(im_shape[1], round(test_shape[1] * .5))
    nose.tools.assert_equal(im_out[0, 0], test_im[0, 0])
    nose.tools.assert_equal(im_out[-1, -1], test_im[-1, -1])


def test_samescale_image():
    im_out = image_utils.rescale_image(test_im, 1)
    im_shape = im_out.shape
    test_shape = test_im.shape
    nose.tools.assert_equal(im_shape[0], test_shape[0])
    nose.tools.assert_equal(im_shape[1], test_shape[1])
    nose.tools.assert_equal(im_out[0, 0], test_im[0, 0])
    nose.tools.assert_equal(im_out[-1, -1], test_im[-1, -1])


def test_rescale_volume():
    # shape (5, 31, 31)
    nd_image = np.repeat(uni_thr_tst_image[np.newaxis], 5, axis=0)
    # upsample isotropically, 0.5 upsampling
    res_volume = image_utils.rescale_nd_image(nd_image, 1.3)
    nose.tools.assert_tuple_equal(res_volume.shape, (6, 40, 40))
    # upsample anisotropically
    res_volume = image_utils.rescale_nd_image(nd_image, [2.1, 1.1, 1.7])
    nose.tools.assert_tuple_equal(res_volume.shape, (10, 34, 53))
    # downsample isotropically, 0.5 downsampling
    res_volume = image_utils.rescale_nd_image(nd_image, 0.7)
    nose.tools.assert_tuple_equal(res_volume.shape, (4, 22, 22))
    # assertion error


@nose.tools.raises(AssertionError)
def test_rescale_volume_vrong_dims():
    nd_image = np.repeat(uni_thr_tst_image[np.newaxis], 5, axis=0)
    image_utils.rescale_nd_image(nd_image, [1.2, 1.8])


def test_center_crop_to_shape():
    im = np.zeros((5, 10, 15))
    output_shape = [5, 6, 9]
    im_center = image_utils.center_crop_to_shape(im, output_shape)
    nose.tools.assert_tuple_equal(im_center.shape, (5, 6, 9))


def test_center_crop_to_shape_2d():
    im = np.zeros((2, 5, 10))
    output_shape = [3, 7]
    im_center = image_utils.center_crop_to_shape(im, output_shape)
    nose.tools.assert_tuple_equal(im_center.shape, (2, 3, 7))


def test_center_crop_to_shape_2d_xyx():
    im = np.zeros((5, 10, 2))
    output_shape = [3, 7]
    im_center = image_utils.center_crop_to_shape(im, output_shape, 'xyz')
    nose.tools.assert_tuple_equal(im_center.shape, (3, 7, 2))


@nose.tools.raises(AssertionError)
def test_center_crop_to_shape_2d_too_big():
    im = np.zeros((2, 5, 10))
    output_shape = [7, 7]
    image_utils.center_crop_to_shape(im, output_shape)


def test_grid_sample_pixel_values():
    im = np.zeros((15, 20))
    row_ids, col_ids, sample_values = image_utils.grid_sample_pixel_values(
        im,
        grid_spacing=5,
    )
    nose.tools.assert_list_equal(row_ids.tolist(), [5, 5, 5, 10, 10, 10])
    nose.tools.assert_list_equal(col_ids.tolist(), [5, 10, 15, 5, 10, 15])
    nose.tools.assert_list_equal(sample_values.tolist(), [0, 0, 0, 0, 0, 0])


class TestImageUtils(unittest.TestCase):

    def setUp(self):
        """Set up a dictionary with images"""

        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        meta_fname = 'frames_meta.csv'
        self.frames_meta = aux_utils.make_dataframe()

        x = np.linspace(-4, 4, 32)
        y = x.copy()
        z = np.linspace(-3, 3, 8)
        xx, yy, zz = np.meshgrid(x, y, z)
        sph = (xx ** 2 + yy ** 2 + zz ** 2)
        sph = (sph <= 8) * (8 - sph)
        sph = (sph / sph.max()) * 255
        sph = sph.astype('uint8')
        self.sph = sph

        self.channel_idx = 1
        self.time_idx = 0
        self.pos_idx = 1
        self.int2str_len = 3

        for z in range(sph.shape[2]):
            im_name = aux_utils.get_im_name(
                channel_idx=self.channel_idx,
                slice_idx=z,
                time_idx=self.time_idx,
                pos_idx=self.pos_idx,
            )
            cv2.imwrite(os.path.join(self.temp_path, im_name), sph[:, :, z])
            meta_row = aux_utils.parse_idx_from_name(im_name=im_name, dir_name=self.temp_path)
            meta_row['zscore_median'] = np.nanmean(sph[:, :, z])
            meta_row['zscore_iqr'] = np.nanstd(sph[:, :, z])
            self.frames_meta = self.frames_meta.append(
                meta_row,
                ignore_index=True
            )
        self.dataset_mean = self.frames_meta['zscore_median'].median()
        self.dataset_std = self.frames_meta['zscore_iqr'].median()
        # Write metadata
        self.frames_meta.to_csv(os.path.join(self.temp_path, meta_fname), sep=',')
        # Get a meta row for testing
        meta_idx = aux_utils.get_meta_idx(
            frames_metadata=self.frames_meta,
            time_idx=self.time_idx,
            channel_idx=self.channel_idx,
            slice_idx=7,
            pos_idx=self.pos_idx)
        self.meta_row = self.frames_meta.loc[meta_idx]
        # Write sph data as zarr
        zarr_writer = io_utils.ZarrWriter(save_dir=self.temp_path)
        zarr_data = np.zeros((2, 1, 2, 8, 32, 32))
        # Fix dimensions to match to zarr format PTCZYX
        zarr_data[1, 0, 1, ...] = np.moveaxis(self.sph, -1, 0)
        for pos_idx in range(2):
            zarr_writer.create_zarr_root('test_sphere_pos{}'.format(pos_idx))
            zarr_writer.init_array(
                position=pos_idx,
                data_shape=(1, 2, 8, 32, 32),
                chunk_size=(1, 1, 1, 32, 32),
                chan_names=['channel1', 'channel2'],
                dtype=zarr_data.dtype,
            )
            zarr_writer.write(zarr_data[pos_idx, ...], p=pos_idx)

        # Write 3D sphere data as npy file
        self.sph_fname = os.path.join(
            self.temp_path,
            'im_c001_z000_t000_p001_3d.npy',
        )
        np.save(self.sph_fname, self.sph, allow_pickle=True, fix_imports=True)
        self.meta_3d = pd.DataFrame.from_dict([{
            'channel_idx': 1,
            'slice_idx': 0,
            'time_idx': 0,
            'channel_name': '3d_test',
            'dir_name': self.temp_path,
            'file_name': 'im_c001_z000_t000_p001_3d.npy',
            'pos_idx': 1,
            'zscore_median': np.nanmean(sph),
            'zscore_iqr': np.nanstd(sph)
        }])

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_read_image(self):
        file_path = os.path.join(
            self.temp_path,
            self.frames_meta['file_name'][0],
        )
        im = image_utils.read_image(file_path)
        np.testing.assert_array_equal(im, self.sph[..., 0])

    def test_read_image_npy(self):
        im = image_utils.read_image(self.sph_fname)
        np.testing.assert_array_equal(im, self.sph)

    def test_read_image_from_row(self):
        im = image_utils.read_image_from_row(self.meta_row)
        np.testing.assert_array_equal(im, self.sph[..., 7])

    def test_read_image_from_row_dir_name(self):
        im = image_utils.read_image_from_row(self.meta_row, self.temp_path)
        np.testing.assert_array_equal(im, self.sph[..., 7])

    def test_read_image_from_row_zarr(self):
        meta_row = self.meta_row.copy()
        meta_row['file_name'] = 'test_sphere_pos{}.zarr'.format(self.pos_idx)
        im = image_utils.read_image_from_row(
            meta_row=meta_row,
        )
        np.testing.assert_array_equal(im, self.sph[..., 7])

    def test_preprocess_image(self):
        im = np.zeros((5, 10, 15, 1))
        im[:, :5, :, :] = 10
        im_proc = image_utils.preprocess_image(
            im,
            hist_clip_limits=(0, 100),
        )
        self.assertEqual(np.mean(im), np.mean(im_proc))
        self.assertTupleEqual(im_proc.shape, (5, 10, 15))

    def test_preprocess_image_norm(self):
        im = np.zeros((5, 10, 15))
        im[:, :5, :] = 10
        im_proc = image_utils.preprocess_image(
            im,
            normalize_im='dataset',
        )
        self.assertEqual(np.mean(im_proc), 0.0)
        self.assertTupleEqual(im.shape, im_proc.shape)

    def test_preprocess_image_mask(self):
        im = np.zeros((5, 10, 15))
        im[:, :5, :] = 10
        im_proc = image_utils.preprocess_image(
            im,
            is_mask=True,
        )
        self.assertEqual(np.mean(im_proc), 0.5)
        self.assertTupleEqual(im.shape, im_proc.shape)
        self.assertTrue(im_proc.dtype == bool)

    def test_read_imstack_from_meta(self):
        im_stack = image_utils.read_imstack_from_meta(
            frames_meta_sub=self.frames_meta,
            dir_name=self.temp_path,
            hist_clip_limits=(0, 100),
        )
        np.testing.assert_array_equal(im_stack, self.sph)

    def test_read_imstack_from_meta_one_im(self):
        im_stack = image_utils.read_imstack_from_meta(
            frames_meta_sub=self.meta_row,
            hist_clip_limits=(0, 100),
        )
        np.testing.assert_array_equal(np.squeeze(im_stack), self.sph[..., 7])

    def test_read_imstack(self):
        fnames = self.frames_meta['file_name'][:3]
        fnames = [os.path.join(self.temp_path, fname) for fname in fnames]
        # non-boolean
        im_stack = image_utils.read_imstack(
            input_fnames=fnames,
            normalize_im=True,
            zscore_mean=self.dataset_mean,
            zscore_std=self.dataset_std,
        )
        exp_stack = normalize.zscore(
            self.sph[:, :, :3],
            im_mean=self.dataset_mean,
            im_std=self.dataset_std,
        )
        np.testing.assert_equal(im_stack.shape, (32, 32, 3))
        np.testing.assert_array_equal(
            exp_stack[:, :, :3],
            im_stack,
        )
        # read a 3D image
        im_stack = image_utils.read_imstack([self.sph_fname])
        np.testing.assert_equal(im_stack.shape, (32, 32, 8))

        # read multiple 3D images
        im_stack = image_utils.read_imstack((self.sph_fname, self.sph_fname))
        np.testing.assert_equal(im_stack.shape, (32, 32, 8, 2))

    def test_preprocess_imstack(self):
        """Test preprocess_imstack"""
        im_stack = image_utils.preprocess_imstack(
            frames_metadata=self.frames_meta,
            depth=3,
            time_idx=self.time_idx,
            channel_idx=self.channel_idx,
            slice_idx=2,
            pos_idx=self.pos_idx,
            dir_name=self.temp_path,
            normalize_im='dataset',
        )
        np.testing.assert_equal(im_stack.shape, (32, 32, 3))
        exp_stack = np.zeros((32, 32, 3))
        # Right now the code normalizes on a z slice basis for all
        # normalization schemes
        for z in range(exp_stack.shape[2]):
            exp_stack[..., z] = normalize.zscore(self.sph[..., z + 1])
        np.testing.assert_array_equal(im_stack, exp_stack)

    def test_preprocess_imstack_3d(self):
        # preprocess a 3D image
        im_stack = image_utils.preprocess_imstack(
            frames_metadata=self.meta_3d,
            depth=1,
            time_idx=0,
            channel_idx=1,
            slice_idx=0,
            pos_idx=1,
            normalize_im='dataset',
        )
        np.testing.assert_equal(im_stack.shape, (32, 32, 8))
        # Normalization for 3D image is done on the entire volume
        exp_stack = normalize.zscore(
            self.sph,
            im_mean=np.nanmean(self.sph),
            im_std=np.nanstd(self.sph),
        )
        np.testing.assert_array_equal(im_stack, exp_stack)
