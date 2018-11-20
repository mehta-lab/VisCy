import cv2
import nose.tools
import numpy as np
import numpy.testing
import os
import pandas as pd
from testfixtures import TempDirectory
import unittest

import micro_dl.input.tile_images as tile_images
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.normalize as norm_util


class TestImageTiler(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory for tiling with flatfield, no mask
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        df_names = ["channel_idx",
                    "slice_idx",
                    "time_idx",
                    "channel_name",
                    "file_name",
                    "pos_idx"]
        frames_meta = pd.DataFrame(
            columns=df_names,
        )
        # Write images as bytes
        self.im = 350 * np.ones((15, 11), dtype=np.uint16)
        self.im2 = 8000 * np.ones((15, 11), dtype=np.uint16)
        res, im_encoded = cv2.imencode('.png', self.im)
        im_encoded = im_encoded.tostring()
        res, im2_encoded = cv2.imencode('.png', self.im2)
        im2_encoded = im2_encoded.tostring()
        self.channel_idx = 1
        self.time_idx = 5
        self.pos_idx1 = 7
        self.pos_idx2 = 8
        int2str_len = 3
        # Write test images with 4 z and 2 pos idx
        for z in range(15, 20):
            im_name = "im_c" + str(self.channel_idx).zfill(int2str_len) + \
            "_z" + str(z).zfill(int2str_len) + \
            "_t" + str(self.time_idx).zfill(int2str_len) + \
            "_p" + str(self.pos_idx1).zfill(int2str_len) + ".png"
            self.tempdir.write(im_name, im_encoded)
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_names),
                ignore_index=True,
            )
        for z in range(15, 20):
            im_name = "im_c" + str(self.channel_idx).zfill(int2str_len) + \
            "_z" + str(z).zfill(int2str_len) + \
            "_t" + str(self.time_idx).zfill(int2str_len) + \
            "_p" + str(self.pos_idx2).zfill(int2str_len) + ".png"
            self.tempdir.write(im_name, im2_encoded)
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_names),
                ignore_index=True,
            )
        # Write metadata
        frames_meta.to_csv(
            os.path.join(self.temp_path, self.meta_name),
            sep=',',
        )
        # Add flatfield
        self.flat_field_dir = os.path.join(self.temp_path, "ff_dir")
        self.tempdir.makedir('ff_dir')
        self.ff_im = 4. * np.ones((15, 11))
        np.save(os.path.join(self.flat_field_dir, 'flat-field_channel-1.npy'),
                self.ff_im,
                allow_pickle=True,
                fix_imports=True)
        # Instantiate tiler class
        self.output_dir = os.path.join(self.temp_path, "tile_dir")
        self.tile_dict = {
            'channels': [1],
            'tile_size': [5, 5],
            'step_size': [4, 4],
            'depths': 3,
            'data_format': 'channels_last',
        }
        self.tile_inst = tile_images.ImageTiler(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            tile_dict=self.tile_dict,
            flat_field_dir=self.flat_field_dir,
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Test image tiler on frames temporary dir
        """
        nose.tools.assert_equal(self.tile_inst.depths, 3)
        nose.tools.assert_equal(self.tile_inst.mask_depth, 1)
        nose.tools.assert_equal(self.tile_inst.tile_size, [5, 5])
        nose.tools.assert_equal(self.tile_inst.step_size, [4, 4])
        nose.tools.assert_false(self.tile_inst.isotropic)
        nose.tools.assert_equal(self.tile_inst.hist_clip_limits, None)
        nose.tools.assert_equal(self.tile_inst.data_format, 'channels_last')
        nose.tools.assert_equal(
            self.tile_inst.str_tile_step,
            'tiles_5-5_step_4-4',
        )
        nose.tools.assert_equal(self.tile_inst.channel_ids, [self.channel_idx])
        nose.tools.assert_equal(self.tile_inst.time_ids, [self.time_idx])
        nose.tools.assert_equal(
            self.tile_inst.flat_field_dir,
            self.flat_field_dir,
        )
        # Depth is 3 so first and last frame will not be used
        numpy.testing.assert_array_equal(
            self.tile_inst.slice_ids,
            np.asarray([16, 17, 18]),
        )
        numpy.testing.assert_array_equal(
            self.tile_inst.pos_ids,
            np.asarray([7, 8]),
        )

        # channel_depth should be a dict containing depths for each channel
        self.assertListEqual(
            list(self.tile_inst.channel_depth),
            [self.channel_idx],
        )
        nose.tools.assert_equal(
            self.tile_inst.channel_depth[self.channel_idx],
            3,
        )

    def test_tile_dir(self):
        nose.tools.assert_equal(self.tile_inst.get_tile_dir(),
                                os.path.join(self.output_dir,
                                             "tiles_5-5_step_4-4"))

    def test_tile_mask_dir(self):
        nose.tools.assert_equal(self.tile_inst.get_tile_mask_dir(), None)

    def test_write_tiled_data(self):
        tiled_data = [('r0-5_c0-5_sl0-3', np.zeros((5, 5, 3), dtype=np.float)),
                      ('r4-9_c0-5_sl0-3', np.ones((5, 5, 3), dtype=np.float))]
        tiled_metadata = self.tile_inst._get_dataframe()
        tile_indices = [(0, 5, 0, 5), (4, 9, 0, 5)]
        tile_dir = self.tile_inst.get_tile_dir()

        out_metadata = self.tile_inst._write_tiled_data(
            tiled_data=tiled_data,
            save_dir=tile_dir,
            time_idx=self.time_idx,
            channel_idx=self.channel_idx,
            slice_idx=17,
            pos_idx=self.pos_idx2,
            tile_indices=tile_indices,
            tiled_metadata=tiled_metadata,
        )

        self.assertListEqual(
            out_metadata.channel_idx.tolist(),
            [self.channel_idx] * 2,
        )
        self.assertListEqual(
            out_metadata.slice_idx.tolist(),
            [17] * 2,
        )
        self.assertListEqual(
            out_metadata.time_idx.tolist(),
            [self.time_idx] * 2,
        )
        self.assertListEqual(
            out_metadata.pos_idx.tolist(),
            [self.pos_idx2] * 2,
        )
        self.assertListEqual(
            out_metadata.row_start.tolist(),
            [0, 4],
        )
        self.assertListEqual(
            out_metadata.col_start.tolist(),
            [0, 0],
        )
        self.assertListEqual(
            out_metadata.file_name.tolist(),
            ['im_c001_z017_t005_p008_r0-5_c0-5_sl0-3.npy',
             'im_c001_z017_t005_p008_r4-9_c0-5_sl0-3.npy'],
        )
        # Load and assert tiles
        tile = np.load(
            os.path.join(tile_dir,
            'im_c001_z017_t005_p008_r0-5_c0-5_sl0-3.npy'),
        )
        numpy.testing.assert_array_equal(tile, tiled_data[0][1])
        tile = np.load(
            os.path.join(tile_dir,
            'im_c001_z017_t005_p008_r4-9_c0-5_sl0-3.npy'),
        )
        numpy.testing.assert_array_equal(tile, tiled_data[1][1])

    def test_get_flat_field(self):
        flat_field_im = self.tile_inst._get_flat_field(channel_idx=1)
        numpy.testing.assert_array_equal(flat_field_im, self.ff_im)

    def test_get_dataframe(self):
        df = self.tile_inst._get_dataframe()
        self.assertListEqual(
            list(df),
            ["channel_idx",
             "slice_idx",
             "time_idx",
             "file_name",
             "pos_idx",
             "row_start",
             "col_start"])

    def test_tile_stack(self):
        self.tile_inst.tile_stack()
        # Read and validate the saved metadata
        tile_dir = self.tile_inst.get_tile_dir()
        frames_meta = pd.read_csv(os.path.join(tile_dir, "frames_meta.csv"))

        self.assertSetEqual(set(frames_meta.channel_idx.tolist()), {1})
        self.assertSetEqual(set(frames_meta.slice_idx.tolist()), {16, 17, 18})
        self.assertSetEqual(set(frames_meta.time_idx.tolist()), {5})
        self.assertSetEqual(set(frames_meta.pos_idx.tolist()), {7, 8})
        # 15 rows and step size 4, so it can take 3 full steps and 1 short step
        self.assertSetEqual(set(frames_meta.row_start.tolist()), {0, 4, 8, 10})
        # 11 cols and step size 4, so it can take 2 full steps and 1 short step
        self.assertSetEqual(set(frames_meta.col_start.tolist()), {0, 4, 6})

        # Read and validate tiles
        im_val = np.mean(norm_util.zscore(self.im / self.ff_im))
        im_norm = im_val * np.ones((5, 5, 3))
        im_val = np.mean(norm_util.zscore(self.im2 / self.ff_im))
        im2_norm = im_val * np.ones((5, 5, 3))
        for i, row in frames_meta.iterrows():
            tile = np.load(os.path.join(tile_dir, row.file_name))
            if row.pos_idx == 7:
                numpy.testing.assert_array_equal(tile, im_norm)
            else:
                numpy.testing.assert_array_equal(tile, im2_norm)


class TestImageMaskTiler(unittest.TestCase):

    def setUp(self):
        """
        Set up a directory for tiling with masks, no flatfield.
        """
        self.tempdir = TempDirectory()
        self.temp_path = self.tempdir.path
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        df_names = ["channel_idx",
                    "slice_idx",
                    "time_idx",
                    "channel_name",
                    "file_name",
                    "pos_idx"]
        frames_meta = pd.DataFrame(
            columns=df_names,
        )
        # Write images as bytes
        self.im = 350 * np.ones((15, 11), dtype=np.uint16)
        self.im2 = 8000 * np.ones((15, 11), dtype=np.uint16)
        res, im_encoded = cv2.imencode('.png', self.im)
        im_encoded = im_encoded.tostring()
        res, im2_encoded = cv2.imencode('.png', self.im2)
        im2_encoded = im2_encoded.tostring()
        self.channel_idx1 = 1
        self.channel_idx2 = 2
        self.time_idx = 5
        self.pos_idx = 7
        int2str_len = 3
        # Write test images with 4 z and 2 pos idx
        for z in range(15, 20):
            im_name = "im_c" + str(self.channel_idx1).zfill(int2str_len) + \
            "_z" + str(z).zfill(int2str_len) + \
            "_t" + str(self.time_idx).zfill(int2str_len) + \
            "_p" + str(self.pos_idx).zfill(int2str_len) + ".png"
            self.tempdir.write(im_name, im_encoded)
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_names),
                ignore_index=True,
            )
        for z in range(15, 20):
            im_name = "im_c" + str(self.channel_idx2).zfill(int2str_len) + \
            "_z" + str(z).zfill(int2str_len) + \
            "_t" + str(self.time_idx).zfill(int2str_len) + \
            "_p" + str(self.pos_idx).zfill(int2str_len) + ".png"
            self.tempdir.write(im_name, im2_encoded)
            frames_meta = frames_meta.append(
                aux_utils.get_ids_from_imname(im_name, df_names),
                ignore_index=True,
            )
        # Write metadata
        frames_meta.to_csv(
            os.path.join(self.temp_path, self.meta_name),
            sep=',',
        )
        # Write masks
        self.mask_dir = os.path.join(self.temp_path, "mask_dir")
        self.tempdir.makedir('mask_dir')
        self.mask_im = np.zeros((15, 11))
        self.mask_im[5:10, 2:5] = 1
        self.mask_channel = 3
        for z in range(15, 20):
            im_name = "im_c" + str(self.mask_channel).zfill(int2str_len) + \
                "_z" + str(z).zfill(int2str_len) + \
                "_t" + str(self.time_idx).zfill(int2str_len) + \
                "_p" + str(self.pos_idx).zfill(int2str_len) + ".npy"
            np.save(os.path.join(self.mask_dir, im_name),
                self.mask_im,
                allow_pickle=True,
                fix_imports=True)
        # Instantiate tiler class
        self.output_dir = os.path.join(self.temp_path, "tile_dir")
        self.tile_dict = {
            'tile_size': [5, 5],
            'step_size': [5, 5],
            'channels': [1, 2],
            'depths': [3, 1],
            'data_format': 'channels_first',
        }
        self.tile_inst = tile_images.ImageTiler(
            input_dir=self.temp_path,
            output_dir=self.output_dir,
            tile_dict=self.tile_dict,
            mask_depth=1,
        )

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        nose.tools.assert_equal(os.path.isdir(self.temp_path), False)

    def test_init(self):
        """
        Test image tiler on frames temporary dir
        """
        self.assertListEqual(self.tile_inst.depths, [3, 1])
        nose.tools.assert_equal(self.tile_inst.mask_depth, 1)
        nose.tools.assert_equal(self.tile_inst.tile_size, [5, 5])
        nose.tools.assert_equal(self.tile_inst.step_size, [5, 5])
        nose.tools.assert_false(self.tile_inst.isotropic)
        nose.tools.assert_equal(self.tile_inst.hist_clip_limits, None)
        nose.tools.assert_equal(self.tile_inst.data_format, 'channels_first')
        nose.tools.assert_equal(
            self.tile_inst.str_tile_step,
            'tiles_5-5_step_5-5',
        )
        nose.tools.assert_equal(
            self.tile_inst.channel_ids,
            [self.channel_idx1, self.channel_idx2],
        )
        nose.tools.assert_equal(self.tile_inst.time_ids, [self.time_idx])
        # Depth is 3 so first and last frame will not be used
        numpy.testing.assert_array_equal(
            self.tile_inst.slice_ids,
            np.asarray([16, 17, 18]),
        )
        numpy.testing.assert_array_equal(
            self.tile_inst.pos_ids,
            np.asarray([self.pos_idx]),
        )

        # channel_depth should be a dict containing depths for each channel
        self.assertListEqual(
            list(self.tile_inst.channel_depth),
            [self.channel_idx1, self.channel_idx2],
        )
        nose.tools.assert_equal(
            self.tile_inst.channel_depth[self.channel_idx1],
            3,
        )
        nose.tools.assert_equal(
            self.tile_inst.channel_depth[self.channel_idx2],
            1,
        )

    def test_tile_dir(self):
        nose.tools.assert_equal(
            self.tile_inst.get_tile_dir(),
            os.path.join(self.output_dir, "tiles_5-5_step_5-5"),
        )

    def test_write_tiled_data(self):
        tiled_data = [('r0-5_c0-5_sl0-3', np.zeros((5, 5, 3), dtype=np.float)),
                      ('r5-10_c0-5_sl0-3', np.ones((5, 5, 3), dtype=np.float))]
        tiled_metadata = self.tile_inst._get_dataframe()
        tile_indices = [(0, 5, 0, 5), (4, 9, 0, 5)]
        tile_dir = self.tile_inst.get_tile_dir()

        out_metadata = self.tile_inst._write_tiled_data(
            tiled_data=tiled_data,
            save_dir=tile_dir,
            time_idx=self.time_idx,
            channel_idx=self.channel_idx1,
            slice_idx=17,
            pos_idx=self.pos_idx,
            tile_indices=tile_indices,
            tiled_metadata=tiled_metadata,
        )

        self.assertListEqual(
            out_metadata.channel_idx.tolist(),
            [self.channel_idx1] * 2,
        )
        self.assertListEqual(
            out_metadata.slice_idx.tolist(),
            [17] * 2,
        )
        self.assertListEqual(
            out_metadata.time_idx.tolist(),
            [self.time_idx] * 2,
        )
        self.assertListEqual(
            out_metadata.pos_idx.tolist(),
            [self.pos_idx] * 2,
        )
        self.assertListEqual(
            out_metadata.row_start.tolist(),
            [0, 4],
        )
        self.assertListEqual(
            out_metadata.col_start.tolist(),
            [0, 0],
        )
        self.assertListEqual(
            out_metadata.file_name.tolist(),
            ['im_c001_z017_t005_p007_r0-5_c0-5_sl0-3.npy',
             'im_c001_z017_t005_p007_r5-10_c0-5_sl0-3.npy'],
        )
        # Load and assert tiles
        tile = np.load(
            os.path.join(tile_dir,
            'im_c001_z017_t005_p007_r0-5_c0-5_sl0-3.npy'),
        )
        # Flip because we did channels_first
        numpy.testing.assert_array_equal(
            tile,
            np.swapaxes(tiled_data[0][1], 0, 2),
        )
        tile = np.load(
            os.path.join(tile_dir,
            'im_c001_z017_t005_p007_r5-10_c0-5_sl0-3.npy'),
        )
        numpy.testing.assert_array_equal(
            tile,
            np.swapaxes(tiled_data[1][1], 0, 2),
        )

    def test_get_flat_field(self):
        flat_field_im = self.tile_inst._get_flat_field(channel_idx=1)
        # We never specified a flatfield dir
        nose.tools.assert_equal(flat_field_im, None)

    def test_tile_mask_stack(self):
        import glob
        print(glob.glob(self.mask_dir + '/*'))
        self.tile_inst.tile_mask_stack(
            min_fraction=0.0,
            mask_dir=self.mask_dir,
            mask_channel=self.mask_channel,
            save_tiled_masks='as_channel',
            )
        # Read and validate the saved metadata
        tile_dir = self.tile_inst.get_tile_dir()
        frames_meta = pd.read_csv(os.path.join(tile_dir, "frames_meta.csv"))

        self.assertSetEqual(
            set(frames_meta.channel_idx.tolist()),
            {self.channel_idx1, self.channel_idx2, self.mask_channel})
        self.assertSetEqual(set(frames_meta.slice_idx.tolist()), {16, 17, 18})
        self.assertSetEqual(set(frames_meta.time_idx.tolist()), {5})
        self.assertSetEqual(set(frames_meta.pos_idx.tolist()), {self.pos_idx})
        # 15 rows and step size 5, so it can take 3 full steps
        self.assertSetEqual(set(frames_meta.row_start.tolist()), {0, 5, 10})
        # 11 cols and step size 5, so it can take 2 full steps and 1 short step
        self.assertSetEqual(set(frames_meta.col_start.tolist()), {0, 5, 6})

        # Read and validate tiles
        im_val = np.mean(norm_util.zscore(self.im))
        im_norm = im_val * np.ones((3, 5, 5))
        im_val = np.mean(norm_util.zscore(self.im2))
        im2_norm = im_val * np.ones((1, 5, 5))
        for i, row in frames_meta.iterrows():
            tile = np.load(os.path.join(tile_dir, row.file_name))
            # Check mask values only
            if row.channel_idx == self.mask_channel:
                nose.tools.assert_true(tile.max() <= 1)
                nose.tools.assert_true(tile.min() >= 0)
                self.assertTupleEqual(tile.shape, (1, 5, 5))
            else:
                if row.channel_idx == self.channel_idx1:
                    numpy.testing.assert_array_equal(tile, im_norm)
                else:
                    numpy.testing.assert_array_equal(tile, im2_norm)
