import nose.tools
import numpy as np
import unittest

import micro_dl.inference.stitch_predictions as stitcher


class TestImageStitcher(unittest.TestCase):

    def setUp(self):
        """Create an instance of ImageStitcher"""

        self.overlap_dict_z = {'overlap_shape': 2,
                               'overlap_operation': 'mean'}

        self.stitch_inst_z = stitcher.ImageStitcher(
            tile_option='tile_z',  # [tile_z, tile_xyz]
            overlap_dict=self.overlap_dict_z,
            image_format='zyx',
            data_format='channels_first'
        )

        self.overlap_dict_zyx = {'overlap_shape': [2, 4, 4],
                                 'overlap_operation': 'mean'}

        self.stitch_inst_zyx = stitcher.ImageStitcher(
            tile_option='tile_xyz',
            overlap_dict=self.overlap_dict_zyx,
            image_format='zyx',
            data_format='channels_first'
        )

    def test_init(self):
        """Test init"""

        nose.tools.assert_equal(self.stitch_inst_z.tile_option,
                                'tile_z')
        nose.tools.assert_dict_equal(
            self.stitch_inst_z.overlap_dict,
            self.overlap_dict_z
        )
        nose.tools.assert_equal(self.stitch_inst_z.data_format,
                                'channels_first')
        nose.tools.assert_list_equal(self.stitch_inst_z.img_dim,
                                     [2, 3, 4])
        nose.tools.assert_equal(self.stitch_inst_z.x_dim, 4)
        nose.tools.assert_equal(self.stitch_inst_z.y_dim, 3)
        nose.tools.assert_equal(self.stitch_inst_z.z_dim, 2)
        nose.tools.assert_equal(self.stitch_inst_z.z_dim_3d, 0)

        nose.tools.assert_equal(self.stitch_inst_zyx.tile_option,
                                'tile_xyz')
        nose.tools.assert_dict_equal(
            self.stitch_inst_zyx.overlap_dict,
            self.overlap_dict_zyx
        )

    def test_place_block_z(self):
        """Test _place_block_z"""

        entire_pred_image = np.zeros((5, 32, 32))

        self.stitch_inst_z._place_block_z(pred_block=np.ones((3, 32, 32)),
                                          pred_image=entire_pred_image,
                                          start_idx=0,
                                          end_idx=3)
        np.testing.assert_array_equal(entire_pred_image[:3, :, :],
                                      np.ones((3, 32, 32)))

        self.stitch_inst_z._place_block_z(pred_block=2 * np.ones((3, 32, 32)),
                                          pred_image=entire_pred_image,
                                          start_idx=1,
                                          end_idx=4)
        # 0.67 * 1 + 0.33 * 2
        nose.tools.assert_equal(np.round(np.unique(entire_pred_image[1]), 2),
                                1.33)
        # 0.33 * 1 + 0.67 * 2
        nose.tools.assert_equal(np.round(np.unique(entire_pred_image[2]), 2),
                                1.67)

        self.stitch_inst_z._place_block_z(pred_block=3 * np.ones((3, 32, 32)),
                                          pred_image=entire_pred_image,
                                          start_idx=2,
                                          end_idx=5)
        # 0.67 * 1.67 + 0.33 * 3
        nose.tools.assert_equal(np.round(np.unique(entire_pred_image[2]), 2),
                                2.11)
        # 0.33 * 2 + 0.67 * 3
        nose.tools.assert_equal(np.round(np.unique(entire_pred_image[3]), 2),
                                2.67)
        nose.tools.assert_equal(np.round(np.unique(entire_pred_image[4]), 2),
                                3)

    def test_stitch_along_z(self):
        """Test _stitch_along_z"""

        tile_imgs_list = [np.ones((3, 32, 32)),
                          2 * np.ones((3, 32, 32)),
                          3 * np.ones((3, 32, 32))]
        block_indices_list = [(0, 3), (2, 5), (3, 6)]

        # place slices 0, 1, 2
        exp_stitched_img = np.ones((6, 32, 32))
        # wtd add slice 2, place slices 3, 4
        exp_stitched_img[2] = 0.5 * 1 + 0.5 * 2
        # wtd_add slice 3, place slices 4, 5
        exp_stitched_img[3] = 0.5 * 2 + 0.5 * 3
        exp_stitched_img[4] = 3
        exp_stitched_img[5] = 3

        self.stitch_inst_z.im_shape = (6, 32, 32)
        self.stitch_inst_z.overlap_dict['overlap_shape'] = 1
        stitched_img = self.stitch_inst_z._stitch_along_z(
            tile_imgs_list=tile_imgs_list,
            block_indices_list=block_indices_list
        )
        np.testing.assert_array_equal(stitched_img,
                                      exp_stitched_img)

        # add a check for place_operation any
        tile_imgs_list = [np.ones((3, 32, 32)),
                          np.ones((3, 32, 32)),
                          np.zeros((3, 32, 32))]
        block_indices_list = [(0, 3), (2, 5), (3, 6)]

        exp_stitched_img = np.ones((6, 32, 32))
        exp_stitched_img[-2:] = 0

        self.stitch_inst_z.overlap_dict['overlap_operation'] = 'any'
        stitched_img = self.stitch_inst_z._stitch_along_z(
            tile_imgs_list=tile_imgs_list,
            block_indices_list=block_indices_list
        )
        np.testing.assert_array_equal(stitched_img,
                                      exp_stitched_img)

    def test_place_block_xyz(self):
        """Test _place_block_xyz"""

        pred_block = np.ones((1, 1, 4, 16, 16))
        pred_image = np.zeros((6, 32, 32))

        self.stitch_inst_zyx._place_block_xyz(
            pred_block=pred_block,
            pred_image=pred_image,
            crop_index=[0, 4, 0, 16, 0, 16]
        )
        np.testing.assert_array_equal(pred_image[:4, :16, :16],
                                      np.ones((4, 16, 16)))

        self.stitch_inst_zyx._place_block_xyz(
            pred_block=2 * pred_block,
            pred_image=pred_image,
            crop_index=[0, 4, 12, 28, 12, 28]
        )
        exp_overlap_array = np.ones(
            self.stitch_inst_zyx.overlap_dict['overlap_shape']
        )
        # 4 pixel overlap: values = 1, 2. wts = 0.2, 0.4, 0.6, 0.8 along x, y
        exp_overlap_array[:, 0, :] = 1.2
        exp_overlap_array[:, 1, :] = 1.4
        exp_overlap_array[:, 2, :] = 1.6
        exp_overlap_array[:, 3, :] = 1.8
        np.testing.assert_array_equal(
            np.round(pred_image[2:4, 12:16, 12:16], 1),
            exp_overlap_array
        )

        self.stitch_inst_zyx._place_block_xyz(
            pred_block=3 * pred_block,
            pred_image=pred_image,
            crop_index=[0, 4, 16, 32, 16, 32]
        )
        exp_overlap_array = np.ones(
            self.stitch_inst_zyx.overlap_dict['overlap_shape']
        )
        # 4 pixel overlap: values = 2, 3. wts = 0.2, 0.4, 0.6, 0.8 along x, y
        exp_overlap_array[:, 0, :] = 2.2
        exp_overlap_array[:, 1, :] = 2.4
        exp_overlap_array[:, 2, :] = 2.6
        exp_overlap_array[:, 3, :] = 2.8
        np.testing.assert_array_equal(
            np.round(pred_image[2:4, 16:20, 16:20], 1),
            exp_overlap_array
        )

        self.stitch_inst_zyx._place_block_xyz(
            pred_block=pred_block,
            pred_image=pred_image,
            crop_index=[2, 6, 0, 16, 0, 16]
        )
        np.testing.assert_array_equal(
            np.round(pred_image[2:4, 16:20, 16:20], 1),
            exp_overlap_array
        )
        # 1.2 * 0.67 + 1 * 0.33, 1.4 * 0.67 + 1 * 0.33,
        # 1.6 * 0.67 + 1 * 0.33, 1.8 * 0.67 + 1 * 0.33
        np.testing.assert_array_equal(
            np.round(np.unique(pred_image[2, 12:16, 12:16]), 2),
            [1.13, 1.27, 1.40, 1.53]
        )
        # 1.2 * 0.33 + 1 * 0.67, 1.4 * 0.33 + 1 * 0.67,
        # 1.6 * 0.33 + 1 * 0.67, 1.8 * 0.33 + 1 * 0.67
        np.testing.assert_array_equal(
            np.round(np.unique(pred_image[3, 12:16, 12:16]), 2),
            [1.07, 1.13, 1.20, 1.27]
        )

    def test_stitch_along_xyz(self):
        """Test _stitch_along_xyz"""

        tile_imgs_list = [np.ones((1, 1, 3, 6, 6)),
                          2 * np.ones((1, 1, 3, 6, 6)),
                          np.ones((1, 1, 3, 6, 6)),
                          2 * np.ones((1, 1, 3, 6, 6))]
        block_indices_list = [(0, 3, 0, 6, 0, 6),
                              (0, 3, 0, 6, 4, 10),
                              (0, 3, 4, 10, 0, 6),
                              (0, 3, 4, 10, 4, 10)]

        self.stitch_inst_zyx.im_shape = (3, 10, 10)
        self.stitch_inst_zyx.overlap_dict['overlap_shape'] = [1, 2, 2]
        stitched_img = self.stitch_inst_zyx._stitch_along_xyz(
            tile_imgs_list=tile_imgs_list,
            block_indices_list=block_indices_list
        )
        stitched_img = np.squeeze(stitched_img)

        # the first slice is as is, no stitching
        exp_z0 = np.ones((10, 10))
        exp_z0[:, 4:] = 2
        np.testing.assert_array_equal(stitched_img[0, ...], exp_z0)

        # second slice, place tile 1. Tile 2: Mean along 2 overlapping cols
        # 4,5 and rows 2-5 [0.67*1 + 0.33*2, 0.33*1 + 0.67*2 = 1.33, 1.67].
        # Tile 3: Mean along 2 overlapping rows 4, 5:
        # 0.67 * 1.33 + 0.33 * 1, 0.67 * 1.67 + 0.33 * 1 = 1.22, 1.44
        # 0.33 * 1.33 + 0.67 * 1, 0.33 * 1.67 + 0.67 * 1 = 1.11, 1.22
        # Tile 4: Mean along 2 overlapping rows 4, 5:
        # 0.67 * 1.22 + 0.33 * 2, 0.67 * 1.44 + 0.33 * 2 = 1.48, 1.63
        # 0.33 * 1.11 + 0.67 * 2, 0.33 * 1.22 + 0.67 * 2 = 1.71, 1.74
        exp_z1 = np.ones((10, 10), dtype=np.float32)
        exp_z1[:, 4:] = 2
        exp_z1[2:, 4] = 1.33
        exp_z1[2:, 5] = 1.67
        exp_z1[4, 4] = 1.48
        exp_z1[4, 5] = 1.63
        exp_z1[5, 4] = 1.7
        exp_z1[5, 5] = 1.74
        np.testing.assert_array_equal(np.round(stitched_img[1], 2), exp_z1)

    def test_stitch_predictions(self):
        """Test stitch_predictions"""

        shape_3d = (3, 10, 10)

        tile_imgs_list = [np.ones((1, 1, 3, 6, 6)),
                          2 * np.ones((1, 1, 3, 6, 6)),
                          np.ones((1, 1, 3, 6, 6))]
                          #2 * np.ones((1, 1, 3, 6, 6))]
        block_indices_list = [(0, 3, 0, 6, 0, 6),
                              (0, 3, 0, 6, 4, 10),
                              (0, 3, 4, 10, 0, 6),
                              (0, 3, 4, 10, 4, 10)]

        nose.tools.assert_raises(AssertionError,
                                 self.stitch_inst_zyx.stitch_predictions,
                                 shape_3d,
                                 tile_imgs_list,
                                 block_indices_list)
        nose.tools.assert_raises(AssertionError,
                                 self.stitch_inst_z.stitch_predictions,
                                 (10, 10),
                                 tile_imgs_list,
                                 block_indices_list)
