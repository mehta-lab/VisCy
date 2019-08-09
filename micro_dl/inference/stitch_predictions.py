"""Stich model predictions either along Z or as tiles"""
import numpy as np


class ImageStitcher:
    """Stitch prediction images for 3D

    USE PREDICT ON LARGER IMAGE FOR 2D AND 2.5D
    """

    def __init__(self, tile_option,
                 overlap_dict,
                 image_format='zyx',
                 data_format='channels_first'):
        """Init

        :param str tile_option: 'tile_z' or 'tile_xyz'
        :param dict overlap_dict: with keys overlap_shape, overlap_operation.
         overlap_shape is an int for tile_z and list of len 3 for tile_xyz.
        :param str image_format: xyz or zyx
        :param str data_format: channels_first or channels_last
        """

        assert tile_option in ['tile_z', 'tile_xyz'], \
            'tile_option not in [tile_z, tile_xyz]'

        allowed_overlap_opn = ['mean', 'any']
        assert ('overlap_operation' in overlap_dict and
                overlap_dict['overlap_operation'] in allowed_overlap_opn), \
            'overlap_operation not provided or not in [mean, any]'
        assert image_format in ['zyx', 'xyz'], 'image_format not in [zyx, xyz]'

        self.tile_option = tile_option
        self.overlap_dict = overlap_dict
        self.data_format = data_format

        img_dim = [2, 3, 4] if self.data_format == 'channels_first' \
            else [1, 2, 3]
        self.img_dim = img_dim
        if data_format == 'channels_first':
            x_dim = 4 if image_format == 'zyx' else 2
            z_dim = 2 if image_format == 'zyx' else 4
            y_dim = 3
        elif data_format == 'channels_last':
            x_dim = 3 if image_format == 'zyx' else 1
            z_dim = 1 if image_format == 'zyx' else 3
            y_dim = 2

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        z_dim_3d = 0 if image_format == 'zyx' else 2
        self.z_dim_3d = z_dim_3d
        self.image_format = image_format

    def _place_block_z(self,
                       pred_block,
                       pred_image,
                       start_idx,
                       end_idx):
        """Place the current block prediction in the larger vol

        pred_image mutated in-place

        :param np.array pred_block: current prediction block
        :param np.array pred_image: full 3D prediction image with zeros
        :param int start_idx: start slice of pred_block
        :param int end_idx: end slice of pred block
        """

        num_overlap = self.overlap_dict['overlap_shape']
        overlap_operation = self.overlap_dict['overlap_operation']
        z_dim = self.z_dim_3d

        # smoothly weight the two images in overlapping slices
        forward_wts = np.linspace(0, 1.0, num_overlap + 2)[1:-1]
        reverse_wts = forward_wts[::-1]
        # initialize all indices to :
        idx_in_img = []
        idx_in_block = []
        for dim_idx in range(len(pred_image.shape)):
            idx_in_img.append(np.s_[:])
            idx_in_block.append(np.s_[:])
        idx_in_img[z_dim] = np.s_[start_idx + num_overlap: end_idx]
        idx_in_block[z_dim] = np.s_[num_overlap:]

        pred_image[tuple(idx_in_img)] = pred_block[tuple(idx_in_block)]
        if start_idx > 0:
            for sl_idx in range(num_overlap):
                idx_in_img[z_dim] = start_idx + sl_idx
                idx_in_block[z_dim] = sl_idx
                if overlap_operation == 'mean':
                    pred_image[tuple(idx_in_img)] = \
                        (reverse_wts[sl_idx] * pred_image[tuple(idx_in_img)] +
                         forward_wts[sl_idx] * pred_block[tuple(idx_in_block)])
                elif overlap_operation == 'any':
                    pred_image[tuple(idx_in_img)] = np.any(
                        [pred_image[tuple(idx_in_img)],
                         pred_block[tuple(idx_in_block)]]
                    )
        else:
            idx_in_img[z_dim] = np.s_[start_idx: start_idx + num_overlap]
            idx_in_block[z_dim] = np.s_[0: num_overlap]
            pred_image[tuple(idx_in_img)] = pred_block[tuple(idx_in_block)]
        return

    def _stitch_along_z(self,
                        tile_imgs_list,
                        block_indices_list):
        """Stitch images along Z with or w/o overlap

        Tile predictions and the stitched img are in 3D

        :param list tile_imgs_list: list with predicted tensors
        :param list block_indices_list: list with tuples of (start, end) idx
        :return np.array stitched_img: 3D image with blocks assembled in place
        """

        stitched_img = np.zeros(self.shape_3d)

        if 'overlap_shape' in self.overlap_dict:
            assert isinstance(self.overlap_dict['overlap_shape'], int), \
                'tile_z only supports an overlap of int slices along z'

        for idx, sub_block in enumerate(tile_imgs_list):
            try:
                cur_sl_idx = block_indices_list[idx]
                self._place_block_z(pred_block=sub_block,
                                    pred_image=stitched_img,
                                    start_idx=cur_sl_idx[0],
                                    end_idx=cur_sl_idx[1])
            except Exception as e:
                raise Exception('error in _stitch_along_z:{}'.format(e))
        return stitched_img

    def _place_block_xyz(self,
                         pred_block,
                         pred_image,
                         crop_index):
        """Place the current block prediction in the larger vol

        pred_image mutated in-place. Tile predictions in 5D and stitched img
        in 3D

        :param np.array pred_block: current prediction block
        :param np.array pred_image: full 3D prediction image with zeros
        :param list crop_index: tuple of len 6 with start, end indices for
         three dimensions
        """

        overlap_shape = self.overlap_dict['overlap_shape']
        overlap_operation = self.overlap_dict['overlap_operation']

        def _init_block_img_idx(task='init'):
            """Helper function to initialize slicing indices

            :param str task: init - initialize with entire range; assign -
             initialize with non-overlapping indices
            """
            # initialize all indices to :
            idx_in_img = []  # 3D
            idx_in_block = []  # 5D

            for dim_idx in range(len(pred_block.shape)):
                idx_in_block.append(np.s_[:])
                if dim_idx < len(pred_image.shape):
                    idx_in_img.append(np.s_[:])
            if task == 'assign':
                for idx_3D, idx_5D in enumerate(self.img_dim):
                    idx_in_img[idx_3D] = np.s_[crop_index[idx_3D * 2]:
                                               crop_index[idx_3D * 2 + 1]]
            return idx_in_block, idx_in_img

        idx_in_block, idx_in_img = _init_block_img_idx()
        # assign non-overlapping regions
        for idx_3D, idx_5D in enumerate(self.img_dim):
            idx_in_img[idx_3D] = np.s_[crop_index[idx_3D * 2] +
                                       overlap_shape[idx_3D]:
                                       crop_index[idx_3D * 2 + 1]]
            idx_in_block[idx_5D] = np.s_[overlap_shape[idx_3D]:]
        pred_image[tuple(idx_in_img)] = pred_block[tuple(idx_in_block)]

        if self.image_format == 'zyx':
            overlap_dim = [self.z_dim, self.y_dim, self.x_dim]
        else:  # 'xyz'
            overlap_dim = [self.x_dim, self.y_dim, self.z_dim]
        idx_in_block, idx_in_img = _init_block_img_idx(task='assign')

        for idx_3d, idx_5d in enumerate(overlap_dim):  # dim_idx, cur_dim
            # 0 - zdim (front), 1 - ydim (top), 2 - xdim (left) if zyx
            forward_wts = np.linspace(0, 1.0, overlap_shape[idx_3d] + 2)[1:-1]
            reverse_wts = forward_wts[::-1]
            if crop_index[2 * idx_3d] > 0:
                for idx in range(overlap_shape[idx_3d]):
                    idx_in_block[idx_5d] = idx
                    idx_in_img[idx_3d] = crop_index[2 * idx_3d] + idx
                    if overlap_operation == 'mean':
                        # smoothly weight the two images in overlapping slices
                        pred_image[tuple(idx_in_img)] = (
                            reverse_wts[idx] * pred_image[tuple(idx_in_img)] +
                            forward_wts[idx] * pred_block[tuple(idx_in_block)]
                        )
                    elif overlap_operation == 'any':
                        pred_block[idx_in_img] = np.any(
                            pred_image[idx_in_img], pred_block[idx_in_block]
                        )
            else:
                idx_in_img[idx_3d] = (
                    np.s_[crop_index[2 * idx_3d]:
                          crop_index[2 * idx_3d] + overlap_shape[idx_3d]]
                )
                idx_in_block[idx_5d] = np.s_[:overlap_shape[idx_3d]]
                pred_image[tuple(idx_in_img)] = pred_block[tuple(idx_in_block)]
            idx_in_img[idx_3d] = np.s_[crop_index[2 * idx_3d] +
                                       overlap_shape[idx_3d]:
                                       crop_index[2 * idx_3d + 1]]
            idx_in_block[idx_5d] = np.s_[overlap_shape[idx_3d]:]
        return

    def _stitch_along_xyz(self,
                          tile_imgs_list,
                          block_indices_list):
        """Stitch images along XYZ with overlap

        :param list tile_imgs_list: list with predicted tensors
        :param list block_indices_list: list with tuples of (start, end) idx
        :return np.array stitched_img: 3D image with blocks assembled in place
        """

        stitched_img = np.zeros(self.shape_3d)
        assert self.data_format is not None, \
            'data format needed for stitching images along xyz'
        for idx, cur_tile in enumerate(tile_imgs_list):
            try:
                cur_crop_idx = block_indices_list[idx]
                self._place_block_xyz(pred_block=cur_tile,
                                      pred_image=stitched_img,
                                      crop_index=cur_crop_idx)
            except Exception as e:
                raise Exception('error in _stitch_along_xyz:{}'.format(e))
        return stitched_img

    def stitch_predictions(self, shape_3d,
                           tile_imgs_list,
                           block_indices_list):
        """Stitch the predicted tiles /blocks for a 3d image

        :param list shape_3d: shape of  3d image
        :param list tile_imgs_list: list of prediction images
        :param list block_indices_list: list of tuple/lists with indices for
         each prediction. Individual list of: len=2 when tile_z (start_slice,
         end_slice), len=6 for tile_xyz with start and end indices for each
         dimension
        :return np.array stitched_img: tile_imgs_list stitched into a 3D image
        """

        assert len(tile_imgs_list) == len(block_indices_list), \
            'missing tile/indices for sub tile/block: {}, {}'.format(
                len(tile_imgs_list), len(block_indices_list)
            )
        assert len(shape_3d) == 3, \
            'only stitching 3D volume is currently supported'
        self.shape_3d = shape_3d

        if self.tile_option == 'tile_z':
            stitched_img = self._stitch_along_z(tile_imgs_list,
                                                block_indices_list)
        elif self.tile_option == 'tile_xyz':
            stitched_img = self._stitch_along_xyz(tile_imgs_list,
                                                  block_indices_list)
        return stitched_img

