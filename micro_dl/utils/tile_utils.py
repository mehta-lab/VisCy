import os

import numpy as np
import pandas as pd

from micro_dl.utils import normalize as normalize, aux_utils as aux_utils
from micro_dl.utils.image_utils import read_image, apply_flat_field_correction


def read_imstack(input_fnames,
                 flat_field_fname=None,
                 hist_clip_limits=None,
                 is_mask=False,
                 normalize_im=True):
    """
    Read the images in the fnames and assembles a stack.
    If images are masks, make sure they're boolean by setting >0 to True

    :param tuple input_fnames: tuple of input fnames with full path
    :param str flat_field_fname: fname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :param bool is_mask: Indicator for if files contain masks
    :param bool normalize_im: Whether to zscore normalize im stack
    :return np.array: input stack flat_field correct and z-scored if regular
        images, booleans if they're masks
    """
    im_stack = []
    for idx, fname in enumerate(input_fnames):
        im = read_image(fname)
        if flat_field_fname is not None:
            # multiple flat field images are passed in case of mask generation
            if isinstance(flat_field_fname, (list, tuple)):
                flat_field_image = np.load(flat_field_fname[idx])
            else:
                flat_field_image = np.load(flat_field_fname)
            if not is_mask and not normalize_im:
                im = apply_flat_field_correction(
                    im,
                    flat_field_image=flat_field_image,
                )
        im_stack.append(im)

    input_image = np.stack(im_stack, axis=-1)
    # remove singular dimension for 3D images
    if len(input_image.shape) > 3:
        input_image = np.squeeze(input_image)
    if not is_mask:
        if hist_clip_limits is not None:
            input_image = normalize.hist_clipping(
                input_image,
                hist_clip_limits[0],
                hist_clip_limits[1]
            )
        if normalize_im:
            input_image = normalize.zscore(input_image)
    else:
        if input_image.dtype != bool:
            input_image = input_image > 0
    return input_image


def preprocess_imstack(frames_metadata,
                       input_dir,
                       depth,
                       time_idx,
                       channel_idx,
                       slice_idx,
                       pos_idx,
                       flat_field_im=None,
                       hist_clip_limits=None,
                       normalize_im=False):
    """
    Preprocess image given by indices: flatfield correction, histogram
    clipping and z-score normalization is performed.

    :param pd.DataFrame frames_metadata: DF with meta info for all images
    :param str input_dir: dir containing input images
    :param int depth: num of slices in stack if 2.5D or depth for 3D
    :param int time_idx: Time index
    :param int channel_idx: Channel index
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param np.array flat_field_im: Flat field image for channel
    :param list hist_clip_limits: Limits for histogram clipping (size 2)
    :param bool normalize_im: indicator to normalize image based on z-score or not
    :return np.array im: 3D preprocessed image
    """
    margin = 0 if depth == 1 else depth // 2
    im_stack = []
    for z in range(slice_idx - margin, slice_idx + margin + 1):
        meta_idx = aux_utils.get_meta_idx(
            frames_metadata,
            time_idx,
            channel_idx,
            z,
            pos_idx,
        )
        file_path = os.path.join(
            input_dir,
            frames_metadata.loc[meta_idx, "file_name"],
        )
        im = read_image(file_path)
        # Only flatfield correct images that will be normalized
        if flat_field_im is not None and not normalize_im:
            im = apply_flat_field_correction(
                im,
                flat_field_image=flat_field_im,
            )
        im_stack.append(im)

    if len(im.shape) == 3:
        # each channel is tiled independently and stacked later in dataset cls
        im_stack = im
        assert depth == 1, 'more than one 3D volume gets read'
    else:
        # Stack images in same channel
        im_stack = np.stack(im_stack, axis=2)
    # normalize
    if hist_clip_limits is not None:
        im_stack = normalize.hist_clipping(
            im_stack,
            hist_clip_limits[0],
            hist_clip_limits[1],
        )
    if normalize_im:
        im_stack = normalize.zscore(im_stack)
    return im_stack


def tile_image(input_image,
               tile_size,
               step_size,
               return_index=False,
               min_fraction=None,
               save_dict=None):
    """
     Tiles the image based on given tile and step size.

     USE MIN_FRACTION WITH INPUT_IMAGE.DTYPE=bool / MASKS

    :param np.array input_image: input image to be tiled
    :param list/tuple/np array tile_size: size of the blocks to be tiled
     from the image
    :param list/tuple/np array step_size: size of the window shift. In case of
     no overlap, the step size is tile_size. If overlap, step_size < tile_size
    :param bool return_index: indicator for returning tile indices
    :param float min_fraction: Minimum fraction of foreground in mask for
    including tile
    :param dict save_dict: dict with keys: time_idx, channel_idx, slice_idx,
     pos_idx, image_format and save_dir for generation output fname
    :return: if not saving: a list with tuples of tiled image id of the format
     rrmin-rmax_ccmin-cmax_slslmin-slmax and tiled image
     Else: save tiles in-place and return a df with tile metadata
     if return_index=True: return a list with tuples of crop indices
    """

    def check_in_range(cur_value, max_value, tile_size):
        """Get the start index for edge tiles

        :param int cur_value: cur index in row / col / slice
        :param int max_value: n_rows, n_cols or n_slices
        :param int tile_size: size of tile along one dimension (row, col,
         slice)
        :return: int start_value - adjusted start_index to fit the edge tile
        """
        cur_length = max_value - cur_value
        miss_length = tile_size - cur_length
        start_value = cur_value - miss_length
        return start_value

    def use_tile(cropped_img, min_fraction):
        """
        Determine if tile should be used given minimum image foreground fraction
        :param np.array cropped_img: Image tile
        :param float min_fraction: Minimum fraction of image being foreground
        :return bool use_tile: Indicator if tile should be used
        """
        use_tile = True
        if min_fraction is not None:
            mask_fraction = np.mean(cropped_img)
            if mask_fraction < min_fraction:
                use_tile = False
        return use_tile

    def get_tile_meta(cropped_img, img_id, save_dict, row, col, sl_start=None):
        cur_metadata = None
        if save_dict is not None:
            file_name = write_tile(cropped_img, save_dict, img_id)
            cur_metadata = {'channel_idx': save_dict['channel_idx'],
                            'slice_idx': save_dict['slice_idx'],
                            'time_idx': save_dict['time_idx'],
                            'file_name': file_name,
                            'pos_idx': save_dict['pos_idx'],
                            'row_start': row,
                            'col_start': col}
            if sl_start is not None:
                cur_metadata['slice_start'] = sl_start
        return cur_metadata

    # Add to tile size and step size in case of 3D images
    im_shape = input_image.shape
    im_depth = im_shape[2]
    if len(im_shape) == 3:
        if len(tile_size) == 2:
            tile_size.append(im_shape[2])
            step_size.append(im_shape[2])
            tile_3d = False
        else:
            if step_size[2] == im_depth:
                tile_3d = False
            else:
                tile_3d = True

    assert len(tile_size) == len(step_size),\
        "Tile {} and step size {} mismatch".format(tile_size, step_size)
    assert np.all(step_size <= tile_size),\
        "Step size {} > tile size {}".format(step_size, tile_size)
    assert np.all(tile_size) > 0,\
        "Tile size must be > 0, not {}".format(tile_size)

    n_rows = im_shape[0]
    n_cols = im_shape[1]
    n_dim = len(input_image.shape)
    if n_dim == 3:
        n_slices = im_shape[2]

    cropped_image_list = []
    cropping_index = []
    tiled_metadata = []
    for row in range(0, n_rows - tile_size[0] + step_size[0], step_size[0]):
        if row + tile_size[0] > n_rows:
            row = check_in_range(row, n_rows, tile_size[0])
        for col in range(0, n_cols - tile_size[1] + step_size[1], step_size[1]):
            if col + tile_size[1] > n_cols:
                col = check_in_range(col, n_cols, tile_size[1])
            img_id = 'r{}-{}_c{}-{}'.format(row, row + tile_size[0],
                                            col, col + tile_size[1])

            cur_index = (row, row + tile_size[0], col, col + tile_size[1])
            cropped_img = input_image[row: row + tile_size[0],
                                      col: col + tile_size[1], ...]
            if n_dim == 3:
                if tile_3d:
                    for sl in range(0,
                                    n_slices - tile_size[2] + step_size[2],
                                    step_size[2]):
                        if sl + tile_size[2] > n_slices:
                            sl = check_in_range(sl, n_slices, tile_size[2])
                        cur_index = (row, row + tile_size[0],
                                     col, col + tile_size[1],
                                     sl, sl + tile_size[2])
                        cur_img_id = '{}_sl{}-{}'.format(img_id, sl,
                                                         sl + tile_size[2])
                        cropped_img = input_image[row: row + tile_size[0],
                                                  col: col + tile_size[1],
                                                  sl: sl + tile_size[2]]
                        if use_tile(cropped_img, min_fraction):
                            cropped_image_list.append([cur_img_id, cropped_img])
                            cropping_index.append(cur_index)
                            cur_tile_meta = get_tile_meta(cropped_img,
                                                          cur_img_id,
                                                          save_dict,
                                                          row, col, sl)
                            tiled_metadata.append(cur_tile_meta)
                else:
                    img_id = '{}_sl{}-{}'.format(img_id, 0, im_depth)
            if use_tile(cropped_img, min_fraction) and not tile_3d:
                cropped_image_list.append([img_id, cropped_img])
                cropping_index.append(cur_index)
                cur_tile_meta = get_tile_meta(cropped_img,
                                              img_id,
                                              save_dict,
                                              row, col)
                tiled_metadata.append(cur_tile_meta)
    if save_dict is None:
        if return_index:
            return cropped_image_list, cropping_index
        return cropped_image_list
    else:
        # create and save meta csv
        tile_meta_df = write_meta(tiled_metadata, save_dict)
        if return_index:
            return tile_meta_df, cropping_index
        return tile_meta_df


def crop_at_indices(input_image,
                    crop_indices,
                    save_dict=None,
                    tile_3d=False):
    """Crop image into tiles at given indices

    :param np.array input_image: input image for cropping
    :param list crop_indices: list of indices for cropping
    :param dict save_dict: dict with keys: time_idx, channel_idx, slice_idx,
     pos_idx, image_format and save_dir for generation output fname
    :param bool tile_3d: boolean flag for adding slice_start_idx to meta
    :return: if not saving tiles: a list with tuples of cropped image id of
     the format rrmin-rmax_ccmin-cmax_slslmin-slmax and cropped image.
     Else saves tiles in-place and returns a df with tile metadata
    """

    n_dim = len(input_image.shape)
    tiles_list = []
    im_depth = input_image.shape[2]
    tiled_metadata = []
    for cur_idx in crop_indices:
        img_id = 'r{}-{}_c{}-{}'.format(cur_idx[0], cur_idx[1],
                                        cur_idx[2], cur_idx[3])

        if n_dim == 3:
            if tile_3d:
                img_id = '{}_sl{}-{}'.format(img_id, cur_idx[4], cur_idx[5])
                cropped_img = input_image[cur_idx[0]: cur_idx[1],
                                          cur_idx[2]: cur_idx[3],
                                          cur_idx[4]: cur_idx[5]]
            else:
                img_id = '{}_sl{}-{}'.format(img_id, 0, im_depth)
                cropped_img = input_image[cur_idx[0]: cur_idx[1],
                                          cur_idx[2]: cur_idx[3], ...]

        if save_dict is not None:
            file_name = write_tile(cropped_img, save_dict, img_id)
            cur_metadata = {'channel_idx': save_dict['channel_idx'],
                            'slice_idx': save_dict['slice_idx'],
                            'time_idx': save_dict['time_idx'],
                            'file_name': file_name,
                            'pos_idx': save_dict['pos_idx'],
                            'row_start': cur_idx[0],
                            'col_start': cur_idx[2]}
            if tile_3d:
                cur_metadata['slice_start'] = cur_idx[4]
            tiled_metadata.append(cur_metadata)
        tiles_list.append([img_id, cropped_img])
    if save_dict is None:
        return tiles_list
    else:
        tile_meta_df = write_meta(tiled_metadata, save_dict)
        return tile_meta_df


def write_tile(tile, save_dict, img_id):
    """
    Write tile function that can be called using threading.

    :param np.array tile: one tile
    :param dict save_dict: dict with keys: time_idx, channel_idx, slice_idx,
     pos_idx, image_format and save_dir for generation output fname
    :param str img_id: tile related indices as string
    :return str op_fname: filename used for saving the tile with entire path
    """

    file_name = aux_utils.get_im_name(time_idx=save_dict['time_idx'],
                                      channel_idx=save_dict['channel_idx'],
                                      slice_idx=save_dict['slice_idx'],
                                      pos_idx=save_dict['pos_idx'],
                                      int2str_len=save_dict['int2str_len'],
                                      extra_field=img_id)
    op_fname = os.path.join(save_dict['save_dir'], file_name)
    if save_dict['image_format'] == 'zyx' and len(tile.shape) > 2:
        tile = np.transpose(tile, (2, 0, 1))
    np.save(op_fname, tile, allow_pickle=True, fix_imports=True)
    return file_name


def write_meta(tiled_metadata, save_dict):
    """Write meta for tiles from an image as a csv

    :param list tiled_metadata: list of meta dicts
    :param dict save_dict: dict with keys: time_idx, channel_idx, slice_idx,
     pos_idx, image_format and save_dir for generation output fname
    :return:
    """

    if np.any(tiled_metadata):
        tile_meta_df = pd.DataFrame.from_dict(tiled_metadata)
        tile_meta_df = tile_meta_df.sort_values(by=['file_name'])
        idx_len = save_dict['int2str_len']
        meta_name = (
            'meta' +
            '_c' + str(save_dict['channel_idx']).zfill(idx_len) +
            '_z' + str(save_dict['slice_idx']).zfill(idx_len) +
            '_t' + str(save_dict['time_idx']).zfill(idx_len) +
            '_p' + str(save_dict['pos_idx']).zfill(idx_len) +
            '.csv')
        tile_meta_df.to_csv(
            os.path.join(save_dict['save_dir'], 'meta_dir', meta_name),
            sep=',',
        )
        return tile_meta_df
