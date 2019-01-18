import os

import numpy as np
import pandas as pd

from micro_dl.utils import normalize as normalize, aux_utils as aux_utils
from micro_dl.utils.image_utils import read_image, \
    apply_flat_field_correction, resize_image


def read_imstack(input_fnames,
                 flat_field_fname=None,
                 hist_clip_limits=None):
    """Read the images in the fnames and assembles a stack
    
    :param tuple input_fnames: tuple of input fnames with full path
    :param str flat_field_fname: fname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :return np.array: input stack flat_field correct and z-scored
    """

    im_stack = []
    for fname in input_fnames:
        im = read_image(fname)
        if flat_field_fname is not None:
            flat_field_image = np.load(flat_field_fname)
            im = apply_flat_field_correction(
                im,
                flat_field_image=flat_field_image,
            )
        im_stack.append(im)
    input_image = np.stack(im_stack, axis=2)
    if hist_clip_limits is not None:
        input_image = normalize.hist_clipping(
            input_image,
            hist_clip_limits[0],
            hist_clip_limits[1]
        )
    if input_image.dtype != bool:
        input_image = normalize.zscore(input_image)
    return input_image


def preprocess_imstack(frames_metadata,
                       input_dir,
                       depth,
                       time_idx,
                       channel_idx,
                       slice_idx,
                       pos_idx,
                       flat_field_im=None,
                       hist_clip_limits=None):
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
    :return np.array im: 2D preprocessed image
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
        if flat_field_im is not None:
            im = apply_flat_field_correction(
                im,
                flat_field_image=flat_field_im,
            )
        im_stack.append(im)
    # Stack images
    im_stack = np.stack(im_stack, axis=2)
    # normalize
    if hist_clip_limits is not None:
        im_stack = normalize.hist_clipping(
            im_stack,
            hist_clip_limits[0],
            hist_clip_limits[1],
        )
    return normalize.zscore(im_stack)


def tile_image(input_image,
               tile_size,
               step_size,
               isotropic=False,
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
    :param bool isotropic: if 3D, make the grid/shape isotropic
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

    tile_3d = False
    # Add to tile size and step size in case of 3D images
    im_shape = input_image.shape
    if len(im_shape) == 3:
        if len(tile_size) == 2:
            tile_size.append(im_shape[2])
        else:
            tile_3d = True

        # Step size in z is assumed to be the same as depth
        if len(step_size) == 2:
            step_size.append(im_shape[2])

    assert len(tile_size) == len(step_size),\
        "Tile {} and step size {} mismatch".format(tile_size, step_size)
    assert np.all(step_size <= tile_size),\
        "Step size {} > tile size {}".format(step_size, tile_size)
    assert np.all(tile_size) > 0,\
        "Tile size must be > 0, not {}".format(tile_size)

    n_rows = im_shape[0]
    n_cols = im_shape[1]
    n_dim = len(im_shape)
    im_depth = im_shape[2]
    if n_dim == 3:
        n_slices = im_depth

    if isotropic:
        isotropic_shape = [tile_size[0], ] * len(tile_size)
        isotropic_cond = not(list(tile_size) == isotropic_shape)
    else:
        isotropic_cond = isotropic

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
                    for sl in range(0, n_slices, step_size[2]):
                        if sl + step_size[2] > n_slices:
                            sl = check_in_range(sl, n_slices, tile_size[2])

                        cur_index = (row, row + tile_size[0],
                                     col, col + tile_size[1],
                                     sl, sl + tile_size[2])
                        img_id = '{}_sl{}-{}'.format(img_id, sl,
                                                     sl + tile_size[2])
                        cropped_img = input_image[row: row + tile_size[0],
                                                  col: col + tile_size[1],
                                                  sl: sl + tile_size[2]]
                else:
                    img_id = '{}_sl{}-{}'.format(img_id, 0, im_depth)
                if isotropic_cond:
                    img_shape = cropped_img.shape
                    isotropic_shape = [img_shape[0], ] * len(img_shape)
                    cropped_img = resize_image(cropped_img, isotropic_shape)
            if use_tile(cropped_img, min_fraction):
                cropped_image_list.append([img_id, cropped_img])
                cropping_index.append(cur_index)
                if save_dict is not None:
                    file_name = write_tile(cropped_img, save_dict, img_id)
                    tiled_metadata.append(
                        {'channel_idx': save_dict['channel_idx'],
                         'slice_idx': save_dict['slice_idx'],
                         'time_idx': save_dict['time_idx'],
                         'file_name': file_name,
                         'pos_idx': save_dict['pos_idx'],
                         'row_start': row,
                         'col_start': col})
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
                    isotropic=False,
                    save_dict=None):
    """Crop image into tiles at given indices

    :param np.array input_image: input image for cropping
    :param list crop_indices: list of indices for cropping
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :param dict save_dict: dict with keys: time_idx, channel_idx, slice_idx,
     pos_idx, image_format and save_dir for generation output fname
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

        cropped_img = input_image[cur_idx[0]: cur_idx[1],
                                  cur_idx[2]: cur_idx[3], ...]
        if n_dim == 3:
            img_id = '{}_sl{}-{}'.format(img_id, 0, im_depth)

            if isotropic:
                img_shape = cropped_img.shape
                isotropic_shape = [img_shape[0], ] * len(img_shape)
                cropped_img = resize_image(cropped_img, isotropic_shape)
        if save_dict is not None:
            file_name = write_tile(cropped_img, save_dict, img_id)
            tiled_metadata.append({'channel_idx': save_dict['channel_idx'],
                                   'slice_idx': save_dict['slice_idx'],
                                   'time_idx': save_dict['time_idx'],
                                   'file_name': file_name,
                                   'pos_idx': save_dict['pos_idx'],
                                   'row_start': cur_idx[0],
                                   'col_start': cur_idx[2]})
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
        meta_name = ('meta'
                     + '_c' + str(save_dict['channel_idx']).zfill(idx_len)
                     + '_z' + str(save_dict['slice_idx']).zfill(idx_len)
                     + '_t' + str(save_dict['time_idx']).zfill(idx_len)
                     + '_p' + str(save_dict['pos_idx']).zfill(idx_len)
                     + '.csv')
        tile_meta_df.to_csv(
            os.path.join(save_dict['save_dir'], 'meta_dir', meta_name),
            sep=",",
        )
        return tile_meta_df
