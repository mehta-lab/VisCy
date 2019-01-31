import cv2
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils
import micro_dl.utils.tile_utils as tile_utils


def mp_create_save_mask(fn_args, workers):
    """Create and save masks with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned dicts from create_save_mask
    """

    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(create_save_mask, *zip(*fn_args))
    return list(res)


def create_save_mask(input_fnames,
                     flat_field_fname,
                     str_elem_radius,
                     mask_dir,
                     mask_channel_idx,
                     time_idx,
                     pos_idx,
                     slice_idx,
                     int2str_len):
    """Create and save mask

    :param tuple input_fnames: tuple of input fnames with full path
    :param str flat_field_fname: fname of flat field image
    :param int str_elem_radius: size of structuring element used for binary
         opening. str_elem: disk or ball
    :param str mask_dir: dir to save masks
    :param int mask_channel_idx: channel number of maskß
    :param int time_idx: time points to use for generating mask
    :param int pos_idx: generate masks for given position / sample ids
    :param int slice_idx: generate masks for given slice ids
    :param int int2str_len: Length of str when converting ints
    :return dict cur_meta for each mask
    """
    im_stack = tile_utils.read_imstack(input_fnames,
                                       flat_field_fname)
    # Combine channel images and generate mask
    summed_image = np.sum(np.stack(im_stack), axis=2)
    summed_image = summed_image.astype('float32')

    mask = image_utils.create_mask(summed_image, str_elem_radius)
    # Create mask name for given slice, time and position
    file_name = aux_utils.get_im_name(time_idx=time_idx,
                                      channel_idx=mask_channel_idx,
                                      slice_idx=slice_idx,
                                      pos_idx=pos_idx,
                                      int2str_len=int2str_len)
    # Save mask for given channels, mask is 2D
    np.save(os.path.join(mask_dir, file_name),
            mask,
            allow_pickle=True,
            fix_imports=True)
    cur_meta = {'channel_idx': mask_channel_idx,
                'slice_idx': slice_idx,
                'time_idx': time_idx,
                'pos_idx': pos_idx,
                'file_name': file_name}
    return cur_meta


def mp_tile_save(fn_args, workers):
    """Tile and save with multiprocessing
    https://stackoverflow.com/questions/42074501/python-concurrent-futures-processpoolexecutor-performance-of-submit-vs-map
    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from tile_and_save
    """

    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(tile_and_save, *zip(*fn_args))
    return list(res)


def tile_and_save(input_fnames,
                  flat_field_fname,
                  hist_clip_limits,
                  time_idx,
                  channel_idx,
                  pos_idx,
                  slice_idx,
                  tile_size,
                  step_size,
                  min_fraction,
                  image_format,
                  isotropic,
                  save_dir,
                  int2str_len=3,
                  is_mask=False):
    """Crop image into tiles at given indices and save

    :param tuple input_fnames: tuple of input fnames with full path
    :param str flat_field_fname: fname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :param int time_idx: time point of input image
    :param int channel_idx: channel idx of input image
    :param int slice_idx: slice idx of input image
    :param int pos_idx: sample idx of input image
    :param list tile_size: size of tile along row, col (& slices)
    :param list step_size: step size along row, col (& slices)
    :param float min_fraction: min foreground volume fraction for keep tile
    :param str image_format: zyx / yxz
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :param str save_dir: output dir to save tiles
    :param int int2str_len: len of indices for creating file names
    :param bool is_mask: Indicates if files are masks
    :return: pd.DataFrame from a list of dicts with metadata
    """

    try:
        input_image = tile_utils.read_imstack(
            input_fnames=input_fnames,
            flat_field_fname=flat_field_fname,
            hist_clip_limits=hist_clip_limits,
            is_mask=is_mask,
        )
        save_dict = {'time_idx': time_idx,
                     'channel_idx': channel_idx,
                     'pos_idx': pos_idx,
                     'slice_idx': slice_idx,
                     'save_dir': save_dir,
                     'image_format': image_format,
                     'int2str_len': int2str_len}

        tile_meta_df = tile_utils.tile_image(
            input_image=input_image,
            tile_size=tile_size,
            step_size=step_size,
            isotropic=isotropic,
            min_fraction=min_fraction,
            save_dict=save_dict,
        )
    except Exception as e:
        err_msg = 'error in t_{}, c_{}, pos_{}, sl_{}'.format(
            time_idx, channel_idx, pos_idx, slice_idx
        )
        err_msg = err_msg + str(e)
        # TODO(Anitha) write to log instead
        print(err_msg)
        raise e
    return tile_meta_df


def mp_crop_save(fn_args, workers):
    """Crop and save images with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from crop_at_indices_save
    """

    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(crop_at_indices_save, *zip(*fn_args))
    return list(res)


def crop_at_indices_save(input_fnames,
                         flat_field_fname,
                         hist_clip_limits,
                         time_idx,
                         channel_idx,
                         pos_idx,
                         slice_idx,
                         crop_indices,
                         image_format,
                         isotropic,
                         save_dir,
                         int2str_len=3,
                         is_mask=False):
    """Crop image into tiles at given indices and save

    :param tuple input_fnames: tuple of input fnames with full path
    :param str flat_field_fname: fname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :param int time_idx: time point of input image
    :param int channel_idx: channel idx of input image
    :param int slice_idx: slice idx of input image
    :param int pos_idx: sample idx of input image
    :param tuple crop_indices: tuple of indices for cropping
    :param str image_format: zyx or yxz
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :param str save_dir: output dir to save tiles
    :param int int2str_len: len of indices for creating file names
    :param bool is_mask: Indicates if files are masks
    :return: pd.DataFrame from a list of dicts with metadata
    """

    try:
        input_image = tile_utils.read_imstack(
            input_fnames=input_fnames,
            flat_field_fname=flat_field_fname,
            hist_clip_limits=hist_clip_limits,
            is_mask=is_mask,
        )
        save_dict = {'time_idx': time_idx,
                     'channel_idx': channel_idx,
                     'pos_idx': pos_idx,
                     'slice_idx': slice_idx,
                     'save_dir': save_dir,
                     'image_format': image_format,
                     'int2str_len': int2str_len}

        tile_meta_df = tile_utils.crop_at_indices(
            input_image=input_image,
            crop_indices=crop_indices,
            isotropic=isotropic,
            save_dict=save_dict,
        )
    except Exception as e:
        err_msg = 'error in t_{}, c_{}, pos_{}, sl_{}'.format(
            time_idx, channel_idx, pos_idx, slice_idx
        )
        err_msg = err_msg + str(e)
        # TODO(Anitha) write to log instead
        print(err_msg)
        raise e

    return tile_meta_df


def mp_resize_save(mp_args, workers):
    """
    Resize and save images with multiprocessing

    :param dict mp_args: Function keyword arguments
    :param int workers: max number of workers
    """
    with ProcessPoolExecutor(workers) as ex:
        {ex.submit(resize_and_save, **kwargs): kwargs for kwargs in mp_args}


def resize_and_save(**kwargs):
    """
    Resizing images and saving them
    :param kwargs: Keyword arguments:
    str file_path: Path to input image
    str write_path: Path to image to be written
    float scale_factor: Scale factor for resizing
    """
    im = image_utils.read_image(kwargs['file_path'])
    im_resized = image_utils.rescale_image(
        im=im,
        scale_factor=kwargs['scale_factor'],
    )
    # Write image
    cv2.imwrite(kwargs['write_path'], im_resized)
