#!/usr/bin/env python

"""
Convert a zarr store to single page tif files. Choose which channels, time points, positions, and z-ids to include.
"""

import os
import cv2
from waveorder.io.reader import WaveorderReader
import pprint
import time
from concurrent.futures import ProcessPoolExecutor


def write_img(img, output_dir, img_name):
    """only supports recon_order image name format currently"""
    if not os.path.exists(output_dir):  # create folder for processed images
        os.makedirs(output_dir)
    if len(img.shape) < 3:
        cv2.imwrite(os.path.join(output_dir, img_name), img)
    else:
        cv2.imwrite(os.path.join(output_dir, img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_position_map(zarr_dir, zarr_name):
    """
    Get position map based on input zarr folder structure.
    {0: {'name': '1-Pos000_000', 'well': 'Row_0/Col_0'}, ...}

    :param str zarr_path: path to zarr folder
    :param str zarr_name: name of zarr store
    :return dct: position map
    """
    position_map = {}
    c = 0
    for col in os.listdir(zarr_dir + zarr_name + '/Row_0'):
        if col.startswith('.'):
            continue
        col_dir = zarr_dir + zarr_name + '/Row_0/' + col
        for file in os.listdir(col_dir):
            if not file.startswith('.'):
                pos_name = file
                position_map[c] = {'name': pos_name, 'well': 'Row_0/' + col}
                c += 1
    return position_map


def get_sms_im_name(time_idx=None,
                    channel_name=None,
                    slice_idx=None,
                    pos_idx=None,
                    extra_field=None,
                    ext='.npy',
                    int2str_len=3):
    """
    Create an image name given parameters and extension
    This function is custom for the computational microscopy (SMS)
    group, who has the following file naming convention:
    File naming convention is assumed to be:
        img_channelname_t***_p***_z***.tif
    This function will alter list and dict in place.

    :param int time_idx: Time index
    :param str channel_name: Channel name
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param str extra_field: Any extra string you want to include in the name
    :param str ext: Extension, e.g. '.png'
    :param int int2str_len: Length of string of the converted integers
    :return st im_name: Image file name
    """

    im_name = "img"
    if channel_name is not None:
        im_name += "_" + str(channel_name)
    if time_idx is not None:
        im_name += "_t" + str(time_idx).zfill(int2str_len)
    if pos_idx is not None:
        im_name += "_p" + str(pos_idx).zfill(int2str_len)
    if slice_idx is not None:
        im_name += "_z" + str(slice_idx).zfill(int2str_len)
    if extra_field is not None:
        im_name += "_" + extra_field
    im_name += ext

    return im_name


def multiprocess(fn, fn_args, n_workers):
    """multiprocessing the input function given a list of function arguments
    :param object fn: function to run in parallel
    :param list of tuple fn_args: list with tuples of function arguments.
        Each tuple contains values to be run in parallel for each argument.
    :param int n_workers: max number of workers
    :return: list of return from function
    """

    with ProcessPoolExecutor(n_workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(fn, *fn_args)
    return list(res)


def process_tcz(reader,
                pos_idx,
                channels,
                chan_ids,
                z_ids,
                t_ids,
                dst_dir):
    print('processing position {}...'.format(pos_idx))
    img_tcz = reader.get_zarr(position=pos_idx)  # Returns sliceable array that hasn't been loaded into memory
    for t_idx in t_ids:
        img_cz = img_tcz[t_idx]
        # pbar = tqdm(total=len(channels), desc="Channels", leave=False)
        for c_idx, chan in zip(chan_ids, channels):
            img_z = img_cz[c_idx]
            for z_idx in z_ids:
                img = img_z[z_idx]
                im_name_dst = get_sms_im_name(
                    time_idx=t_idx,
                    channel_name=chan,
                    slice_idx=z_idx,
                    pos_idx=pos_idx,
                    ext='.tif',
                )
                write_img(img, dst_dir, im_name_dst)
            # pbar.update(1)


def main(input_path,
        output_path,
        conditions,
        channels,
        chan_ids,
        z_ids,
        t_ids,
        pos_ids,
        n_workers):

    for condition in conditions:
        print('processing condition {}...'.format(condition))
        dst_dir = os.path.join(output_path, os.path.splitext(condition)[0])
        os.makedirs(dst_dir, exist_ok=True)
        reader = WaveorderReader(os.path.join(input_path, condition), data_type='zarr')
        reader.reader.position_map = get_position_map(input_path, condition)
        n_pos = len(pos_ids)
        fn_args = [(reader,) * n_pos,
                   tuple(pos_ids),
                   (tuple(channels),) * n_pos,
                   (tuple(chan_ids),) * n_pos,
                   (tuple(z_ids),) * n_pos,
                   (tuple(t_ids),) * n_pos,
                   (dst_dir,) * n_pos]
        multiprocess(process_tcz, fn_args=fn_args, n_workers=n_workers)


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    input_path = r'/hpc/projects/comp_micro/projects/HEK/2022_03_15_orgs_nuc_mem_63x_04NA/all_pos_zarr/'  # directory with zarr file
    output_path = r'/hpc/projects/comp_micro/projects/HEK/2022_03_15_orgs_nuc_mem_63x_04NA/all_pos_single_page/'  # save directory
    conditions = ['all_pos_Phase1e-3_Denconv_Nuc8e-4_Mem8e-4_pad15_bg50.zarr']  # name of zarr file
    channels = ['phase', 'membrane', 'nucleus']  # name of target channels
    chan_ids = [0, 2, 3]  # channel ids
    z_ids = list(range(97))  # choose z-ids
    t_ids = [0]  # choose time points
    pos_ids = list(range(336))  # choose positions
    n_workers = 84
    time_start = time.time()
    main(input_path,
        output_path,
        conditions,
        channels,
        chan_ids,
        z_ids,
        t_ids,
        pos_ids,
        n_workers)
    time_el = time.time() - time_start
    print('processing time: {}s'.format(time_el))

