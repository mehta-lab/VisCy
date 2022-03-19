"""Pool multiple datasets into a single dataset for training"""
import argparse
import os
import yaml
import pandas as pd
from shutil import copy, copy2
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.meta_utils as meta_utils

def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        help='path to inference yaml configuration file',
    )

    args = parser.parse_args()
    return args

def pool_dataset(config):
    """
    :param dict args: dict with input options
    :return:
    """

    config_fname = config
    with open(config_fname, 'r') as f:
        pool_config = yaml.safe_load(f)
    dst_dir = pool_config['destination']
    num_workers = pool_config['num_workers']
    pool_mode = pool_config['pool_mode']
    frames_meta_dst_path = os.path.join(dst_dir, 'frames_meta.csv')
    ints_meta_dst_path = os.path.join(dst_dir, 'ints_meta.csv')
    pos_idx_cur = 0
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(frames_meta_dst_path) and pool_mode == 'add':
        frames_meta_dst = pd.read_csv(frames_meta_dst_path, index_col=0)
        ints_meta_dst = pd.read_csv(ints_meta_dst_path, index_col=0)
        pos_idx_cur = frames_meta_dst['pos_idx'].max() + 1
    else:
        frames_meta_dst = aux_utils.make_dataframe(nbr_rows=None)
        ints_meta_dst = pd.DataFrame()
    for src_key in pool_config:
        if 'source' in src_key:
            src_dir = pool_config[src_key]['dir']
            src_pos_ids = pool_config[src_key]['pos_ids']
            frames_meta_src = meta_utils.frames_meta_generator(
                src_dir,
                name_parser=pool_config['name_parser'],
            )
            ints_meta_src = meta_utils.ints_meta_generator(
                src_dir,
                name_parser=pool_config['name_parser'],
                num_workers=num_workers,
            )
            if src_pos_ids == 'all':
                src_pos_ids = frames_meta_src['pos_idx'].unique()
            src_pos_ids.sort()
            pos_idx_map = dict(zip(src_pos_ids, range(pos_idx_cur, pos_idx_cur + len(src_pos_ids))))
            # select positions to pool and update their indices
            frames_meta_src_new = frames_meta_src.copy()
            frames_meta_src_new = frames_meta_src_new[frames_meta_src['pos_idx'].isin(src_pos_ids)]
            frames_meta_src_new['pos_idx'] = frames_meta_src_new['pos_idx'].map(pos_idx_map)
            ints_meta_src_new = ints_meta_src.copy()
            ints_meta_src_new = ints_meta_src_new[ints_meta_src['pos_idx'].isin(src_pos_ids)]
            ints_meta_src_new['pos_idx'] = ints_meta_src_new['pos_idx'].map(pos_idx_map)
            # update file names and copy the files
            for row_idx in list(frames_meta_src_new.index):
                meta_row = frames_meta_src_new.loc[row_idx]
                im_name_dst = aux_utils.get_sms_im_name(
                    time_idx=meta_row['time_idx'],
                    channel_name=meta_row['channel_name'],
                    slice_idx=meta_row['slice_idx'],
                    pos_idx=meta_row['pos_idx'],
                    ext='.tif',
                    )
                frames_meta_src_new.loc[row_idx, 'file_name'] = im_name_dst
                im_name_src = frames_meta_src.loc[row_idx, 'file_name']
                # copy(os.path.join(src_dir, im_name_src),
                #       os.path.join(dst_dir, im_name_dst))
                os.link(os.path.join(src_dir, im_name_src),
                     os.path.join(dst_dir, im_name_dst))

            frames_meta_dst = frames_meta_dst.append(
                frames_meta_src_new,
                ignore_index=True,
            )
            ints_meta_dst = ints_meta_dst.append(
                ints_meta_src_new,
                ignore_index=True,
            )
            pos_idx_cur = pos_idx_map[src_pos_ids[-1]] + 1
    frames_meta_dst.to_csv(frames_meta_dst_path, sep=",")
    ints_meta_dst.to_csv(ints_meta_dst_path, sep=",")

if __name__ == '__main__':
    args = parse_args()
    pool_dataset(args.config)
