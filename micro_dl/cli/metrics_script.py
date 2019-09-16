#!/usr/bin/env/python

import argparse
import numpy as np
import os
import pandas as pd
import yaml

import micro_dl.inference.evaluation_metrics as metrics
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing model weights, config and csv files',
    )
    parser.add_argument(
        '--model_fname',
        type=str,
        default=None,
        help='File name of weights in model dir (.hdf5). If None grab newest.',
    )
    parser.add_argument(
        '--test_data',
        dest='test_data',
        action='store_true',
        help="Use test indices in split_samples.json",
    )
    parser.add_argument(
        '--all_data',
        dest='test_data',
        action='store_false',
    )
    parser.set_defaults(test_data=True)
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help="Directory containing target images",
    )
    parser.add_argument(
        '--metrics',
        type=str,
        required=True,
        nargs='*',
        help='Metrics for model evaluation'
    )
    parser.add_argument(
        '--orientations',
        type=str,
        default='xyz',
        nargs='*',
        help='Evaluate metrics along these orientations (xy, xz, yz, xyz)'
    )
    return parser.parse_args()


def compute_metrics(model_dir,
                    image_dir,
                    metrics_list,
                    orientations_list,
                    test_data=True):
    """
    Compute specified metrics for given orientations for predictions, which
    are assumed to be stored in model_dir/predictions. Targets are stored in
    image_dir.
    Writes metrics csv files for each orientation in model_dir/predictions.

    :param str model_dir: Assumed to contain config, split_samples.json and
        subdirectory predictions/
    :param str image_dir: Directory containing target images with frames_meta.csv
    :param list metrics_list: See inference/evaluation_metrics.py for options
    :param list orientations_list: Any subset of {xy, xz, yz, xyz}
        (see evaluation_metrics)
    :param bool test_data: Uses test indices in split_samples.json,
    otherwise all indices
    """
    # Load config file
    config_name = os.path.join(model_dir, 'config.yml')
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f)
    # Load frames metadata and determine indices
    frames_meta = pd.read_csv(os.path.join(image_dir, 'frames_meta.csv'))

    if isinstance(metrics_list, str):
        metrics_list = [metrics_list]
    metrics_inst = metrics.MetricsEstimator(metrics_list=metrics_list)

    split_idx_name = config['dataset']['split_by_column']
    if test_data:
        idx_fname = os.path.join(model_dir, 'split_samples.json')
        try:
            split_samples = aux_utils.read_json(idx_fname)
            test_ids = split_samples['test']
        except FileNotFoundError as e:
            print("No split_samples file. Will predict all images in dir.")
    else:
        test_ids = np.unique(frames_meta[split_idx_name])

    # Find other indices to iterate over than split index name
    # E.g. if split is position, we also need to iterate over time and slice
    test_meta = pd.read_csv(os.path.join(model_dir, 'test_metadata.csv'))
    metadata_ids = {split_idx_name: test_ids}
    iter_ids = ['slice_idx', 'pos_idx', 'time_idx']

    for id in iter_ids:
        if id != split_idx_name:
            metadata_ids[id] = np.unique(test_meta[id])

    # Create image subdirectory to write predicted images
    pred_dir = os.path.join(model_dir, 'predictions')

    target_channel = config['dataset']['target_channels'][0]

    # If network depth is > 3 determine depth margins for +-z
    depth = 1
    if 'depth' in config['network']:
        depth = config['network']['depth']

    # Get channel name and extension for predictions
    pred_fnames = [f for f in os.listdir(pred_dir) if f.startswith('im_')]
    meta_row = aux_utils.parse_idx_from_name(pred_fnames[0])
    pred_channel = meta_row['channel_idx']
    _, ext = os.path.splitext(pred_fnames[0])

    if isinstance(orientations_list, str):
        orientations_list = [orientations_list]
    available_orientations = {'xy', 'xz', 'yz', 'xyz'}
    assert set(orientations_list).issubset(available_orientations), \
        "Orientations must be subset of {}".format(available_orientations)

    fn_mapping = {
        'xy': metrics_inst.estimate_xy_metrics,
        'xz': metrics_inst.estimate_xz_metrics,
        'yz': metrics_inst.estimate_yz_metrics,
        'xyz': metrics_inst.estimate_xyz_metrics,
    }
    metrics_mapping = {
        'xy': metrics_inst.get_metrics_xy,
        'xz': metrics_inst.get_metrics_xz,
        'yz': metrics_inst.get_metrics_yz,
        'xyz': metrics_inst.get_metrics_xyz,
    }
    df_mapping = {
        'xy': pd.DataFrame(),
        'xz': pd.DataFrame(),
        'yz': pd.DataFrame(),
        'xyz': pd.DataFrame(),
    }

    # Iterate over all indices for test data
    for time_idx in metadata_ids['time_idx']:
        for pos_idx in metadata_ids['pos_idx']:
            target_fnames = []
            pred_fnames = []
            for slice_idx in metadata_ids['slice_idx']:
                im_idx = aux_utils.get_meta_idx(
                    frames_metadata=frames_meta,
                    time_idx=time_idx,
                    channel_idx=target_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                )
                target_fname = os.path.join(
                    image_dir,
                    frames_meta.loc[im_idx, 'file_name'],
                )
                target_fnames.append(target_fname)
                pred_fname = aux_utils.get_im_name(
                    time_idx=time_idx,
                    channel_idx=pred_channel,
                    slice_idx=slice_idx,
                    pos_idx=pos_idx,
                    ext=ext,
                )
                pred_fname = os.path.join(pred_dir, pred_fname)
                pred_fnames.append(pred_fname)

            target_stack = image_utils.read_imstack(
                input_fnames=tuple(target_fnames),
            )
            pred_stack = image_utils.read_imstack(
                input_fnames=tuple(pred_fnames),
                normalize_im=False,
            )

            if depth == 1:
                # Remove singular z dimension for 2D image
                target_stack = np.squeeze(target_stack)
                pred_stack = np.squeeze(pred_stack)
            if target_stack.dtype == np.float64:
                target_stack = target_stack.astype(np.float32)
            pred_name = "t{}_p{}".format(time_idx, pos_idx)
            for orientation in orientations_list:
                metric_fn = fn_mapping[orientation]
                metric_fn(
                    target=target_stack,
                    prediction=pred_stack,
                    pred_name=pred_name,
                )
                df_mapping[orientation] = df_mapping[orientation].append(
                    metrics_mapping[orientation](),
                    ignore_index=True,
                )

    # Save non-empty dataframes
    for orientation in orientations_list:
        metrics_df = df_mapping[orientation]
        df_name = 'metrics_{}.csv'.format(orientation)
        metrics_name = os.path.join(pred_dir, df_name)
        metrics_df.to_csv(metrics_name, sep=",", index=False)


if __name__ == '__main__':
    args = parse_args()
    compute_metrics(
        model_dir=args.model_dir,
        image_dir=args.image_dir,
        metrics_list=args.metrics,
        orientations_list=args.orientations,
        test_data=args.test_data,
    )
