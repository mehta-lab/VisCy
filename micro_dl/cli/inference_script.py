#!/usr/bin/env/python
"""Model inference on larger images with and w/o stitching"""
import argparse
import glob
import os
import yaml

from micro_dl.inference import image_inference as image_inf
import micro_dl.utils.train_utils as train_utils


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=None,
                        help=('Optional: specify the gpu to use: 0,1,...',
                              ', -1 for debugging. Default: pick best GPU'))
    parser.add_argument('--gpu_mem_frac', type=float, default=None,
                        help='Optional: specify gpu memory fraction to use')
    parser.add_argument(
        '--data_split',
        type=str,
        default='test',
        nargs='?',
        choices=['train', 'val', 'test', 'all'],
        help='data split to predict on'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='path to inference yaml configuration file',
    )

    args = parser.parse_args()
    return args


def run_prediction(args, gpu_ids, gpu_mem_frac):
    """

    :param dict args: dict with input options
    :param int gpu_ids: gpu id to use
    :param float gpu_mem_frac: gpu memory fraction to use
    :return:
    """

    config_fname = args.config
    with open(config_fname, 'r') as f:
        inf_config = yaml.safe_load(f)

    train_config_fname = glob.glob(
        os.path.join(inf_config['model_dir'], '*.yml')
    )
    assert len(train_config_fname) == 1, \
        'more than one train config yaml found in model dir'

    model_fname = None
    if model_fname in inf_config:
        model_fname = os.path.join(inf_config['model_dir'],
                                   inf_config['model_fname'])
    image_pred_inst = image_inf.ImagePredictor(
        config=train_config_fname[0],
        model_fname=model_fname,
        image_dir=inf_config['image_dir'],
        data_split=args.data_split,
        image_param_dict=inf_config['image_params_dict'],
        gpu_id=gpu_ids,
        gpu_mem_frac=gpu_mem_frac,
        metrics_list=inf_config['metrics'],
        metrics_orientations=inf_config['metrics_orientations'],
        mask_param_dict=inf_config['mask_params_dict'],
        vol_inf_dict=inf_config['vol_inf_dict'])
    image_pred_inst.run_prediction()


if __name__ == '__main__':
    args = parse_args()
    # Get GPU ID and memory fraction
    gpu_id, gpu_mem_frac = train_utils.select_gpu(
        args.gpu,
        args.gpu_mem_frac,
    )
    run_prediction(args, gpu_id, gpu_mem_frac)
