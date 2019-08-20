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
        '--config',
        type=str,
        help='path to inference yaml configuration file',
    )

    args = parser.parse_args()
    return args


def run_inference(config_fname,
                   gpu_ids,
                   gpu_mem_frac):
    """

    :param str config_fname: Full path to config yaml file
    :param int gpu_ids: gpu id to use
    :param float gpu_mem_frac: gpu memory fraction to use
    """

    with open(config_fname, 'r') as f:
        inference_config = yaml.safe_load(f)
    # Load train config from model dir
    train_config_fname = glob.glob(
        os.path.join(inference_config['model_dir'], '*.yml')
    )
    assert len(train_config_fname) == 1, \
        'more than one train config yaml found in model dir'
    with open(train_config_fname[0], 'r') as f:
        train_config = yaml.safe_load(f)

    inference_inst = image_inf.ImagePredictor(
        train_config=train_config,
        inference_config=inference_config,
        gpu_id=gpu_ids,
        gpu_mem_frac=gpu_mem_frac,
    )
    inference_inst.run_prediction()


if __name__ == '__main__':
    args = parse_args()
    # Get GPU ID and memory fraction
    gpu_id, gpu_mem_frac = train_utils.select_gpu(
        args.gpu,
        args.gpu_mem_frac,
    )
    run_inference(
        config_fname=args.config,
        gpu_ids=gpu_id,
        gpu_mem_frac=gpu_mem_frac,
    )
