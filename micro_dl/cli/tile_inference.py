#!/usr/bin/env/python
"""Model inference"""
import argparse
import os
import pandas as pd
import pickle
import yaml

from micro_dl.input.dataset import BaseDataSet, DataSetWithMask
from micro_dl.train.model_inference import ModelEvaluator
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.train_utils import select_gpu

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
                        help='Optional: specify the gpu memory fraction to use')
    parser.add_argument('--config', type=str,
                       help='path to yaml configuration file')
    parser.add_argument('--model_fname', type=str, default=None,
                       help='fname with full path to model weights .hdf5')

    parser.add_argument('--num_batches', type=int, default=2,
                        help='run prediction on tiles for num_batches')

    parser.add_argument('--flat_field', dest='flat_field', action='store_true',
                        help='Indicator to correct for flat field')

    parser.add_argument('--no_flat_field', dest='flat_field',
                        action='store_false')
    
    parser.set_defaults(flat_field=True)
    
    parser.add_argument('--focal_plane_idx', type=int, default=0,
                        help='idx for focal plane')
    parser.add_argument('--base_image_dir', type=str, default=None,
                        help='base dir with whole/entire images')

    parser.add_argument('--image_meta_fname', type=str, default=None,
                        help='csv holding meta for all images in study')

    args = parser.parse_args()
    return args


def run_inference(args, gpu_id, gpu_mem_frac):
    """Evaluate model performance"""

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    df_test = pd.read_csv(os.path.join(config['trainer']['model_dir'],
                                       'test_metadata.csv'))

    if 'masked_loss' in config['trainer']:
        ds_test = DataSetWithMask(input_fnames=df_test['fpaths_input'],
                                  target_fnames=df_test['fpaths_target'],
                                  mask_fnames=df_test['fpaths_mask'],
                                  batch_size=config['trainer']['batch_size'])
    else:
        ds_test = BaseDataSet(input_fnames=df_test['fpaths_input'],
                              target_fnames=df_test['fpaths_target'],
                              batch_size=config['trainer']['batch_size'])

    ev_inst = ModelEvaluator(config,
                             model_fname=args.model_fname,
                             gpu_ids=gpu_id,
                             gpu_mem_frac=gpu_mem_frac)
    test_perf_metrics = ev_inst.evaluate_model(ds_test)

    ev_inst.predict_on_tiles(ds_test, nb_batches=args.num_batches)
    idx_fname = os.path.join(config['trainer']['model_dir'],
                             'split_samples.json')
    split_samples = aux_utils.read_json(idx_fname)

    image_meta = pd.read_csv(args.image_meta_fname)
    # for regression tasks change place_operation to 'mean'
    ev_inst.predict_on_full_image(image_meta=image_meta,
                                  test_samples=split_samples['test'],
                                  focal_plane_idx=args.focal_plane_idx,
                                  flat_field_correct=args.flat_field,
                                  base_image_dir=args.base_image_dir,
                                  place_operation='max')
    return test_perf_metrics


if __name__ == '__main__':
    args = parse_args()
    # Get GPU ID and memory fraction
    gpu_id, gpu_mem_frac = select_gpu(
        args.gpu,
        args.gpu_mem_frac,
    )
    model_perf = run_inference(args, gpu_id, gpu_mem_frac)
    print('model performance on test images:', model_perf)
