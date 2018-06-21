#!/usr/bin/env python
"""Train neural network models in keras"""
import argparse
import os
import pandas as pd
import pickle
import yaml

from micro_dl.input.dataset import BaseDataSet, DataSetWithMask
from micro_dl.input.training_table import (
    BaseTrainingTable, TrainingTableWithMask
)
from micro_dl.train.trainer import BaseKerasTrainer
from micro_dl.utils.train_utils import check_gpu_availability


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help=('specify the gpu to use: 0,1,...',
                              ', -1 for debugging'))
    parser.add_argument('--gpu_mem_frac', type=float, default=1.,
                        help='specify the gpu memory fraction to use')
    parser.add_argument('--action', type=str, default='train',
                        choices=('train', 'tune_hyperparam'),
                        help=('action to take on the model: train,'
                              'tune_hyperparam'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str,
                       help='path to yaml configuration file')
    group.add_argument('--model', type=str, default=None,
                       help='path to checkpoint file/directory')
    parser.add_argument('--port', type=int, default=-1,
                        help='port to use for the tensorboard callback')
    args = parser.parse_args()
    return args


def read_config(config_fname):
    """Read the config file in yml format

    TODO: validate config!

    :param str config_fname: fname of config yaml with its full path
    :return:
    """

    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    return config


def train_xy(df_meta, config):
    """Train using fit_generator"""

    tt = BaseTrainingTable(df_meta, config['dataset']['input_channels'],
                           config['dataset']['target_channels'],
                           config['dataset']['split_by_column'],
                           config['dataset']['split_ratio'])
    if 'val' in config['dataset']['split_ratio']:
        train_metadata, val_metadata, test_metadata, split_idx = \
            tt.train_test_split()
        val_gen_params = {}
        if 'augmentations' in config['trainer']:
            val_gen_params['augmentations'] = (
                config['trainer']['augmentations']
            )

        val_dataset = BaseDataSet(input_fnames=val_metadata['fpaths_input'],
                                  target_fnames=val_metadata['fpaths_target'],
                                  batch_size=config['trainer']['batch_size'],
                                  **val_gen_params)
        train_gen_params = val_gen_params.copy()
        val_metadata.to_csv(
            os.path.join(config['trainer']['model_dir'], 'val_metadata.csv'),
            sep=','
        )
    else:
        train_metadata, test_metadata, split_idx = tt.train_test_split()
        val_dataset = None
        train_gen_params = {}
        if 'augmentations' in config['trainer']:
            train_gen_params['augmentations'] = (
                config['trainer']['augmentations']
            )
    test_metadata.to_csv(
        os.path.join(config['trainer']['model_dir'], 'test_metadata.csv'),
        sep=','
    )

    train_dataset = BaseDataSet(input_fnames=train_metadata['fpaths_input'],
                                target_fnames=train_metadata['fpaths_target'],
                                batch_size=config['trainer']['batch_size'],
                                **train_gen_params)
    test_dataset = BaseDataSet(input_fnames=test_metadata['fpaths_input'],
                               target_fnames=test_metadata['fpaths_target'],
                               batch_size=config['trainer']['batch_size'])
    return train_dataset, val_dataset, test_dataset, split_idx


def train_xyweights(df_meta, config):
    """Train using fit_generator"""

    tt = TrainingTableWithMask(df_meta, config['dataset']['input_channels'],
                               config['dataset']['target_channels'],
                               config['dataset']['mask_channels'],
                               config['dataset']['split_by_column'],
                               config['dataset']['split_ratio'])
    if 'val' in config['dataset']['split_ratio']:
        train_metadata, val_metadata, test_metadata, split_idx = \
            tt.train_test_split()
        val_gen_params = {}
        if 'augmentations' in config['trainer']:
            val_gen_params['augmentations'] = (
                config['trainer']['augmentations']
            )

        val_dataset = DataSetWithMask(
            input_fnames=val_metadata['fpaths_input'],
            target_fnames=val_metadata['fpaths_target'],
            mask_fnames=val_metadata['fpaths_mask'],
            batch_size=config['trainer']['batch_size'],
            **val_gen_params
        )
        train_gen_params = val_gen_params.copy()
        val_metadata.to_csv(
            os.path.join(config['trainer']['model_dir'], 'val_metadata.csv'),
            sep=','
        )
    else:
        train_metadata, test_metadata, split_idx = tt.train_test_split()
        val_dataset = None
        train_gen_params = {}
        if 'augmentations' in config['trainer']:
            train_gen_params['augmentations'] = (
                config['trainer']['augmentations']
            )
    test_metadata.to_csv(os.path.join(config['trainer']['model_dir'],
                                      'test_metadata.csv'),
                         sep=',')

    train_dataset = DataSetWithMask(
        input_fnames=train_metadata['fpaths_input'],
        target_fnames=train_metadata['fpaths_target'],
        mask_fnames=train_metadata['fpaths_mask'],
        batch_size=config['trainer']['batch_size'],
        **train_gen_params
    )
    test_dataset = DataSetWithMask(
        input_fnames=test_metadata['fpaths_input'],
        target_fnames=test_metadata['fpaths_target'],
        mask_fnames=test_metadata['fpaths_mask'],
        batch_size=config['trainer']['batch_size']
    )
    return train_dataset, val_dataset, test_dataset, split_idx


def run_action(args):
    """Performs training or tune hyper parameters

    :param Namespace args: namespace containing the arguments passed
    """

    action = args.action
    config = read_config(args.config)
    if action == 'train':

        df_meta_fname = os.path.join(config['dataset']['data_dir'],
                                     'tiled_images_info.csv')
        df_meta = pd.read_csv(df_meta_fname)

        if 'weighted_loss' in config['trainer']:
            train_dataset, val_dataset, test_dataset, split_indices = \
                train_xyweights(df_meta, config)
        else:
            train_dataset, val_dataset, test_dataset, split_indices = \
                train_xy(df_meta, config)
        split_idx_fname = os.path.join(config['trainer']['model_dir'],
                                       'split_indices.pkl')

        with open(split_idx_fname, 'wb') as f:
            pickle.dump(split_indices, f)

        if 'model_name' in config['trainer']:
            model_name = config['trainer']['model_name']
        else:
            model_name = None

        trainer = BaseKerasTrainer(config=config,
                                   model_dir=config['trainer']['model_dir'],
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   model_name=model_name,
                                   gpu_ids=args.gpu,
                                   gpu_mem_frac=args.gpu_mem_frac)
        trainer.train()

    elif action == 'tune_hyperparam':
        raise NotImplementedError
    else:
        raise TypeError(('action {} not permitted. options: train or '
                        'tune_hyperparam').format(action))


if __name__ == '__main__':
    args = parse_args()
    gpu_available = False
    if args.gpu >= 0:
        gpu_available = check_gpu_availability(args.gpu, args.gpu_mem_frac)
    if not isinstance(args.gpu, int):
        raise NotImplementedError
    if gpu_available:
        run_action(args)
