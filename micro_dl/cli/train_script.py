#!/usr/bin/env python
"""Train neural network models in keras"""
import argparse
from keras import Model
import keras.backend as K
from keras.utils import plot_model
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import yaml

from micro_dl.input.dataset import BaseDataSet, DataSetWithMask
from micro_dl.input.training_table import (
    BaseTrainingTable, TrainingTableWithMask
)
from micro_dl.train.model_inference import load_model
from micro_dl.train.trainer import BaseKerasTrainer
from micro_dl.utils.aux_utils import import_class, validate_config
from micro_dl.utils.train_utils import check_gpu_availability, \
    set_keras_session


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
    group.add_argument('--model_fname', type=str, default=None,
                       help='path to checkpoint file')
    parser.add_argument('--port', type=int, default=-1,
                        help='port to use for the tensorboard callback')
    args = parser.parse_args()
    return args


def read_config(config_fname):
    """Read the config file in yml format

    TODO: validate config!

    :param str config_fname: fname of config yaml with its full path
    :return: dict config
    """

    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    return config


def create_datasets(df_meta, dataset_config, trainer_config):
    """Create train, val and test datasets

    Saves val_metadata.csv and test_metadata.csv for checking model performance

    :param pd.DataFrame df_meta: Dataframe containing info on split tiles
    :param dict dataset_config: dict with dataset related params
    :param dict trainer_config: dict with params related to training
    :return:
     :BaseDataSet train_dataset
     :BaseDataSet val_dataset
     :BaseDataSet test dataset
     :dict split_idx: dict with keys [train, val, test] and list of sample
      numbers as values
    """

    tt = BaseTrainingTable(df_meta,
                           dataset_config['input_channels'],
                           dataset_config['target_channels'],
                           dataset_config['split_by_column'],
                           dataset_config['split_ratio'])
    if 'val' in dataset_config['split_ratio']:
        train_metadata, val_metadata, test_metadata, split_samples = \
            tt.train_test_split()
        val_gen_params = {}
        if 'augmentations' in trainer_config:
            val_gen_params['augmentations'] = (
                trainer_config['augmentations']
            )

        val_dataset = BaseDataSet(input_fnames=val_metadata['fpaths_input'],
                                  target_fnames=val_metadata['fpaths_target'],
                                  batch_size=trainer_config['batch_size'],
                                  **val_gen_params)
        train_gen_params = val_gen_params.copy()
        val_metadata.to_csv(
            os.path.join(trainer_config['model_dir'], 'val_metadata.csv'),
            sep=','
        )
    else:
        train_metadata, test_metadata, split_samples = tt.train_test_split()
        val_dataset = None
        train_gen_params = {}
        if 'augmentations' in trainer_config:
            train_gen_params['augmentations'] = (
                trainer_config['augmentations']
            )
    test_metadata.to_csv(
        os.path.join(trainer_config['model_dir'], 'test_metadata.csv'),
        sep=','
    )

    train_dataset = BaseDataSet(input_fnames=train_metadata['fpaths_input'],
                                target_fnames=train_metadata['fpaths_target'],
                                batch_size=trainer_config['batch_size'],
                                **train_gen_params)
    test_dataset = BaseDataSet(input_fnames=test_metadata['fpaths_input'],
                               target_fnames=test_metadata['fpaths_target'],
                               batch_size=trainer_config['batch_size'])
    return train_dataset, val_dataset, test_dataset, split_samples


def create_datasets_with_mask(df_meta, dataset_config, trainer_config):
    """Create train, val and test datasets

    :param pd.DataFrame df_meta: Dataframe containing info on split tiles
    :param dict dataset_config: dict with dataset related params
    :param dict trainer_config: dict with params related to training
    :return:
     :BaseDataSet train_dataset: y_true has mask concatenated at the end
     :BaseDataSet val_dataset
     :BaseDataSet test dataset
     :dict split_idx: dict with keys [train, val, test] and list of sample
      numbers as values
    """

    if 'min_fraction' in dataset_config:
        min_fraction = dataset_config['min_fraction']
    else:
        min_fraction = None
    tt = TrainingTableWithMask(df_meta,
                               dataset_config['input_channels'],
                               dataset_config['target_channels'],
                               dataset_config['mask_channels'],
                               dataset_config['split_by_column'],
                               dataset_config['split_ratio'],
                               min_fraction)
    if 'val' in dataset_config['split_ratio']:
        train_metadata, val_metadata, test_metadata, split_samples = \
            tt.train_test_split()
        val_gen_params = {}
        if 'label_weights' in dataset_config:
            val_gen_params['label_weights'] = dataset_config['label_weights']
        if 'augmentations' in trainer_config:
            val_gen_params['augmentations'] = (
                trainer_config['augmentations']
            )

        val_dataset = DataSetWithMask(
            input_fnames=val_metadata['fpaths_input'],
            target_fnames=val_metadata['fpaths_target'],
            mask_fnames=val_metadata['fpaths_mask'],
            batch_size=trainer_config['batch_size'],
            **val_gen_params
        )
        train_gen_params = val_gen_params.copy()
        val_metadata.to_csv(
            os.path.join(trainer_config['model_dir'], 'val_metadata.csv'),
            sep=','
        )
    else:
        train_metadata, test_metadata, split_samples = tt.train_test_split()
        val_dataset = None
        train_gen_params = {}
        if 'label_weights' in dataset_config:
            train_gen_params['label_weights'] = dataset_config['label_weights']
        if 'augmentations' in trainer_config:
            train_gen_params['augmentations'] = (
                trainer_config['augmentations']
            )
    test_metadata.to_csv(os.path.join(trainer_config['model_dir'],
                                      'test_metadata.csv'),
                         sep=',')

    train_dataset = DataSetWithMask(
        input_fnames=train_metadata['fpaths_input'],
        target_fnames=train_metadata['fpaths_target'],
        mask_fnames=train_metadata['fpaths_mask'],
        batch_size=trainer_config['batch_size'],
        **train_gen_params
    )
    test_gen_params = {}
    if 'label_weights' in dataset_config:
        test_gen_params['label_weights'] = dataset_config['label_weights']
    test_dataset = DataSetWithMask(
        input_fnames=test_metadata['fpaths_input'],
        target_fnames=test_metadata['fpaths_target'],
        mask_fnames=test_metadata['fpaths_mask'],
        batch_size=trainer_config['batch_size'],
        **test_gen_params
    )
    return train_dataset, val_dataset, test_dataset, split_samples


def create_network(network_config, gpu_id):
    """Create an instance of the network

    :param dict network_config: dict with network related params
    :param int gpu_id: gpu to use
    """

    params = ['class', 'filter_size', 'activation', 'pooling_type',
              'batch_norm', 'height', 'width', 'data_format',
              'final_activation']
    param_indicator = validate_config(network_config, params)
    assert np.all(param_indicator), \
        'Params absent in network_config: %s'.format(
            params[param_indicator == 0]
        )
    network_cls = network_config['class']
    network_cls = import_class('networks', network_cls)
    network = network_cls(network_config)
    # assert if network shape matches dataset shape?
    inputs, outputs = network.build_net()
    with tf.device('/gpu:{}'.format(gpu_id)):
        model = Model(inputs=inputs, outputs=outputs)
    return model


def run_action(args):
    """Performs training or tune hyper parameters

    Lambda layers throw errors when converting to yaml!
    model_yaml = self.model.to_yaml()

    :param Namespace args: namespace containing the arguments passed
    """

    action = args.action
    config = read_config(args.config)
    dataset_config = config['dataset']
    trainer_config = config['trainer']
    network_config = config['network']
    if action == 'train':
        df_meta_fname = os.path.join(dataset_config['data_dir'],
                                     'tiled_images_info.csv')
        df_meta = pd.read_csv(df_meta_fname)

        if 'masked_loss' in trainer_config:
            train_dataset, val_dataset, test_dataset, split_samples = \
                create_datasets_with_mask(df_meta,
                                          dataset_config,
                                          trainer_config)
        else:
            train_dataset, val_dataset, test_dataset, split_samples = \
                create_datasets(df_meta, dataset_config, trainer_config)

        split_idx_fname = os.path.join(trainer_config['model_dir'],
                                       'split_samples.pkl')
        with open(split_idx_fname, 'wb') as f:
            pickle.dump(split_samples, f)

        K.set_image_data_format(network_config['data_format'])

        if args.gpu == -1:
            sess = None
        else:
            sess = set_keras_session(gpu_ids=args.gpu,
                                     gpu_mem_frac=args.gpu_mem_frac)

        if args.model_fname:
            # load model only loads the weights, have to save intermediate
            # states of gradients to resume training
            model = load_model(network_config, args.model_fname)
        else:
            model = create_network(network_config, args.gpu)
            os.makedirs(trainer_config['model_dir'], exist_ok=True)
            plot_model(model,
                       to_file=os.path.join(trainer_config['model_dir'],
                                            'model_graph.png'),
                       show_shapes=True, show_layer_names=True)
            with open(os.path.join(trainer_config['model_dir'],
                                   'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        num_target_channels = network_config['num_target_channels']
        trainer = BaseKerasTrainer(sess=sess,
                                   train_config=trainer_config,
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   model=model,
                                   num_target_channels=num_target_channels,
                                   gpu_ids=args.gpu,
                                   gpu_mem_frac=args.gpu_mem_frac)
        trainer.train()

    elif action == 'tune_hyperparam':
        raise NotImplementedError
    else:
        raise TypeError(('action {} not permitted. options: train or '
                        'tune_hyperparam').format(action))


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    # Currently only supporting one GPU as input
    if not isinstance(args.gpu, int):
        raise NotImplementedError
    # If debug mode, run without checking GPUs
    if args.gpu == -1:
        run_action(args)
    # Get currently available GPU memory fractions and determine if
    # requested amount of memory is available
    gpu_mem_frac = args.gpu_mem_frac
    if isinstance(gpu_mem_frac, float):
        gpu_mem_frac = [gpu_mem_frac]
    gpu_available, curr_mem_frac = check_gpu_availability(
        args.gpu,
        gpu_mem_frac,
    )
    # Allow run if gpu_available
    if gpu_available:
        run_action(args)
    else:
        raise ValueError(
            "Not enough memory available. Requested/current fractions:",
            "\n".join([str(c) + " / " + "{0:.4g}".format(m)
                       for c, m in zip(gpu_mem_frac, curr_mem_frac)]),
        )
