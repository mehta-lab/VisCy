#!/usr/bin/env python
"""Train neural network models in keras"""
import argparse
from keras import Model
import keras.backend as K
from keras.utils import plot_model
import os
import pandas as pd
import tensorflow as tf
import yaml

from micro_dl.input.dataset import BaseDataSet, DataSetWithMask
from micro_dl.input.training_table import BaseTrainingTable
from micro_dl.train.model_inference import load_model
from micro_dl.train.trainer import BaseKerasTrainer
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.train_utils as train_utils


def parse_args():
    """
    Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help=('Default: find GPU with most memory.',
              'Optional: specify the gpu to use: 0,1,...',
              ', -1 for debugging'),
    )
    parser.add_argument(
        '--gpu_mem_frac',
        type=float,
        default=None,
        help=('Default: max memory fraction for given GPU ID.'
              'Optional: specify the gpu memory fraction to use [0, 1]'),
    )
    parser.add_argument(
        '--action',
        type=str,
        default='train',
        choices=('train', ),  # as only train supported currently
        help='action to take on the model: train, ',
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config',
        type=str,
        help='path to yaml configuration file',
    )
    group.add_argument(
        '--model_fname',
        type=str,
        default=None,
        help='full path of saved model to resume training',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=-1,
        help='port to use for the tensorboard callback',
    )
    return parser.parse_args()


def create_datasets(df_meta,
                    tile_dir,
                    dataset_config,
                    trainer_config,
                    image_format,
                    masked_loss):
    """Create train, val and test datasets

    Saves val_metadata.csv and test_metadata.csv for checking model performance

    :param pd.DataFrame df_meta: Dataframe containing info on split tiles
    :param str tile_dir: directory containing training image tiles
    :param dict dataset_config: dict with dataset related params
    :param dict trainer_config: dict with params related to training
    :param str image_format: Tile shape order: 'yxz' or 'zyx'
    :param bool masked_loss: Whether or not to use masks
    :return: Dict containing
     :BaseDataSet df_train: training dataset
     :BaseDataSet df_val: validation dataset
     :BaseDataSet df_test: test dataset
     :dict split_idx: dict with keys [train, val, test] and list of sample
      numbers as values
    """
    mask_channels = None
    if masked_loss:
        mask_channels = dataset_config['mask_channels']

    random_seed = None
    if 'random_seed' in dataset_config:
        random_seed = dataset_config['random_seed']

    tt = BaseTrainingTable(
        df_metadata=df_meta,
        input_channels=dataset_config['input_channels'],
        target_channels=dataset_config['target_channels'],
        split_by_column=dataset_config['split_by_column'],
        split_ratio=dataset_config['split_ratio'],
        mask_channels=mask_channels,
        random_seed=random_seed
    )
    all_metadata, split_samples = tt.train_test_split()
    csv_names = ['train_metadata.csv', 'val_metadata.csv', 'test_metadata.csv']
    df_names = ['df_train', 'df_val', 'df_test']
    all_datasets = {}
    for i in range(3):
        metadata = all_metadata[df_names[i]]
        if isinstance(metadata, type(None)):
            all_datasets[df_names[i]] = None
        else:
            if masked_loss:
                dataset = DataSetWithMask(
                    tile_dir=tile_dir,
                    input_fnames=metadata['fpaths_input'],
                    target_fnames=metadata['fpaths_target'],
                    mask_fnames=metadata['fpaths_mask'],
                    dataset_config=dataset_config,
                    batch_size=trainer_config['batch_size'],
                    image_format=image_format,
                )
            else:
                dataset = BaseDataSet(
                    tile_dir=tile_dir,
                    input_fnames=metadata['fpaths_input'],
                    target_fnames=metadata['fpaths_target'],
                    dataset_config=dataset_config,
                    batch_size=trainer_config['batch_size'],
                    image_format=image_format,
                )
            metadata.to_csv(
                os.path.join(trainer_config['model_dir'], csv_names[i]),
                sep=','
            )
            all_datasets[df_names[i]] = dataset

    return all_datasets, split_samples


def create_network(network_config, gpu_id):
    """Create an instance of the network

    :param dict network_config: dict with network related params
    :param int gpu_id: gpu to use
    """

    params = ['class', 'filter_size', 'activation', 'pooling_type',
              'batch_norm', 'height', 'width', 'data_format',
              'final_activation']
    param_check, msg = aux_utils.validate_config(network_config, params)
    if not param_check:
        raise ValueError(msg)

    network_cls = network_config['class']
    network_cls = aux_utils.import_object('networks', network_cls)
    network = network_cls(network_config)
    # assert if network shape matches dataset shape?
    inputs, outputs = network.build_net()
    with tf.device('/gpu:{}'.format(gpu_id)):
        model = Model(inputs=inputs, outputs=outputs)
    return model


def get_image_dir_format(dataset_config):
    """Get dir with input images for generating full path from frames_meta

    If the tiled dir is passed as data dir there will be no
    preprocessing_info.json. If json present use it, else read images from the
    given dir.
    """

    # tile dir pass directly as data_dir
    tile_dir = dataset_config['data_dir']
    image_format = 'zyx'

    # If the parent dir with tile dir, mask dir is passed as data_dir,
    # it should contain a json with directory names
    json_fname = os.path.join(dataset_config['data_dir'],
                              'preprocessing_info.json')
    if os.path.exists(json_fname):
        preprocessing_info = aux_utils.read_json(json_filename=json_fname)

        # Preprocessing_info is a list of jsons. Use the last json. If a tile
        # (training data) dir is specified and exists in info json use that
        recent_json = preprocessing_info[-1]
        pp_config = recent_json['config']
        if 'tile' in pp_config and 'tile_dir' in pp_config['tile']:
            tile_dir = pp_config['tile']['tile_dir']

        # Get shape order from recent_json
        if 'image_format' in pp_config['tile']:
            image_format = pp_config['tile']['image_format']

    return tile_dir, image_format


def run_action(args, gpu_ids, gpu_mem_frac):
    """Performs training or tune hyper parameters

    Lambda layers throw errors when converting to yaml!
    model_yaml = self.model.to_yaml()

    :param Namespace args: namespace containing the arguments passed
    :param int gpu_ids: GPU ID
    :param float gpu_mem_frac: Available GPU memory fraction
    """
    action = args.action
    config = aux_utils.read_config(args.config)
    dataset_config = config['dataset']
    trainer_config = config['trainer']
    network_config = config['network']

    # Safety check: 2D UNets needs to have singleton dimension squeezed
    if network_config['class'] == 'UNet2D':
        dataset_config['squeeze'] = True
    elif network_config['class'] == 'UNetStackTo2D':
        dataset_config['squeeze'] = False

    # Check if masked loss exists
    masked_loss = False
    if 'masked_loss' in trainer_config:
        masked_loss = trainer_config["masked_loss"]

    tile_dir, image_format = get_image_dir_format(dataset_config)

    if action == 'train':
        # Create directory where model will be saved
        if not os.path.exists(trainer_config['model_dir']):
            os.makedirs(trainer_config['model_dir'], exist_ok=True)
        # Get tile directory from preprocessing info and load metadata
        tiles_meta = pd.read_csv(os.path.join(tile_dir, 'frames_meta.csv'))
        tiles_meta = aux_utils.sort_meta_by_channel(tiles_meta)
        # Generate training, validation and test data sets
        all_datasets, split_samples = create_datasets(
            tiles_meta,
            tile_dir,
            dataset_config,
            trainer_config,
            image_format,
            masked_loss,
        )
        # Save train, validation and test indices
        split_idx_fname = os.path.join(trainer_config['model_dir'],
                                       'split_samples.json')
        aux_utils.write_json(split_samples, split_idx_fname)

        K.set_image_data_format(network_config['data_format'])

        if gpu_ids == -1:
            sess = None
        else:
            sess = train_utils.set_keras_session(
                gpu_ids=gpu_ids,
                gpu_mem_frac=gpu_mem_frac,
            )

        if args.model_fname:
            # load model only loads the weights, have to save intermediate
            # states of gradients to resume training
            model = load_model(network_config, args.model_fname)
        else:
            with open(os.path.join(trainer_config['model_dir'],
                                   'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            model = create_network(network_config, gpu_ids)
            plot_model(model,
                       to_file=os.path.join(trainer_config['model_dir'],
                                            'model_graph.png'),
                       show_shapes=True, show_layer_names=True)

        num_target_channels = network_config['num_target_channels']
        trainer = BaseKerasTrainer(sess=sess,
                                   train_config=trainer_config,
                                   train_dataset=all_datasets['df_train'],
                                   val_dataset=all_datasets['df_val'],
                                   model=model,
                                   num_target_channels=num_target_channels,
                                   gpu_ids=args.gpu,
                                   gpu_mem_frac=args.gpu_mem_frac)
        trainer.train()

    else:
        raise TypeError(('action {} not permitted. options: only train'
                         'supported currently').format(action))


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    # Get GPU ID and memory fraction
    gpu_id, gpu_mem_frac = train_utils.select_gpu(
        args.gpu,
        args.gpu_mem_frac,
    )
    run_action(args, gpu_id, gpu_mem_frac)

