#!/usr/bin/env python
"""Train neural network models in keras"""
import argparse
import yaml

from utils.train_utils import check_gpu_availability


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify the gpu to use: 0,1,...')
    parser.add_argument('--gpu_mem_frac', type=float, default=1,
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

    :param str config_fname: fname of config yaml with its full path
    """

    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    return config


def pre_process(meta_preprocess):
    """Split and crop volumes from lif data

    :param dict meta_preprocess: dict with keys [input_fname, base_output_dir,
     split_volumes, crop_volumes]
    """

    #CHECK HOW TO IMPORT THIS CLASS DYNAMICALLY
    preprocess_cls = meta_preprocess['class']
    preprocessor = preprocess_cls(
        base_output_dir=meta_preprocess['base_output_dir'],
        verbose=meta_preprocess['verbose']
    )
    if meta_preprocess['split_volumes']:
        preprocessor.save_image_volumes(meta_preprocess['input_fname'])
    if meta_preprocess['crop_volumes']:
        preprocessor.crop_image_volumes(
            meta_preprocess['crop_volumes']['channels'],
            meta_preprocess['crop_volumes']['tile_size'],
            meta_preprocess['crop_volumes']['step_size'],
            meta_preprocess['crop_volumes']['normalize']
        )

def run_action(args):
    """Performs training or tune hyper parameters

    :param Namespace args: namespace containing the arguments passed
    :return: None
    """

    action = args.action
    config = read_config(args.config)
    if action=='train':
        if config['dataset']['preprocess']:
            preprocess_meta = config['dataset']['preprocess']
            prepocess_meta['verbose'] = config['verbose']
            pre_process(preprocess_meta)
        training_table_class = config['dataset']['training_table_class']
        for channel in config['dataset']['input_channels']:
            df_fname = os.path.join(config['dataset']['data_dir'], )
        df_train, df_test = training_table_class()

        CsvBatchGenerator()




        # From the data set portion initiate the data manager
        # dm = DataManager(main_path, verbose_level)
        # dm.create_model(params)
        # From the model portion, initiate the model
    elif action=='tune_hyperparam':
        raise NotImplementedError
    else:
        raise TypeError(('action {} not permitted. options: train or '
                        'tune_hyperparam').format(action))

if __name__ == '__main__':
    args = parse_args()
    gpu_availability = check_gpu_availability(args.gpu,
                                              args.gpu_mem_frac)
    if not isinstance(args.gpu, int):
        raise NotImplementedError
    run_action(args)




