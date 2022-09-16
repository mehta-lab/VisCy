from socket import ALG_SET_AEAD_ASSOCLEN
import yaml
import micro_dl.inference.image_inference as image_inf
import micro_dl.torch_unet.utils.inference as torch_inference_utils

import argparse

def read_config(config_path):
    '''
    One-line to safely open config files for argument reading
    
    :param str config_path: abs or relative path to configuration file
    
    :return dict config: a dictionary of config information read from input file
    '''
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """
    Parse command line arguments
    In python namespaces are implemented as dictionaries
    
    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='path to yaml configuration file',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch_config = read_config(args.config)
    
    #read configuration parameters and metadata
    preprocess_config = read_config(torch_config['preprocess_config_path'])
    train_config = read_config(torch_config['train_config_path'])
    inference_config = read_config(torch_config['inference_config_path'])
    
    network_config = torch_config['model']

    #instantiate and prep TorchPredictor interfacing object
    torch_predictor = torch_inference_utils.TorchPredictor(network_config = network_config)
    torch_predictor.load_model_torch()
        
    #instantiate ImagePredictor object and run inference
    image_predictor = image_inf.ImagePredictor(train_config, inference_config, preprocess_config,
                                              framework='torch', torch_predictor=torch_predictor)
    image_predictor.run_prediction()
