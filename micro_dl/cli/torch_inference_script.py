#%%
import os
import yaml
import datetime
import torch
import micro_dl.inference.image_inference as image_inf
import micro_dl.torch_unet.utils.inference as torch_inference_utils
import micro_dl.utils.train_utils as train_utils

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
        '--gpu',
        type=int,
        default=None,
        help=('Optional: specify the gpu to use: 0,1,...',
              ', -1 for debugging. Default: pick best GPU'),
    )
    parser.add_argument(
        '--gpu_mem_frac',
        type=float,
        default=None,
        help='Optional: specify gpu memory fraction to use',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='path to yaml configuration file',
    )
    args = parser.parse_args()
    return args

def check_save_folder(inf_cfig, prep_cfig):
    '''
    Helper method to ensure that save folder exists.
    If no save folder specified in inference_config, force saving in data
    directory with dynamic name and timestamp.
    
    :param pd.dataframe inference_config: inference config file (not) containing save_folder_name
    :param pd.dataframe preprocess_config: preprocessing config file containing input_dir
    '''
    
    if 'save_folder_name' not in inf_cfig:
        assert 'input_dir' in prep_cfig, 'Error in autosaving: \'input_dir\'' \
            'unspecified in preprocess config'
        now = str(datetime.datetime.now()).replace(' ', '_').replace(':','_').replace('-','_')[:-10]
        save_dir = os.path.join(prep_cfig['input_dir'], f'../prediction_{now}')
        
        if os.path.exists(save_dir):
            os.makedirs(save_dir)
        prep_cfig['save_folder_name'] = save_dir
        print(f'No save folder specified in inference config: automatically saving predictions in : \n\t{save_dir}')
__name__ = 0
if __name__ == '__main__':
    args = parse_args()
    torch_config = read_config(args.config)
    
    # Get GPU ID and memory fraction
    gpu_id, gpu_mem_frac = train_utils.select_gpu(
        args.gpu,
        args.gpu_mem_frac,
    )
    device = torch.device(gpu_id)
    
    #read configuration parameters and metadata
    preprocess_config = read_config(torch_config['preprocess_config_path'])
    train_config = read_config(torch_config['train_config_path'])
    inference_config = read_config(torch_config['inference_config_path'])
    
    network_config = torch_config['model']
    
    #if no save_folder_name specified, automatically incur saving in data folder
    check_save_folder(inference_config, preprocess_config)

    #instantiate and prep TorchPredictor interfacing object
    torch_predictor = torch_inference_utils.TorchPredictor(network_config = network_config, device = device)
    torch_predictor.load_model_torch()
        
    #instantiate ImagePredictor object and run inference
    image_predictor = image_inf.ImagePredictor(train_config, inference_config, torch_predictor, 
                                               preprocess_config = preprocess_config)
    image_predictor.run_prediction()

#%%
torch_config = read_config('/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2022_09_27_A549_NuclStain/09_30_2022_15_09/torch_config_2D.yml')
#torch_config = read_config('/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset/09_30_2022_12_06/torch_config_25D.yml')
device = torch.device('cuda:0')

#read configuration parameters and metadata
preprocess_config = read_config(torch_config['preprocess_config_path'])
train_config = read_config(torch_config['train_config_path'])
inference_config = read_config(torch_config['inference_config_path'])

network_config = torch_config['model']

#if no save_folder_name specified, automatically incur saving in data folder
check_save_folder(inference_config, preprocess_config)

#instantiate and prep TorchPredictor interfacing object
torch_predictor = torch_inference_utils.TorchPredictor(network_config = network_config, device = device)
torch_predictor.load_model_torch()
    
#instantiate ImagePredictor object and run inference
image_predictor = image_inf.ImagePredictor(train_config, inference_config, torch_predictor, 
                                            preprocess_config = preprocess_config)
image_predictor.run_prediction()
# %%
