import argparse
import yaml

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.torch_unet.utils.training as train

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
    parser.add_argument(
        '--gpu',
        type=str,
        help='intended gpu device number',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch_config = aux_utils.read_config(args.config)
    network_config = torch_config['model']
    training_config = torch_config['training']
    
    #Instantiate training object
    trainer = train.TorchTrainer(torch_config)
    
    #generate dataloaders and init model
    trainer.generate_dataloaders()
    trainer.load_model()
    
    #train
    trainer.train()
