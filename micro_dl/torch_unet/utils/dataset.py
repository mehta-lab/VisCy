#frameworks
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

#io
import os
import sys
from PIL import Image
import pandas as pd

#microDL dependencies
import micro_dl.cli.train_script as train
import micro_dl.utils.aux_utils as aux_utils

class TorchDataset(Dataset): 
    '''
    Based off of torch.utils.data.Dataset: 
        - https://pytorch.org/docs/stable/data.html
        
    Custom dataset class that draws samples from a  'micro_dl.input.dataset.BaseDataSet' object, and converts them to pytorch
    inputs.
    
    Also takes lists of transformation classes. Transformations should be primarily designed for data refactoring and type conversion,
    since augmentations are already applied to tensorflow BaseDataSet object.
    
    '''
    def __init__(self, train_config, tf_dataset = None, transforms=None, target_transforms=None):
        '''
        Init object.
        Params:
            train_config -> str: path to .yml config file from which to create BaseDataSet object if none given
            tf_dataset -> micro_dl.inpuit.dataset.BaseDataSet: tensorflow-based dataset from which to convert
            transforms -> Transform object: transforms to be applied to every sample *after tf_dataset transforms*
            target_transforms -> Transform object: transforms to be applied to every target *after tf_dataset transforms*
        '''
        self.tf_dataset = None
        self.transforms = transforms
        self.target_transforms = target_transforms
        
        if tf_dataset != None:
            self.tf_dataset = tf_dataset
            self.train_dataset = None
            self.test_dataset = None
            self.val_dataset = None
        else:
            config = aux_utils.read_config('../micro_dl/config_train_25D.yml')
            
            dataset_config, trainer_config = config['dataset'], config['trainer']
            tile_dir, image_format = train.get_image_dir_format(dataset_config)
            
            tiles_meta = pd.read_csv(os.path.join(tile_dir, 'frames_meta.csv'))
            tiles_meta = aux_utils.sort_meta_by_channel(tiles_meta)
            
            masked_loss = False
            if masked_loss in trainer_config:
                masked_loss = trainer_config['masked_loss']

            all_datasets, split_samples = train.create_datasets(tiles_meta, tile_dir, dataset_config, 
                                                                trainer_config, image_format, masked_loss)
            
            self.train_dataset = TorchDataset(None, tf_dataset = all_datasets['df_train'],
                                                 transforms = self.transforms,
                                                 target_transforms = self.target_transforms)
            self.test_dataset = TorchDataset(None, tf_dataset = all_datasets['df_test'],
                                             transforms = self.transforms,
                                             target_transforms = self.target_transforms)
            self.val_dataset = TorchDataset(None, tf_dataset = all_datasets['df_val'],
                                            transforms = self.transforms,
                                            target_transforms = self.target_transforms)
            
    def __len__(self):
        if self.tf_dataset:
            return len(self.tf_dataset)
        else:
            return sum([1 if self.train_dataset else 0, 1 if self.test_dataset else 0, 1 if self.val_dataset else 0])

    def __getitem__(self, idx):
        # if acting as dataset object
        if self.tf_dataset:
            sample = self.tf_dataset[idx]
            sample_input = sample[0]
            sample_target = sample[1]
            
            if self.transforms:
                for transform in self.transforms:
                    sample_input = transform(sample_input)
                    
            if self.target_transforms:
                for transform in self.target_transforms:
                    sample_target = transform(sample_target)
            
            return (sample_input, sample_target)
        
        # if acting as container object of dataset objects
        else:
            keys = {}
            if self.val_dataset:
                keys['val'] = self.val_dataset
            if self.train_dataset:
                keys['train'] = self.train_dataset
            if self.test_dataset:
                keys['test'] = self.test_dataset
            
            if idx in keys:
                return keys[idx]
            else:
                raise KeyError(f'This object is a container. Acceptable keys:{[k for k in keys]}')
                return
    
class Resize(object):
    '''
    Transformation. Resises called sample to 'output_size'.
    
    Note: Dangerous. Actually transforms data instead of cropping.
    '''
    def __init__(self, output_size = (256, 256)):
        self.output_size = output_size
    def __call__(self, sample):
        sample = cv2.resize(sample, self.output_size)
        sample = cv2.resize(sample, self.output_size)
        return sample

class RandTile(object):
    '''
    ******BROKEN*******
    Transformation. Selects and returns random tile size 'tile_size' from input.
    
    Note: some issues matching the target. Need to rethink implementation. Not necessary if preprocessing is tiling images anyways.
    '''
    def __init__(self, tile_size = (256,256), input_format = 'zxy'):
        self.tile_size = tile_size
    def __call__(self,sample):
        if input_format == 'zxy':
            x_ind, y_ind= -2, -1
        elif input_format == 'xyz':
            x_ind, y_ind = -3, -2
        
        x_shape, y_shape = sample.shape[x_ind], sample.shape[y_ind]
        assert self.tile_size[0] < y_shape and self.tile_size[1] < x_shape, f'Sample size {(x_shape, y_shape)} must be greater than tile size {self.tile_size}.'
               
        randx = np.random.randint(0, x_shape - self.tile_size[1])
        randy = np.random.randint(0, y_shape - self.tile_size[0])
        
        sample = sample[randy:randy+self.tile_size[0], randx:randx+self.tile_size[1]]
        return sample
    
class Normalize(object):
    '''
    Transformation. Normalizes input sample to max value
    '''
    def __call__(self, sample):
        if type(sample) == type(np.asarray([1,1])):
            sample = sample/np.max(sample)
        elif type(sample) == type(toTensor(np.asarray([1,1]))):
            sample = sample.numpy()
            sample = toTensor(sample)
        else:
            raise Exception('unhandled sample type. Try numpy.ndarray or torch.tensor')
        return sample
    
class RandFlip(object):
    '''
    Transformation. Flips input in random direction and returns.
    '''
    def __call__(self, sample):
        rand = np.random.randint(0,2,2)
        if rand[0]==1:
            sample = np.flipud(sample)
        if rand[1]==1:
            sample = np.fliplr(sample)
        return sample
    
class ChooseBands(object):
    '''
    Transformation. Selects specified z (or lambda) band range from 3d input image.
    Note: input format should be
    '''
    def __init__(self, bands = (0, 30), input_format = 'zxy'):
        assert input_format in {'zxy', 'xyz'}, 'unacceptable input format; try \'zxy\' or \'xyz\''
        self.bands = bands
    def __call__(self, sample):
        if input_format == 'zxy':
            sample = sample[...,self.bands[0]:self.bands[1],:,:]
        elif input_format == 'xyz':
            sample = sample[...,self.bands[0]:self.bands[1]]
        return sample
    
class ToTensor(object):
    '''
    Transformation. Converts input to torch.Tensor and returns. By default also places tensor on gpu.
    '''
    def __init__(self, device = torch.device('cuda')):
        self.device = device
    def __call__(self, sample):
        sample = torch.tensor(sample.copy(), dtype=torch.float32).to(self.device)
        return sample