#frameworks
from ast import Assert
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

#io
import os
import sys
from PIL import Image
import pandas as pd

#tools
import collections
from collections.abc import Iterable

#microDL dependencies
import micro_dl.cli.train_script as train
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.masks as mask_utils
import micro_dl.utils.normalize as norm_utils

class TorchDataset(Dataset): 
    '''
    Based off of torch.utils.data.Dataset: 
        - https://pytorch.org/docs/stable/data.html
        
    Custom dataset class that draws samples from a  'micro_dl.input.dataset.BaseDataSet' object, and converts them to pytorch
    inputs.
    
    Also takes lists of transformation classes. Transformations should be primarily designed for data refactoring and type conversion,
    since augmentations are already applied to tensorflow BaseDataSet object.
    
    The dataset object will cache samples so that the processing required to produce samples does not need to be repeated. This drastically
    speeds up training time.
    
    '''
    def __init__(self, train_config = None, tf_dataset = None, transforms=None, target_transforms=None):
        '''
        Init object.
        Params:
            train_config -> str: path to .yml config file from which to create BaseDataSet object if none given
            tf_dataset -> micro_dl.input.dataset.BaseDataSet: tensorflow-based dataset containing samples to convert
            transforms -> iterable(Transform object): transforms to be applied to every sample 
                                                      *after tf_dataset transforms*
            target_transforms -> iterable(Transform object): transforms to be applied to every target 
                                                             *after tf_dataset transforms*
        '''
        assert train_config or tf_dataset, 'Must provide either train config file or tf dataset'

        self.tf_dataset = None
        self.transforms = transforms
        self.target_transforms = target_transforms
        
        if tf_dataset != None:
            self.tf_dataset = tf_dataset
            self.sample_cache = collections.defaultdict(lambda : None)
            self.train_dataset = None
            self.test_dataset = None
            self.val_dataset = None
            self.mask = True
        else:
            config = aux_utils.read_config(train_config)
            
            dataset_config, trainer_config = config['dataset'], config['trainer']
            tile_dir, image_format = train.get_image_dir_format(dataset_config)
            
            tiles_meta = pd.read_csv(os.path.join(tile_dir, 'frames_meta.csv'))
            tiles_meta = aux_utils.sort_meta_by_channel(tiles_meta)
            
            masked_loss = False
            # if 'masked_loss' in trainer_config:
            #     masked_loss = trainer_config['masked_loss']

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
        '''
        Returns number of sample (or sample stack)/target pairs in dataset
        '''
        if self.tf_dataset:
            return len(self.tf_dataset)
        else:
            return sum([1 if self.train_dataset else 0, 1 if self.test_dataset else 0, 1 if self.val_dataset else 0])

    def __getitem__(self, idx):
        '''
        If acting as a dataset object, returns the sample target pair at 'idx'
        in dataset, after applying augment/transformations to sample/target pairs.
        
        If acting as a dataset container object, returns subsidary dataset
        objects.
        
        Params:
            - idx: index of dataset item to transform and return
        '''
        # if acting as dataset object
        if self.tf_dataset:
            if self.sample_cache[idx]:
                assert len(self.sample_cache[idx]) > 0, 'Sample caching error'
            else:
                sample = self.tf_dataset[idx]
                sample_input = sample[0]
                sample_target = sample[1]

                if self.transforms:
                    for transform in self.transforms:
                        sample_input = transform(sample_input)

                if self.target_transforms:
                    for transform in self.target_transforms:
                        sample_target = transform(sample_target)

                #depending on the transformation we might return lists or tuples, which we must unpack
                self.sample_cache[idx] = self.unpack(sample_input, sample_target)
        
            return self.sample_cache[idx]
        
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
            
    def unpack(self, sample_input, sample_target):
        ''' 
        Helper function for unpacking tuples returned by some transformation objects
        (e.g. GenerateMasks) into outputs.
        
        Unpacking before returning allows transformation functions which produce variable amounts of
        additional tensor information to pack that information in tuples with the sample and target
        tensors. 
        
        Params:
            - sample_input -> torch.Tensor or tuple(torch.Tensor): input sample to unpack
            - sample_target -> torch.Tensor or tuple(torch.Tensor): target sample to unpack
        '''
        inp, targ = type(sample_input), type(sample_target)
        
        if inp == list or inp == tuple:
            if targ == list or targ == tuple:
                return (*sample_input, *sample_target)
            else:
                return (*sample_input, sample_target)
        else:
            if targ == list or targ == tuple:
                return (sample_input, *sample_target)
            else:
                return (sample_input, sample_target)



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
        self.input_format = input_format
    def __call__(self,sample):
        if self.input_format == 'zxy':
            x_ind, y_ind= -2, -1
        elif self.input_format == 'xyz':
            x_ind, y_ind = -3, -2
        
        x_shape, y_shape = sample.shape[x_ind], sample.shape[y_ind]
        assert self.tile_size[0] < y_shape and self.tile_size[1] < x_shape, f'Sample size {(x_shape, y_shape)} must be greater than tile size {self.tile_size}.'
               
        randx = np.random.randint(0, x_shape - self.tile_size[1])
        randy = np.random.randint(0, y_shape - self.tile_size[0])
        
        sample = sample[randy:randy+self.tile_size[0], randx:randx+self.tile_size[1]]
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
        self.input_format = input_format
    def __call__(self, sample):
        if self.input_format == 'zxy':
            sample = sample[...,self.bands[0]:self.bands[1],:,:]
        elif self.input_format == 'xyz':
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

    
class GenerateMasks(object):
    '''
    Appends target channel thresholding based masks for each sample to the sample in a third channel, ordered respective to the order of each sample within its minibatch.
    Masks generated are torch tensors.
    
    Params:
        - masking_type -> token{'rosin', 'otsu'}: type of thresholding to apply:
                                                    1.) Rosin/unimodal: https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf
                                                    2.) Otsu: https://en.wikipedia.org/wiki/Otsu%27s_method
        - clipping -> Boolean: whether or not to clip the extraneous values in the data before thresholding
        - clip_amount -> int, tuple: amount to clip from both ends of brighness histogram as a percentage (%)
                                     If clipping==True but clip_amount == 0, clip for default amount (2%)
    '''
    def __init__(self, masking_type = 'rosin', clipping = False, clip_amount = 0):
        assert masking_type in {'rosin', 'unimodal', 'otsu'}, f'Unaccepted masking type: {masking_type}'
        self.masking_type = masking_type
        self.clipping = clipping
        self.clip_amount = clip_amount
    def __call__(self, sample):
        original_sample = sample
        
        #convert to numpy
        if type(sample) != type(np.asarray([1,1])):
            sample = sample.detach().cpu().numpy()
            
        # clip top and bottom 2% of images for better thresholding
        if self.clipping:
            if type(self.clip_amount) == tuple:
                sample = norm_utils.hist_clipping(sample, self.clip_amount[0], 100 - self.clip_amount[1])
            else:
                if self.clip_amount != 0:
                    sample = norm_utils.hist_clipping(sample, self.clip_amount, 100 - self.clip_amount)
                else:
                    sample = norm_utils.hist_clipping(sample)
        
        #generate masks
        masks = []
        for i in range(sample.shape[0]):
            if self.masking_type == 'otsu':
                masks.append(mask_utils.create_otsu_mask(sample[i,0,0]))
            elif self.masking_type == 'rosin' or self.masking_type == 'unimodal':
                masks.append(mask_utils.create_unimodal_mask(sample[i,0,0]))
            else:
                raise NotImplementedError(f'Masking type {self.masking_type} not implemented.')
                break
        masks = ToTensor()(np.asarray(masks)).unsqueeze(1).unsqueeze(1)
        
        return [original_sample, masks]

class Normalize(object):
    '''
    Normalizes the sample sample according to the mode in init.
    
    Params:
        - mode -> token{'one', 'max', 'zero'}: type of normalization to apply
                        - one: normalizes sample values proportionally between 0 and 1
                        - zeromax: centers sample around zero according to half of its normalized (between -1 and 1) maximum
                        - median: centers samples around zero, according to their respective median, then normalizes (between -1 and 1)
                        - mean: centers samples around zero, according to their respective means, then normalizes (between -1 and 1)
    '''
    def __init__(self, mode = 'max'):
        self.mode = mode
    def __call__(self, sample, scaling = 1):
        '''
        Forward call of Normalize
        Params:
            - sample -> torch.Tensor or numpy.ndarray: sample to normalize
            - scaling -> float: value to scale output normalization by
        '''
        #determine module
        if isinstance(sample, torch.Tensor):
            module = torch
        elif isinstance(sample, np.ndarray):
            module = np
        else:
            raise NotImplementedError('Only numpy array and torch tensor inputs handled.')
        
        #apply normalization
        if self.mode == 'one':
            sample = (sample - module.min(sample))/(module.max(sample) - module.min(sample))
        elif self.mode == 'zeromax':
            sample = (sample - module.min(sample))/(module.max(sample) - module.min(sample))
            sample = sample - module.max(sample)/2
        elif self.mode == 'median':
            sample = (sample - module.median(sample))
            sample = sample / module.max(module.abs(sample))
        elif self.mode == 'mean':
            sample = (sample - module.mean(sample))
            sample = sample / module.max(module.abs(sample))
        else:
            raise NotImplementedError(f'Unhandled mode type: \'{self.mode}\'.')

        return sample * scaling
    

class RandomNoise(object):
    '''
    Augmentation for applying random noise. High variance.
    
    Params:
        - noise_type -> token{'gauss', 's&p', 'poisson', 'speckle'}: type of noise to apply (see token names)
        - sample -> numpy.ndarray or torch.tensor: input sample
    
    returns:
        - noisy sample of type input type
    '''
    def __init__(self, noise_type):
        self.noise_type = noise_type
    def __call__(self, sample):
        pt = False
        if isinstance(sample, torch.Tensor):
            pt = True
            sample = sample.detach().cpu().numpy()

        if self.noise_type == "gauss":
            row,col,ch= sample.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = sample + gauss
            return noisy
        
        elif self.noise_type == "s&p":
            row,col,ch = sample.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(sample)
            
            # Salt mode
            num_salt = np.ceil(amount * sample.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in sample.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* sample.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in sample.shape]
            out[coords] = 0
            return out
        
        elif self.noise_typ == "poisson":
            vals = len(np.unique(sample))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(sample * vals) / float(vals)
            return noisy
        
        elif self.noise_typ =="speckle":
            row,col,ch = sample.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = sample + sample * gauss
            return noisy
        
        if pt:
            sample = ToTensor()(sample)
        return sample