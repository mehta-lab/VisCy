import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

class MyDataset(Dataset): 
    '''
    Custom dataset class that takes lists of transformation classes. Can design transformations for data augmentation 
    as well as data reformatting to be passed in.
    
    Contains custom __getitem__ to act as sample selection in iterable.
    
    '''
    def __init__(self, img_dict, transform=None, target_transform=None):
        self.img_dict = img_dict
        self.keys = [key for key in img_dict]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        key = self.keys[idx]
        phase = np.asarray(Image.open(self.img_dict[key][3]))
        #retardance = np.asarray(Image.open(self.img_dict[key][2]))
        #actin = np.asarray(Image.open(self.img_dict[key][1]))
        nuclei = np.asarray(Image.open(self.img_dict[key][0]))
        
        sample = {'image': phase, 'target': nuclei}
        if self.transform:
            sample = self.transform(sample)
        for key in sample:
            sample[key] = torch.unsqueeze(sample[key], 0)
        return sample
    
class Resize(object):
    '''
    Transformation. Resises called sample to 'output_size'.
    '''
    def __init__(self, output_size = (256, 256)):
        self.output_size = output_size
    def __call__(self, sample):
        sample['image'] = cv2.resize(sample['image'], self.output_size)
        sample['target'] = cv2.resize(sample['target'], self.output_size)
        return sample

class RandTile(object):
    '''
    Transformation. Selects and returns random tile size 'tile_size' from input.
    '''
    def __init__(self, tile_size = (256,256)):
        self.tile_size = tile_size
    def __call__(self,sample):
        assert self.tile_size[0] < sample['image'].shape[-1] and self.tile_size[1] < sample['image'].shape[-2], 'Sample size must be greater than tile size.'
               
        x, y = sample['image'].shape[-2], sample['image'].shape[-1]
        randx = np.random.randint(0, x - self.tile_size[1])
        randy = np.random.randint(0, y - self.tile_size[0])
        
        sample['image'] = sample['image'][randy:randy+self.tile_size[0], randx:randx+self.tile_size[1]]
        sample['target'] = sample['target'][randy:randy+self.tile_size[0], randx:randx+self.tile_size[1]]
        return sample
    
class Normalize(object):
    '''
    Transformation. Normalizes input sample to max value
    '''
    def __call__(self, sample):
        sample['image'] = sample['image']/np.max(sample['image'])
        sample['target'] = sample['target']/np.max(sample['target'])
        return sample
    
class RandFlip(object):
    '''
    Transformation. Flips input in random direction and returns.
    '''
    def __call__(self, sample):
        rand = np.random.randint(0,2,2)
        image = sample['image'].copy()
        target = sample['target'].copy()
        if rand[0]==1:
            image = np.flipud(image)
            target = np.flipud(target)
        if rand[1]==1:
            image = np.fliplr(image)
            target = np.fliplr(target)
        sample['image'] = image
        sample['target'] = target
        return sample
    
class chooseBands(object):
    '''
    Transformation. Selects specified z (or lambda) band range from 3d input image.
    '''
    def __init__(self, bands = (0, 30)):
        self.bands = bands
    def __call__(self, sample):
        sample['image'] = sample['image'][...,self.bands[0]:self.bands[1]]
        return sample
    
class toTensor(object):
    '''
    Transformation. Converts input to torch.Tensor and returns.
    '''
    def __init__(self, device = torch.device('cuda')):
        self.device = device
    def __call__(self, sample):
        sample['image'] = torch.tensor(sample['image'].copy(), dtype=torch.float32).to(self.device)
        sample['target'] = torch.tensor(sample['target'].copy(), dtype=torch.float32).to(self.device)
        return sample