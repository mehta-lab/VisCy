# Last edit: Christian Foley, 08/30/2022

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from micro_dl.torch_unet.networks.layers.ConvBlock3D import *


class Unet25d(nn.Module):
    def __name__(self):
        return 'Unet25d'
    
    def __init__(self,
                 in_channels = 1,
                 out_channels = 1,
                 in_stack_depth = 5,
                 out_stack_depth = 1,
                 xy_kernel_size = (3,3),
                 residual = False, 
                 dropout = 0.2,
                 num_blocks = 4,
                 num_block_layers = 3,
                 num_filters = [], 
                 task = 'seg'):
        '''
        Instance of 2.5D Unet. 
        1.) https://elifesciences.org/articles/55502
        
        Architecture takes in stack of 2d inputs given as a 3d tensor and returns a 2d interpretation. Learns 3d information based upon input stack, but 
        speeds up training by compressing 3d information before the decoding path. Uses interruption conv layers in the Unet skip paths to compress
        information with z-channel convolution.
        
        Parameters
            - in_channels -> int: number of feature channels in
            - out_channels -> int: number of feature channels out
            - input_stack_depth -> int: depth of input
            - output_stack_depth -> int: depth of output
            - xy_kernel_size -> int or tuple(int, int): size of x and y dimensions of conv kernels in blocks
            - residual -> boolean: see name
            - down_mode -> token{'avgpool', 'maxpool', 'conv'}: type of downsampling in encoder path
            - up_mode -> token{see link}: type of upsampling in decoder path (https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
            - activation -> token{'relu', 'elu', 'selu', 'leakyrelu'}: activation function to use in convolutional blocks
            - num_blocks -> int: number of convolutional blocks on encoder and decoder paths
            - num_block_layers -> int: number of layers per block
            - num_filters -> list[int]: sequence of filters/feature levels at each conv block depth
            - task -> token{'recon','seg','reg'}: network task (for virtual staining this is regression)
            - bottom_block_spatial -> boolean: whether or not the bottom feature compression block learns spatial information as well
            
        '''
        super(Unet25d, self).__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.ks = xy_kernel_size
        self.residual = residual
        self.dropout = dropout
        self.task = task
        
        #----- set static parameters -----#
        self.block_padding = 'same'
        down_mode = 'avgpool' #TODO set static avgpool
        up_mode = 'trilinear' #TODO set static trilinear
        activation = 'relu' #TODO set static relu
        self.bottom_block_spatial = False #TODO set static
        
        
        #----- Standardize Filter Sequence -----#
        if len(num_filters) != 0:
            assert len(num_filters) == num_blocks, 'Length of num_filters must be equal to num_blocks + 1 (number of convolutional blocks per path).'
            self.num_filters = num_filters
        else:
            self.num_filters = [pow(2,i)*16 for i in range(num_blocks + 1)]
            self.num_filters
        forward_filters = [in_channels] + self.num_filters
        backward_filters = [self.num_filters[-(i+1)] + self.num_filters[-(i+2)] for i in range(len(self.num_filters)) if i < len(self.num_filters) - 1] + [out_channels]
        
        #----- Downsampling steps -----#
        self.down_list = []
        if down_mode == 'maxpool':
            for i in range(num_blocks):
                self.down_list.append(nn.MaxPool3d(kernel_size=(1,2,2)))
        elif down_mode == 'avgpool':
            for i in range(num_blocks):
                self.down_list.append(nn.AvgPool3d(kernel_size=(1,2,2)))
        elif down_mode == 'conv':
            raise NotImplementedError('Not yet implemented!')
            #TODO: implement.
        self.register_modules(self.down_list, 'down_samp')
        
        
        #----- Upsampling steps -----#
        self.up_list = []
        for i in range(num_blocks):
            self.up_list.append(nn.Upsample(scale_factor=(1,2,2), mode = up_mode, align_corners=False))
        
        #----- Convolutional blocks -----# Forward Filters [1, 16, 32, 64, 128, 256] -> Backward Filters [128+256, 64+128, 32+64, 16+32, 1]
        self.down_conv_blocks = []
        for i in range(num_blocks):
            self.down_conv_blocks.append(ConvBlock3D(forward_filters[i], forward_filters[i+1],
                                                     dropout = self.dropout, residual = self.residual, activation = activation,
                                                     kernel_size = (3, self.ks[0], self.ks[1]), num_layers = num_block_layers))
        self.register_modules(self.down_conv_blocks, 'down_conv_block')
        
        if self.bottom_block_spatial:
            #TODO: residual must be false or dimensionality breaks. Fix later
            self.bottom_transition_block = ConvBlock3D(self.num_filters[-2], self.num_filters[-1], num_layers = 1, residual = False,
                                                   kernel_size = (1 + in_stack_depth - out_stack_depth, self.ks[0], self.ks[1]), padding = (0,1,1))
        else:
            self.bottom_transition_block = nn.Conv3d(self.num_filters[-2], self.num_filters[-1],
                                                   kernel_size = (1 + in_stack_depth - out_stack_depth, 1, 1), padding = 0)

        self.up_conv_blocks = []
        for i in range(num_blocks):
            self.up_conv_blocks.append(ConvBlock3D(backward_filters[i], forward_filters[-(i+2)], 
                                                   dropout = self.dropout, residual = self.residual, activation = activation,
                                                   kernel_size = (1, self.ks[0], self.ks[1]), num_layers = num_block_layers))
        self.register_modules(self.up_conv_blocks, 'up_conv_block')   
        
        
        #----- Skip Interruption Conv Blocks -----#
        self.skip_conv_layers = []
        for i in range(num_blocks):
            self.skip_conv_layers.append(nn.Conv3d(forward_filters[i+1], forward_filters[i+1],
                                                   kernel_size = (1 + in_stack_depth - out_stack_depth, 1, 1), padding = 'valid'))
        self.register_modules(self.skip_conv_layers, 'skip_conv_layer')   
        
        
        #----- Terminal Block and Activation Layer -----#
        if self.task == 'reg':
            self.terminal_block = ConvBlock3D(forward_filters[1], out_channels,
                                              dropout = self.dropout, residual = False, activation = 'linear',
                                              kernel_size = (1,3,3), norm = 'none', num_layers = 1)
            
            #TODO This line is for compatibility with a previous model. remove before release
            self.linear_activation = nn.Linear(256, 256) 
        else:
            self.terminal_block = ConvBlock3D(forward_filters[1], out_channels, 
                                              dropout = self.dropout, residual = self.residual,
                                              activation = activation, kernel_size = (1,3,3), num_layers = 1)
            
            
    def forward(self, x, validate_input=False):
        '''
        Forward call of network
            
        Call order:
            => num_block 3D convolutional blocks, with downsampling in between (encoder)
            => skip connections between corresponding blocks on encoder and decoder paths
            => num_block 2D (3d with 1 z-channel) convolutional blocks, with upsampling between them (decoder)
            => terminal block collapses to output dimensions
        
        Params:
            - x -> torch.tensor: input image stack
            - validate_input -> bool: Deactivates assertions which are redundat if forward pass is being traced by
                                tensorboard writer. 

        '''
        #handle input exceptions
        if validate_input:
            assert x.shape[-1] == x.shape[-2], 'Input must be square in xy'
            assert x.shape[-4] == self.in_channels, f'Input channels must equal network' \
                f'input channels: {self.in_channels}'
        
        #encoder
        skip_tensors = []
        for i in range(self.num_blocks):
            x = self.down_conv_blocks[i](x, validate_input = validate_input)
            skip_tensors.append(x)
            x = self.down_list[i](x)
        
        #transition block
        x = self.bottom_transition_block(x)
        
        #skip interruptions
        for i in range(self.num_blocks):
            skip_tensors[i] = self.skip_conv_layers[i](skip_tensors[i])
        
        #decoder
        for i in range(self.num_blocks):
            x = self.up_list[i](x)
            x = torch.cat([x, skip_tensors[-1*(i+1)]], 1)
            x = self.up_conv_blocks[i](x, validate_input = validate_input)            
        
        # output channel collapsing layer
        x = self.terminal_block(x)
        
        return x
            
    def register_modules(self, module_list, name):
        '''
        Helper function that registers modules stored in a list to the model object.
        Used to enable model graph creation with non-sequential model types and dynamic layer numbers
        
        Params:
            - module_list -> list[torch.nn.module]: list of modules present in the network
            - name -> str: name to register module under
        '''
        for i, module in enumerate(module_list):
            self.add_module(f'{name}_{str(i)}', module)