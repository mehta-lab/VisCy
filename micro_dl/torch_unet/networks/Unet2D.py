# Last edit: Christian Foley, 08/30/2022

import torch
import torch.nn as nn
from micro_dl.torch_unet.networks.layers.ConvBlock2D import *

class Unet2d(nn.Module):
    def __name__(self):
        return 'Unet2d'
    
    def __init__(self,
                 in_channels = 1,
                 out_channels = 1,
                 kernel_size = (3,3),
                 residual = False, 
                 dropout = 0.2,
                 num_blocks = 4,
                 num_block_layers = 2,
                 num_filters = [],
                 task = 'seg'):
        '''
        Instance of 2D Unet. Implementedfor e with variable input/output channels and depth (block numbers).
        Follows 2D UNet Architecture: 
            1) Unet: https://arxiv.org/pdf/1505.04597.pdf
            2) residual Unet: https://arxiv.org/pdf/1711.10684.pdf
        
        Parameters
            - in_channels -> int: number of feature channels in
            - out_channels -> int: number of feature channels out
            - kernel_size -> int or tuple(int, int): size of x and y dimensions of conv kernels in blocks
            - residual -> boolean: see name
            - down_mode -> token{'avgpool','maxpool','conv'}: type of downsampling in encoder path
            - up_mode -> token{'bilinear','tconv','conv'}: type of upsampling in decoder path
            - activation -> token{'relu','elu','selu','leakyrelu'}: activation function to use in convolutional blocks
            - num_blocks -> int: number of convolutional blocks on encoder and decoder paths
            - num_block_layers -> int: number of layers per block
            - num_filters -> list[int]: list of filters/feature levels at each conv block depth
            - task -> token{'recon','seg','reg'}: network task (for virtual staining this is regression)
            - bottom_block_spatial -> boolean: whether or not the bottom feature compression block learns spatial information as well
            
        '''
        
        super(Unet2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.residual = residual
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.task = task
        
         #----- set static parameters -----#
        self.block_padding = 'same'
        down_mode = 'avgpool' #TODO set static avgpool
        up_mode = 'bilinear' #TODO set static trilinear
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
                self.down_list.append(nn.MaxPool2d(kernel_size=2))
        elif down_mode == 'avgpool':
            for i in range(num_blocks):
                self.down_list.append(nn.AvgPool2d(kernel_size=2))
        elif down_mode == 'conv':
            raise NotImplementedError('Not yet implemented!')
            #TODO: implement.
        self.register_modules(self.down_list, 'down_samp')
        
        
        #----- Upsampling steps -----#
        self.up_list = []
        if up_mode == 'bilinear':
            for i in range(num_blocks):
                self.up_list.append(nn.Upsample(mode=up_mode, scale_factor=2, align_corners=False))
        elif up_mode == 'conv':
            raise NotImplementedError('Not yet implemented!')
            #TODO: implement
        elif up_mode == 'tconv':
            raise NotImplementedError('Not yet implemented!')
            #TODO: implement
        else:
            raise NotImplementedError(f'Upsampling mode \'{up_mode}\' not supported.')
        
        
        #----- Convolutional blocks -----# Forward Filters [16, 32, 64, 128, 256] -> Backward Filters [128+256, 64+128, 32+64, 16+32, 1]
        self.down_conv_blocks = []
        for i in range(num_blocks):
            self.down_conv_blocks.append(ConvBlock2D(forward_filters[i], forward_filters[i+1], 
                                                     dropout=self.dropout, residual = self.residual, activation = activation,
                                                     kernel_size = self.kernel_size, num_layers=self.num_block_layers))
        self.register_modules(self.down_conv_blocks, 'down_conv_block')
        
        self.bottom_transition_block = ConvBlock2D(self.num_filters[-2], self.num_filters[-1], 
                                                   dropout=self.dropout, residual = self.residual, activation = activation,
                                                   kernel_size = self.kernel_size, num_layers=self.num_block_layers)

        self.up_conv_blocks = []
        for i in range(num_blocks):
            self.up_conv_blocks.append(ConvBlock2D(backward_filters[i], forward_filters[-(i+2)],
                                                   dropout=self.dropout, residual = self.residual, activation = activation,
                                                   kernel_size = self.kernel_size, num_layers=self.num_block_layers))
        self.register_modules(self.up_conv_blocks, 'up_conv_block')            
        
        
        #----- Terminal Block and Activation Layer -----#
        # 
        if self.task == 'reg':
            self.terminal_block = ConvBlock2D(forward_filters[1], out_channels,
                                              dropout = self.dropout, residual = self.residual,
                                              activation = 'linear', num_layers = 1, norm='none',
                                              kernel_size=self.kernel_size)
        else:
            self.terminal_block = ConvBlock2D(forward_filters[1], out_channels,
                                              dropout = self.dropout, residual = self.residual,
                                              activation = activation, num_layers = 1, norm='none',
                                              kernel_size=self.kernel_size)
        
    def forward(self, x, validate_input = False):
        '''
        Forward call of network
            - x -> Torch.tensor: input image stack
            
        Call order:
            => num_block 2D convolutional blocks, with downsampling in between (encoder)
            => num_block 2D convolutional blocks, with upsampling between them (decoder)
            => skip connections between corresponding blocks on encoder and decoder paths
            => terminal block collapses to output dimensions
            
        Params:
            - x -> torch.tensor: input image
            - validate_input -> bool: Deactivates assertions which are redundat if forward pass is being traced by
                                tensorboard writer. 
        '''
        #handle input exceptions
        if validate_input:
            assert x.shape[-1] == x.shape[-2], 'Input must be square in xy'
            assert x.shape[-3] == self.in_channels, f'Input channels must equal network' \
                f' input channels: {self.in_channels}'
        
        #encoder
        skip_tensors = []
        for i in range(self.num_blocks):
            x = self.down_conv_blocks[i](x, validate_input = validate_input)
            skip_tensors.append(x)
            x = self.down_list[i](x)
        
        #transition block
        x = self.bottom_transition_block(x)
        
        #decoder
        for i in range(self.num_blocks):
            x = self.up_list[i](x)
            x = torch.cat([x, skip_tensors[-1*(i+1)]], 1)
            x = self.up_conv_blocks[i](x, validate_input = validate_input)
        
        # output channel collapsing layer
        x = self.terminal_block(x)
        
        return x
    
    def model(self):
        '''
        Allows calling of parameters inside ConvBlock2D object: 'model.model().parameters()'

        Sequential order:
            => num_block 2D convolutional blocks, with downsampling in between (encoder)
            => num_block 2D convolutional blocks, with upsampling between them (decoder)
            => skip connections between corresponding blocks on encoder and decoder paths
            => terminal block compresses to output dimensions
        
        We can make a list of layer modules and unpack them into nn.Sequential.
            Note: this is distinct from the forward call because we want to use the forward call with addition, since this
                  is a residual network. The forward call performs the resid calculation, and all the parameters can be seen
                  by the optimizer when given this model.        
        '''

        
        def get_block_layers(conv_block):
            block_layers = conv_block.model().children()
            return list(block_layers)
        
        layers = []

        #encoder blocks
        for i in range(self.num_blocks):
            layers.extend(get_block_layers(self.down_conv_blocks[i]))
            layers.append(self.down_list[i])
            
        #transition block
        layers.extend(get_block_layers(self.bottom_transition_block))
        
        #decoder blocks
        for i in range(self.num_blocks):
            #layers.append(self.up_list[i]) <-- We don't include the upsampling because it is a functional interpolation, not a module
            layers.extend(get_block_layers(self.up_conv_blocks[i]))
        
        #terminal block & activation
        layers.extend(get_block_layers(self.terminal_block))
        
        if self.task == 'reg':
            layers.append(self.linear_activation)
        
        return nn.Sequential(*layers)
    
    def register_modules(self, module_list, name):
        '''
        Helper function that registers modules stored in a list to the model object.
        Used to enable model graph creation with non-sequential model types
        '''
        for i, module in enumerate(module_list):
            self.add_module(f'{name}_{str(i)}', module)