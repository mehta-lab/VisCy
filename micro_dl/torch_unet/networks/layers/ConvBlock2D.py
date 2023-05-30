import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 dropout=False,
                 norm='batch',
                 residual=True,
                 activation='relu',
                 transpose=False,
                 kernel_size = 3,
                 num_repeats = 3,
                 filter_steps = 'first',
                 layer_order = 'can'):

        """
        Convolutional block for lateral layers in Unet
        
        Format for layer initialization is as follows:
            if layer type specified 
            => for number of layers 
            => add layer to list of that layer type
            => register elements of list
        This is done to allow for dynamic layer number specification in the conv blocks, which
        allows us to change the parameter numbers of the network.
        
        :param int in_filters: number of images in in stack
        :param int out_filters: number of images in out stack
        :param float dropout: dropout probability (False => 0)
        :param str norm: normalization type: 'batch', 'instance'
        :param bool residual: as name
        :param str activation: activation function: 'relu', 'leakyrelu', 'elu', 'selu' 
        :param bool transpose: as name
        :param int/tuple kernel_size: convolutional kernel size
        :param int num_repeats: number of times the layer_order layer sequence is repeated in the block
        :param str filter_steps: determines where in the block the filters inflate channels (learn
                                    abstraction information): 'linear','first','last'
        :param str layer_order: order of conv, norm, and act layers in block: 'can', 'cna', 'nca', etc
        """

        super(ConvBlock2D, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.dropout = dropout
        self.norm = norm
        self.residual = residual
        self.activation = activation
        self.transpose = transpose
        self.num_repeats = num_repeats
        self.filter_steps = filter_steps
        self.layer_order = layer_order
        
        #---- Handle Kernel ----#
        ks = kernel_size
        if isinstance(ks, int):
            assert ks%2==1, 'Kernel dims must be odd'
        elif isinstance(ks, tuple):
            for i in range(len(ks)):
                assert ks[i]%2==1,'Kernel dims must be odd'
            assert i == 1, 'kernel_size length must be 2'
        else:
            raise AttributeError("'kernel_size' must be either int or tuple")
        self.kernel_size = kernel_size
        
        #----- Init Dropout -----#
        if self.dropout:
            self.drop_list = []
            for i in range(self.num_repeats):
                self.drop_list.append(nn.Dropout2d(int(self.dropout)))
        
        #---- Init linear filter steps ----#
        steps = np.linspace(in_filters, out_filters, num_repeats+1).astype(int)
        
        #----- Init Normalization Layers -----#
        # The parameters governing the initiation logic flow are:
        #                 self.norm
        #                 self.num_repeats
        #                 self.filter_steps
        self.norm_list = [None for i in range(num_repeats)]
        if self.norm == 'batch':
            for i in range(self.num_repeats):
                if self.filter_steps == 'linear':
                    self.norm_list[i] = nn.BatchNorm2d(steps[i+1])
                elif self.filter_steps == 'first':
                    self.norm_list[i] = nn.BatchNorm2d(steps[-1])
                elif self.filter_steps == 'last':
                    if i < self.num_repeats - 1:
                        self.norm_list[i] = nn.BatchNorm2d(steps[0])
                    else:
                        self.norm_list[i] = nn.BatchNorm2d(steps[-1])
        elif self.norm == 'instance':
            for i in range(self.num_repeats):
                if self.filter_steps == 'linear':
                    self.norm_list[i] = nn.InstanceNorm2d(steps[i+1])
                elif self.filter_steps == 'first':
                    self.norm_list[i] = nn.InstanceNorm2d(steps[-1])
                elif self.filter_steps == 'last':
                    if i < self.num_repeats - 1:
                        self.norm_list[i] = nn.InstanceNorm2d(steps[0])
                    else:
                        self.norm_list[i] = nn.InstanceNorm2d(steps[-1])
        self.register_modules(self.norm_list, f'{norm}_norm')
        
        
        #----- Init Conv Layers -----#
        # init conv layers and determine transposition during convolution
        # The parameters governing the initiation logic flow are:
        #                 self.transpose
        #                 self.num_repeats
        #                 self.filter_steps
        # See above for definitions.
        #-------#
        
        self.conv_list = []
        if self.filter_steps == 'linear': #learn progressively over steps
            for i in range(self.num_repeats):
                depth_pair = (steps[i], steps[i+1]) if i+1 < num_repeats else (steps[i], steps[-1])
                if self.transpose:
                    self.conv_list.append(nn.ConvTranspose2d(depth_pair[0],
                                                             depth_pair[1], 
                                                             kernel_size=kernel_size, 
                                                             padding='same'))
                else:
                    self.conv_list.append(nn.Conv2d(depth_pair[0],
                                                    depth_pair[1], 
                                                    kernel_size=kernel_size, 
                                                    padding='same'))
                    
        elif self.filter_steps == 'first': #learn in the first convolution
            if self.transpose:
                raise NotImplementedError('PyTorch-side problem with \'same\' padding in ConvTranspose2d.')
                for i in range(self.num_repeats):
                    if i == 0:
                        self.conv_list.append(nn.ConvTranspose2d(in_filters,
                                                                 out_filters, 
                                                                 kernel_size=kernel_size, 
                                                                 padding='same'))
                    else:
                        self.conv_list.append(nn.ConvTranspose2d(out_filters, out_filters, 
                                                                 kernel_size=kernel_size, 
                                                                 padding='same'))
            else:
                for i in range(self.num_repeats):
                    if i == 0:
                        self.conv_list.append(nn.Conv2d(in_filters,
                                                        out_filters, 
                                                        kernel_size=kernel_size, 
                                                        padding='same'))
                    else:
                        self.conv_list.append(nn.Conv2d(out_filters,
                                                        out_filters, 
                                                        kernel_size=kernel_size, 
                                                        padding='same'))
                        
        elif self.filter_steps == 'last': #learn in the last convolution
            if self.transpose:
                raise NotImplementedError('Problem with \'same\' padding in ConvTranspose2d.')
                for i in range(self.num_repeats):
                    if i == self.num_repeats-1:
                        self.conv_list.append(nn.ConvTranspose2d(in_filters,
                                                                 out_filters, 
                                                                 kernel_size=kernel_size, 
                                                                 padding='same'))
                    else:
                        self.conv_list.append(nn.ConvTranspose2d(out_filters,
                                                                 out_filters, 
                                                                 kernel_size=kernel_size, 
                                                                 padding='same'))
            else:
                for i in range(self.num_repeats):
                    if i == self.num_repeats-1:
                        self.conv_list.append(nn.Conv2d(in_filters,
                                                        out_filters, 
                                                        kernel_size=kernel_size, 
                                                        padding='same'))
                    else:
                        self.conv_list.append(nn.Conv2d(in_filters,
                                                        in_filters, 
                                                        kernel_size=kernel_size, 
                                                        padding='same'))
        self.register_modules(self.conv_list, 'Conv2d')
        
        
        #----- Init Residual Layer -----#
        self.resid_conv = nn.Conv2d(self.in_filters,
                                    self.out_filters, 
                                     kernel_size=1, 
                                     padding=0)
        
        
        #----- Init Activation Layers -----#
        self.act_list = []
        if self.activation == 'relu':
            for i in range(self.num_repeats):
                self.act_list.append(nn.ReLU())
        elif self.activation == 'leakyrelu':
            for i in range(self.num_repeats):
                self.act_list.append(nn.LeakyReLU())
        elif self.activation == 'elu':
            for i in range(self.num_repeats):
                self.act_list.append(nn.ELU())
        elif self.activation == 'selu':
            for i in range(self.num_repeats):
                self.act_list.append(nn.SELU())
        elif self.activation != 'linear':
            raise NotImplementedError(f'Activation type {self.activation} not supported.')
        self.register_modules(self.act_list, f'{self.activation}_act')
    
    def forward(self, x, validate_input = False):
        """
        Forward call of convolutional block
        
        Order of layers within the block is defined by the 'layer_order' parameter, which is a string of
        'c's, 'a's and 'n's in reference to convolution, activation, and normalization layers. This sequence
        is repeated num_repeats times.
        
        Recommended layer order:   convolution -> activation -> normalization
        
        Regardless of layer order, the final layer sequence in the block always ends in activation. This
        allows for usage of passthrough layers or a final output activation function determined separately.
        
        Residual blocks:
            if input channels are greater than output channels, we use a 1x1 convolution on
                input to get desired feature channels
            if input channels are less than output channels, we zero-pad input channels to
                output channel size
            
        :param torch.tensor x: input tensor
        :param bool validate_input: Deactivates assertions which are redundat if forward pass is being
                                    traced by tensorboard writer. 
        """
        if validate_input:
            if isinstance(self.kernel_size, int):
                assert x.shape[-1] > self.kernel_size and x.shape[-2] > self.kernel_size, f'Input size'\
                    f' {x.shape} too small for kernel of size {self.kernel_size}'
            elif isinstance(self.kernel_size, tuple):
                assert x.shape[-1] > self.kernel_size[-1] and x.shape[-2] > self.kernel_size[-2], f'Input size'\
                    f' {x.shape} too small for kernel of size {self.kernel_size}'
        
        x_0 = x
        for i in range(self.num_repeats):
            order = list(self.layer_order)
            while(len(order) > 0):
                layer = order.pop(0)
                if layer == 'c':
                    x = self.conv_list[i](x)
                    if self.dropout:
                        x = self.drop_list[i](x)
                elif layer == 'a':
                    if i < self.num_repeats - 1 or self.activation != 'linear':
                        x = self.act_list[i](x)
                elif layer == 'n' and self.norm_list[i]:
                    x = self.norm_list[i](x)
        
        #residual summation after final activation/normalization
        if self.residual:
            if self.in_filters > self.out_filters:
                x_0 = self.resid_conv(x_0)
            elif self.in_filters < self.out_filters:
                x_0 = F.pad(x_0,
                            (*[0]*4, self.out_filters-self.in_filters,*[0]*3),
                            mode = 'constant',
                            value = 0)
            x = torch.add(x_0, x)
        
        return x
    
    def model(self):
        """
        Allows calling of parameters inside ConvBlock object: 'ConvBlock.model().parameters()''
        
        Layer order:       convolution -> normalization -> activation
        
        We can make a list of layer modules and unpack them into nn.Sequential.
        Note: this is distinct from the forward call because we want to use the forward call with
                addition, since this is a residual block. The forward call performs the resid
                calculation, and all the parameters can be seen by the optimizer when given this model.
        """
        layers = []
        
        for i in range(self.num_repeats):
            layers.append(self.conv_list[i])
            if self.dropout:
                layers.append(self.drop_list[i])
            if self.norm[i]:
                layers.append(self.norm_list[i])
            if i < len(self.act_list):
                layers.append(self.act_list[i])
        
        return nn.Sequential(*layers)
    
    def register_modules(self, module_list, name):
        """
        Helper function that registers modules stored in a list to the model object so that the can
        be seen by PyTorch optimizer.
        
        Used to enable model graph creation with non-sequential model types and dynamic layer numbers
        
        :param list(torch.nn.module) module_list: list of modules to register/make visible
        :param str name: name of module type
        """
        for i, module in enumerate(module_list):
            self.add_module(f'{name}_{str(i)}', module)