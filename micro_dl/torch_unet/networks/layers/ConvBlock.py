# Last edit: Christian Foley, 08/26/2022
import torch
import torch.nn as nn
import numpy as np

# convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout=False, norm='batch', residual=True, activation='relu',
                 transpose=False, num_layers = 3, filter_steps = 'first'):

        '''
        Convolutional block for lateral layers in Unet
            - in_filters -> int: number of images in in stack
            - out_filters -> int: number of images in out stack
            - dropout -> float: dropout probability (False = 0)
            - norm -> token{'batch', 'instance', 'mixed'}: normalization type
            - residual -> boolean: as name
            - activation -> token{'relu', 'leakyrelu', 'elu', 'selu'}: activation function
            - final_activation -> token{'linear', 'nonlinear'}: final layer activation, 'linear' for regression tasks
            - transpose -> boolean: as name
            - num_layers -> int: as name
            - filter_steps -> token{'linear','first','last'}: determines where in the block the filters inflate
                                                            channels (learn abstraction info)

        Format for layer initialization is as follows:
            if layer type specified 
            => for number of layers 
            => add layer to list of that layer type
        This is done to allow for dynamic layer number specification in the conv blocks, which
        allows us to change the parameter numbers of the network.
        '''

        super(ConvBlock, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.dropout = dropout
        self.norm = norm
        self.residual = residual
        self.activation = activation
        self.transpose = transpose
        self.num_layers = num_layers
        self.filter_steps = filter_steps
        
        #----- Init Dropout -----#
        if self.dropout:
            self.drop_list = []
            for i in range(self.num_layers):
                self.drop_list.append(nn.Dropout2d(int(self.dropout)))
        
        
        #----- Init Normalization Layers -----#
        #      => note: prefer batch normalization, but normalization can be mixed
        #
        self.norm_list = [None for i in range(num_layers)]
        if norm == 'batch':
            for i in range(self.num_layers):
                self.norm_list[i] = nn.BatchNorm2d(out_filters)
        elif norm == 'instance':
            for i in range(self.num_layers):
                self.norm_list[i] = nn.InstanceNorm2d(out_filters, affine=True)
        elif norm == 'mixed':
            for i in range(self.num_layers):
                if i < self.num_layers//2:
                    self.norm_list[i] = nn.BatchNorm2d(out_filters)
                else:
                    self.norm_list[i] = nn.InstanceNorm2d(out_filters, affine=True)
        
        #----- Init Conv Layers -----#
        #
        # init conv layers and determine transposition during convolution
        # The parameters governing the initiation logic flow are:
        #                 self.transpose
        #                 self.num_layers
        #                 self.filter steps
        # See above for definitions.
        #-------#
        
        self.conv_list = []
        steps = np.linspace(in_filters, out_filters, num_layers).astype(int)
        if self.filter_steps == 'linear':
            if self.transpose:
                for i in range(self.num_layers):
                    depth_pair = (steps[i], steps[i+1]) if i+1 < num_layers else (steps[i],steps[i])
                    self.conv_list.append(nn.ConvTranspose2d(depth_pair[0], depth_pair[1], kernel_size=3, padding=1))
            else:
                for i in range(self.num_layers):
                    depth_pair = (steps[i], steps[i+1]) if i+1 < num_layers else (steps[i],steps[i])
                    self.conv_list.append(nn.Conv2d(depth_pair[0], depth_pair[1], kernel_size=3, padding=1))
                    
        elif self.filter_steps == 'first':
            if self.transpose:
                for i in range(self.num_layers):
                    if i == 0:
                        self.conv_list.append(nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3, padding=1))
                    else:
                        self.conv_list.append(nn.ConvTranspose2d(out_filters, out_filters, kernel_size=3, padding=1))
            else:
                for i in range(self.num_layers):
                    if i == 0:
                        self.conv_list.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1))
                    else:
                        self.conv_list.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1))
                        
        elif self.filter_steps == 'last':
            if self.transpose:
                for i in range(self.num_layers):
                    if i == self.num_layers-1:
                        self.conv_list.append(nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3, padding=1))
                    else:
                        self.conv_list.append(nn.ConvTranspose2d(out_filters, out_filters, kernel_size=3, padding=1))
            else:
                for i in range(self.num_layers):
                    if i == self.num_layers-1:
                        self.conv_list.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1))
                    else:
                        self.conv_list.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1))
        
        #----- Init Activation Layers -----#
        self.act_list = []
        if self.activation == 'relu':
            for i in range(self.num_layers):
                self.act_list.append(nn.ReLU())
        elif self.activation == 'leakyrelu':
            for i in range(self.num_layers):
                self.act_list.append(nn.LeakyReLU())
        elif self.activation == 'elu':
            for i in range(self.num_layers):
                self.act_list.append(nn.ELU())
        elif self.activation == 'selu':
            for i in range(self.num_layers):
                self.act_list.append(nn.SELU())
        
    
    def forward(self, x):
        '''
        Forward call of convolutional block
            - x -> Torch.tensor: sample image stack
            
        Layer order:      convolution -> normalization -> activation
        
        '''
        
        x_0 = x
        for i in range(self.num_layers):
            x = self.conv_list[i](x)
            if self.dropout:
                x = self.drop_list(x)
            if self.norm[i]:
                x = self.norm_list[i](x)
            if i < self.num_layers - 1:
                x = self.act_list[i](x)
        
        #residual summation comes before final activation layer
        if self.residual:
            if self.out_filters == 1: #preserve information for 5 -> 1 conv blocks
                x += torch.unsqueeze(torch.mean(x_0, axis = 1), 1)
            else:
                x[:, 0:min(x_0.shape[1], x.shape[1]), :, :] += x_0[:, 0:min(x_0.shape[1], x.shape[1]), :, :]
            
        #last activation could be linear in prediction block
        if self.activation != 'linear':
            x = self.act_list[-1](x)
        
        return x
    
    def model(self):
        '''
        Allows calling of parameters inside ConvBlock object: 'ConvBlock.model().parameters()''
        
        Layer order:       convolution -> normalization -> activation
        
        We can make a list of layer modules and unpack them into nn.Sequential.
            Note: this is distinct from the forward call because we want to use the forward call with addition, since this
                  is a residual block. The forward call performs the resid calculation, and all the parameters can be seen
                  by the optimizer when given this model.
        '''
        layers = []
        
        for i in range(self.num_layers):
            layers.append(self.conv_list[i])
            if self.dropout:
                layers.append(self.drop_list[i])
            if self.norm[i]:
                layers.append(self.norm_list[i])
            if i < len(self.act_list):
                layers.append(self.act_list[i])
        
        return nn.Sequential(*layers)
            