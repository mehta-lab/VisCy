# Last edit: Christian Foley, 08/26/2022
import torch

class ResNet(torch.nn.Module):
    def __init__(self, module):
        '''
        Base ResNet structure for building residual blocks
            module -> torch.nn.module: prebuilt module that is not residual. For example, an instance of nn.Sequential.
        '''
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs