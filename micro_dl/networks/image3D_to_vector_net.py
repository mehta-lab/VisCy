"""Image 3D to vector / scalar conv net"""
import numpy as np

from micro_dl.networks.base_image_to_vector_net import BaseImageToVectorNet


class Image3DToVectorNet(BaseImageToVectorNet):
    """Uses 3D images as input"""

    def __init__(self, network_config, predict=False):
        """Init

        :param dict network_config: dict with all network associated parameters
        """

        super().__init__(network_config, predict)
        if not predict and self.config['num_dims'] == 3 and \
                'num_initial_filters' in self.config:
            depth = self.config['depth']
            assert(np.mod(np.log2(network_config['depth']), 1) <= 0), \
                'Input image dimensions has to be in powers of 2 as the' \
                'receptive field is the entire image'
            assert network_config['width'] == network_config['depth'], \
                'Expecting an isotropic shape'

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        if self.config['data_format'] == 'channels_first':
            shape = (self.config['num_input_channels'],
                     self.config['depth'],
                     self.config['height'],
                     self.config['width'])
        else:
            shape = (self.config['depth'],
                     self.config['height'],
                     self.config['width'],
                     self.config['num_input_channels'])
        return shape
