"""Unet 2D"""
from micro_dl.networks.base_unet import BaseUNet


class UNet2D(BaseUNet):
    """2D UNet

    [batch_size, num_channels, y, x] or [batch_size, y, x, num_channels]
    """

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        if self.config['data_format'] == 'channels_first':
            shape = (self.config['num_input_channels'],
                     self.config['height'],
                     self.config['width'])
        else:
            shape = (self.config['height'],
                     self.config['width'],
                     self.config['num_input_channels'])
        return shape
