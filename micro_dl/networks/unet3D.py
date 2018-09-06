"""Unet 3D"""
from micro_dl.networks.base_unet import BaseUNet


class UNet3D(BaseUNet):
    """3D UNet

    [batch_size, num_channels, z, y, x] or [batch_size, z, y, x, num_channels]
    """

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
