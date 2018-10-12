"""Unet 3D"""
from micro_dl.networks.base_unet import BaseUNet


class UNet3D(BaseUNet):
    """3D UNet

    [batch_size, num_channels, z, y, x] or [batch_size, z, y, x, num_channels]
    """

    def __init__(self, network_config, predict=False):
        """Init

        :param dict network_config: dict with all network associated parameters
        """

        super().__init__(network_config, predict)
        if not predict and self.config['num_dims'] == 3:
            depth = self.config['depth']
            feature_depth_at_last_block = depth // (2 ** self.num_down_blocks)
            assert feature_depth_at_last_block >= 2, \
                'network depth is incompatible with input depth'

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
