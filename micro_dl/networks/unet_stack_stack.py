"""Unet for 3D volumes with anisotropic shape"""
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Conv3D, Input

from micro_dl.networks.base_unet import BaseUNet


class UNetStackToStack(BaseUNet):
    """Unet for anisotropic stacks"""

    def __init__(self, network_config):
        """Init

        :param dict network_config: dict with all network associated parameters
        """

        assert 'depth' in network_config, 'depth is missing in network_config'
        num_slices = network_config['depth']
        msg = 'Depth of the input has to be in powers of 2 as this network ' \
              'upsamples and downsamples in factors of 2'
        if np.mod(np.log2(num_slices), 1) > 0:
            raise ValueError(msg)

        super().__init__(network_config)

    def _get_filter_shape(self, filter_size, input_layer):
        """Get the filter shape depending on z dimension of input layer

        :param keras.layer input_layer: as named
        """

        if self.config['data_format'] == 'channels_first':
            z_dim = 4
        else:
            z_dim = 3

        num_slices = input_layer.shape.as_list()[z_dim]
        if num_slices <= 2:
            if isinstance(filter_size, (tuple, list)):
                filter_shape = (filter_size[0], filter_size[1], 1)
            else:
                # assuming it is an int
                filter_shape = (filter_size, filter_size, 1)
            down_up_sample_shape = (2, 2, 1)
        else:
            filter_shape = filter_size
            down_up_sample_shape = (2, 2, 2)
        return filter_shape, down_up_sample_shape

    def build_net(self):
        """Assemble the network"""

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ---------- Downsampling + middle blocks ---------
        init_filter_size = self.config['filter_size']
        skip_layers_list = []
        for block_idx in range(self.num_down_blocks + 1):
            block_name = 'down_block_{}'.format(block_idx + 1)
            filter_shape, downsample_shape = self._get_filter_shape(
                init_filter_size, input_layer
            )
            self.config['filter_size'] = filter_shape
            with tf.name_scope(block_name):
                layer, cur_skip_layers = super()._downsampling_block(
                    input_layer=input_layer, block_idx=block_idx
                )
            skip_layers_list.append(cur_skip_layers)
            input_layer = layer
        del skip_layers_list[-1]

        # ------------- Upsampling / decoding blocks -------------
        for block_idx in reversed(range(self.num_down_blocks)):
            cur_skip_layers = skip_layers_list[block_idx]
            block_name = 'up_block_{}'.format(block_idx)
            filter_shape, upsample_shape = self._get_filter_shape(
                init_filter_size, cur_skip_layers
            )
            self.config['filter_size'] = filter_shape

            with tf.name_scope(block_name):
                layer = super()._upsampling_block(
                    input_layers=input_layer,
                    skip_layers=cur_skip_layers,
                    block_idx=block_idx,
                    filter_shape=filter_shape,
                    upsampling_shape=upsample_shape
                )
            input_layer = layer

        # ------------ output block ------------------------
        final_activation = self.config['final_activation']
        with tf.name_scope('output'):
            layer = Conv3D(filters=1,
                           kernel_size=(1, 1, 1),
                           padding='same',
                           kernel_initializer='he_normal',
                           data_format=self.config['data_format'])(input_layer)
        outputs = Activation(final_activation)(layer)
        return inputs, outputs

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
