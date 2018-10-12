"""Unet for 3D volumes with anisotropic shape"""
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Conv3D, Input

from micro_dl.networks.base_unet import BaseUNet
from micro_dl.networks.conv_blocks import conv_block, residual_conv_block
from micro_dl.utils.network_utils import get_keras_layer


class UNetStackToStack(BaseUNet):
    """Unet for anisotropic stacks"""

    def __init__(self, network_config, predict=False):
        """Init

        :param dict network_config: dict with all network associated parameters
        """

        assert 'depth' in network_config and network_config['depth'] > 1, \
            'depth is missing in network_config'

        num_slices = network_config['depth']
        msg = 'Depth of the input has to be in powers of 2 as this network' \
              'upsamples and downsamples in factors of 2'
        if not predict:
            if np.mod(np.log2(num_slices), 1) > 0:
                raise ValueError(msg)

        assert ('padding' not in network_config or
                network_config['padding'] == 'same'), \
            'Due to anisotropic filter shape only padding=same allowed here'

        super().__init__(network_config)
        num_down_blocks = len(network_config['num_filters_per_block'])
        self.num_down_blocks = num_down_blocks

    def _get_filter_shape(self, filter_size, input_layer):
        """Get the filter shape depending on z dimension of input layer

        :param keras.layer input_layer: as named
        """

        if self.config['data_format'] == 'channels_first':
            z_dim = 2
        else:
            z_dim = 1

        num_slices = input_layer.shape.as_list()[z_dim]
        if num_slices == 1:
            if isinstance(filter_size, (tuple, list)):
                filter_shape = (1, filter_size[0], filter_size[1])
            else:
                # assuming it is an int
                filter_shape = (1, filter_size, filter_size)
            down_up_sample_shape = (1, 2, 2)
        else:
            filter_shape = filter_size
            down_up_sample_shape = (2, 2, 2)
        return filter_shape, down_up_sample_shape

    def _downsampling_block(self,
                            input_layer,
                            block_idx,
                            filter_shape=None,
                            downsample_shape=None):
        """Downsampling blocks of U-net

        :param keras.layer input_layer: must be the output of Input layer
        :param int block_idx: as named
        :param tuple filter_shape: filter size is an int for most cases.
         filter_shape enables passing anisotropic filter shapes
        :return keras.layer layer: output layer of bridge/middle block
         skip_layers_list: list of all skip layers
        """

        if self.config['residual']:
            layer = residual_conv_block(layer=input_layer,
                                        network_config=self.config,
                                        block_idx=block_idx)

        else:
            layer = conv_block(layer=input_layer,
                               network_config=self.config,
                               block_idx=block_idx)
        skip_layers = layer
        if block_idx < self.num_down_blocks - 1:
            pool_object = get_keras_layer(type=self.config['pooling_type'],
                                          num_dims=self.config['num_dims'])
            layer = pool_object(pool_size=downsample_shape,
                                data_format=self.config['data_format'])(layer)
        return layer, skip_layers

    def build_net(self):
        """Assemble the network"""

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ---------- Downsampling + middle blocks ---------
        init_filter_size = self.config['filter_size']
        skip_layers_list = []
        for block_idx in range(self.num_down_blocks):
            block_name = 'down_block_{}'.format(block_idx + 1)
            filter_shape, downsample_shape = self._get_filter_shape(
                self.config['filter_size'],
                input_layer
            )
            with tf.name_scope(block_name):
                layer, cur_skip_layers = self._downsampling_block(
                    input_layer=input_layer,
                    block_idx=block_idx,
                    filter_shape=filter_shape,
                    downsample_shape=downsample_shape
                )
            skip_layers_list.append(cur_skip_layers)
            input_layer = layer

        # ------------- Upsampling / decoding blocks -------------
        for block_idx in reversed(range(self.num_down_blocks - 1)):
            cur_skip_layers = skip_layers_list[block_idx]
            block_name = 'up_block_{}'.format(block_idx)
            filter_shape, upsample_shape = self._get_filter_shape(
                init_filter_size, cur_skip_layers
            )

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
