"""Predict the center slice from a stack of 3-5 slices"""
import tensorflow as tf
import warnings
from keras.layers import Activation, Conv3D, Input

from micro_dl.networks.base_unet import BaseUNet
from micro_dl.networks.conv_blocks import conv_block, residual_conv_block
from micro_dl.utils.network_utils import get_keras_layer


class UNetStackTo2D(BaseUNet):
    """Implements a U-net that takes a stack and predicts the center slice"""

    def __init__(self, network_config, predict=False):
        """Init

        :param dict network_config: dict with all network associated parameters
        """

        assert 'depth' in network_config and network_config['depth'] > 1, \
            'depth is missing in network config'
        assert network_config['depth'] % 2 != 0, \
            'depth is even. Expecting an odd value to predict center slice'
        assert ('padding' not in network_config or
                network_config['padding'] == 'same'), \
            'Due to anisotropic filter shape only padding=same allowed here'

        super().__init__(network_config, predict)
        num_down_blocks = len(network_config['num_filters_per_block'])
        self.num_down_blocks = num_down_blocks

        if network_config['depth'] > 5:
            warnings.warn('using more than 5 slices to predict center slice',
                          Warning)

    def _skip_block(self, input_layer, num_slices, num_filters):
        """Converts skip layers from 3D to 2D: 1x1 along Z

        The contracting path of this U-net uses 3D images of shape
        [x, y, depth]. The expanding path reduces the shape to [x, y, 1]

        :param keras.layers input_layer: layers to be used in skip connection
        :param int num_slices: as named
        :param int num_filters: as named
        :return: convolved layer with valid padding
        """

        filter_shape = (num_slices, 1, 1)
        layer = Conv3D(filters=num_filters,
                       kernel_size=filter_shape,
                       padding='valid',
                       data_format=self.config['data_format'])(input_layer)
        return layer

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

        assert filter_shape is not None and downsample_shape is not None, \
            'anisotropic filter_shape and downsample_shape are required'
        if filter_shape is not None:
            self.config['filter_size'] = filter_shape

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
        """Assemble the network

        Treat the downsampling blocks as 3D and the upsampling blocks as 2D.
        All blocks use 3D filters: either 3x3x3 or 3x3x1.
        Another variant is: if the stack could be treated as channels similar
        to RGB and 2D convolutions are sufficient to extract features. This
        could be done by using UNet2D with num_input_channels = 3 (N) and
        num_output_channels = 1
        """

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ---------- Downsampling + middle blocks ---------
        filter_size = self.config['filter_size']
        num_slices = self.config['depth']
        filter_shape = (num_slices, filter_size, filter_size)

        skip_layers_list = []
        for block_idx in range(self.num_down_blocks):
            block_name = 'down_block_{}'.format(block_idx + 1)
            with tf.name_scope(block_name):
                layer, cur_skip_layers = self._downsampling_block(
                    input_layer=input_layer,
                    block_idx=block_idx,
                    filter_shape=filter_shape,
                    downsample_shape=(1, 2, 2)
                )
            skip_layers_list.append(cur_skip_layers)
            input_layer = layer

        #  ---------- skip block before upsampling ---------
        block_name = 'skip_block_{}'.format(
            len(self.config['num_filters_per_block'])
        )
        with tf.name_scope(block_name):
            layer = self._skip_block(
                input_layer=input_layer,
                num_slices=num_slices,
                num_filters=self.config['num_filters_per_block'][-1]
            )
        input_layer = layer

        # ------------- Upsampling / decoding blocks -------------
        upsampling_shape = (1, 2, 2)
        self.config['filter_size'] = (1, filter_size, filter_size)
        for block_idx in reversed(range(self.num_down_blocks - 1)):
            cur_skip_layers = skip_layers_list[block_idx]
            cur_skip_layers = self._skip_block(
                input_layer=cur_skip_layers,
                num_slices=num_slices,
                num_filters=self.config['num_filters_per_block'][block_idx]
            )
            block_name = 'up_block_{}'.format(block_idx)
            with tf.name_scope(block_name):
                layer = super()._upsampling_block(
                    input_layers=input_layer,
                    skip_layers=cur_skip_layers,
                    block_idx=block_idx,
                    upsampling_shape=upsampling_shape
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
            shape = (1,
                     self.config['depth'],
                     self.config['height'],
                     self.config['width'])
        else:
            shape = (self.config['depth'],
                     self.config['height'],
                     self.config['width'], 1)
        return shape
