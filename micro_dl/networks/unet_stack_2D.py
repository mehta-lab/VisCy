"""Predict the center slice from a stack of 3-5 slices"""
import tensorflow as tf
import warnings
from keras.layers import Activation, Conv3D, Input

from micro_dl.networks.base_unet import BaseUNet


class UNetStackTo2D(BaseUNet):
    """Implements a U-net that takes a stack and predicts the center slice"""

    def __init__(self, network_config):
        """Init

        :param dict network_config: dict with all network associated parameters
        """

        assert 'depth' in network_config, 'depth is missing in network config'
        assert network_config['depth'] % 2 != 0, \
            'depth is even. Expecting an odd value to predict center slice'
        super().__init__(network_config)
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

        filter_shape = (1, 1, num_slices)
        layer = Conv3D(filters=num_filters,
                       kernel_size=filter_shape,
                       padding='valid',
                       data_format=self.config['data_format'])(input_layer)
        return layer

    def build_net(self):
        """Assemble the network

        Treat the downsampling blocks as 3D and the upsampling blocks as 2D.
        All blocks use 3D filters: either 3x3x3 or 3x3x1
        """

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ---------- Downsampling + middle blocks ---------
        filter_size = self.config['filter_size']
        num_slices = self.config['depth']
        filter_shape = (filter_size, filter_size, num_slices)

        skip_layers_list = []
        for block_idx in range(self.num_down_blocks + 1):
            block_name = 'down_block_{}'.format(block_idx + 1)
            with tf.name_scope(block_name):
                layer, cur_skip_layers = super()._downsampling_block(
                    input_layer=input_layer,
                    block_idx=block_idx,
                    filter_shape=filter_shape,
                    downsample_shape=(2, 2, 1)
                )
            skip_layers_list.append(cur_skip_layers)
            input_layer = layer
        del skip_layers_list[-1]

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
        upsampling_shape = (2, 2, 1)
        self.config['filter_size'] = (filter_size, filter_size, 1)
        for block_idx in reversed(range(self.num_down_blocks)):
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
