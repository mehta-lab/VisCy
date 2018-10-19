"""Network for regressing a vector from a set of images"""
import numpy as np
import tensorflow as tf
import warnings
from keras.layers import BatchNormalization, Conv2D, Conv3D, Dense, Dropout, \
    Input, Flatten
import keras.regularizers as k_regularizers

from micro_dl.networks.base_conv_net import BaseConvNet
from micro_dl.networks.conv_blocks import conv_block,\
    residual_conv_block, residual_downsample_conv_block
from micro_dl.utils.aux_utils import validate_config
from micro_dl.utils.network_utils import get_keras_layer, \
    create_activation_layer


class BaseImageToVectorNet(BaseConvNet):
    """Network for regression/classification from a set of images"""

    def __init__(self, network_config, predict=False):
        """Init

        This network resembles the down-sampling half of a (res-)Unet rather
        than VGG or res-net or inception. Config can be tweaked to get
        VGG-like (num_convs_per_block=3, residual=False) or
        res-net like (num_convs_per_block=1, residual=True) networks.
        No Inception module!!

        :param dict network_config: dict with all network associated parameters
        """

        super().__init__(network_config)

        req_params = ['num_convs_per_block',
                      'residual',
                      'block_sequence']
        req_dense_params = ['type', 'regression_length']

        if not predict:
            param_check, msg = validate_config(network_config, req_params)
            if not param_check:
                raise ValueError(msg)
            dense_param_check, msg = validate_config(network_config['dense'],
                                                     req_dense_params)
            if not dense_param_check:
                raise ValueError(msg)

            assert network_config['dense']['type'] in ['dense', 'conv'], \
                'Dense layers implemented as dense or 1x1 convs'

            if network_config['pooling_type'] == 'conv':
                assert 'res_pool_type' in network_config, \
                    'pooling type for the res connection is not provided'

        if network_config['dense']['type'] == 'conv':
            network_config['dense']['dropout'] = network_config['dropout']
        self.config = network_config

        max_num_conv_blocks = int(np.log2(self.config['height']))
        self.max_num_conv_blocks = max_num_conv_blocks

        if 'depth' in self.config and self.config['depth'] > 1:
            self.config['num_dims'] = 3
            self.conv = Conv3D
        else:
            self.config['num_dims'] = 2
            self.conv = Conv2D

        if 'num_initial_filters' in self.config:
            msg = 'Input image dimensions has to be in powers of 2 as the' \
                  'receptive field is the entire image'
            if not predict:
                assert 'num_filters_per_block' not in self.config, \
                    'Both num_initial_filters & num_filters_per_block provided'
                assert network_config['width'] == network_config['height'], \
                    'Expecting a square image for receptive field = full image'
                assert (np.mod(np.log2(network_config['width']), 1) <= 0 and
                        np.mod(np.log2(network_config['height']), 1) <= 0), msg
                warnings.warn('Assuming the network receptive field is the '
                              'entire image. If not, provide num_filters_per_'
                              'block instead of num_initial_filters', Warning)

            num_init_filters = self.config['num_initial_filters']
            num_filters_per_block = (
                [int(num_init_filters * 2 ** block_idx)
                 for block_idx in range(max_num_conv_blocks)]
            )
            num_conv_blocks = max_num_conv_blocks

        elif 'num_filters_per_block' in self.config:
            num_filters_per_block = self.config['num_filters_per_block']
            if not predict:
                assert len(num_filters_per_block) <= max_num_conv_blocks, \
                    'network depth is incompatible with the input size'
                assert 'num_units' in self.config['dense'], \
                    'num_units for dense layers not provided'
            num_conv_blocks = len(num_filters_per_block)

        else:
            raise ValueError('Both num_initial_filters and '
                             'num_filters_per_block not in network_config')
        self.config['num_filters_per_block'] = num_filters_per_block
        self.num_conv_blocks = num_conv_blocks

    def _get_regularizer_instance(self):
        """Get kernel/activity regularizer"""

        kernel_reg_dict = self.config['dense']['kernel_regularizer']
        kernel_reg_object = getattr(k_regularizers, kernel_reg_dict['type'])
        if 'lambda' in kernel_reg_dict:
            kernel_reg_inst = kernel_reg_object(kernel_reg_dict['lambda'])
        else:
            # default lambda=0.01
            kernel_reg_inst = kernel_reg_object()
        return kernel_reg_inst

    def _downsample_layer(self, layer):
        """Downsample a keras layer"""

        pool_object = get_keras_layer(
            type=self.config['pooling_type'],
            num_dims=self.config['num_dims']
        )
        layer = pool_object(
            pool_size=(2,) * self.config['num_dims'],
            data_format=self.config['data_format']
        )(layer)
        return layer

    def build_net(self):
        """Assemble the network"""

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ----------------------- convolution blocks --------------------
        for block_idx in range(self.num_conv_blocks):
            block_name = 'conv_block_{}'.format(block_idx + 1)
            with tf.name_scope(block_name):
                if self.config['residual']:
                    if self.config['pooling_type'] == 'conv':
                        self.config['pooling_type'] = \
                            self.config['res_pool_type']
                        layer = residual_downsample_conv_block(
                            layer=input_layer,
                            network_config=self.config,
                            block_idx=block_idx,
                        )
                        if block_idx == self.num_conv_blocks:
                            layer = self._downsample_layer(layer)
                        self.config['pooling_type'] = 'conv'
                    else:
                        layer = residual_conv_block(layer=input_layer,
                                                    network_config=self.config,
                                                    block_idx=block_idx)
                        if block_idx < self.num_conv_blocks:
                             layer = self._downsample_layer(layer)
                else:
                    layer = conv_block(layer=input_layer,
                                       network_config=self.config,
                                       block_idx=block_idx)
                    if block_idx < self.num_conv_blocks:
                        layer = self._downsample_layer(layer)
            input_layer = layer

        # ----------------------- dense blocks ------------------------
        num_units = input_layer.shape.as_list()[1:]
        num_units = np.prod(num_units)
        regression_length = self.config['dense']['regression_length']

        if 'num_units' in self.config['dense'] and \
                len(self.config['dense']['num_units']) > 1:
            dense_units = self.config['dense']['num_units']
        else:
            if self.num_conv_blocks == self.max_num_conv_blocks:
                if num_units / 16 > regression_length:
                    dense_units = np.array([num_units / 4,
                                            num_units / 8, num_units / 16],
                                           dtype='int')
                elif num_units / 8 > regression_length:
                    dense_units = np.array([num_units / 2, num_units / 4,
                                            num_units / 8], dtype='int')
                elif num_units / 4 >= regression_length:
                    dense_units = np.array([num_units / 2, num_units / 4],
                                           dtype='int')
                else:
                    raise ValueError(
                        'num features extracted < 4 * regression_length'
                    )
            else:
                raise ValueError('Invalid num_dense_units in config')

        prev_dense_layer = layer
        if self.config['dense']['type'] == 'dense':
            prev_dense_layer = Flatten()(layer)

        for dense_idx in range(len(dense_units)):
            block_name = 'dense_{}'.format(dense_idx + 1)
            if 'kernel_regularizer' in self.config['dense']:
                kernel_reg_inst = self._get_regularizer_instance()
            else:
                kernel_reg_inst = None
            # can't choose b/w dropout and regularization, use both ¯\_(ツ)_/¯
            with tf.name_scope(block_name):
                if self.config['dense']['type'] == 'dense':
                    layer = Dense(
                        units=dense_units[dense_idx],
                        kernel_initializer=self.config['init'],
                        kernel_regularizer=kernel_reg_inst
                    )(prev_dense_layer)
                else:
                    kernel_shape = (1, ) * self.config['num_dims']
                    layer = self.conv(
                        filters=dense_units[dense_idx],
                        kernel_size=kernel_shape,
                        padding='same',
                        kernel_initializer='he_normal',
                        data_format=self.config['data_format']
                    )(prev_dense_layer)

                layer = BatchNormalization()(layer)
                activation_layer_instance = create_activation_layer(
                    self.config['activation']
                )
                layer = activation_layer_instance(layer)
                if 'dropout' in self.config['dense']:
                    layer = Dropout(self.config['dense']['dropout'])(layer)
            prev_dense_layer = layer

        # --------------------- output block -------------------------
        final_activation = self.config['final_activation']
        with tf.name_scope('output'):
            if self.config['dense']['type'] == 'dense':
                outputs = Dense(regression_length,
                                kernel_initializer='he_normal',
                                activation=final_activation)(prev_dense_layer)
            else:
                kernel_shape = (1,) * self.config['num_dims']
                outputs = self.conv(
                    filters=regression_length,
                    padding='same',
                    kernel_size=kernel_shape,
                    kernel_initializer='he_normal',
                    data_format=self.config['data_format']
                )(prev_dense_layer)
                outputs = Flatten()(outputs)
        return inputs, outputs
