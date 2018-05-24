"""Implementation of U-net"""
from abc import ABCMeta, abstractmethod
import tensorflow as tf

import keras.backend as K
from keras.layers import (
    Activation, AveragePooling2D, AveragePooling3D, BatchNormalization, Conv2D,
    Conv3D, Dropout, Input, Lambda, MaxPooling2D, MaxPooling3D, UpSampling2D,
    UpSampling3D
)
from keras.layers.merge import Add, Concatenate


class BaseUNet(metaclass=ABCMeta):
    """Base U-net implementation

    1) Unet: https://arxiv.org/pdf/1505.04597.pdf
    2) residual Unet: https://arxiv.org/pdf/1711.10684.pdf
    border_mode='same' preferred over 'valid'. Else have to interpolate the
    last block to match the input image size.
    """

    def __init__(self, config):
        """Init

        :param yaml config: yaml with all network associated parameters
        """

        num_down_blocks = len(config['network']['num_filters_per_block']) - 1
        # assuming height=width
        width = config['network']['width']
        feature_width_at_last_block = width / (2 ** (num_down_blocks))
        msg = 'network depth is incompatible with the input size'
        assert feature_width_at_last_block >= 2, msg

        self.config = config
        if 'depth' in config['network']:
            self.num_dims = 3
            self.Conv = Conv3D
            self.UpSampling = UpSampling3D
        else:
            self.num_dims = 2
            self.Conv = Conv2D
            self.UpSampling = UpSampling2D

        self._set_pooling_type()
        self.num_down_blocks = num_down_blocks
        self.skip_merge_type = config['network']['skip_merge_type']
        self.data_format = config['network']['data_format']
        self._set_skip_merge_type()
        if config['network']['data_format'] == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1

        if 'num_input_channels' not in config['network']:
            config['network']['num_input_channels'] = (
                len(config['dataset']['input_channels'])
            )

        if 'num_target_channels' not in config['network']:
            config['network']['num_target_channels'] = (
                len(config['dataset']['target_channels'])
            )

    @staticmethod
    @abstractmethod
    def _pad_channels(input_layer, num_desired_channels,
                      final_layer, channel_axis):
        """Zero pad along channels before residual/skip merge"""

        raise NotImplementedError

    def _merge_residual(self, final_layer, input_layer):
        """Add residual connection from input to last layer

        Residual layers are always added (no concat supported currently)
        :param keras.layers final_layer: last layer
        :param keras.layers input_layer: input_layer
        :return: input_layer 1x1 / padded to match the shape of final_layer
         and added
        """

        num_final_layers = int(final_layer.get_shape()[self.channel_axis])
        num_input_layers = int(input_layer.get_shape()[self.channel_axis])
        if num_input_layers > num_final_layers:
            # use 1x 1 to get to the desired num of feature maps
            input_layer = self.Conv(filters=num_final_layers,
                                    kernel_size=(1, ) * self.num_dims,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    data_format=self.data_format)(input_layer)
        elif num_input_layers < num_final_layers:
            # padding with zeros along channels
            input_layer = Lambda(
                self._pad_channels,
                arguments={'num_desired_channels': num_final_layers,
                           'final_layer': final_layer,
                           'channel_axis': self.channel_axis})(input_layer)
        layer = Add()([final_layer, input_layer])
        return layer

    def _conv_block(self, block_idx, layer, num_convs_per_block, num_filters,
                    filter_size, activation='relu', batch_norm=True,
                    init='he_normal', dropout_prob=0.0, residual=False):
        """Downsampling blocks used in Unet

        Consists of n convolutions per block. Each convolution is followed by
        batchnorm and activation (and dropout if provided).

        :param int block_idx: index of the block (depth in the network)
        :param keras.layers layer: current input layer
        :param int num_convs_per_block: as named
        :param int num_filters: as named
        :param tuple filter_size: kernel_size = (filter_size, )*self.num_dims,
         isotropic kernel only!
        :param str activation: activation type (no advanced activations
         supported currently, to be added)
        :param bool batch_norm: indicator for batch norm
        :param str init: method used for initializing weights
        :param float dropout_prob: as named
        :param bool residual: indicator for residual block
        :return: keras.layers after convolution->BN->activ and pooling
        """

        input_layer = layer
        for conv_idx in range(num_convs_per_block):
            if residual and block_idx>0 and conv_idx==0:
                stride = (2, ) * self.num_dims
                input_layer = self.Pooling(
                    pool_size=(2,) * self.num_dims,
                    data_format=self.data_format
                )(input_layer)
            else:
                stride = (1, ) * self.num_dims
            layer = self.Conv(filters=num_filters, kernel_size=filter_size,
                              strides=stride, padding='same',
                              kernel_initializer=init,
                              data_format=self.data_format)(layer)
            if batch_norm:
                # data is assumed to be in channels_first format
                layer = BatchNormalization(axis=self.channel_axis)(layer)
            layer = Activation(activation)(layer)
            if dropout_prob:
                layer = Dropout(dropout_prob)(layer)

        if residual:
            layer = self._merge_residual(layer, input_layer)
        else:
            # downsample by 2
            pool_size = (2, ) * self.num_dims
            layer = self.Pooling(pool_size=pool_size,
                                 data_format=self.data_format)(layer)
        return layer

    def _upsampling_block(self, layer, skip_layers, num_convs_per_block,
                          num_filters, filter_size, activation='relu',
                          batch_norm=True, init='he_normal', dropout_prob=0.0,
                          residual=False):
        """Upsampling blocks of Unet

        The skip layers could be either concatenated or added
        Same as _conv_block
        :param keras.layers skip_layers: skip layers from the downsampling path
        :return: keras.layers after upsampling, merging, conv->BN->activ
        """

        layer_upsampled = self.UpSampling(size=(2, ) * self.num_dims,
                                          data_format=self.data_format)(layer)
        if self.skip_merge_type == Concatenate:
            layer = self.skip_merge_type(axis=self.channel_axis)(
                [layer_upsampled, skip_layers]
            )
        else:
            num_upsamp_layers = int(
                layer_upsampled.get_shape()[self.channel_axis]
            )
            ##
            skip_layers = Lambda(
                self._pad_channels,
                arguments={'num_desired_channels': num_upsamp_layers,
                           'final_layer': layer_upsampled,
                           'channel_axis': self.channel_axis})(skip_layers)
            layer = self.skip_merge_type()([layer_upsampled, skip_layers])
        input_layer = layer
        for conv_idx in range(num_convs_per_block):
            layer = self.Conv(filters=num_filters, kernel_size=filter_size,
                              padding='same', kernel_initializer=init,
                              data_format=self.data_format)(layer)
            if batch_norm:
                # data is assumed to be in channels_first format
                layer = BatchNormalization(axis=self.channel_axis)(layer)
            layer = Activation(activation)(layer)
            if dropout_prob:
                layer = Dropout(dropout_prob)(layer)
        if residual:
            layer = self._merge_residual(layer, input_layer)
        return layer

    def build_net(self):
        """Assemble the network"""

        num_convs_per_block = self.config['network']['num_convs_per_block']
        filter_size = self.config['network']['filter_size']
        activation = self.config['network']['activation']
        batch_norm = self.config['network']['batch_norm']
        dropout_prob = self.config['network']['dropout']
        residual = self.config['network']['residual']
        num_filters_per_block = self.config['network']['num_filters_per_block']

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        #---------- Downsampling/Encoding blocks ---------
        skip_layers_list = []
        for block_idx in range(self.num_down_blocks):
            block_name = 'down_block_{}'.format(block_idx+1)
            with tf.name_scope(block_name):
                layer = self._conv_block(
                    block_idx=block_idx, layer=input_layer,
                    num_convs_per_block=num_convs_per_block,
                    num_filters=num_filters_per_block[block_idx],
                    filter_size=(filter_size, ) * self.num_dims,
                    activation=activation, batch_norm=batch_norm,
                    dropout_prob=dropout_prob, residual=residual
                )
            skip_layers_list.append(layer)
            input_layer = layer

        #----------- Bridge/Middle block ---------------
        with tf.name_scope('bridge_block'):
            layer = self._conv_block(
                block_idx=len(num_filters_per_block) - 1, layer=input_layer,
                num_convs_per_block=num_convs_per_block,
                num_filters=num_filters_per_block[-1],
                filter_size=(filter_size, ) * self.num_dims,
                activation=activation, batch_norm=batch_norm,
                dropout_prob=dropout_prob, residual=residual
            )
        input_layer = layer

        #------------- Upsampling/decoding blocks -------------
        for block_idx in reversed(range(self.num_down_blocks)):
            cur_skip_layers = skip_layers_list[block_idx]
            block_name = 'up_block_{}'.format(block_idx)
            with tf.name_scope(block_name):
                layer = self._upsampling_block(
                    layer=input_layer, skip_layers=cur_skip_layers,
                    num_convs_per_block=num_convs_per_block,
                    num_filters=num_filters_per_block[block_idx],
                    filter_size=(filter_size, ) * self.num_dims,
                    activation=activation, batch_norm=batch_norm,
                    dropout_prob=dropout_prob, residual=residual
                )
            input_layer = layer

        #------------ output block ------------------------
        final_activation = self.config['network']['final_activation']
        num_output_channels = self.config['network']['num_target_channels']
        with tf.name_scope('output'):
            layer = self.Conv(filters=num_output_channels,
                              kernel_size=(1, ) * self.num_dims,
                              padding='same', kernel_initializer='he_normal',
                              data_format=self.data_format)(input_layer)
            outputs = Activation(final_activation)(layer)

        return inputs, outputs

    def _set_skip_merge_type(self):
        """Set if skip layers are to be added or concatenated"""

        skip = self.config['network']['skip_merge_type']
        self.skip_merge_type = {'add': Add, 'concat': Concatenate}[skip]

    @abstractmethod
    def _set_pooling_type(self):
        """Set the pooling type defined in the config"""

        raise NotImplementedError

    @property
    @abstractmethod
    def _get_input_shape(self):
        """Return shape of the input"""

        raise NotImplementedError


class UNet2D(BaseUNet):
    """2D UNet"""

    def ___init__(self, config):
        """Init"""

        super().__init__(config=config)

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        shape = (
            self.config['network']['num_input_channels'],
            self.config['network']['height'],
            self.config['network']['width']
        )
        return shape

    def _set_pooling_type(self):
        """Set the pooling type"""

        pool = self.config['network']['pooling_type']
        self.Pooling = {
            'max': MaxPooling2D,
            'average': AveragePooling2D
        }[pool]

    @staticmethod
    def _pad_channels(input_layer, num_desired_channels,
                      final_layer, channel_axis):
        """Zero pad along channels before residual/skip merge"""

        input_zeros = K.zeros_like(final_layer)
        num_input_layers = int(input_layer.get_shape()[channel_axis])
        new_zero_channels = int((num_desired_channels - num_input_layers) / 2)
        if num_input_layers % 2 == 0:
            zero_pad_layers = input_zeros[:, :new_zero_channels, :, :]
            layer_padded = Concatenate(axis=channel_axis)(
                [zero_pad_layers, input_layer, zero_pad_layers]
            )
        else:
            zero_pad_layers = input_zeros[:, :new_zero_channels+1, :, :]
            layer_padded = Concatenate(axis=channel_axis)(
                [zero_pad_layers, input_layer, zero_pad_layers[:, :-1, :, :]]
            )
        return layer_padded


class UNet3D(BaseUNet):
    """3D UNet"""

    def __init__(self, config):
        """Init"""
        
        super().__init__(config=config)

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        shape = (
            self.config['network']['num_input_channels'],
            self.config['network']['depth'],
            self.config['network']['height'],
            self.config['network']['width']
        )
        return shape

    def _set_pooling_type(self):
        """Set the pooling type"""

        pool = self.config['network']['pooling_type']
        self.Pooling = {
            'max': MaxPooling3D,
            'average': AveragePooling3D
        }[pool]

    @staticmethod
    def _pad_channels(input_layer, num_desired_channels,
                      final_layer, channel_axis):
        """Zero pad along channels before residual/skip merge"""

        input_zeros = K.zeros_like(final_layer)
        num_input_layers = int(input_layer.get_shape()[channel_axis])
        new_zero_channels = int((num_desired_channels - num_input_layers) / 2)
        if num_input_layers % 2 == 0:
            zero_pad_layers = input_zeros[:, :new_zero_channels, :, :, :]
            layer_padded = Concatenate(axis=channel_axis)(
                [zero_pad_layers, input_layer, zero_pad_layers]
            )
        else:
            zero_pad_layers = input_zeros[:, :new_zero_channels + 1, :, :, :]
            layer_padded = Concatenate(axis=channel_axis)(
                [zero_pad_layers, input_layer, zero_pad_layers[:, :-1, :, :, :]]
            )
        return layer_padded
