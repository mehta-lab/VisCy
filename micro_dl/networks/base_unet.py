"""Base class for U-net"""
import tensorflow as tf
from keras.layers import Activation, Input, UpSampling2D, UpSampling3D, Lambda

from micro_dl.networks.base_conv_net import BaseConvNet
from micro_dl.networks.conv_blocks import conv_block, residual_conv_block, \
    residual_downsample_conv_block, skip_merge
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.network_utils import get_keras_layer


class BaseUNet(BaseConvNet):
    raise DeprecationWarning('Tensorflow models are no longer supported as of 2.0.0')
    """Base U-net implementation

    1) Unet: https://arxiv.org/pdf/1505.04597.pdf
    2) residual Unet: https://arxiv.org/pdf/1711.10684.pdf
    border_mode='same' preferred over 'valid'. Else have to interpolate the
    last block to match the input image size.
    """

    def __init__(self, network_config, predict=False):
        """Init

        :param dict network_config: dict with all network associated parameters
        :param bool predict: indicator for what the model is used for:
         train/predict
        """

        super().__init__(network_config, predict)
        req_params = ['num_filters_per_block',
                      'num_convs_per_block',
                      'skip_merge_type',
                      'upsampling',
                      'num_target_channels',
                      'residual',
                      'block_sequence']

        self.config = network_config
        num_down_blocks = len(self.config['num_filters_per_block']) - 1

        if not predict:
            param_check, msg = aux_utils.validate_config(
                network_config,
                req_params,
            )
            if not param_check:
                raise ValueError(msg)
            width = network_config['width']
            feature_width_at_last_block = width // (2 ** num_down_blocks)
            assert feature_width_at_last_block >= 2, \
                'network depth is incompatible with width'
            feature_height_at_last_block = \
                network_config['height'] // (2 ** num_down_blocks)
            assert feature_height_at_last_block >= 2, \
                'network depth is incompatible with height'
        #  keras upsampling repeats the rows and columns in data. leads to
        #  checkerboard in upsampled images. repeat - use keras builtin
        #  nearest_neighbor, bilinear: interpolate using custom layers
        upsampling = self.config['upsampling']
        msg = 'invalid upsampling, not in repeat/bilinear/nearest_neighbor'
        assert upsampling in ['bilinear', 'nearest_neighbor', 'repeat'], msg

        def _init_2D():
            self.config['num_dims'] = 2
            if upsampling == 'repeat':
                self.UpSampling = UpSampling2D
            else:
                self.UpSampling = aux_utils.import_object(
                    'networks',
                    'InterpUpSampling2D',
                )
            return

        def _init_3D():
            self.config['num_dims'] = 3
            if upsampling == 'repeat':
                self.UpSampling = UpSampling3D
            else:
                self.UpSampling = aux_utils.import_object(
                    'networks',
                    'InterpUpSampling3D',
                )
        if 'depth' in self.config:
            if self.config['depth'] > 1:
                _init_3D()
            elif self.config['depth'] == 1:
                _init_2D()
        else:
            # If no depth set, default to 2D
            _init_2D()

        self.num_down_blocks = num_down_blocks

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

        if filter_shape is not None:
            self.config['filter_size'] = filter_shape

        if downsample_shape is None:
            downsample_shape = (2,) * self.config['num_dims']

        if self.config['residual']:
            layer = residual_downsample_conv_block(
                layer=input_layer,
                network_config=self.config,
                block_idx=block_idx,
                downsample_shape=downsample_shape
            )
            skip_layers = layer
        else:
            layer = conv_block(layer=input_layer,
                               network_config=self.config,
                               block_idx=block_idx)
            skip_layers = layer
            if block_idx < self.num_down_blocks:
                pool_object = get_keras_layer(type=self.config['pooling_type'],
                                              num_dims=self.config['num_dims'])
                layer = pool_object(
                    pool_size=downsample_shape,
                    data_format=self.config['data_format']
                )(layer)

        return layer, skip_layers

    def _upsampling_block(self,
                          input_layers,
                          skip_layers,
                          block_idx,
                          filter_shape=None,
                          upsampling_shape=None):
        """Upsampling blocks of U net

        The skip layers could be either concatenated or added

        :param keras.layes input_layers: input layer to be upsampled
        :param keras.layers skip_layers: skip layers from the downsampling path
        :param int block_idx: block in the downsampling path to be used for
         skip connection
        :param tuple filter_shape: as named
        :param tuple upsampling_shape: allows for anisotropic upsampling
        :return: keras.layers after upsampling, skip-merge, conv block
        """

        if filter_shape is not None:
            self.config['filter_size'] = filter_shape

        if upsampling_shape is None:
            upsampling_shape = (2, ) * self.config['num_dims']

        # upsampling
        if self.config['upsampling'] == 'repeat':
            layer_upsampled = self.UpSampling(
                size=upsampling_shape,
                data_format=self.config['data_format']
            )(input_layers)
        else:
            layer_upsampled = self.UpSampling(
                size=upsampling_shape,
                data_format=self.config['data_format'],
                interp_type=self.config['upsampling']
            )(input_layers)

        # skip-merge
        layer = skip_merge(skip_layers=skip_layers,
                           upsampled_layers=layer_upsampled,
                           skip_merge_type=self.config['skip_merge_type'],
                           data_format=self.config['data_format'],
                           num_dims=self.config['num_dims'],
                           padding=self.config['padding'])

        # conv
        if self.config['residual']:
            layer = residual_conv_block(layer=layer,
                                        network_config=self.config,
                                        block_idx=block_idx)
        else:
            layer = conv_block(layer=layer,
                               network_config=self.config,
                               block_idx=block_idx)
        return layer

    def build_net(self):
        """Assemble the network"""

        with tf.name_scope('input'):
            input_layer = inputs = Input(shape=self._get_input_shape)

        # ---------- Downsampling + middle blocks ---------
        skip_layers_list = []
        for block_idx in range(self.num_down_blocks + 1):
            block_name = 'down_block_{}'.format(block_idx + 1)
            with tf.name_scope(block_name):
                layer, cur_skip_layers = self._downsampling_block(
                    input_layer=input_layer, block_idx=block_idx
                )
            skip_layers_list.append(cur_skip_layers)
            input_layer = layer
        del skip_layers_list[-1]

        # ------------- Upsampling / decoding blocks -------------
        for block_idx in reversed(range(self.num_down_blocks)):
            cur_skip_layers = skip_layers_list[block_idx]
            block_name = 'up_block_{}'.format(block_idx)
            with tf.name_scope(block_name):
                layer = self._upsampling_block(input_layers=input_layer,
                                               skip_layers=cur_skip_layers,
                                               block_idx=block_idx)
            input_layer = layer

        # ------------ output block ------------------------
        final_activation = self.config['final_activation']
        num_output_channels = self.config['num_target_channels']
        temperature = 1
        if 'temperature' in self.config:
            temperature = self.config['temperature']
        conv_object = get_keras_layer(type='conv',
                                      num_dims=self.config['num_dims'])
        with tf.name_scope('output'):
            layer = conv_object(
                filters=num_output_channels,
                kernel_size=(1,) * self.config['num_dims'],
                padding=self.config['padding'],
                kernel_initializer=self.config['init'],
                data_format=self.config['data_format'])(input_layer)
            layer = Lambda(lambda x: x / temperature)(layer)
            outputs = Activation(final_activation)(layer)
        return inputs, outputs
