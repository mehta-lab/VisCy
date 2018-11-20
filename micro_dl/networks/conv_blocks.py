"""Collection of different conv blocks typically used in conv nets"""
from keras.layers import BatchNormalization, Dropout, Lambda
from keras.layers.merge import Add, Concatenate
import tensorflow as tf

from micro_dl.utils.aux_utils import get_channel_axis
from micro_dl.utils.network_utils import create_activation_layer, \
    get_keras_layer


def conv_block(layer, network_config, block_idx):
    """Convolution block

    Allowed block-seq: [conv-BN-activation, conv-activation-BN,
     BN-activation-conv]
    To accommodate params of advanced activations, activation is a dict with
     keys 'type' and 'params'.
    For a complete list of keys in network_config, refer to
    BaseConvNet.__init__() in base_conv_net.py

    :param keras.layers layer: current input layer
    :param dict network_config: dict with network related keys
    :param int block_idx: block index in the network
    :return: keras.layers after performing operations in block-sequence
     repeated for num_convs_per_block times
    TODO: data_format from network_config won't work for full 3D models in predict
    if depth is set to None
    """

    conv = get_keras_layer(type='conv', num_dims=network_config['num_dims'])
    block_sequence = network_config['block_sequence'].split('-')
    for _ in range(network_config['num_convs_per_block']):
        for cur_layer_type in block_sequence:
            if cur_layer_type == 'conv':
                layer = conv(
                    filters=network_config['num_filters_per_block'][block_idx],
                    kernel_size=network_config['filter_size'],
                    padding=network_config['padding'],
                    kernel_initializer=network_config['init'],
                    data_format=network_config['data_format'])(layer)
            elif cur_layer_type == 'bn' and network_config['batch_norm']:
                layer = BatchNormalization(
                    axis=get_channel_axis(network_config['data_format'])
                )(layer)
            else:
                activation_layer_instance = create_activation_layer(
                    network_config['activation']
                )
                layer = activation_layer_instance(layer)

        if network_config['dropout']:
            layer = Dropout(network_config['dropout'])(layer)

    return layer


def downsample_conv_block(layer,
                          network_config,
                          block_idx,
                          downsample_shape=None):
    """Conv-BN-activation block

    :param keras.layers layer: current input layer
    :param dict network_config: please check conv_block()
    :param int block_idx: block index in the network
    :param tuple downsample_shape: anisotropic downsampling kernel shape
    :return: keras.layers after downsampling and conv_block
    """

    conv = get_keras_layer(type='conv', num_dims=network_config['num_dims'])
    block_sequence = network_config['block_sequence'].split('-')
    for conv_idx in range(network_config['num_convs_per_block']):
        for cur_layer_type in block_sequence:
            if cur_layer_type == 'conv':
                if block_idx > 0 and conv_idx == 0:
                    if downsample_shape is None:
                        stride = (2, ) * network_config['num_dims']
                    else:
                        stride = downsample_shape
                else:
                    stride = (1, ) * network_config['num_dims']
                layer = conv(
                    filters=network_config['num_filters_per_block'][block_idx],
                    kernel_size=network_config['filter_size'],
                    strides=stride,
                    padding=network_config['padding'],
                    kernel_initializer=network_config['init'],
                    data_format=network_config['data_format'])(layer)
            elif cur_layer_type == 'bn' and network_config['batch_norm']:
                layer = BatchNormalization(
                    axis=get_channel_axis(network_config['data_format'])
                )(layer)
            else:
                activation_layer_instance = create_activation_layer(
                    network_config['activation']
                )
                layer = activation_layer_instance(layer)

        if network_config['dropout']:
            layer = Dropout(network_config['dropout'])(layer)
    return layer


def pad_channels(input_layer, final_layer, channel_axis):
    """Zero pad along channels before residual/skip add

    :param keras.layers input_layer: input layer to be padded with zeros / 1x1
    to match shape of final layer
    :param keras.layers final_layer: layer whose shape has to be matched
    :param int channel_axis: dimension along which to pad
    :return: keras.layer layer_padded - layer with the same shape as final
     layer
    """

    num_input_layers = tf.shape(input_layer)[channel_axis]
    num_final_layers = tf.shape(final_layer)[channel_axis]
    num_zero_channels = num_final_layers - num_input_layers
    tensor_zeros = tf.zeros_like(final_layer)
    tensor_zeros, _ = tf.split(tensor_zeros,
                               [num_zero_channels, num_input_layers],
                               axis=channel_axis)
    delta = tf.cond(tf.equal(tf.mod(num_zero_channels, 2), 0),
                    lambda: 0, lambda: 1)
    top_block, bottom_block = tf.split(
        tensor_zeros,
        [(num_zero_channels + delta) // 2,
         (num_zero_channels - delta) // 2],
        axis=channel_axis
    )
    layer_padded = tf.concat([top_block, input_layer, bottom_block],
                             axis=channel_axis)
    op_shape = final_layer.get_shape().as_list()
    layer_padded.set_shape(tuple(op_shape))
    return layer_padded


def _crop_layer(input_layer, final_layer, data_format, num_dims):
    """Crop input layer to match shape of final layer

    ONLY SYMMETRIC CROPPING IS HANDLED HERE!

    :param keras.layers final_layer: last layer of conv block or skip layers
     in Unet
    :param keras.layers input_layer: input_layer to the block
    :param str data_format: [channels_first, channels_last]
    :param int num_dims: as named
    :return: keras.layer, input layer cropped if shape is different than final
     layer, else input layer as is
    """

    input_shape = tf.shape(input_layer)
    final_shape = tf.shape(final_layer)
    # offsets for the top left corner of the crop
    if data_format == 'channels_first':
        offsets = [0, 0, (input_shape[2] - final_shape[2]) // 2,
                   (input_shape[3] - final_shape[3]) // 2]
        crop_shape = [-1, input_shape[1], final_shape[2], final_shape[3]]
        if num_dims == 3:
            offsets.append((input_shape[4] - final_shape[4]) // 2)
            crop_shape.append(final_shape[4])
    else:
        offsets = [0, (input_shape[1] - final_shape[1]) // 2,
                   (input_shape[2] - final_shape[2]) // 2]
        crop_shape = [-1, final_shape[1], final_shape[2]]
        if num_dims == 3:
            offsets.append((input_shape[3] - final_shape[3]) // 2)
            crop_shape.append(final_shape[3])
        offsets.append(0)
        crop_shape.append(input_shape[-1])

    # https://github.com/tensorflow/tensorflow/issues/19376
    input_cropped = tf.slice(input_layer, offsets, crop_shape)

    op_shape = final_layer.get_shape().as_list()
    channel_axis = get_channel_axis(data_format)
    op_shape[channel_axis] = input_layer.get_shape().as_list()[channel_axis]
    input_cropped.set_shape(tuple(op_shape))

    return input_cropped


def _merge_residual(final_layer,
                    input_layer,
                    data_format,
                    num_dims,
                    kernel_init,
                    padding):
    """Add residual connection from input to last layer
    :param keras.layers final_layer: last layer
    :param keras.layers input_layer: input_layer
    :param str data_format: [channels_first, channels_last]
    :param int num_dims: as named
    :param str kernel_init: kernel initializer from config
    :param str padding: same or valid
    :return: input_layer 1x1 / padded to match the shape of final_layer
     and added
    """

    channel_axis = get_channel_axis(data_format)
    conv_object = get_keras_layer(type='conv',
                                  num_dims=num_dims)
    num_final_layers = int(final_layer.get_shape()[channel_axis])
    num_input_layers = int(input_layer.get_shape()[channel_axis])
    # crop input if padding='valid'
    if padding == 'valid':
        input_layer = Lambda(_crop_layer,
                             arguments={'final_layer': final_layer,
                                        'data_format': data_format,
                                        'num_dims': num_dims})(input_layer)

    if num_input_layers > num_final_layers:
        # use 1x 1 to get to the desired num of feature maps
        input_layer = conv_object(
            filters=num_final_layers,
            kernel_size=(1, ) * num_dims,
            padding='same',
            kernel_initializer=kernel_init,
            data_format=data_format)(input_layer)
    elif num_input_layers < num_final_layers:
        # padding with zeros along channels
        input_layer = Lambda(
                      pad_channels,
                      arguments={'final_layer': final_layer,
                                 'channel_axis': channel_axis})(input_layer)
    layer = Add()([final_layer, input_layer])
    return layer


def skip_merge(skip_layers,
               upsampled_layers,
               skip_merge_type,
               data_format,
               num_dims,
               padding):
    """Skip connection concatenate/add to upsampled layer
    :param keras.layer skip_layers: as named
    :param keras.layer upsampled_layers: as named
    :param str skip_merge_type: [add, concat]
    :param str data_format: [channels_first, channels_last]
    :param int num_dims: as named
    :param str padding: same or valid
    :return: keras.layer skip merged layer
    """

    channel_axis = get_channel_axis(data_format)
    # crop input if padding='valid'
    if padding == 'valid':
        skip_layers = Lambda(_crop_layer,
                             arguments={'final_layer': upsampled_layers,
                                        'data_format': data_format,
                                        'num_dims': num_dims})(skip_layers)

    if skip_merge_type == 'concat':
        layer = Concatenate(axis=channel_axis)([upsampled_layers,
                                                skip_layers])
    else:
        skip_layers = Lambda(
            pad_channels,
            arguments={'final_layer': upsampled_layers,
                       'channel_axis': channel_axis})(skip_layers)
        layer = Add()([upsampled_layers, skip_layers])
    return layer


def residual_conv_block(layer, network_config, block_idx):
    """Convolution block where the last layer is merged (+) with input layer

    :param keras.layers layer: current input layer
    :param dict network_config: please check conv_block()
    :param int block_idx: block index in the network
    :return: keras.layers after conv-block and residual merge
    """

    input_layer = layer
    final_layer = conv_block(layer, network_config, block_idx)
    layer = _merge_residual(final_layer=final_layer,
                            input_layer=input_layer,
                            data_format=network_config['data_format'],
                            num_dims=network_config['num_dims'],
                            kernel_init=network_config['init'],
                            padding=network_config['padding'])
    return layer


def residual_downsample_conv_block(layer, network_config, block_idx,
                                   downsample_shape=None):
    """Convolution block where the last layer is merged (+) with input layer

    :param keras.layers layer: current input layer
    :param dict network_config: please check conv_block()
    :param int block_idx: block index in the network
    :param tuple downsample_shape: anisotropic downsampling kernel shape
    :return: keras.layers after conv-block and residual merge
    """

    if downsample_shape is None:
        downsample_shape = (2, ) * network_config['num_dims']

    if block_idx == 0:
        input_layer = layer
        final_layer = conv_block(layer, network_config, block_idx)
    else:
        final_layer = downsample_conv_block(layer=layer,
                                            network_config=network_config,
                                            block_idx=block_idx,
                                            downsample_shape=downsample_shape)

        pool_layer = get_keras_layer(type=network_config['pooling_type'],
                                     num_dims=network_config['num_dims'])
        downsampled_input_layer = pool_layer(
            pool_size=downsample_shape,
            data_format=network_config['data_format']
        )(layer)
        input_layer = downsampled_input_layer

    layer = _merge_residual(final_layer=final_layer,
                            input_layer=input_layer,
                            data_format=network_config['data_format'],
                            num_dims=network_config['num_dims'],
                            kernel_init=network_config['init'],
                            padding=network_config['padding'])
    return layer



