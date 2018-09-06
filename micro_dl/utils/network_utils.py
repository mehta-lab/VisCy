"""Network related util functions"""
from keras.layers import (Activation, AveragePooling2D, AveragePooling3D,
                          Conv2D, Conv3D, MaxPooling2D, MaxPooling3D,
                          UpSampling2D, UpSampling3D)
import keras.layers.advanced_activations as advanced_activations
from keras import activations as basic_activations


def get_keras_layer(type, num_dims):
    """Get the 2D or 3D keras layer

    :param str stype: type of layer [conv, pooling, upsampling]
    :param int num_dims: dimensionality of the image [2 ,3]
    :return: keras.layer
    """

    assert num_dims in [2, 3], 'num_dims >3, keras handles up to num_dims=3'
    assert type in ('conv', 'max', 'average', 'upsampling')
    if num_dims == 2:
        if type == 'conv':
            return Conv2D
        elif type == 'max':
            return MaxPooling2D
        elif type == 'average':
            return AveragePooling2D
        else:
            return UpSampling2D
    else:
        if type == 'conv':
            return Conv3D
        elif type == 'max':
            return MaxPooling3D
        elif type == 'average':
            return AveragePooling3D
        else:
            return UpSampling3D


def create_activation_layer(activation_dict):
    """Get the keras activation / advanced activation

    :param dict activation_dict: Nested dict with keys: type -> activation type
    and params -> dict activation related params such as alpha, theta,
    alpha_initializer, alpha_regularizer etc from advanced activations
    :return keras.layer: instance of activation layer
    """

    if hasattr(advanced_activations, activation_dict['type']):
        activation_layer = getattr(advanced_activations,
                                   activation_dict['type'])
        if 'params' in activation_dict:
            activation_layer_instance = activation_layer(
                activation_dict['params']
            )
        else:
            activation_layer_instance = activation_layer()
    elif hasattr(basic_activations, activation_dict['type']):
        activation_layer_instance = Activation(activation_dict['type'])
    else:
        raise ValueError('%s is not a valid activation type'
                         % activation_dict['type'])
    return activation_layer_instance
