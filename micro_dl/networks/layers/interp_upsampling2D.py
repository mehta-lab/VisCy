"""Nearest/Bilinear interpolation in 2D"""
from abc import ABCMeta
from keras import backend as K
from keras.engine import Layer, InputSpec
import numpy as np
import tensorflow as tf


class InterpUpSampling2D(Layer, metaclass=ABCMeta):
    """Interpolates the feature map for upsampling"""

    def __init__(self, size=(2, 2), interp_type='nearest',
                 data_format='channels_last', **kwargs):
        """Init

        https://github.com/aurora95/Keras-FCN/blob/master/utils/BilinearUpSampling.py
        https://keunwoochoi.wordpress.com/2016/11/18/for-beginners-writing-a-custom-keras-layer/
        https://keras.io/layers/writing-your-own-keras-layers/
        resizing by int only!

        :param int/tuple size: upsampling factor
        :param str interp_type: type of interpolation [bilinear,
         nearest neighbour]
        :param str data_format: allowed options are 'channels_last',
         'channels_first'
        kwargs: for all kw args in layers
        """

        msg = 'data_format is not in channels_first/last'
        assert data_format in ['channels_last', 'channels_first'], msg
        self.data_format = data_format

        msg = 'only nearest neighbour & bilinear interpolation are allowed'
        assert interp_type in ['nearest_neighbor', 'bilinear'], msg
        self.interp_type = interp_type

        if isinstance(size, tuple):
            chk = [isinstance(val, int) for val in size]
            assert all(chk), 'only int values are allowed in size'

        if isinstance(size, int):
            size = (size, ) * 2
        self.size = size
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build layer

        There are no weights for bilinear interpolation. InputSpec specifies
        the ndim, dtype and shape of every input to a layer

        :param tuple/list/np.array input_shape: shape of the input tensor
        """

        self.input_spec = [InputSpec(shape=input_shape, ndim=4)]
        super().build(input_shape)

    def _get_output_shape(self, input_shape):
        """Compute shape of output in channels_last format

        :param tuple/list/np.array input_shape: shape of the input tensor
        :return: width and height of the upsampled image
        """

        width = int(self.size[0] * input_shape[1]
                    if input_shape[1] is not None else None)
        height = int(self.size[1] * input_shape[2]
                     if input_shape[2] is not None else None)
        return width, height

    def compute_output_shape(self, input_shape):
        """Compute shape output

        :param tuple/list/np.array input_shape: shape of the input tensor
        :return: width and height of the upsampled image
        """

        input_shape = np.array(input_shape)
        if self.data_format == 'channels_first':
            # convert to channels_last
            input_shape = input_shape[[0, 2, 3, 1]]
            width, height = self._get_output_shape(input_shape)
            #  switching back
            input_shape = input_shape[[0, 3, 1, 2]]
            return tuple([input_shape[0], input_shape[1], width, height])
        else:
            width, height = self._get_output_shape(input_shape)
            return tuple([input_shape[0], width, height, input_shape[3]])

    def _interp_image(self, x):
        """Interpolate the image in channel_last format

        :param keras.layers x: input layer for upsampling
        :return: resized tensor
        """

        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(np.array(self.size[0:2]).astype('int32'))
        if self.interp_type == 'bilinear':
            x = tf.image.resize_bilinear(x, new_shape, align_corners=True)
        else:
            x = tf.image.resize_nearest_neighbor(x, new_shape,
                                                 align_corners=True)
        x.set_shape((original_shape[0],
                     original_shape[1] * self.size[0],
                     original_shape[2] * self.size[1],
                     original_shape[3]))
        return x

    def call(self, x, mask=None):
        """Layer's logic

        tf.image.resize_bilinear uses channels_last and has border issues!
        https://github.com/tensorflow/tensorflow/issues/6720

        :param keras.layers x: input layer for upsampling
        :return: upsampled tensor
        """

        original_shape = K.int_shape(x)
        if self.data_format == 'channels_first':
            #  convert to channel_last
            x = tf.transpose(x, [0, 2, 3, 1])
            x = self._interp_image(x)
            #  switch back to channels_first
            x = tf.transpose(x, [0, 3, 1, 2])
            x.set_shape((original_shape[0], original_shape[1],
                         original_shape[2] * self.size[0],
                         original_shape[3] * self.size[1]))
            return x
        else:
            x = self._interp_image(x)
            return x

    def get_config(self):
        """Return config"""

        base_config = super().get_config()
        base_config['size'] = self.size
        base_config['data_format'] = self.data_format
        base_config['interp_type'] = self.interp_type
        return base_config
