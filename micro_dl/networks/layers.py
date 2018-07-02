"""Bilinear interpolation for upsampling"""
from abc import ABCMeta
import keras.backend as K
from keras.engine.topology import InputSpec, Layer
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
        """Compute shape of input in channels_last format

        :param tuple/list/np.array input_shape: shape of the input tensor
        :return: width and height of the upsampled image
        """

        width = int(self.size[0] * input_shape[1]
                    if input_shape[2] is not None else None)
        height = int(self.size[1] * input_shape[2]
                     if input_shape[3] is not None else None)
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


class InterpUpSampling3D(InterpUpSampling2D):
    """Interpolates the feature map for upsampling"""

    def __init__(self, size=(2, 2, 2), interp_type='nearest',
                 data_format='channels_last', **kwargs):
        """Init

        :param int/tuple size: upsampling factor
        :param str interp_type: type of interpolation [bilinear,
         nearest neighbour]
        :param str data_format: allowed options are 'channels_last',
         'channels_first'
        kwargs: for all kw args in layers
        """

        super().__init__(size, interp_type, data_format, **kwargs)
        if isinstance(size, int):
            size = (size, ) * 3
        self.size = size

    def build(self, input_shape):
        """Build layer

        There are no weights for bilinear interpolation

        :param tuple/list/np.array input_shape: shape of the input tensor
        """

        self.input_spec = [InputSpec(shape=input_shape, ndim=5)]
        super().build(input_shape)

    def _get_output_shape(self, input_shape):
        """Compute shape of input in channels_last format

        :param tuple/list/np.array input_shape: shape of the input tensor
        :return: width and height of the upsampled image
        """

        width, height = super()._get_output_shape(input_shape)
        depth = int(self.size[2] * input_shape[3]
                    if input_shape[3] is not None else None)
        return width, height, depth

    def compute_output_shape(self, input_shape):
        """Compute shape of output

        :param tuple/list/np.array input_shape: shape of the input tensor
        :return: width and height of the upsampled image
        """

        input_shape = np.array(input_shape)
        if self.data_format == 'channels_first':
            #  convert to channels_last
            input_shape = input_shape[[0, 2, 3, 4, 1]]
            width, height, depth = self._get_output_shape(input_shape)
            #  switch back
            input_shape = input_shape[[0, 4, 1, 2, 3]]
            return tuple([input_shape[0], input_shape[1], width, height,
                          depth])
        else:
            width, height, depth = self._get_output_shape(input_shape)
            return tuple([input_shape[0], width, height, depth,
                          input_shape[4]])

    def _interp_image(self, x):
        """Interpolate the image in channel_last format

        :param keras.layers x: input layer for upsampling
        :return: resized tensor
        """

        b_size, x_size, y_size, z_size, c_size = x.shape.as_list()
        x_size_new = x_size * self.size[0]
        y_size_new = y_size * self.size[1]
        z_size_new = z_size * self.size[2]
        # resize y-z
        squeeze_b_x = tf.reshape(x, [-1, y_size, z_size, c_size])
        resize_b_x = super()._interp_image(squeeze_b_x)
        #  yikes, tf doesn't like None in reshape
        #  https://github.com/tensorflow/tensorflow/issues/7253
        resume_b_x = tf.reshape(
            tensor=resize_b_x,
            shape=tf.convert_to_tensor((tf.shape(resize_b_x)[0], x_size,
                                        y_size_new, z_size_new, c_size))
        )
        # resize x
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size])
        resize_b_z = super()._interp_image(squeeze_b_z)
        resume_b_z = tf.reshape(
            tensor=resize_b_z,
            shape=tf.convert_to_tensor((tf.shape(resize_b_z)[0], z_size_new,
                                        y_size_new, x_size_new, c_size))
        )
        output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
        return output_tensor

    def call(self, x, mask=None):
        """Layer's logic

        https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html
        https://stackoverflow.com/questions/43814367/resize-3d-data-in-tensorflow-like-tf-image-resize-images

        :param keras.layers x: input layer for upsampling
        :return: upsampled tensor
        """

        if self.data_format == 'channels_last':
            b_size, x_size, y_size, z_size, c_size = x.shape.as_list()
        else:
            b_size, c_size, x_size, y_size, z_size = x.shape.as_list()
        x_size_new = self.size[0] * x_size
        y_size_new = self.size[1] * y_size
        z_size_new = self.size[2] * z_size

        if (x_size == x_size_new) and (y_size == y_size_new) and (
                z_size == z_size_new):
            # already in the target shape
            return x

        if self.data_format == 'channels_first':
            #  convert to channels_last
            x = tf.transpose(x, [0, 2, 3, 4, 1])
            upsampled_x = self._interp_image(x)
            output_tensor = tf.transpose(upsampled_x, [0, 4, 1, 2, 3])
            return output_tensor
        else:
            output_tensor = self._interp_image(x)
            return output_tensor

    def get_config(self):
        """Return config"""

        base_config = super().get_config()
        base_config['size'] = self.size
        return base_config
