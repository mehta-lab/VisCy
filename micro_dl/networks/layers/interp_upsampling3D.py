"""Nearest/Bilinear interpolation in 3D"""
from keras.engine import InputSpec
import numpy as np
import tensorflow as tf

from micro_dl.networks.layers.interp_upsampling2D import InterpUpSampling2D


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

        depth = int(self.size[0] * input_shape[1]) \
            if input_shape[1] is not None else None
        height = int(self.size[1] * input_shape[2]) \
            if input_shape[2] is not None else None
        width = int(self.size[2] * input_shape[3]) \
            if input_shape[3] is not None else None
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
            return tuple([input_shape[0],
                          input_shape[1],
                          depth, height, width])
        else:
            width, height, depth = self._get_output_shape(input_shape)
            return tuple([input_shape[0], depth, height, width,
                          input_shape[4]])

    def _interp_image(self, x, size=None):
        """Interpolate the image in channel_last format

        :param keras.layers x: input layer for upsampling
        :return: resized tensor
        """

        b_size, z_size, y_size, x_size, c_size = x.shape.as_list()
        x_size_new = x_size * self.size[2]
        y_size_new = y_size * self.size[1]
        z_size_new = z_size * self.size[0]
        # resize y-x
        squeeze_b_z = tf.reshape(
            x, tf.convert_to_tensor([-1, y_size, x_size, c_size])
        )
        resize_b_z = super()._interp_image(squeeze_b_z)
        #  tf doesn't like None in reshape
        #  https://github.com/tensorflow/tensorflow/issues/7253
        resume_b_z = tf.reshape(
            tensor=resize_b_z,
            shape=tf.convert_to_tensor([-1, z_size,
                                        y_size_new,
                                        x_size_new,
                                        c_size])
        )
        # resize y-z, only z as y is already resized in the previous step
        #   first reorient
        reoriented = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_x = tf.reshape(reoriented, [-1, y_size_new, z_size, c_size])
        resize_b_x = super()._interp_image(squeeze_b_x, (1, self.size[0]))
        resume_b_x = tf.reshape(
            tensor=resize_b_x,
            shape=tf.convert_to_tensor((-1, x_size_new,
                                        y_size_new,
                                        z_size_new,
                                        c_size))
        )
        output_tensor = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
        return output_tensor

    def call(self, x, mask=None):
        """Layer's logic

        https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html
        https://stackoverflow.com/questions/43814367/resize-3d-data-in-tensorflow-like-tf-image-resize-images

        :param keras.layers x: input layer for upsampling
        :return: upsampled tensor
        """

        if self.data_format == 'channels_last':
            b_size, z_size, y_size, x_size, c_size = x.shape.as_list()
        else:
            b_size, c_size, z_size, y_size, x_size = x.shape.as_list()
        x_size_new = self.size[2] * x_size
        y_size_new = self.size[1] * y_size
        z_size_new = self.size[0] * z_size

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
