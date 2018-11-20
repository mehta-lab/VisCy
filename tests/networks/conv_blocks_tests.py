"""Tests for  conv_blocks"""
import keras.backend as K
from keras import Model
from keras import layers as k_layers
import nose.tools
import numpy as np
import unittest

from micro_dl.networks import conv_blocks as conv_blocks
from micro_dl.utils.aux_utils import get_channel_axis
from micro_dl.utils.network_utils import get_keras_layer


class TestConvBlocks(unittest.TestCase):

    def setUp(self):
        """Set up input shapes for both 2D and 3D and network_config"""

        self.in_shape_2d = (1, 64, 64)
        self.in_shape_3d = (1, 64, 64, 32)
        self.network_config = {'num_dims': 2,
                               'block_sequence': 'conv-bn-activation',
                               'num_filters_per_block': [16, 32],
                               'num_convs_per_block': 2,
                               'filter_size': 3,
                               'init': 'he_normal',
                               'padding': 'same',
                               'activation': {'type': 'relu'},
                               'batch_norm': True,
                               'dropout': 0.2,
                               'data_format': 'channels_first',
                               'num_input_channels': 1,
                               'residual': True,
                               'pooling_type': 'max'}
        self.expected_num_weights = (
                len(self.network_config['num_filters_per_block']) *
                self.network_config['num_convs_per_block'] * 4
        )

        self.expected_2dweight_shapes = self._get_expected_wt_shape(2)
        self.expected_3dweight_shapes = self._get_expected_wt_shape(3)

    def _get_expected_wt_shape(self, num_dims):
        """Get shape of filters + bias and BN beta and gamma"""

        num_filters_per_block = \
            self.network_config['num_filters_per_block'].copy()
        num_filters_per_block.insert(
            0, self.network_config['num_input_channels']
        )
        expected_shapes = []
        for block_idx, block_num_filters in \
                enumerate(self.network_config['num_filters_per_block']):
            for conv_idx in \
                    range(self.network_config['num_convs_per_block']):
                if num_dims == 2:
                    # filter shape
                    expected_shapes.append(
                        (self.network_config['filter_size'],
                         self.network_config['filter_size'],
                         num_filters_per_block[conv_idx + block_idx],
                         block_num_filters)
                    )
                else:  # 3D
                    expected_shapes.append(
                        (self.network_config['filter_size'],
                         self.network_config['filter_size'],
                         self.network_config['filter_size'],
                         num_filters_per_block[conv_idx + block_idx],
                         block_num_filters)
                    )
                # filter bias
                expected_shapes.append((block_num_filters,))
                # bn_gamma
                expected_shapes.append((block_num_filters,))
                # bn_beta
                expected_shapes.append((block_num_filters,))
        return expected_shapes

    def _test_same_out_shape(self, layers_list, exp_shapes):
        """Checks for the shape of output layers with same padding

        :param list layers_list: output of functional blocks of a NN stored as
         a list
        :param list exp_shapes: list of lists with the shape of each output
         layer
        """

        num_blocks = len(layers_list)
        for block_idx in range(num_blocks):
            np.testing.assert_array_equal(
                layers_list[block_idx].get_shape().as_list()[1:],
                exp_shapes[block_idx]
            )

    def _test_valid_out_shape(self, layers_list, exp_shapes):
        """Checks for the shape of output layers with valid padding

        :param list layers_list: output of functional blocks of a NN stored as
         a list
        :param list exp_shapes: list of lists with the shape of each output
         layer
        """

        exp_shapes[0][1:] = exp_shapes[0][1:] - 4
        exp_shapes[1][1:] = exp_shapes[1][1:] - 8
        self._test_same_out_shape(layers_list=layers_list,
                                  exp_shapes=exp_shapes)

    def _test_weight_shape(self, weight_arrays, expected_shapes):
        """Check for shape of filters + bias and BN beta and gamma

        :param list weight_arrays: list of np.arrays
        :param list expected_shapes: list of tuples
        """

        for idx, weight in enumerate(weight_arrays):
            nose.tools.assert_equal(weight.shape,
                                    expected_shapes[idx])

    def _create_model(self, in_shape, block_function):
        """Create a model with the functional blocks

        :param tuple ip_shape: as named
        :param function block_function: function from conv_blocks
        :return:
         op_layers: list of keras layers, output of the blocks being tested
         weight_arrays: list of np.arrays with trainable weights of the model
        """

        in_layer = k_layers.Input(shape=in_shape, dtype='float32')
        out_layer1 = block_function(in_layer, self.network_config, 0)
        out_layer2 = block_function(out_layer1, self.network_config, 1)
        out_layers = [out_layer1, out_layer2]

        model = Model(in_layer, out_layer2)
        weight_tensors = model.trainable_weights
        sess = K.get_session()
        weight_arrays = sess.run(weight_tensors)
        return out_layers, weight_arrays

    def test_conv_block(self):
        """Test conv_block()"""

        for idx, in_shape in enumerate([self.in_shape_2d, self.in_shape_3d]):
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            num_filters_per_block = \
                self.network_config['num_filters_per_block']

            exp_out_shapes = []
            for block_idx in range(len(num_filters_per_block)):
                cur_out_shape = np.array(in_shape)
                cur_out_shape[0] = num_filters_per_block[block_idx]
                exp_out_shapes.append(cur_out_shape)

            out_layers, weight_arrays = self._create_model(
                in_shape, conv_blocks.conv_block
            )

            # test for op layer shape
            self._test_same_out_shape(layers_list=out_layers,
                                      exp_shapes=exp_out_shapes)

            # valid padding
            self.network_config['padding'] = 'valid'
            out_layers, _ = self._create_model(in_shape,
                                              conv_blocks.conv_block)
            self._test_valid_out_shape(layers_list=out_layers,
                                       exp_shapes=exp_out_shapes)
            self.network_config['padding'] = 'same'

            # test for num of trainable weights
            nose.tools.assert_equal(len(weight_arrays),
                                    self.expected_num_weights)

            # test for shape of trainable weights
            if idx == 0:
                self._test_weight_shape(weight_arrays,
                                        self.expected_2dweight_shapes)
            else:
                self._test_weight_shape(weight_arrays,
                                        self.expected_3dweight_shapes)

    def test_residual_conv_block(self):
        """Test residual_conv_block()

        Adding a residual connection doesn't increase the number of trainable
        #params.
        """

        for idx, in_shape in enumerate([self.in_shape_2d, self.in_shape_3d]):
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            num_filters_per_block = \
                self.network_config['num_filters_per_block'].copy()

            exp_out_shapes = []
            for block_idx in range(len(num_filters_per_block)):
                cur_out_shape = list(in_shape)
                cur_out_shape[0] = num_filters_per_block[block_idx]
                exp_out_shapes.append(cur_out_shape)

            op_layers, weight_arrays = self._create_model(
                in_shape, conv_blocks.residual_conv_block
            )

            # test for op layer shape
            self._test_same_out_shape(layers_list=op_layers,
                                      exp_shapes=exp_out_shapes)
            # valid padding
            self.network_config['padding'] = 'valid'
            op_layers, _ = self._create_model(in_shape,
                                              conv_blocks.residual_conv_block)
            self.network_config['padding'] = 'same'

            # test for num of trainable weights
            nose.tools.assert_equal(len(weight_arrays),
                                    self.expected_num_weights)

            # test for shape of trainable weights
            if idx == 0:
                self._test_weight_shape(weight_arrays,
                                        self.expected_2dweight_shapes)
            else:
                self._test_weight_shape(weight_arrays,
                                        self.expected_3dweight_shapes)

    def test_downsample_conv_block(self):
        """Test downsample_conv_block()"""

        for idx, in_shape in enumerate([self.in_shape_2d, self.in_shape_3d]):
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            num_filters_per_block = \
                self.network_config['num_filters_per_block']

            exp_out_shapes = []
            for block_idx in range(len(num_filters_per_block)):
                cur_out_shape = np.array(in_shape)
                cur_out_shape[0] = num_filters_per_block[block_idx]
                if block_idx > 0:
                    cur_out_shape[1:] = cur_out_shape[1:] / 2
                exp_out_shapes.append(cur_out_shape)

            out_layers, weight_arrays = self._create_model(
                in_shape, conv_blocks.downsample_conv_block
            )

            # test for op layer shape
            self._test_same_out_shape(layers_list=out_layers,
                                      exp_shapes=exp_out_shapes)
            # valid_padding
            self.network_config['padding'] = 'valid'
            out_layers, _ = self._create_model(
                in_shape, conv_blocks.downsample_conv_block
            )
            layer0_shape = [val - 4 for val in in_shape[1:]]
            nose.tools.assert_equal(out_layers[0].get_shape().as_list()[2:],
                                    layer0_shape)
            # downsampling by conv with stride = 2 and a filter size = 3,
            # removes 3 pixels (as opposed to 2 pixels after ds by 2) with
            # valid padding. ex, in=[1,256,256]->16x125x125->16x123x123.
            # Requires non-symmetric cropping for merging skip layers, not
            # handled currently.
            layer1_shape = [val / 2 - 3 for val in layer0_shape]
            nose.tools.assert_equal(out_layers[1].get_shape().as_list()[2:],
                                    layer1_shape)
            self.network_config['padding'] = 'same'

            # test for num of trainable weights
            nose.tools.assert_equal(len(weight_arrays),
                                    self.expected_num_weights)

            # test for shape of trainable weights
            if idx == 0:
                self._test_weight_shape(weight_arrays,
                                        self.expected_2dweight_shapes)
            else:
                self._test_weight_shape(weight_arrays,
                                        self.expected_3dweight_shapes)

    def test_residual_downsample_conv_block(self):
        """Test residual_downsample_conv_block()"""

        for idx, in_shape in enumerate([self.in_shape_2d, self.in_shape_3d]):
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            num_filters_per_block = \
                self.network_config['num_filters_per_block'].copy()

            exp_out_shapes = []
            for block_idx in range(len(num_filters_per_block)):
                cur_out_shape = np.array(in_shape)
                cur_out_shape[0] = num_filters_per_block[block_idx]
                if block_idx > 0:
                    cur_out_shape[1:] = cur_out_shape[1:] / 2
                exp_out_shapes.append(cur_out_shape)

            out_layers, weight_arrays = self._create_model(
                in_shape, conv_blocks.residual_downsample_conv_block
            )

            # test for op layer shape
            self._test_same_out_shape(layers_list=out_layers,
                                      exp_shapes=exp_out_shapes)

            # test for num of trainable weights
            nose.tools.assert_equal(len(weight_arrays),
                                    self.expected_num_weights)

            # test for shape of trainable weights
            if idx == 0:
                self._test_weight_shape(weight_arrays,
                                        self.expected_2dweight_shapes)
            else:
                self._test_weight_shape(weight_arrays,
                                        self.expected_3dweight_shapes)

    def test_pad_channels(self):
        """Test pad_channels()

        zero-pads the layer along the channel dimension when padding=same.
        zero-pads + crops when padding=valid
        """

        for idx, in_shape in enumerate([self.in_shape_2d, self.in_shape_3d]):
            # create a model that gives padded layer as output
            self.network_config['num_dims'] = \
                self.network_config['num_dims'] + idx

            in_layer = k_layers.Input(shape=in_shape, dtype='float32')
            conv_layer = get_keras_layer('conv',
                                             self.network_config['num_dims'])
            out_layer = conv_layer(
                filters=self.network_config['num_filters_per_block'][0],
                kernel_size=self.network_config['filter_size'],
                padding='same',
                data_format=self.network_config['data_format']
            )(in_layer)

            channel_axis = get_channel_axis(self.network_config['data_format'])
            layer_padded = k_layers.Lambda(
                conv_blocks.pad_channels,
                arguments={'final_layer': out_layer,
                           'channel_axis': channel_axis})(in_layer)
            # layer padded has zeros in all channels except 8
            model = Model(in_layer, layer_padded)
            test_shape = list(in_shape)
            test_shape.insert(0, 1)
            test_image = np.ones(shape=test_shape)
            sess = K.get_session()
            # forward pass
            out = model.predict(test_image, batch_size=1)
            # test shape: should be the same as conv_layer
            out_shape = list(in_shape)
            out_shape[0] = self.network_config['num_filters_per_block'][0]
            np.testing.assert_array_equal(
                out_layer.get_shape().as_list()[1:], out_shape
            )
            out = np.squeeze(out)
            # only slice 8 is not zero
            nose.tools.assert_equal(np.sum(out), np.sum(out[8]))
            np.testing.assert_array_equal(out[8], np.squeeze(test_image))
            nose.tools.assert_equal(np.sum(out[8]), np.prod(in_shape))

    def test_merge_residual(self):
        """Test _merge_residual()"""

        for in_shape in [(1, 16, 16), (24, 16, 16)]:
            for padding in ['same', 'valid']:
                in_layer = k_layers.Input(shape=in_shape, dtype='float32')
                out_layer = k_layers.Conv2D(
                    filters=self.network_config['num_filters_per_block'][0],
                    kernel_size=self.network_config['filter_size'],
                    kernel_initializer='Ones',
                    padding=padding,
                    data_format=self.network_config['data_format']
                )(in_layer)
                res_layer = conv_blocks._merge_residual(
                    final_layer=out_layer,
                    input_layer=in_layer,
                    data_format=self.network_config['data_format'],
                    num_dims=2,
                    kernel_init='Ones',
                    padding=padding
                )
                model = Model(in_layer, res_layer)
                test_shape = list(in_shape)
                test_shape.insert(0, 1)
                test_image = np.ones(shape=test_shape)
                sess = K.get_session()
                # forward pass
                out = model.predict(test_image, batch_size=1)
                if in_shape[0] == 1:
                    # only the center slice is res added with input. Before
                    # res add, all channels will be identical
                    if padding == 'same':
                        np.testing.assert_array_equal(
                            out[:, 8, :, :], out[:, 7, :, :] + test_image[0]
                        )
                    else:
                        np.testing.assert_array_equal(
                            out[:, 8, :, :],
                            out[:, 7, :, :] + test_image[0, 0, 1:-1, 1:-1]
                        )
                        np.testing.assert_array_equal(
                            out.shape[2:], test_image[0, 0, 1:-1, 1:-1].shape
                        )
                    np.testing.assert_array_equal(out[0, 7], out[0, 10])
                if in_shape[0] == 24:
                    # input -> 1x1 to match the num of layers. res_layer must
                    # be > input center slices
                    if padding == 'same':
                        np.testing.assert_array_less(test_image[:, 4:20, :, :],
                                                     out)
                    else:
                        np.testing.assert_array_less(
                            test_image[:, 4:20, 1:-1, 1:-1], out
                        )

    def test_skip_merge(self):
        """Test skip_merge()"""

        in_shape = (1, 16, 16)
        for skip_type in ['add', 'concat']:
            for padding in ['same', 'valid']:
                in_layer = k_layers.Input(shape=in_shape, dtype='float32')
                out_layer = k_layers.Conv2D(
                    filters=self.network_config['num_filters_per_block'][0],
                    kernel_size=self.network_config['filter_size'],
                    kernel_initializer='Ones',
                    padding=padding,
                    data_format=self.network_config['data_format']
                 )(in_layer)
                res_layer = conv_blocks.skip_merge(
                    skip_layers=in_layer,
                    upsampled_layers=out_layer,
                    skip_merge_type=skip_type,
                    data_format=self.network_config['data_format'],
                    num_dims=2,
                    padding=padding
                )
                model = Model(in_layer, res_layer)
                test_shape = list(in_shape)
                test_shape.insert(0, 1)
                test_image = np.ones(shape=test_shape)
                sess = K.get_session()
                # forward pass
                out = model.predict(test_image, batch_size=1)
                if skip_type == 'add':
                    # only the center slice is skip added with input. Before
                    # skip add, all channels will be identical. with add,
                    # _merge_residual and skip_merge are identical
                    if padding == 'same':
                        np.testing.assert_array_equal(
                            out[:, 8, :, :], out[:, 7, :, :] + test_image[0]
                        )
                    else:
                        np.testing.assert_array_equal(
                            out[:, 8, :, :],
                            out[:, 7, :, :] + test_image[0, 0, 1:-1, 1:-1]
                        )
                        np.testing.assert_array_equal(
                            out.shape[2:], test_image[0, 0, 1:-1, 1:-1].shape
                        )
                    np.testing.assert_array_equal(out[0, 7], out[0, 10])

                if skip_type == 'concat':
                    if padding == 'same':
                        nose.tools.assert_equal(out.shape,
                                                (1, 17, 16, 16))
                    else:
                        nose.tools.assert_equal(out.shape,
                                                (1, 17, 14, 14))
