"""Tests for UNet2D"""
import keras.backend as K
from keras import Model
import nose.tools
import numpy as np
import unittest

from micro_dl.networks import UNet2D


class TestUNet2D(unittest.TestCase):

    def setUp(self):
        """Set up network_config, model and model layers"""

        self.network_config = {'num_dims': 2,
                               'num_input_channels': 1,
                               'data_format': 'channels_first',
                               'height': 48,
                               'width': 64,
                               'batch_norm': True,
                               'dropout': 0.2,
                               'pooling_type': 'average',
                               'block_sequence': 'conv-bn-activation',
                               'num_filters_per_block': [8, 16, 32],
                               'num_convs_per_block': 2,
                               'residual': True,
                               'num_target_channels': 1,
                               'upsampling': 'bilinear',
                               'skip_merge_type': 'concat',
                               'final_activation': 'linear'}
        self.net = UNet2D(self.network_config)
        inputs, outputs = self.net.build_net()
        self.model = Model(inputs, outputs)
        self.model_layers = self.model.layers

    def test_UNet2D_init(self):
        """Test initialization"""

        # assert default values
        nose.tools.assert_equal(self.net.config['filter_size'], 3)
        nose.tools.assert_equal(self.net.config['padding'], 'same')
        nose.tools.assert_equal(self.net.config['init'], 'he_normal')
        nose.tools.assert_dict_equal(self.net.config['activation'],
                                     {'type': 'relu'})

        # check for missing input
        self.network_config.pop('block_sequence')
        nose.tools.assert_raises(ValueError, UNet2D, self.network_config)
        self.network_config['block_sequence'] = 'conv-bn-activation'

        # check for network depth
        self.network_config['num_filters_per_block'] = [8, 16, 32, 48, 64, 72]
        nose.tools.assert_raises(AssertionError, UNet2D, self.network_config)
        self.network_config['num_filters_per_block'] = [8, 16, 32]

        # check for interpolation type
        self.network_config['upsampling'] = 'trilinear'
        nose.tools.assert_raises(AssertionError, UNet2D, self.network_config)
        self.network_config['upsampling'] = 'bilinear'

    def test_UNet2D_get_input_shape(self):
        """Test input and output shape of network"""

        exp_in_shape = (self.network_config['num_input_channels'],
                        self.network_config['height'],
                        self.network_config['width'])

        exp_out_shape = (self.network_config['num_target_channels'],
                         self.network_config['height'],
                         self.network_config['width'])
        in_shape = self.model_layers[0].output_shape[1:]
        out_shape = self.model_layers[-1].output_shape[1:]
        nose.tools.assert_equal(in_shape, exp_in_shape)
        nose.tools.assert_equal(out_shape, exp_out_shape)

        # padding = 'valid'
        exp_out_shape = (self.network_config['num_input_channels'],
                         self.network_config['height'] - (4 + 8) * 2 - 16,
                         self.network_config['width'] - (4 + 8) * 2 - 16)
        self.network_config['padding'] = 'valid'
        self.network_config['residual'] = False
        net_valid = UNet2D(self.network_config)
        inputs, outputs = net_valid.build_net()
        model = Model(inputs, outputs)
        model_layers = model.layers
        in_shape = model_layers[0].output_shape[1:]
        out_shape = model_layers[-1].output_shape[1:]
        nose.tools.assert_equal(in_shape, exp_in_shape)
        nose.tools.assert_equal(out_shape, exp_out_shape)

    def test_UNet2D_same_shapes(self):
        """Test the shape of intermediate layers"""

        # conv-act-bn-do x 2 + lambda_pad + add + (avg for res)
        conv_idx = [0, 4]
        cur_feature_shape = [self.network_config['num_input_channels'],
                             self.network_config['height'],
                             self.network_config['width']]
        down_block_idx = [[1, 11], [11, 22], [22, 33]]
        upsamp_idx = [45, 33]
        concat_idx = [46, 34]
        for idx, down_idx in enumerate(down_block_idx):
            # conv layers
            cur_down_block = self.model_layers[down_idx[0]: down_idx[1]]
            cur_feature_shape[0] = \
                self.network_config['num_filters_per_block'][idx]

            if idx > 0:
                cur_feature_shape[1] = int(cur_feature_shape[1] / 2)
                cur_feature_shape[2] = int(cur_feature_shape[2] / 2)
                pool_feature_shape = (
                    self.network_config['num_filters_per_block'][idx - 1],
                    cur_feature_shape[1],
                    cur_feature_shape[2]
                )
                # pool layer
                nose.tools.assert_equal(
                    cur_down_block[7].output_shape[1:],
                    pool_feature_shape
                )

            nose.tools.assert_equal(
                cur_down_block[conv_idx[0]].output_shape[1:],
                tuple(cur_feature_shape)
            )
            nose.tools.assert_equal(
                cur_down_block[conv_idx[1]].output_shape[1:],
                tuple(cur_feature_shape)
            )
            # res add
            nose.tools.assert_equal(cur_down_block[-1].output_shape[1:],
                                    tuple(cur_feature_shape))
            # upsamp and merge layers
            if idx < 2:
                num_features = (
                        cur_feature_shape[0] +
                        self.network_config['num_filters_per_block'][idx + 1]
                )
                nose.tools.assert_equal(
                    self.model_layers[upsamp_idx[idx]].output_shape[1:],
                    (self.network_config['num_filters_per_block'][idx + 1],
                     cur_feature_shape[1],
                     cur_feature_shape[2])
                )
                nose.tools.assert_equal(
                    self.model_layers[concat_idx[idx]].output_shape[1],
                    num_features
                )

    def test_UNet2D_shape_mismatches(self):
        """Test for shape mismatches with padding=valid and/or residual"""

        # shape mismatch between conv downsampled layer and input pooled layer
        self.network_config['padding'] = 'valid'
        tst = UNet2D(self.network_config)
        nose.tools.assert_raises(AssertionError,
                                 tst.build_net)

        # even without conv downsampling, there could be shape mismatches if
        # downsampling results in odd num of channels, as in upsampling we'll
        # have even num of channels
        self.network_config['residual'] = False
        self.network_config['height'] = 54
        self.network_config['width'] = 54
        tst = UNet2D(self.network_config)
        nose.tools.assert_raises(AssertionError,
                                 tst.build_net)

    def test_UNet2D_valid_shapes(self):
        """Test intermediate shapes with valid padding"""

        self.network_config['padding'] = 'valid'
        self.network_config['residual'] = False
        net_valid = UNet2D(self.network_config)
        inputs, outputs = net_valid.build_net()
        model = Model(inputs, outputs)
        model_layers = model.layers

        # conv-act-bn-do x 2 + lambda_pad + add + (avg for res)
        conv_idx = [0, 4]
        cur_feature_shape = [self.network_config['num_input_channels'],
                             self.network_config['height'],
                             self.network_config['width']]
        down_block_idx = [[1, 8], [10, 17], [19, 26]]
        pool_idx = [9, 18]
        upsamp_idx = [38, 27]
        concat_idx = [40, 29]
        crop_idx = [39, 28]
        exp_upsamp_shape = [[12, 28], [10, 18]]

        for idx, down_idx in enumerate(down_block_idx):
            # conv layers
            cur_down_block = model_layers[down_idx[0]: down_idx[1]]
            cur_feature_shape[0] = \
                self.network_config['num_filters_per_block'][idx]

            cur_feature_shape[1] = cur_feature_shape[1] - 2
            cur_feature_shape[2] = cur_feature_shape[2] - 2
            nose.tools.assert_equal(
                cur_down_block[conv_idx[0]].output_shape[1:],
                tuple(cur_feature_shape)
            )

            cur_feature_shape[1] = cur_feature_shape[1] - 2
            cur_feature_shape[2] = cur_feature_shape[2] - 2
            nose.tools.assert_equal(
                cur_down_block[conv_idx[1]].output_shape[1:],
                tuple(cur_feature_shape)
            )

            if idx < len(self.network_config['num_filters_per_block'])-1:
                cur_feature_shape[1] = int(cur_feature_shape[1] / 2)
                cur_feature_shape[2] = int(cur_feature_shape[2] / 2)
                # pool layer
                nose.tools.assert_equal(
                    model_layers[pool_idx[idx]].output_shape[1:],
                    tuple(cur_feature_shape)
                )
            # upsamp and merge layers
            if idx < 2:
                num_features = (
                        cur_feature_shape[0] +
                        self.network_config['num_filters_per_block'][
                            idx + 1]
                )
                nose.tools.assert_equal(
                    model_layers[upsamp_idx[idx]].output_shape[1:],
                    (self.network_config['num_filters_per_block'][idx + 1],
                     exp_upsamp_shape[idx][0],
                     exp_upsamp_shape[idx][1])
                )
                nose.tools.assert_equal(
                    model_layers[concat_idx[idx]].output_shape[1],
                    num_features
                )
                nose.tools.assert_equal(
                    model_layers[crop_idx[idx]].output_shape[1:],
                    (self.network_config['num_filters_per_block'][idx],
                     exp_upsamp_shape[idx][0],
                     exp_upsamp_shape[idx][1])
                )

    def test_all_weights_trained(self):
        """Test if all weights are getting trained

        https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
        """

        image = np.ones((1, 1, 48, 64))
        target = np.random.random((1, 1, 48, 64))
        weight_tensors = self.model.trainable_weights
        sess = K.get_session()
        weight_arrays_b4 = sess.run(weight_tensors)
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.train_on_batch(image, target)
        weight_arrays_after = sess.run(weight_tensors)
        for before, after in zip(weight_arrays_b4, weight_arrays_after):
            # assert some part of the matrix changes
            assert (before != after).any()
