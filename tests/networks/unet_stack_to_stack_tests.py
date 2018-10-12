"""Tests for UNetStackToStack"""

import keras.backend as K
from keras import Model
import nose.tools
import numpy as np
import unittest

from micro_dl.networks import UNetStackToStack


class TestUNetStackToStack(unittest.TestCase):

    def setUp(self):
        """Set up network_config, model and model layers"""

        self.network_config = {'num_dims': 3,
                               'num_input_channels': 1,
                               'data_format': 'channels_first',
                               'depth': 8,
                               'height': 64,
                               'width': 64,
                               'batch_norm': True,
                               'dropout': 0.2,
                               'pooling_type': 'average',
                               'block_sequence': 'conv-bn-activation',
                               'num_filters_per_block': [8, 16, 32, 48],
                               'num_convs_per_block': 2,
                               'residual': True,
                               'num_target_channels': 1,
                               'upsampling': 'bilinear',
                               'skip_merge_type': 'concat',
                               'final_activation': 'linear'}
        self.net = UNetStackToStack(self.network_config)
        inputs, outputs = self.net.build_net()
        self.model = Model(inputs, outputs)
        self.model_layers = self.model.layers

    def test_UNetStackToStack_init(self):
        """Test initialization"""

        # check for network depth
        self.network_config['depth'] = 0
        nose.tools.assert_raises(AssertionError,
                                 UNetStackToStack,
                                 self.network_config)
        self.network_config['depth'] = 6
        nose.tools.assert_raises(ValueError,
                                 UNetStackToStack,
                                 self.network_config)
        self.network_config['depth'] = 8
        self.network_config['padding'] = 'valid'
        nose.tools.assert_raises(AssertionError,
                                 UNetStackToStack,
                                 self.network_config)
        self.network_config['padding'] = 'same'
        nose.tools.assert_equal(self.net.num_down_blocks,
                                len(self.net.config['num_filters_per_block']))

    def test_UNetStackToStack_get_input_shape(self):
        """Test input and output shape of network"""

        """Test input and output shape of network"""

        exp_in_shape = (self.network_config['num_input_channels'],
                        self.network_config['depth'],
                        self.network_config['height'],
                        self.network_config['width'])

        exp_out_shape = (self.network_config['num_target_channels'],
                         self.network_config['depth'],
                         self.network_config['height'],
                         self.network_config['width'])
        in_shape = self.model_layers[0].output_shape[1:]
        out_shape = self.model_layers[-1].output_shape[1:]
        nose.tools.assert_equal(in_shape, exp_in_shape)
        nose.tools.assert_equal(out_shape, exp_out_shape)

    def test_UNetStackToStack_intermediate_shapes(self):
        """Test the shape of intermediate layers"""

        # conv-act-bn-do x 2 + lambda_pad + add + (avg for res)
        conv_idx = [0, 4]
        cur_feature_shape = [self.network_config['num_input_channels'],
                             self.network_config['depth'],
                             self.network_config['height'],
                             self.network_config['width']]
        down_block_idx = [[1, 12], [12, 23], [23, 34], [34, 44]]
        upsamp_idx = [68, 56, 44]
        concat_idx = [69, 57, 45]

        for idx, down_idx in enumerate(down_block_idx):
            # conv layers
            cur_down_block = self.model_layers[down_idx[0]: down_idx[1]]
            cur_feature_shape[0] = \
                self.network_config['num_filters_per_block'][idx]
            nose.tools.assert_equal(
                cur_down_block[conv_idx[0]].output_shape[1:],
                tuple(cur_feature_shape)
            )
            nose.tools.assert_equal(
                cur_down_block[conv_idx[1]].output_shape[1:],
                tuple(cur_feature_shape)
            )

            # res add
            nose.tools.assert_equal(cur_down_block[-2].output_shape[1:],
                                    tuple(cur_feature_shape))

            # upsamp and merge layers
            if idx < 3:
                num_features = (
                    self.network_config['num_filters_per_block'][idx] +
                    self.network_config['num_filters_per_block'][idx + 1]
                )
                nose.tools.assert_equal(
                    self.model_layers[concat_idx[idx]].output_shape[1],
                    num_features
                )

                nose.tools.assert_equal(
                    self.model_layers[upsamp_idx[idx]].output_shape[1:],
                    (self.network_config['num_filters_per_block'][idx + 1],
                     cur_feature_shape[1],
                     cur_feature_shape[2],
                     cur_feature_shape[3])
                )

                cur_feature_shape[1] = max(1, cur_feature_shape[1] // 2)
                cur_feature_shape[2] = cur_feature_shape[2] // 2
                cur_feature_shape[3] = cur_feature_shape[3] // 2

                pool_feature_shape = (
                     self.network_config['num_filters_per_block'][idx],
                     cur_feature_shape[1],
                     cur_feature_shape[2],
                     cur_feature_shape[3]
                )
                # pool layer
                nose.tools.assert_equal(cur_down_block[-1].output_shape[1:],
                                        pool_feature_shape)

    def test_all_weights_trained(self):
        """Test if all weights are getting trained

        https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
        """

        image = np.ones((1, 1, 8, 64, 64), dtype='float')
        target = np.random.random((1, 1, 8, 64, 64)).astype('float')
        weight_tensors = self.model.trainable_weights
        sess = K.get_session()
        weight_arrays_b4 = sess.run(weight_tensors)
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.train_on_batch(image, target)
        weight_arrays_after = sess.run(weight_tensors)
        for before, after in zip(weight_arrays_b4, weight_arrays_after):
            # assert some part of the matrix changes
            assert (before != after).any()
