"""Tests for Image3DToVectorNet"""
from keras import Model
import nose.tools
import numpy as np
import unittest

from micro_dl.networks import Image3DToVectorNet


class TestImage3DToVectorNet(unittest.TestCase):

    def setUp(self):
        """Set up network_config, model and model layers"""

        self.network_config = {'num_input_channels': 1,
                               'data_format': 'channels_first',
                               'height': 64,
                               'width': 64,
                               'depth': 64,
                               'batch_norm': True,
                               'dropout': 0.2,
                               'pooling_type': 'average',
                               'block_sequence': 'conv-bn-activation',
                               'num_initial_filters': 16,
                               'num_convs_per_block': 2,
                               'residual': True,
                               'dense': {
                                   'type': 'dense',
                                   'dropout': 0.5,
                                   'regression_length': 7,
                                   'kernel_regularizer': {
                                       'type': 'l2',
                                       'lambda': 0.0001
                                   },
                               },
                               'final_activation': 'linear'}

        self.net = Image3DToVectorNet(self.network_config)
        inputs, outputs = self.net.build_net()
        self.model = Model(inputs, outputs)
        self.model_layers = self.model.layers

    def test_Image3DToVectorNet_init(self):
        """Test initialization"""

        # check if depth = width
        self.network_config['depth'] = 32
        nose.tools.assert_raises(AssertionError,
                                 Image3DToVectorNet,
                                 self.network_config)
        self.network_config['depth'] = 64

        # check if depth is a power of 2
        self.network_config['depth'] = 48
        nose.tools.assert_raises(AssertionError,
                                 Image3DToVectorNet,
                                 self.network_config)
        self.network_config['depth'] = 64

    def test_Image3DToVectorNet_get_input_shape(self):
        """Test input and output shape of network"""

        exp_in_shape = (self.network_config['num_input_channels'],
                        self.network_config['depth'],
                        self.network_config['height'],
                        self.network_config['width'])

        exp_out_shape = (self.network_config['dense']['regression_length'],)
        in_shape = self.model_layers[0].output_shape[1:]
        out_shape = self.model_layers[-1].output_shape[1:]
        nose.tools.assert_equal(in_shape, exp_in_shape)
        nose.tools.assert_equal(out_shape, exp_out_shape)

    def test_Image3DToVectorNet_init_filters_res_shapes(self):
        """Test for intermediate shapes"""

        conv_idx = [0, 4]
        cur_feature_shape = [self.network_config['num_input_channels'],
                             self.network_config['depth'],
                             self.network_config['height'],
                             self.network_config['width']]
        down_block_idx = [[1, 12], [12, 23], [23, 34],
                          [34, 45], [45, 56], [56, 67]]
        dense_block_idx = np.array([68, 72, 76, 80])
        dense_units = [128, 64, 32, 7]

        for idx, down_idx in enumerate(down_block_idx):
            cur_down_block = self.model_layers[down_idx[0]: down_idx[1]]
            cur_feature_shape[0] = \
                self.net.config['num_filters_per_block'][idx]
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

            if idx < 6:
                cur_feature_shape[1] = cur_feature_shape[1] // 2
                cur_feature_shape[2] = cur_feature_shape[2] // 2
                cur_feature_shape[3] = cur_feature_shape[3] // 2

                pool_feature_shape = (
                    self.net.config['num_filters_per_block'][idx],
                    cur_feature_shape[1],
                    cur_feature_shape[2],
                    cur_feature_shape[3]
                )
                # pool layer
                nose.tools.assert_equal(cur_down_block[-1].output_shape[1:],
                                        pool_feature_shape)

        for dense_type in ['dense', 'conv']:
            if dense_type == 'conv':
                dense_block_idx = dense_block_idx - 1
                self.network_config['dense']['type'] = 'conv'
                self.network_config.pop('num_filters_per_block')
                self.net = Image3DToVectorNet(self.network_config)
                inputs, outputs = self.net.build_net()
                self.model = Model(inputs, outputs)
                self.model_layers = self.model.layers

            for idx, dense_idx in enumerate(dense_block_idx):
                cur_dense_block = self.model_layers[dense_idx]
                nose.tools.assert_equal(cur_dense_block.output_shape[1],
                                        dense_units[idx])

    def test_Image3DToVectorNet_filters_per_block_shapes(self):
        """Test for intermediate shapes"""

        self.network_config.pop('num_initial_filters')
        self.network_config['num_filters_per_block'] = [8, 16, 32, 48]
        self.network_config['pooling_type'] = 'conv'
        self.network_config['res_pool_type'] = 'average'
        self.network_config['dense']['num_units'] = [32, 16]
        self.network_config['dense']['type'] = 'conv'
        self.net = Image3DToVectorNet(self.network_config)
        inputs, outputs = self.net.build_net()
        self.model = Model(inputs, outputs)
        self.model_layers = self.model.layers

        conv_idx = [0, 4]
        cur_feature_shape = [self.network_config['num_input_channels'],
                             self.network_config['depth'],
                             self.network_config['height'],
                             self.network_config['width']]
        down_block_idx = [[1, 11], [11, 22], [22, 33], [33, 44]]
        dense_block_idx = np.array([44, 48, 52])
        dense_units = [32, 16, 7]

        for idx, down_idx in enumerate(down_block_idx):
            # conv layers
            cur_down_block = self.model_layers[down_idx[0]: down_idx[1]]
            cur_feature_shape[0] = \
                self.network_config['num_filters_per_block'][idx]

            if idx > 0:
                cur_feature_shape[1] = cur_feature_shape[1] // 2
                cur_feature_shape[2] = cur_feature_shape[2] // 2
                cur_feature_shape[3] = cur_feature_shape[3] // 2

                pool_feature_shape = (
                    self.network_config['num_filters_per_block'][idx - 1],
                    cur_feature_shape[1],
                    cur_feature_shape[2],
                    cur_feature_shape[3]
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

        for idx, dense_idx in enumerate(dense_block_idx):
            cur_dense_block = self.model_layers[dense_idx]
            nose.tools.assert_equal(cur_dense_block.output_shape[1:],
                                    (dense_units[idx],
                                     cur_feature_shape[1],
                                     cur_feature_shape[2],
                                     cur_feature_shape[3]))

    # TODO: fix the test for all weights getting trained. The weights
    # don't seem to budge when trying to predict random vectors or vector of
    # ones at various learning rates
