"""Base class for all networks"""
from abc import ABCMeta, abstractmethod

from micro_dl.utils.aux_utils import validate_config


class BaseConvNet(metaclass=ABCMeta):
    """Base class for all networks"""

    @abstractmethod
    def __init__(self, network_config, predict=False):
        """Init

        :param dict network_config: dict with all network associated parameters
         str class: class of the network to be used
         int num_input_channels: as named
         str data_format: as named. [channels_last, channels_first]
         int height: as named
         int width: as named
         int depth: as named (only for 3D networks)
         str padding: default is 'same'. [same, valid]
         str init: method used for initializing weights. default 'he_normal'
         dict activation: keys: type: activation type (default: relu), and
          params: other advanced activation related params
         bool batch_norm: indicator for batch norm
         str pooling_type: ['max', 'average']
         int/tuple filter_size: tuple for anisotropic filters. default = 3
         float dropout: as named. default=0.0
         str block_sequence: order of conv, BN and activation
         int num_filters_per_block: as named
         int num_convs_per_block: as named.
         int num_dims: dimensionality of the filter
         bool residual: make the blocks residual
         str skip_merge_type: [add, concat] for Unet variants
         str upsampling: [repeat, bilinear, nearest_neighbor] for Unet variants
         int num_target_channels: for unet variants
         str kernel_regularizer: for networks with dense layers. Instace of
          keras.regularizers [l1, l2 or l1l2]
         float dropout_dense: dropout probability for dense layers
        :param bool predict: indicator for what the model is used for:
         train/predict
        """

        req_params = ['batch_norm', 'pooling_type', 'height', 'width',
                      'data_format', 'num_input_channels', 'final_activation']

        if not predict:
            param_check, msg = validate_config(network_config, req_params)
            if not param_check:
                raise ValueError(msg)
        self.config = network_config

        assert network_config['data_format'] in ['channels_first',
                                                 'channels_last'], \
            'invalid data format. Not in [channels_first or channels_last]'

        # fill in default values
        if 'filter_size' not in network_config:
            network_config['filter_size'] = 3
        if 'activation' not in network_config:
            network_config['activation'] = {'type': 'relu'}
        if 'padding' not in network_config:
            network_config['padding'] = 'same'
        if 'init' not in network_config:
            network_config['init'] = 'he_normal'
        if 'dropout' not in network_config:
            network_config['dropout'] = 0.0

        dropout_prob = network_config['dropout']
        assert 0.0 <= dropout_prob < 1, 'invalid dropout probability'
        self.dropout_prob = dropout_prob

    @abstractmethod
    def build_net(self):
        """Assemble/build the network from layers"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _get_input_shape(self):
        """Return the shape of the input"""
        raise NotImplementedError
