import micro_dl.torch_unet.networks.Unet25D as Unet25D
import micro_dl.torch_unet.networks.Unet2D as Unet2D
import os
import torch
import matplotlib.pyplot as plt


def model_init(network_config, device=torch.device("cuda"), debug_mode=False):
    """
    Initializes network model from a configuration dictionary.

    :param dict network_config: dict containing the configuration parameters for
                                the model
    :param torch.device device: device to store model parameters on (must be same
                                as data)
    """

    assert (
        "architecture" in network_config
    ), "Must specify network architecture: 2D, 2.5D"

    if network_config["architecture"] == "2.5D":
        default_model = ModelDefaults25D()
        model_class = Unet25D.Unet25d
        model = define_model(
            model_class,
            default_model,
            network_config,
        )
    elif network_config["architecture"] == "2D":
        default_model = ModelDefaults2D()
        model_class = Unet2D.Unet2d
        model = define_model(
            model_class,
            default_model,
            network_config,
        )
    else:
        raise NotImplementedError("Only 2.5D and 2D architectures available.")

    model.debug_mode = debug_mode

    model.to(device)

    return model


def define_model(model_class, model_defaults, config):
    """
    Returns an instance of the model given the parameter config and specified
    defaults. The model weights are not on cpu at this point.

    :param nn.Module model_class: actual model class to pass defaults into
    :param ModelDefaults model_defaults: default model corresponding to config
    :param dict config: _description_
    """
    kwargs = {}
    for param_name in vars(model_defaults):
        if param_name in config:
            kwargs[param_name] = config[param_name]
        else:
            kwargs[param_name] = model_defaults.get(param_name)

    return model_class(**kwargs)


class ModelDefaults:
    def __init__(self):
        """
        Parent class of the model defaults objects.
        """
        self.in_channels = 1
        self.out_channels = 1

    def get(self, varname):
        """
        Logic for getting an attribute of the default parameters class

        :param str varname: name of attribute
        """
        return getattr(self, varname)


class ModelDefaults2D(ModelDefaults):
    def __init__(self):
        """
        Instance of model defaults class, containing all of the default
        hyper-parameters for the 2D unet

        All parameters in this default model CAN be accessed by name through
        the model config
        """
        super(ModelDefaults, self).__init__()

        self.kernel_size = (3, 3)
        self.residual = False
        self.dropout = 0.2
        self.num_blocks = 4
        self.num_block_layers = 2
        self.num_filters = []
        self.task = "reg"


class ModelDefaults25D(ModelDefaults):
    def __init__(self):
        """
        Instance of default model class, containing all of the default
        hyper-parameters for the 2D unet.

        All parameters in this default model CAN be accessed by name through
        the model config
        """

        self.in_stack_depth = 5
        self.out_stack_depth = 1
        self.xy_kernel_size = (3, 3)
        self.residual = False
        self.dropout = 0.2
        self.num_blocks = 4
        self.num_block_layers = 2
        self.num_filters = []
        self.task = "reg"
