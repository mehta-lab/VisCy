import micro_dl.torch_unet.networks.Unet25D as Unet25D
import micro_dl.torch_unet.networks.Unet2D as Unet2D
import os
import torch
import matplotlib.pyplot as plt


def model_init(network_config, device=torch.device("cuda"), debug_mode=False):
    """
    Initializes network model from a configuration dictionary.

    :param dict network_config: dict containing the configuration parameters for the model
    :param torch.device device: device to store model parameters on (must be same as data)
    """

    if network_config["architecture"] == "2.5D":
        model = Unet25D.Unet25d(
            in_channels=network_config["in_channels"],
            out_channels=network_config["out_channels"],
            residual=network_config["residual"],
            task=network_config["task"],
        )
    elif network_config["architecture"] == "2D":
        model = Unet2D.Unet2d(
            in_channels=network_config["in_channels"],
            out_channels=network_config["out_channels"],
            residual=network_config["residual"],
            task=network_config["task"],
        )
    else:
        raise NotImplementedError("Only 2.5D and 2D architectures available.")

    model.debug_mode = debug_mode

    model.to(device)

    return model
