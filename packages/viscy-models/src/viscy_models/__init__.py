"""VisCy Models - Neural network architectures for virtual staining microscopy."""

from importlib.metadata import version

__version__ = version("viscy-models")

from viscy_models.contrastive import ContrastiveEncoder, ResNet3dEncoder
from viscy_models.unet import FullyConvolutionalMAE, UNeXt2, Unet2d, Unet25d
from viscy_models.vae import BetaVae25D, BetaVaeMonai

__all__ = [
    "BetaVae25D",
    "BetaVaeMonai",
    "ContrastiveEncoder",
    "FullyConvolutionalMAE",
    "ResNet3dEncoder",
    "UNeXt2",
    "Unet2d",
    "Unet25d",
]
