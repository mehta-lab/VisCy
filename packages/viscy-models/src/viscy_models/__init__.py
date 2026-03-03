"""VisCy Models - Neural network architectures for virtual staining microscopy."""

from importlib.metadata import version

__version__ = version("viscy-models")

from viscy_models.contrastive import ContrastiveEncoder, ResNet3dEncoder
from viscy_models.foundation import DINOv3Model
from viscy_models.unet import FullyConvolutionalMAE, Unet2d, Unet25d, UNeXt2
from viscy_models.vae import BetaVae25D, BetaVaeMonai

__all__ = [
    "BetaVae25D",
    "BetaVaeMonai",
    "ContrastiveEncoder",
    "DINOv3Model",
    "FullyConvolutionalMAE",
    "ResNet3dEncoder",
    "UNeXt2",
    "Unet2d",
    "Unet25d",
]
