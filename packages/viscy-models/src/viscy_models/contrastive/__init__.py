"""Contrastive learning architectures."""

from viscy_models.contrastive.encoder import ContrastiveEncoder, projection_mlp
from viscy_models.contrastive.loss import NTXentHCL, NTXentLoss, TemporalStraighteningLoss
from viscy_models.contrastive.predictor import Predictor
from viscy_models.contrastive.resnet3d import ResNet3dEncoder

__all__ = [
    "ContrastiveEncoder",
    "NTXentHCL",
    "NTXentLoss",
    "Predictor",
    "ResNet3dEncoder",
    "TemporalStraighteningLoss",
    "projection_mlp",
]
