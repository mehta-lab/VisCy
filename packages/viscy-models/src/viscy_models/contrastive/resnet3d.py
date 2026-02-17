"""3D ResNet contrastive encoder using MONAI backends."""

import torch.nn as nn
from monai.networks.nets.resnet import ResNetFeatures
from torch import Tensor

from viscy_models.contrastive.encoder import projection_mlp

__all__ = ["ResNet3dEncoder"]


class ResNet3dEncoder(nn.Module):
    """3D ResNet encoder network that uses MONAI's ResNetFeatures.

    Parameters
    ----------
    backbone : str
        Name of the backbone model.
    in_channels : int, optional
        Number of input channels.
    embedding_dim : int, optional
        Embedded feature dimension that matches backbone output channels,
        by default 512 (ResNet-18).
    projection_dim : int, optional
        Projection dimension for computing loss, by default 128.
    pretrained : bool, optional
        Whether to load pretrained weights for the backbone, by default False.
    """

    def __init__(
        self,
        backbone: str,
        in_channels: int = 1,
        embedding_dim: int = 512,
        projection_dim: int = 128,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ResNetFeatures(
            backbone, pretrained=pretrained, spatial_dims=3, in_channels=in_channels
        )
        self.projection = projection_mlp(embedding_dim, embedding_dim, projection_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input image.

        Returns
        -------
        tuple[Tensor, Tensor]
            The embedding tensor and the projection tensor.
        """
        feature_map = self.encoder(x)[-1]
        embedding = self.encoder.avgpool(feature_map)
        embedding = embedding.view(embedding.size(0), -1)
        projections = self.projection(embedding)
        return (embedding, projections)
