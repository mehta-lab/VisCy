from typing import Literal

import timm
import torch.nn as nn
from torch import Tensor

from viscy.unet.networks.unext2 import StemDepthtoChannels


class ContrastiveEncoder(nn.Module):
    """
    Contrastive encoder network that uses ConvNeXt v1 and ResNet backbones from timm.

    Parameters
    ----------
    backbone : Literal["convnext_tiny", "convnextv2_tiny", "resnet50"]
        Name of the timm backbone architecture
    in_channels : int, optional
        Number of input channels
    in_stack_depth : int, optional
        Number of input Z slices
    stem_kernel_size : tuple[int, int, int], optional
        Stem kernel size, by default (5, 4, 4)
    stem_stride : tuple[int, int, int], optional
        Stem stride, by default (5, 4, 4)
    embedding_dim : int, optional
        Embedded feature dimension that matches backbone output channels,
        by default 768 (convnext_tiny)
    projection_dim : int, optional
        Projection dimension for computing loss, by default 128
    drop_path_rate : float, optional
        probability that residual connections are dropped during training,
        by default 0.0
    """

    def __init__(
        self,
        backbone: Literal["convnext_tiny", "convnextv2_tiny", "resnet50"],
        in_channels: int,
        in_stack_depth: int,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        stem_stride: tuple[int, int, int] = (5, 4, 4),
        embedding_dim: int = 768,
        projection_dim: int = 128,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=False,
            drop_path_rate=drop_path_rate,
            num_classes=embedding_dim,
        )
        if "convnext" in backbone:
            in_channels_encoder = encoder.stem[0].out_channels
            # Remove the convolution layer of stem, but keep the layernorm.
            encoder.stem[0] = nn.Identity()
        elif "resnet" in backbone:
            # Adapt stem and projection head of resnet here.
            # replace the stem designed for RGB images with a stem designed to handle 3D multi-channel input.
            in_channels_encoder = encoder.conv1.out_channels
            encoder.conv1 = nn.Identity()
        # Save projection head separately and erase the projection head contained within the encoder.
        projection = nn.Sequential(
            nn.Linear(encoder.head.fc.in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )
        encoder.head.fc = nn.Identity()
        # Create a new stem that can handle 3D multi-channel input.
        self.stem = StemDepthtoChannels(
            in_channels=in_channels,
            in_stack_depth=in_stack_depth,
            in_channels_encoder=in_channels_encoder,
            stem_kernel_size=stem_kernel_size,
            stem_stride=stem_stride,
        )
        # Append modified encoder.
        self.encoder = encoder
        # Append modified projection head.
        self.projection = projection

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input image

        Returns
        -------
        tuple[Tensor, Tensor]
            The embedding tensor and the projection tensor
        """
        x = self.stem(x)
        embedding = self.encoder(x)
        projections = self.projection(embedding)
        return (embedding, projections)
