"""Contrastive encoder using timm 2D backbones with 3D-to-2D stem."""

import warnings
from typing import Literal

import timm
import torch.nn as nn
from torch import Tensor

from viscy_models.components.stems import StemDepthtoChannels

__all__ = ["projection_mlp", "ProjectionMLP", "ContrastiveEncoder"]


def projection_mlp(in_dims: int, hidden_dims: int, out_dims: int) -> nn.Module:
    """Build a two-layer projection MLP with batch normalization.

    .. deprecated::
        Use :class:`ProjectionMLP` instead. This function returns a flat
        ``nn.Sequential`` whose state dict keys (``projection.0.*``,
        ``projection.4.*``) match legacy checkpoints. New code and configs
        should use ``ProjectionMLP`` directly.

    Parameters
    ----------
    in_dims : int
        Input feature dimension.
    hidden_dims : int
        Hidden layer dimension.
    out_dims : int
        Output projection dimension.

    Returns
    -------
    nn.Module
        Sequential MLP: Linear -> BN -> ReLU -> Linear -> BN.
    """
    warnings.warn(
        "projection_mlp() is deprecated and will be removed in a future release. Use ProjectionMLP instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.BatchNorm1d(hidden_dims),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dims, out_dims),
        nn.BatchNorm1d(out_dims),
    )


class ProjectionMLP(nn.Module):
    """Two-layer projection MLP with configurable normalization and activation.

    Designed to be directly instantiable from a YAML config (e.g. LightningCLI).

    Use ``norm="bn"`` (default) for standard contrastive pretraining.
    Use ``norm="ln"`` for cross-scope finetuning where batches mix samples
    from different microscopes — LayerNorm normalizes per-sample so domain
    mixing does not contaminate the normalization statistics.

    Use ``activation="relu"`` (default) for standard training.
    Use ``activation="gelu"`` for consistency with ConvNeXt backbones
    (which use GELU internally).

    Parameters
    ----------
    in_dims : int
        Input feature dimension (must match encoder ``embedding_dim``).
    hidden_dims : int
        Hidden layer dimension.
    out_dims : int
        Output projection dimension.
    norm : Literal["bn", "ln"]
        Normalization type. ``"bn"`` = BatchNorm1d, ``"ln"`` = LayerNorm.
        Default: ``"bn"``.
    activation : Literal["relu", "gelu", "silu"]
        Hidden activation function. Default: ``"relu"``.
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        norm: Literal["bn", "ln"] = "bn",
        activation: Literal["relu", "gelu", "silu"] = "relu",
    ) -> None:
        super().__init__()
        norm1: nn.Module
        norm2: nn.Module
        if norm == "bn":
            norm1 = nn.BatchNorm1d(hidden_dims)
            norm2 = nn.BatchNorm1d(out_dims)
        elif norm == "ln":
            norm1 = nn.LayerNorm(hidden_dims)
            norm2 = nn.LayerNorm(out_dims)
        else:
            raise ValueError(f"norm must be 'bn' or 'ln', got '{norm}'")
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "silu":
            act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"activation must be 'relu', 'gelu', or 'silu', got '{activation}'")
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            norm1,
            act,
            nn.Linear(hidden_dims, out_dims),
            norm2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, in_dims)``.

        Returns
        -------
        Tensor
            Projected tensor of shape ``(B, out_dims)``.
        """
        return self.net(x)


class ContrastiveEncoder(nn.Module):
    """Contrastive encoder network using ConvNeXt v1 and ResNet backbones from timm.

    Parameters
    ----------
    backbone : Literal["convnext_tiny", "convnextv2_tiny", "resnet50"]
        Name of the timm backbone architecture.
    in_channels : int
        Number of input channels.
    in_stack_depth : int
        Number of input Z slices.
    stem_kernel_size : tuple[int, int, int], optional
        Stem kernel size, by default (5, 4, 4).
    stem_stride : tuple[int, int, int], optional
        Stem stride, by default (5, 4, 4).
    embedding_dim : int, optional
        Embedded feature dimension that matches backbone output channels,
        by default 768 (convnext_tiny).
    projection_dim : int, optional
        Projection dimension for computing loss, by default 128.
    drop_path_rate : float, optional
        Probability that residual connections are dropped during training,
        by default 0.0.
    pretrained : bool, optional
        Whether to load pretrained weights for the backbone, by default False.
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
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
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
            # Replace the stem designed for RGB images with a stem designed
            # to handle 3D multi-channel input.
            in_channels_encoder = encoder.conv1.out_channels
            encoder.conv1 = nn.Identity()
        # Save projection head separately and erase the projection head
        # contained within the encoder.
        # Use encoder.num_features for uniform API across all timm backbones
        # (fixes bug where encoder.head.fc.in_features fails for resnet50).
        projection = nn.Sequential(
            nn.Linear(encoder.num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )
        if "convnext" in backbone:
            encoder.head.fc = nn.Identity()
        elif "resnet" in backbone:
            encoder.fc = nn.Identity()
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
        x = self.stem(x)
        embedding = self.encoder(x)
        projections = self.projection(embedding)
        return (embedding, projections)
