"""Head modules for VisCy models.

Includes spatial heads (2D→3D projection) and embedding-space heads
(projection MLPs, cosine classifiers).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.utils import normal_init
from torch import Tensor, nn

from viscy_models.components.blocks import icnr_init

__all__ = [
    "CosineClassifier",
    "MLP",
    "PixelToVoxelHead",
    "PixelToVoxelShuffleHead",
    "UnsqueezeHead",
]


class CosineClassifier(nn.Module):
    """L2-normalised linear head with learnable temperature.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality.
    num_classes : int
        Number of output classes.
    init_scale : float
        Initial value of the temperature scale (before log).
    learn_scale : bool
        Whether to make the temperature a learnable parameter.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        init_scale: float = 20.0,
        learn_scale: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.normal_(self.weight, std=0.01)
        if learn_scale:
            self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        else:
            self.register_buffer("log_scale", torch.tensor(math.log(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return torch.exp(self.log_scale) * (x @ w.t())


class MLP(nn.Module):
    """Configurable MLP with optional classification head and penultimate-layer extraction.

    Supports two modes:

    - **Projection mode** (``num_classes=None``, default): maps embeddings to a
      projection space for contrastive loss. Output norm is applied to the final
      layer via ``norm``.
    - **Classification mode** (``num_classes`` set): adds a classification head
      (linear or cosine) on top of the backbone. Use :meth:`encode` to extract
      L2-normalised penultimate-layer representations.

    Parameters
    ----------
    in_dims : int
        Input feature dimension.
    hidden_dims : int | list[int]
        Hidden layer width. A single ``int`` gives one hidden layer (matching
        the legacy two-layer behaviour); a ``list`` gives one layer per element.
    out_dims : int
        Output dimension of the final projection layer (ignored in classification
        mode — the backbone output feeds directly into ``head``).
    norm : Literal["bn", "ln"]
        Normalization applied after each hidden layer. ``"bn"`` = BatchNorm1d,
        ``"ln"`` = LayerNorm.
    activation : Literal["relu", "gelu", "silu"]
        Hidden activation function.
    dropout : float
        Dropout rate applied after each hidden layer. ``0.0`` disables dropout.
    num_classes : int or None
        When set, adds a classification head and enables :meth:`encode`.
        When ``None`` (default), the MLP acts as a projection head.
    cosine_classifier : bool
        Use :class:`CosineClassifier` instead of a plain linear head.
        Only used when ``num_classes`` is set.
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int | list[int],
        out_dims: int | None = None,
        norm: Literal["bn", "ln"] = "bn",
        activation: Literal["relu", "gelu", "silu"] = "relu",
        dropout: float = 0.0,
        num_classes: int | None = None,
        cosine_classifier: bool = True,
    ) -> None:
        if num_classes is None and out_dims is None:
            raise ValueError("out_dims is required in projection mode (num_classes=None).")
        super().__init__()
        self.input_dim = in_dims

        hidden_list = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)

        def _norm(dim: int) -> nn.Module:
            if norm == "bn":
                return nn.BatchNorm1d(dim)
            elif norm == "ln":
                return nn.LayerNorm(dim)
            raise ValueError(f"norm must be 'bn' or 'ln', got '{norm}'")

        def _act() -> nn.Module:
            if activation == "relu":
                return nn.ReLU(inplace=True)
            elif activation == "gelu":
                return nn.GELU()
            elif activation == "silu":
                return nn.SiLU(inplace=True)
            raise ValueError(f"activation must be 'relu', 'gelu', or 'silu', got '{activation}'")

        layers: list[nn.Module] = []
        prev_dim = in_dims
        for h in hidden_list:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(_norm(h))
            layers.append(_act())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        if num_classes is None:
            # Projection mode: final linear + norm
            layers.append(nn.Linear(prev_dim, out_dims))
            layers.append(_norm(out_dims))
            self.backbone = nn.Sequential(*layers)
            self.head: nn.Module | None = None
        else:
            # Classification mode: backbone stops before head
            self.backbone = nn.Sequential(*layers)
            if cosine_classifier:
                self.head = CosineClassifier(prev_dim, num_classes)
            else:
                self.head = nn.Linear(prev_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through backbone and optional head.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, in_dims)``.

        Returns
        -------
        Tensor
            Projected or classified output.
        """
        x = self.backbone(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def encode(self, x: Tensor) -> Tensor:
        """Return L2-normalised penultimate-layer representations.

        Only valid when ``num_classes`` was set at construction.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, in_dims)``.

        Returns
        -------
        Tensor
            L2-normalised backbone output of shape ``(B, hidden_dims[-1])``.

        Raises
        ------
        RuntimeError
            If called on a projection-mode MLP (``num_classes=None``).
        """
        if self.head is None:
            raise RuntimeError("encode() is only available in classification mode (num_classes != None).")
        return F.normalize(self.backbone(x), dim=1)


class PixelToVoxelHead(nn.Module):
    """Pixel-shuffle head that upsamples 2D features to 3D voxel output."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_stack_depth: int,
        expansion_ratio: int,
        pool: bool,
    ) -> None:
        super().__init__()
        first_scale = 2
        self.upsample = UpSample(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels // first_scale**2,
            scale_factor=first_scale,
            mode="pixelshuffle",
            pre_conv=None,
            apply_pad_pool=pool,
        )
        mid_channels = out_channels * expansion_ratio * 2**2
        self.conv = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=in_channels // first_scale**2 // (out_stack_depth + 2),
                out_channels=mid_channels,
                kernel_size=3,
                padding=(0, 1, 1),
            ),
            nn.Conv3d(mid_channels, out_channels * 2**2, 1),
        )
        normal_init(self.conv[0])
        icnr_init(self.conv[-1], 2, upsample_dims=2)
        self.out = nn.PixelShuffle(2)
        self.out_stack_depth = out_stack_depth

    def forward(self, x: Tensor) -> Tensor:
        """Upsample 2D feature map and reshape to 3D voxel output."""
        x = self.upsample(x)
        d = self.out_stack_depth + 2
        b, c, h, w = x.shape
        x = x.reshape((b, c // d, d, h, w))
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.out(x)
        return x.transpose(1, 2)


class UnsqueezeHead(nn.Module):
    """Unsqueeze 2D (B, C, H, W) feature map to 3D (B, C, 1, H, W) output."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Add a singleton depth dimension to convert 2D to 3D."""
        x = x.unsqueeze(2)
        return x


class PixelToVoxelShuffleHead(nn.Module):
    """Pixel-shuffle head that reshapes 2D features into a 3D volume."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_stack_depth: int = 5,
        xy_scaling: int = 4,
        pool: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.out_stack_depth = out_stack_depth
        self.upsample = UpSample(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_stack_depth * out_channels,
            scale_factor=xy_scaling,
            mode="pixelshuffle",
            pre_conv=None,
            apply_pad_pool=pool,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Upsample 2D feature map and reshape to 3D voxel output."""
        x = self.upsample(x)
        b, _, h, w = x.shape
        x = x.reshape(b, self.out_channels, self.out_stack_depth, h, w)
        return x
