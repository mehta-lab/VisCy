"""Shared 3D convolutional building blocks for U-Net architectures.

Provides configurable convolution blocks, residual blocks, timestep
embedding, and a convolutional bottleneck module. All modules are
pure PyTorch with no optional dependencies.
"""

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["Block", "ConvBottleneck3D", "ResnetBlock", "TimestepEmbedder"]


def _make_norm(norm: Literal["group", "batch"], channels: int, groups: int) -> nn.Module:
    """Create a normalization layer.

    Parameters
    ----------
    norm : {"group", "batch"}
        Normalization type.
    channels : int
        Number of channels.
    groups : int
        Number of groups for GroupNorm (ignored for BatchNorm).

    Returns
    -------
    nn.Module
        Normalization layer.
    """
    if norm == "group":
        return nn.GroupNorm(groups, channels)
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    raise ValueError(f"Unknown norm type: {norm!r}, expected 'group' or 'batch'")


def _make_activation(activation: Literal["silu", "relu"]) -> nn.Module:
    """Create an activation layer.

    Parameters
    ----------
    activation : {"silu", "relu"}
        Activation type.

    Returns
    -------
    nn.Module
        Activation layer.
    """
    if activation == "silu":
        return nn.SiLU()
    if activation == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation type: {activation!r}, expected 'silu' or 'relu'")


class Block(nn.Module):
    """Conv3d + Norm + Activation block with optional affine conditioning.

    Parameters
    ----------
    dim : int
        Input channels.
    dim_out : int
        Output channels.
    norm : {"group", "batch"}
        Normalization type.
    activation : {"silu", "relu"}
        Activation type.
    groups : int
        Number of groups for GroupNorm (ignored for BatchNorm).
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        norm: Literal["group", "batch"] = "group",
        activation: Literal["silu", "relu"] = "silu",
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = _make_norm(norm, dim_out, groups)
        self.act = _make_activation(activation)

    def forward(self, x: Tensor, scale_shift: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, D, H, W)``.
        scale_shift : tuple[Tensor, Tensor] | None
            Optional ``(scale, shift)`` tensors for affine conditioning.

        Returns
        -------
        Tensor
            Output of shape ``(B, dim_out, D, H, W)``.
        """
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """Double-conv block with optional residual connection and timestep conditioning.

    When ``residual=True`` (default), output is ``block2(block1(x)) + proj(x)``
    where ``proj`` is a 1x1 conv if ``dim != dim_out``, else identity.

    When ``residual=False``, output is ``block2(block1(x))`` — equivalent to
    a plain double-convolution block (like FNet's ``_DoubleConv3d``).

    Parameters
    ----------
    dim : int
        Input channels.
    dim_out : int
        Output channels.
    time_emb_dim : int | None
        Timestep embedding dimension. If provided, timestep embeddings are
        projected to scale/shift parameters for adaptive normalization.
    residual : bool
        Whether to add a residual connection.
    norm : {"group", "batch"}
        Normalization type for inner blocks.
    activation : {"silu", "relu"}
        Activation type for inner blocks.
    groups : int
        Number of groups for GroupNorm (ignored for BatchNorm).
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: int | None = None,
        residual: bool = True,
        norm: Literal["group", "batch"] = "group",
        activation: Literal["silu", "relu"] = "silu",
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None
        block_kwargs = dict(norm=norm, activation=activation, groups=groups)
        self.block1 = Block(dim, dim_out, **block_kwargs)
        self.block2 = Block(dim_out, dim_out, **block_kwargs)
        self.res_conv = (nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()) if residual else None

    def forward(self, x: Tensor, time_emb: Tensor | None = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, D, H, W)``.
        time_emb : Tensor | None
            Timestep embedding of shape ``(B, time_emb_dim)``.

        Returns
        -------
        Tensor
            Output of shape ``(B, dim_out, D, H, W)``.
        """
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb[:, :, None, None, None]
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        if self.res_conv is not None:
            return h + self.res_conv(x)
        return h


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations.

    Uses sinusoidal frequency embeddings followed by an MLP.

    Parameters
    ----------
    hidden_size : int
        Output embedding dimension.
    frequency_embedding_size : int
        Dimension of intermediate sinusoidal embeddings.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Parameters
        ----------
        t : Tensor
            Scalar timesteps of shape ``(B,)``.

        Returns
        -------
        Tensor
            Embeddings of shape ``(B, hidden_size)``.
        """
        args = t[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(t_freq)


class ConvBottleneck3D(nn.Module):
    """Convolutional bottleneck for 3D U-Net architectures.

    Wraps a single ``ResnetBlock`` with matching norm/activation/residual
    settings. Implements the unified bottleneck interface
    ``forward(x, time_embeds=None)``.

    Parameters
    ----------
    channels : int
        Input and output channels.
    time_emb_dim : int | None
        Timestep embedding dimension (passed through to ``ResnetBlock``).
    residual : bool
        Whether the inner block uses a residual connection.
    norm : {"group", "batch"}
        Normalization type.
    activation : {"silu", "relu"}
        Activation type.
    groups : int
        Number of groups for GroupNorm (ignored for BatchNorm).
    """

    def __init__(
        self,
        channels: int,
        *,
        time_emb_dim: int | None = None,
        residual: bool = True,
        norm: Literal["group", "batch"] = "group",
        activation: Literal["silu", "relu"] = "silu",
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = ResnetBlock(
            channels,
            channels,
            time_emb_dim=time_emb_dim,
            residual=residual,
            norm=norm,
            activation=activation,
            groups=groups,
        )

    def forward(self, x: Tensor, time_embeds: Tensor | None = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, D, H, W)``.
        time_embeds : Tensor | None
            Timestep embeddings of shape ``(B, time_emb_dim)``.

        Returns
        -------
        Tensor
            Output of shape ``(B, C, D, H, W)``.
        """
        return self.block(x, time_embeds)
