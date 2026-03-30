"""3D convolutional building blocks for CellDiff U-Net encoder/decoder."""

import torch
import torch.nn as nn
from einops import rearrange


class Block(nn.Module):
    """Conv3d + GroupNorm + SiLU block with optional affine conditioning.

    Parameters
    ----------
    dim : int
        Input channels.
    dim_out : int
        Output channels.
    groups : int
        Number of groups for GroupNorm.
    """

    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, C, D, H, W)``.
        scale_shift : tuple[torch.Tensor, torch.Tensor] | None
            Optional ``(scale, shift)`` tensors for affine conditioning.

        Returns
        -------
        torch.Tensor
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
    """Residual block with optional timestep conditioning.

    Parameters
    ----------
    dim : int
        Input channels.
    dim_out : int
        Output channels.
    time_emb_dim : int | None
        Timestep embedding dimension. If provided, timestep embeddings are
        projected to scale/shift parameters for adaptive normalization.
    """

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int | None = None) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, C, D, H, W)``.
        time_emb : torch.Tensor | None
            Timestep embedding of shape ``(B, time_emb_dim)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, dim_out, D, H, W)``.
        """
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)
