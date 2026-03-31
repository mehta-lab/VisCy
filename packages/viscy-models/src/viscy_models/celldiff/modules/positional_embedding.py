"""Sine/cosine positional embedding functions for 3D grids.

Adapted from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
"""

import numpy as np
from numpy.typing import NDArray


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: list[int],
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> NDArray[np.float64]:
    """Generate 3D sinusoidal positional embeddings.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension. Must be divisible by 8.
    grid_size : list[int]
        Grid dimensions ``[D, H, W]``.
    cls_token : bool
        Whether to prepend extra token positions.
    extra_tokens : int
        Number of extra tokens to prepend (zeros).

    Returns
    -------
    NDArray[np.float64]
        Positional embeddings of shape ``(D*H*W, embed_dim)``
        or ``(extra_tokens + D*H*W, embed_dim)`` if cls_token is True.
    """
    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    # indexing="ij" produces D-varies-slowest order matching PatchEmbed3D's
    # C-order flattening of (B, C, D, H, W) → (B, C, D*H*W).
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing="ij")  # d, h, w
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim: int, grid: NDArray[np.float32]) -> NDArray[np.float64]:
    """Generate 3D sinusoidal embeddings from a pre-built meshgrid.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension. Must be divisible by 8.
    grid : NDArray[np.float32]
        Meshgrid array of shape ``(3, 1, D, H, W)``.

    Returns
    -------
    NDArray[np.float64]
        Embeddings of shape ``(D*H*W, embed_dim)``.
    """
    if embed_dim % 8 != 0:
        raise ValueError(f"embed_dim must be divisible by 8, got {embed_dim}")

    dim_d = embed_dim // 4
    dim_h = 3 * embed_dim // 8
    dim_w = 3 * embed_dim // 8

    emb_d = get_1d_sincos_pos_embed_from_grid(dim_d, grid[0])  # (D*H*W, D/4)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid[1])  # (D*H*W, 3*D/8)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid[2])  # (D*H*W, 3*D/8)

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (D*H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: NDArray[np.floating]) -> NDArray[np.float64]:
    """Generate 1D sinusoidal embeddings from position values.

    Parameters
    ----------
    embed_dim : int
        Output dimension for each position. Must be divisible by 2.
    pos : NDArray[np.floating]
        Positions to encode, shape ``(M,)`` or broadcastable.

    Returns
    -------
    NDArray[np.float64]
        Embeddings of shape ``(M, embed_dim)``.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
