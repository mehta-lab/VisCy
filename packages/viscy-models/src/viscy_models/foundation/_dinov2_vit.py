"""Vendored DINOv2 vision transformer (eval-only subset) for CELL-DINO.

Vendored from https://github.com/facebookresearch/dinov2 (Apache-2.0).
Stripped to what is needed to load CELL-DINO checkpoints and run frozen
inference: ``DinoVisionTransformer`` + ``Block`` + standard ``Attention``
(no xformers, no NestedTensorBlock, no stochastic depth machinery, no
CausalAttentionBlock).

Module layout matches the original so the published CELL-DINO state_dicts
(``cls_token``, ``pos_embed``, ``patch_embed.proj.*``, ``blocks.X.Y.*``,
``norm.*``) load cleanly.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _make_2tuple(x: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError(f"expected 2-tuple, got {x!r}")
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, C, H, W) -> (B, N, D)."""

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        image_hw = _make_2tuple(img_size)
        patch_hw = _make_2tuple(patch_size)
        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = (image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        ph, pw = self.patch_size
        if h % ph != 0 or w % pw != 0:
            raise ValueError(f"input {h}x{w} not divisible by patch {ph}x{pw}")
        x = self.proj(x)
        h_out, w_out = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, h_out, w_out, self.embed_dim)
        return x


class LayerScale(nn.Module):
    """Per-channel learnable gain applied after attention/MLP."""

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.full((dim,), float(init_values)))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    """Standard multi-head self-attention using PyTorch SDPA (no xformers)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        x = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        x = x.transpose(1, 2).contiguous().view(b, n, c)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Block(nn.Module):
    """Pre-norm transformer block with LayerScale (eval-only, no drop_path)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class BlockChunk(nn.ModuleList):
    """Sequential block group used when ``block_chunks > 0`` (matches state_dict layout)."""

    def forward(self, x: Tensor) -> Tensor:
        for blk in self:
            x = blk(x)
        return x


class DinoVisionTransformer(nn.Module):
    """Eval-only DINOv2 ViT.

    Parameters
    ----------
    img_size, patch_size, in_chans : int
        Patch-embedding geometry.  CELL-DINO ``channel_adaptive_dino_vitl16``
        was trained at ``img_size=224, patch_size=16, in_chans=1``.
    embed_dim, depth, num_heads, mlp_ratio : int / float
        Backbone size.  ViT-L = ``(1024, 24, 16, 4.0)``.
    init_values : float
        LayerScale init.  CELL-DINO uses ``1.0``.
    block_chunks : int
        Splits blocks into ``block_chunks`` ``BlockChunk`` groups so that
        parameter names match published checkpoints (``blocks.<chunk>.<idx>.*``).
        CELL-DINO uses ``4``.
    channel_adaptive : bool
        Stored as ``self.bag_of_channels``.  Wrapper code outside this module
        is responsible for the ``(B,C,H,W) -> (B*C,1,H,W)`` reshape; this flag
        is recorded so the wrapper can branch.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        block_chunks: int = 1,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        channel_adaptive: bool = False,
    ) -> None:
        super().__init__()
        if num_register_tokens < 0:
            raise ValueError("num_register_tokens must be >= 0")
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.bag_of_channels = channel_adaptive

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        blocks_list = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ]
        if block_chunks > 0:
            if depth % block_chunks != 0:
                # `range(0, depth, depth // block_chunks)` produces more than
                # `block_chunks` iterations when this divides unevenly, which
                # silently breaks the published state_dict layout.
                raise ValueError(f"depth ({depth}) must be divisible by block_chunks ({block_chunks}).")
            self.chunked_blocks = True
            chunked = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos = pos_embed[:, 0]
        patch_pos = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        m = int(math.sqrt(n))
        if n != m * m:
            raise AssertionError("non-square positional grid not supported")
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / m
            sy = float(h0 + self.interpolate_offset) / m
            kwargs = {"scale_factor": (sx, sy)}
        else:
            kwargs = {"size": (w0, h0)}
        patch_pos = nn.functional.interpolate(
            patch_pos.reshape(1, m, m, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos.unsqueeze(0), patch_pos), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x: Tensor, masks: Optional[Tensor] = None) -> Tensor:
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]),
                dim=1,
            )
        return x

    def forward_features(self, x: Tensor, masks: Optional[Tensor] = None) -> dict:
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x: Tensor, is_training: bool = False) -> Tensor | dict:
        ret = self.forward_features(x)
        return ret if is_training else self.head(ret["x_norm_clstoken"])


def vit_large(
    patch_size: int = 16,
    in_chans: int = 3,
    channel_adaptive: bool = False,
    **kwargs,
) -> DinoVisionTransformer:
    """ViT-L factory matching ``dinov2.models.vision_transformer.vit_large``."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        in_chans=in_chans,
        channel_adaptive=channel_adaptive,
        **kwargs,
    )
