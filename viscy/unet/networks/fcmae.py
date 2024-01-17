"""
Fully Convolutional Masked Autoencoder as described in ConvNeXt V2
based on the official JAX example in
https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md#implementing-fcmae-with-masked-convolution-in-jax
and timm's dense implementation of the encoder in ``timm.models.convnext``
"""


from typing import Callable, Literal, Sequence

import torch
import torch.nn.functional as F
from timm.layers import (
    DropPath,
    GlobalResponseNormMlp,
    LayerNorm2d,
    create_conv2d,
    trunc_normal_,
)
from timm.models.convnext import Downsample
from torch import BoolTensor, Size, Tensor, nn


def generate_mask(target: Size, stride: int, mask_ratio: float) -> BoolTensor:
    """
    :param Size target: target shape
    :param int stride: total stride
    :param float mask_ratio: ratio of the pixels to mask
    :return BoolTensor: boolean mask (N, H*W)
    """
    m_height = target[-2] // stride
    m_width = target[-1] // stride
    mask_numel = m_height * m_width
    masked_elements = int(mask_numel * mask_ratio)
    mask = torch.rand(target[0], mask_numel).argsort(1) < masked_elements
    return mask.reshape(target[0], 1, m_height, m_width)


def upsample_mask(mask: BoolTensor, target: Size) -> BoolTensor:
    """
    :param BoolTensor mask: low-resolution boolean mask (B1HW)
    :param Size target: target size (BCHW)
    :return BoolTensor: upsampled boolean mask (B1HW)
    """
    if target[-2:] != mask.shape[-2:]:
        if not all(i % j == 0 for i, j in zip(target, mask.shape)):
            raise ValueError(
                f"feature map shape {target} must be divisible by "
                f"mask shape {mask.shape}."
            )
        mask = mask.repeat_interleave(
            target[-2] // mask.shape[-2], dim=-2
        ).repeat_interleave(target[-1] // mask.shape[-1], dim=-1)
    return mask


def masked_patchify(features: Tensor, mask: BoolTensor | None = None) -> Tensor:
    """
    :param Tensor features: input image features (BCHW)
    :param BoolTensor mask: boolean mask (B1HW)
    :return Tensor: masked channel-last features (BLC, L = H * W * mask_ratio)
    """
    if mask is None:
        return features.flatten(2).permute(0, 2, 1)
    b, c = features.shape[:2]
    # (B, C, H, W) -> (B, H, W, C)
    features = features.permute(0, 2, 3, 1)
    # (B, H, W, C) -> (B * L, C) -> (B, L, C)
    features = features[~mask[:, 0]].reshape(b, -1, c)

    # kernel_size = tuple(features.shape[-i] // mask.shape[-i] for i in (2, 1))
    # # (B, C, H, W) -> (B, C * H_patch * Wp, H_grid * Wg)
    # features = F.unfold(features, kernel_size=kernel_size, stride=kernel_size)
    # patch_size = kernel_size[0] * kernel_size[1]
    # # (B, C * Hp * Wp, Hg * Wg) -> (B, C, Hp * Wp, Hg * Wg) -> (B, Hg * Wg, C, Hp * Wp)
    # features = features.view(b, c, patch_size, -1).permute(0, 3, 1, 2)
    # # (B, 1, Hg, Wg) -> (B, Hg*Wg)
    # idx = ~mask.flatten(1)
    # # (B, Hg * Wg, C, Hp * Wp) -> (B * L, C, Hp * Wp) -> (B, L, C, Hp * Wp)
    # features = features[idx].view(b, -1, c, patch_size)
    # # (B, L, C, Hp * Wp) -> (B, L, Hp * Wp, C) -> (B, L * Hp * Wp, C)
    # features = features.permute(0, 1, 3, 2).reshape(b, -1, c)
    return features


def masked_unpatchify(
    features: Tensor, out_shape: Size, mask: BoolTensor | None = None
) -> Tensor:
    if mask is None:
        # (B, L, C) -> (B, C, L) -> (B, C, H, W)
        return features.permute(0, 2, 1).reshape(out_shape)
    b, c, w, h = out_shape
    out = torch.zeros((b, w, h, c), device=features.device, dtype=features.dtype)
    # (B, L, C) -> (B * L, C)
    features = features.reshape(-1, c)
    out[~mask[:, 0]] = features
    # (B, H, W, C) -> (B, C, H, W)
    return out.permute(0, 3, 1, 2)


class MaskedConvNeXtV2Block(nn.Module):
    """Masked ConvNeXt V2 Block.

    :param int in_channels: input channels
    :param int | None out_channels: output channels, defaults to None
    :param int kernel_size: depth-wise convolution kernel size, defaults to 7
    :param int stride: downsample stride, defaults to 1
    :param int mlp_ratio: MLP expansion ratio, defaults to 4
    :param float drop_path: drop path rate, defaults to 0.0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 7,
        stride: int = 1,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.dwconv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            depthwise=True,
        )
        self.layernorm = nn.LayerNorm(out_channels)
        mid_channels = mlp_ratio * out_channels
        self.mlp = GlobalResponseNormMlp(
            in_features=out_channels,
            hidden_features=mid_channels,
            out_features=out_channels,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if in_channels != out_channels or stride > 1:
            self.shortcut = Downsample(in_channels, out_channels, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        :param Tensor x: input tensor (BCHW)
        :param BoolTensor | None mask: boolean mask, defaults to None
        :return Tensor: output tensor (BCHW)
        """
        shortcut = self.shortcut(x)
        if mask is not None:
            x *= ~mask
        x = self.dwconv(x)
        if mask is not None:
            x *= ~mask
        out_shape = x.shape
        x = masked_project(x, mask)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x, mask)
        x = self.pwconv2(x)
        x = self.drop_path(x) + shortcut
        return x


class MaskedConvNeXtV2Stage(nn.Module):
    """Masked ConvNeXt V2 Stage.

    :param int in_channels: input channels
    :param int out_channels: output channels
    :param int kernel_size: depth-wise convolution kernel size, defaults to 7
    :param int stride: downsampling factor of this stage, defaults to 2
    :param int num_blocks: number of residual blocks, defaults to 2
    :param Sequence[float] | None drop_path_rates: drop path rates of each block,
        defaults to None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        num_blocks: int = 2,
        drop_path_rates: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        elif len(drop_path_rates) != num_blocks:
            raise ValueError(
                "length of drop_path_rates must be equal to "
                f"the number of blocks {num_blocks}, got {len(drop_path_rates)}."
            )
        if in_channels != out_channels or stride > 1:
            downsample_kernel_size = stride if stride > 1 else 1
            self.downsample = nn.Sequential(
                LayerNorm2d(in_channels),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=downsample_kernel_size,
                    stride=stride,
                    padding=0,
                ),
            )
            in_channels = out_channels
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                MaskedConvNeXtV2Block(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    drop_path=drop_path_rates[i],
                )
            )
            in_channels = out_channels

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        :param Tensor x: input tensor (BCHW)
        :param BoolTensor | None mask: boolean mask, defaults to None
        :return Tensor: output tensor (BCHW)
        """
        x = self.downsample(x)
        if mask is not None:
            mask = upsample_mask(mask, x.shape)
        for block in self.blocks:
            x = block(x, mask)
        return x


class MaskedAdaptiveProjection(nn.Module):
    """
    Masked patchifying layer for projecting 2D or 3D input into 2D feature maps.

    :param int in_channels: input channels
    :param int out_channels: output channels
    :param Sequence[int, int] | int kernel_size_2d: kernel width and height
    :param int kernel_depth: kernel depth for 3D input
    :param int in_stack_depth: input stack depth for 3D input
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_2d: tuple[int, int] | int = 4,
        kernel_depth: int = 5,
        in_stack_depth: int = 5,
    ) -> None:
        super().__init__()
        ratio = in_stack_depth // kernel_depth
        if isinstance(kernel_size_2d, int):
            kernel_size_2d = [kernel_size_2d] * 2
        kernel_size_3d = [kernel_depth, *kernel_size_2d]
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels // ratio,
            kernel_size=kernel_size_3d,
            stride=kernel_size_3d,
        )
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_2d,
            stride=kernel_size_2d,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor, mask: BoolTensor = None) -> Tensor:
        """
        :param Tensor x: input tensor (BCDHW)
        :param BoolTensor mask: boolean mask (B1HW), defaults to None
        :return Tensor: output tensor (BCHW)
        """
        # no need to mask before convolutions since patches do not spill over
        if x.shape[2] > 1:
            x = self.conv3d(x)
            b, c, d, h, w = x.shape
            # project Z/depth into channels
            # return a view when possible (contiguous)
            x = x.reshape(b, c * d, h, w)
        else:
            x = self.conv2d(x.squeeze(2))
        out_shape = x.shape
        if mask is not None:
            mask = upsample_mask(mask, x.shape)
            x = x[mask]
        else:
            x = x.flatten(2)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        if mask is not None:
            out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
            out[mask] = x
            return out
        return x.reshape(out_shape)


class MaskedMultiscaleEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_blocks: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        stem_kernel_size_2d = 4
        self.stem = nn.Sequential(
            MaskedAdaptiveProjection(
                in_channels, dims[0], kernel_size_2d=stem_kernel_size_2d, kernel_depth=5
            ),
            LayerNorm2d(dims[0]),
        )
        self.stages = nn.ModuleList()
        chs = [dims[0], *dims]
        for i, num_blocks in enumerate(stage_blocks):
            stride = 1 if i == 0 else 2
            self.stages.append(
                MaskedConvNeXtV2Stage(
                    chs[i],
                    chs[i + 1],
                    kernel_size=7,
                    stride=stride,
                    num_blocks=num_blocks,
                    drop_path_rates=[drop_path_rate] * num_blocks,
                )
            )
        self.total_stride = stem_kernel_size_2d * 2 ** (len(self.stages) - 1)

    def forward(self, x: Tensor, mask_ratio: float = 0.0) -> Tensor:
        """
        :param Tensor x: input tensor (BCHW)
        :param float mask_ratio: ratio of the feature maps to mask,
            defaults to 0.0 (no masking)
        :return Tensor: output tensor (BCHW)
        """
        if mask_ratio > 0.0:
            mask = generate_mask(x.shape, self.total_stride, mask_ratio)
        else:
            mask = None
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x, mask)
            features.append(x)
        if mask is not None:
            return features, mask
        return features
