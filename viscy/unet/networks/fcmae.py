"""
Fully Convolutional Masked Autoencoder as described in ConvNeXt V2
based on the official JAX example in
https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md#implementing-fcmae-with-masked-convolution-in-jax
also referring to timm's dense implementation of the encoder in ``timm.models.convnext``
"""


from typing import Callable, Literal, Sequence

import torch
from timm.layers import DropPath, LayerNorm2d, create_conv2d, trunc_normal_
from timm.models.convnext import Downsample
from torch import BoolTensor, Tensor, nn


def _upsample_mask(mask: BoolTensor, features: Tensor) -> BoolTensor:
    mask = mask[..., :, :][None, None]
    if features.shape[-2:] != mask.shape[-2:]:
        if not all(i % j == 0 for i, j in zip(features.shape[-2:], mask.shape[-2:])):
            raise ValueError(
                f"feature map shape {features.shape} must be divisible by "
                f"mask shape {mask.shape}."
            )
        mask = mask.repeat_interleave(
            features.shape[-2] // mask.shape[-2], dim=-2
        ).repeat_interleave(features.shape[-1] // mask.shape[-1], dim=-1)
    return mask


class MaskedGlobalResponseNorm(nn.Module):
    """
    Masked Global Response Normalization.

    :param int dim: number of input channels
    :param float eps: small value added for numerical stability,
        defaults to 1e-6
    :param bool channels_last: BHWC (True) or BCHW (False) dimension ordering,
        defaults to False
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, channels_last: bool = False
    ) -> None:
        super().__init__()
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            weights_shape = (1, 1, 1, dim)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            weights_shape = (1, dim, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(weights_shape))
        self.beta = nn.Parameter(torch.zeros(weights_shape))
        self.eps = eps

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        :param Tensor x: input tensor, BHWC or BCHW
        :param BoolTensor | None mask: boolean mask, defaults to None
        :return Tensor: normalized tensor
        """
        samples = x if mask is None else x * ~mask
        g_x = samples.norm(p=2, dim=self.spatial_dim, keepdim=True)
        n_x = g_x / (g_x.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return x + torch.addcmul(self.beta, self.gamma, x * n_x)


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
        self.layernorm = LayerNorm2d(out_channels)
        self.pwconv1 = nn.Conv2d(out_channels, mlp_ratio * out_channels, kernel_size=1)
        self.act = nn.GELU()
        self.grn = MaskedGlobalResponseNorm(mlp_ratio * out_channels)
        self.pwconv2 = nn.Conv2d(mlp_ratio * out_channels, out_channels, kernel_size=1)
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
            mask = _upsample_mask(mask, x)
        for block in self.blocks:
            x = block(x, mask)
        return x


class MaskedMultiscaleEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_blocks: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        chs = [in_channels, *dims]
        for i, num_blocks in enumerate(stage_blocks):
            self.stages.append(
                MaskedConvNeXtV2Stage(
                    chs[i],
                    chs[i + 1],
                    kernel_size=7,
                    stride=2,
                    num_blocks=num_blocks,
                    drop_path_rates=[drop_path_rate] * num_blocks,
                )
            )

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        :param Tensor x: input tensor (BCHW)
        :param BoolTensor | None mask: boolean mask, defaults to None
        :return Tensor: output tensor (BCHW)
        """
        features = []
        for stage in self.stages:
            x = stage(x, mask)
            features.append(x)
        return features
