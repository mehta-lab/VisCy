"""Fully Convolutional Masked Autoencoder as described in ConvNeXt V2.

Based on the official JAX example in
https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md#implementing-fcmae-with-masked-convolution-in-jax
and timm's dense implementation of the encoder in ``timm.models.convnext``.
"""

import math
from typing import Sequence

import torch
from timm.models.convnext import (
    Downsample,
    DropPath,
    GlobalResponseNormMlp,
    LayerNorm2d,
    create_conv2d,
    trunc_normal_,
)
from torch import BoolTensor, Size, Tensor, nn

from viscy_models.components.blocks import UNeXt2Decoder
from viscy_models.components.heads import PixelToVoxelHead, PixelToVoxelShuffleHead


def _init_weights(module: nn.Module) -> None:
    """Initialize weights of the given module."""
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def generate_mask(target: Size, stride: int, mask_ratio: float, device: str) -> BoolTensor:
    """Generate a random binary mask at low resolution.

    Parameters
    ----------
    target : Size
        Target shape.
    stride : int
        Total stride.
    mask_ratio : float
        Ratio of the pixels to mask.
    device : str
        Device to create the mask on.

    Returns
    -------
    BoolTensor
        Boolean mask (B1HW).
    """
    m_height = target[-2] // stride
    m_width = target[-1] // stride
    mask_numel = m_height * m_width
    masked_elements = int(mask_numel * mask_ratio)
    mask = torch.rand(target[0], mask_numel, device=device).argsort(1) < masked_elements
    return mask.reshape(target[0], 1, m_height, m_width)


def upsample_mask(mask: BoolTensor, target: Size) -> BoolTensor:
    """Upsample a low-resolution mask to match the target shape.

    Parameters
    ----------
    mask : BoolTensor
        Low-resolution boolean mask (B1HW).
    target : Size
        Target size (BCHW).

    Returns
    -------
    BoolTensor
        Upsampled boolean mask (B1HW).
    """
    if target[-2:] != mask.shape[-2:]:
        if not all(i % j == 0 for i, j in zip(target, mask.shape)):
            raise ValueError(f"feature map shape {target} must be divisible by mask shape {mask.shape}.")
        mask = mask.repeat_interleave(target[-2] // mask.shape[-2], dim=-2).repeat_interleave(
            target[-1] // mask.shape[-1], dim=-1
        )
    return mask


def masked_patchify(features: Tensor, unmasked: BoolTensor | None = None) -> Tensor:
    """Patchify features, keeping only unmasked positions.

    Parameters
    ----------
    features : Tensor
        Input image features (BCHW).
    unmasked : BoolTensor | None, optional
        Boolean foreground mask (B1HW).

    Returns
    -------
    Tensor
        Masked channel-last features (BLC, L = H * W * mask_ratio).
    """
    if unmasked is None:
        return features.flatten(2).permute(0, 2, 1)
    b, c = features.shape[:2]
    # (B, C, H, W) -> (B, H, W, C)
    features = features.permute(0, 2, 3, 1)
    # (B, H, W, C) -> (B * L, C) -> (B, L, C)
    features = features[unmasked[:, 0]].reshape(b, -1, c)
    return features


def masked_unpatchify(features: Tensor, out_shape: Size, unmasked: BoolTensor | None = None) -> Tensor:
    """Unpatchify channel-last features back to spatial layout.

    Parameters
    ----------
    features : Tensor
        Dense channel-last features (BLC).
    out_shape : Size
        Output shape (BCHW).
    unmasked : BoolTensor | None, optional
        Boolean foreground mask, defaults to None.

    Returns
    -------
    Tensor
        Masked features (BCHW).
    """
    if unmasked is None:
        return features.permute(0, 2, 1).reshape(out_shape)
    b, c, w, h = out_shape
    out = torch.zeros((b, w, h, c), device=features.device, dtype=features.dtype)
    # (B, L, C) -> (B * L, C)
    features = features.reshape(-1, c)
    out[unmasked[:, 0]] = features
    # (B, H, W, C) -> (B, C, H, W)
    return out.permute(0, 3, 1, 2)


class MaskedConvNeXtV2Block(nn.Module):
    """Masked ConvNeXt V2 Block.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int | None, optional
        Output channels, defaults to None.
    kernel_size : int, optional
        Depth-wise convolution kernel size, defaults to 7.
    stride : int, optional
        Downsample stride, defaults to 1.
    mlp_ratio : int, optional
        MLP expansion ratio, defaults to 4.
    drop_path : float, optional
        Drop path rate, defaults to 0.0.
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

    def forward(self, x: Tensor, unmasked: BoolTensor | None = None) -> Tensor:
        """Forward pass through the masked block.

        Parameters
        ----------
        x : Tensor
            Input tensor (BCHW).
        unmasked : BoolTensor | None, optional
            Boolean foreground mask, defaults to None.

        Returns
        -------
        Tensor
            Output tensor (BCHW).
        """
        shortcut = self.shortcut(x)
        if unmasked is not None:
            x *= unmasked
        x = self.dwconv(x)
        if unmasked is not None:
            x *= unmasked
        out_shape = x.shape
        x = masked_patchify(x, unmasked=unmasked)
        x = self.layernorm(x)
        x = self.mlp(x.unsqueeze(1)).squeeze(1)
        x = masked_unpatchify(x, out_shape=out_shape, unmasked=unmasked)
        x = self.drop_path(x) + shortcut
        return x


class MaskedConvNeXtV2Stage(nn.Module):
    """Masked ConvNeXt V2 Stage.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    kernel_size : int, optional
        Depth-wise convolution kernel size, defaults to 7.
    stride : int, optional
        Downsampling factor of this stage, defaults to 2.
    num_blocks : int, optional
        Number of residual blocks, defaults to 2.
    drop_path_rates : Sequence[float] | None, optional
        Drop path rates of each block, defaults to None.
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

    def forward(self, x: Tensor, unmasked: BoolTensor | None = None) -> Tensor:
        """Forward pass through the stage.

        Parameters
        ----------
        x : Tensor
            Input tensor (BCHW).
        unmasked : BoolTensor | None, optional
            Boolean foreground mask, defaults to None.

        Returns
        -------
        Tensor
            Output tensor (BCHW).
        """
        x = self.downsample(x)
        if unmasked is not None:
            unmasked = upsample_mask(unmasked, x.shape)
        for block in self.blocks:
            x = block(x, unmasked)
        return x


class MaskedAdaptiveProjection(nn.Module):
    """Masked patchifying layer for projecting 2D or 3D input into 2D feature maps.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    kernel_size_2d : tuple[int, int] | int, optional
        Kernel width and height, defaults to 4.
    kernel_depth : int, optional
        Kernel depth for 3D input, defaults to 5.
    in_stack_depth : int, optional
        Input stack depth for 3D input, defaults to 5.
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

    def forward(self, x: Tensor, unmasked: BoolTensor = None) -> Tensor:
        """Forward pass through the adaptive projection.

        Parameters
        ----------
        x : Tensor
            Input tensor (BCDHW).
        unmasked : BoolTensor, optional
            Boolean foreground mask (B1HW), defaults to None.

        Returns
        -------
        Tensor
            Output tensor (BCHW).
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
        if unmasked is not None:
            unmasked = upsample_mask(unmasked, x.shape)
        x = masked_patchify(x, unmasked=unmasked)
        x = self.norm(x)
        x = masked_unpatchify(x, out_shape=out_shape, unmasked=unmasked)
        return x


class MaskedMultiscaleEncoder(nn.Module):
    """Multiscale encoder with optional sparse masking for MAE pretraining."""

    def __init__(
        self,
        in_channels: int,
        stage_blocks: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        stem_kernel_size: Sequence[int] = (5, 4, 4),
        in_stack_depth: int = 5,
    ) -> None:
        super().__init__()
        self.stem = MaskedAdaptiveProjection(
            in_channels,
            dims[0],
            kernel_size_2d=stem_kernel_size[1:],
            kernel_depth=stem_kernel_size[0],
            in_stack_depth=in_stack_depth,
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
        self.total_stride = stem_kernel_size[1] * 2 ** (len(self.stages) - 1)
        self.apply(_init_weights)

    def forward(self, x: Tensor, mask_ratio: float = 0.0) -> tuple[list[Tensor], BoolTensor | None]:
        """Forward pass through the multiscale encoder.

        Parameters
        ----------
        x : Tensor
            Input tensor (BCDHW).
        mask_ratio : float, optional
            Ratio of the feature maps to mask, defaults to 0.0 (no masking).

        Returns
        -------
        tuple[list[Tensor], BoolTensor | None]
            Output tensors (list of BCHW) and boolean foreground mask
            (None if no masking).
        """
        if mask_ratio > 0.0:
            mask = generate_mask(x.shape, self.total_stride, mask_ratio, device=x.device)
            b, c, d, h, w = x.shape
            unmasked = ~mask
            mask = upsample_mask(mask, (b, 1, h, w))
        else:
            mask = unmasked = None
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x, unmasked=unmasked)
            features.append(x)
        return features, mask


class FullyConvolutionalMAE(nn.Module):
    """Fully Convolutional Masked Autoencoder (FCMAE) for self-supervised pretraining."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_blocks: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        encoder_drop_path_rate: float = 0.0,
        stem_kernel_size: Sequence[int] = (5, 4, 4),
        in_stack_depth: int = 5,
        decoder_conv_blocks: int = 1,
        pretraining: bool = True,
        head_conv: bool = False,
        head_conv_expansion_ratio: int = 4,
        head_conv_pool: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = MaskedMultiscaleEncoder(
            in_channels=in_channels,
            stage_blocks=encoder_blocks,
            dims=dims,
            drop_path_rate=encoder_drop_path_rate,
            stem_kernel_size=stem_kernel_size,
            in_stack_depth=in_stack_depth,
        )
        decoder_channels = list(dims)
        decoder_channels.reverse()
        if head_conv:
            decoder_channels[-1] = (in_stack_depth + 2) * in_channels * 2**2 * head_conv_expansion_ratio
        else:
            decoder_channels[-1] = out_channels * in_stack_depth * stem_kernel_size[-1] ** 2
        self.decoder = UNeXt2Decoder(
            decoder_channels,
            norm_name="instance",
            mode="pixelshuffle",
            conv_blocks=decoder_conv_blocks,
            strides=[2] * (len(dims) - 1) + [stem_kernel_size[-1]],
            upsample_pre_conv=None,
        )
        if head_conv:
            self.head = PixelToVoxelHead(
                in_channels=decoder_channels[-1],
                out_channels=out_channels,
                out_stack_depth=in_stack_depth,
                expansion_ratio=head_conv_expansion_ratio,
                pool=head_conv_pool,
            )
        else:
            self.head = PixelToVoxelShuffleHead(
                in_channels=decoder_channels[-1],
                out_channels=out_channels,
                out_stack_depth=in_stack_depth,
                xy_scaling=stem_kernel_size[-1],
                pool=True,
            )
        self.out_stack_depth = in_stack_depth
        # TODO: replace num_blocks with explicit strides for all models
        self.num_blocks = len(dims) * int(math.log2(stem_kernel_size[-1]))
        self.pretraining = pretraining

    def forward(self, x: Tensor, mask_ratio: float = 0.0) -> Tensor:
        """Forward pass through the FCMAE.

        Parameters
        ----------
        x : Tensor
            Input tensor (BCDHW).
        mask_ratio : float, optional
            Ratio of the feature maps to mask, defaults to 0.0.

        Returns
        -------
        Tensor
            Reconstructed output tensor.
        """
        x, mask = self.encoder(x, mask_ratio=mask_ratio)
        x.reverse()
        x = self.decoder(x)
        x = self.head(x)
        if self.pretraining:
            return x, mask
        return x
