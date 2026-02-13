"""Tests for block modules in viscy_models._components.blocks."""

import torch
from torch import nn

from viscy_models._components.blocks import (
    UNeXt2Decoder,
    UNeXt2UpStage,
    _get_convnext_stage,
    icnr_init,
)


def test_icnr_init():
    """icnr_init initializes conv weights with ICNR pattern."""
    conv = nn.Conv2d(16, 64, 3, padding=1)
    original_weight = conv.weight.data.clone()
    icnr_init(conv, upsample_factor=2, upsample_dims=2)
    # Weight should be modified (not all zeros, different from original)
    assert conv.weight.shape == (64, 16, 3, 3)
    assert not torch.allclose(conv.weight.data, original_weight)
    assert not torch.all(conv.weight.data == 0)


def test_get_convnext_stage(device):
    """_get_convnext_stage creates a ConvNeXt stage with correct forward pass."""
    stage = _get_convnext_stage(in_channels=96, out_channels=96, depth=2).to(device)
    assert isinstance(stage, nn.Module)
    x = torch.randn(1, 96, 32, 32, device=device)
    out = stage(x)
    assert out.shape == (1, 96, 32, 32)


def test_unext2_up_stage_pixelshuffle(device):
    """UNeXt2UpStage upsamples and merges with skip connection.

    Input: (1, 192, 16, 16), skip: (1, 96, 32, 32).
    Pixelshuffle 2x: 192 -> 192//(2^2) = 48 channels at 32x32.
    Cat with skip: 48 + 96 = 144 channels.
    ConvNeXt stage: 144 -> 96 channels at 32x32.
    Output: (1, 96, 32, 32).
    """
    stage = UNeXt2UpStage(
        in_channels=192,
        skip_channels=96,
        out_channels=96,
        scale_factor=2,
        mode="pixelshuffle",
        conv_blocks=2,
        norm_name="instance",
        upsample_pre_conv=None,
    ).to(device)
    inp = torch.randn(1, 192, 16, 16, device=device)
    skip = torch.randn(1, 96, 32, 32, device=device)
    out = stage(inp, skip)
    assert out.shape == (1, 96, 32, 32)


def test_unext2_decoder(device):
    """UNeXt2Decoder chains multiple UpStages to decode features.

    Input features: [384@8x8, 192@16x16, 96@32x32] (high-to-low resolution).
    Stage 0: 384->192 at 16x16, skip=192@16x16.
    Stage 1: 192->96 at 32x32, skip=96@32x32.
    Output: (1, 96, 32, 32).
    """
    decoder = UNeXt2Decoder(
        num_channels=[384, 192, 96],
        norm_name="instance",
        mode="pixelshuffle",
        conv_blocks=2,
        strides=[2, 2],
        upsample_pre_conv=None,
    ).to(device)
    features = [
        torch.randn(1, 384, 8, 8, device=device),
        torch.randn(1, 192, 16, 16, device=device),
        torch.randn(1, 96, 32, 32, device=device),
    ]
    out = decoder(features)
    assert out.shape == (1, 96, 32, 32)
