"""Tests for stem modules in viscy_models.components.stems."""

import pytest
import torch

from viscy_models.components.stems import StemDepthtoChannels, UNeXt2Stem


def test_unext2_stem_output_shape(device):
    """UNeXt2Stem projects 3D input to 2D feature map with correct shape.

    Input: (1, 1, 5, 256, 256) with kernel_size=(5,4,4), stride=(5,4,4).
    Conv3d output: D=(5-5)/5+1=1, H=(256-4)/4+1=64, W=64, C=96//1=96.
    Reshape: (1, 96*1, 64, 64) = (1, 96, 64, 64).
    """
    stem = UNeXt2Stem(
        in_channels=1,
        out_channels=96,
        kernel_size=(5, 4, 4),
        in_stack_depth=5,
    ).to(device)
    x = torch.randn(1, 1, 5, 256, 256, device=device)
    out = stem(x)
    assert out.shape == (1, 96, 64, 64)


def test_stem_depth_to_channels_output_shape(device):
    """StemDepthtoChannels maps depth to channels with correct shape.

    Input: (1, 1, 15, 256, 256) with kernel_size=(5,4,4), stride=(5,4,4).
    stem3d_out_depth = (15-5)//5+1 = 3.
    stem3d_out_channels = 96//3 = 32.
    Conv3d output: (1, 32, 3, 64, 64).
    Reshape: (1, 96, 64, 64).
    """
    stem = StemDepthtoChannels(
        in_channels=1,
        in_stack_depth=15,
        in_channels_encoder=96,
        stem_kernel_size=(5, 4, 4),
        stem_stride=(5, 4, 4),
    ).to(device)
    x = torch.randn(1, 1, 15, 256, 256, device=device)
    out = stem(x)
    assert out.shape == (1, 96, 64, 64)


def test_stem_depth_to_channels_mismatch():
    """StemDepthtoChannels raises ValueError when channels don't divide evenly.

    in_channels_encoder=97, in_stack_depth=15:
    stem3d_out_depth = (15-5)//5+1 = 3.
    stem3d_out_channels = 97//3 = 32.
    channel_mismatch = 97 - 3*32 = 1 != 0 -> ValueError.
    """
    with pytest.raises(ValueError, match="more channels"):
        StemDepthtoChannels(
            in_channels=1,
            in_stack_depth=15,
            in_channels_encoder=97,
            stem_kernel_size=(5, 4, 4),
            stem_stride=(5, 4, 4),
        )
