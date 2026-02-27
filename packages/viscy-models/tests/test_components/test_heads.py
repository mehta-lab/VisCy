"""Tests for head modules in viscy_models.components.heads."""

import torch

from viscy_models.components.heads import (
    PixelToVoxelHead,
    PixelToVoxelShuffleHead,
    UnsqueezeHead,
)


def test_pixel_to_voxel_head_output_shape(device):
    """PixelToVoxelHead produces 5D output with correct channels and depth.

    Uses params matching UNeXt2's actual usage:
    in_channels = (out_stack_depth + 2) * out_channels * 2^2 * expansion_ratio
               = (5 + 2) * 2 * 4 * 4 = 224.
    After pixelshuffle 2x: 224//4=56 channels at 128x128.
    Reshape to 3D: 56//7=8 channels at depth=7.
    After 3D conv (padding=(0,1,1)): depth=5, then pixelshuffle 2x -> 256x256.
    Output: (1, 2, 5, 256, 256).
    """
    head = PixelToVoxelHead(
        in_channels=224,
        out_channels=2,
        out_stack_depth=5,
        expansion_ratio=4,
        pool=False,
    ).to(device)
    x = torch.randn(1, 224, 64, 64, device=device)
    out = head(x)
    assert out.ndim == 5
    assert out.shape[0] == 1
    assert out.shape[1] == 2
    assert out.shape[2] == 5
    assert out.shape == (1, 2, 5, 256, 256)


def test_unsqueeze_head(device):
    """UnsqueezeHead adds a depth=1 dimension at position 2."""
    head = UnsqueezeHead().to(device)
    x = torch.randn(2, 16, 32, 32, device=device)
    out = head(x)
    assert out.shape == (2, 16, 1, 32, 32)


def test_pixel_to_voxel_shuffle_head_output_shape(device):
    """PixelToVoxelShuffleHead upsamples 2D to 3D with pixel shuffle.

    Uses params matching FCMAE's actual usage:
    in_channels = out_channels * out_stack_depth * xy_scaling^2 = 2 * 5 * 4^2 = 160.
    UpSample pixelshuffle 4x: out_channels=5*2=10.
    Need in_channels/scale^2 = 160/16 = 10 = target out. Correct.
    Reshape: (1, 2, 5, 64, 64).
    """
    head = PixelToVoxelShuffleHead(
        in_channels=160,
        out_channels=2,
        out_stack_depth=5,
        xy_scaling=4,
        pool=False,
    ).to(device)
    x = torch.randn(1, 160, 16, 16, device=device)
    out = head(x)
    assert out.shape == (1, 2, 5, 64, 64)
