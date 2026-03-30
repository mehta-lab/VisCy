"""Tests for CELLDiffNet flow-matching backbone."""

import pytest
import torch

pytest.importorskip("einops")
pytest.importorskip("diffusers")

from viscy_models.celldiff import CELLDiffNet  # noqa: E402


def test_forward(small_config):
    """Forward with (x, cond, t) -> (B, in_channels, D, H, W)."""
    model = CELLDiffNet(in_channels=1, **small_config)
    x = torch.randn(2, 1, 8, 64, 64)
    cond = torch.randn(2, 1, 8, 64, 64)
    t = torch.rand(2)
    y = model(x, cond, t)
    assert y.shape == (2, 1, 8, 64, 64)


def test_forward_multi_channel(small_config):
    """Multi-channel input: in_channels=2 produces matching output."""
    model = CELLDiffNet(in_channels=2, **small_config)
    x = torch.randn(1, 2, 8, 64, 64)
    cond = torch.randn(1, 1, 8, 64, 64)
    t = torch.rand(1)
    y = model(x, cond, t)
    assert y.shape == (1, 2, 8, 64, 64)


def test_num_blocks(small_config):
    """.num_blocks returns the number of downsampling stages."""
    model = CELLDiffNet(**small_config)
    assert model.num_blocks == 2


def test_wrong_spatial_raises(small_config):
    """Forward rejects input with wrong spatial dimensions."""
    model = CELLDiffNet(in_channels=1, **small_config)
    x = torch.randn(1, 1, 8, 32, 32)
    cond = torch.randn(1, 1, 8, 32, 32)
    t = torch.rand(1)
    with pytest.raises(ValueError, match="does not match expected"):
        model(x, cond, t)


def test_wrong_cond_channels_raises(small_config):
    """Forward rejects conditioning with wrong channel count."""
    model = CELLDiffNet(in_channels=1, **small_config)
    x = torch.randn(1, 1, 8, 64, 64)
    cond = torch.randn(1, 3, 8, 64, 64)  # 3 channels, expected 1
    t = torch.rand(1)
    with pytest.raises(ValueError, match="cond must have 1 channel"):
        model(x, cond, t)


def test_indivisible_patch_size_raises(small_config):
    """Constructor rejects spatial sizes not divisible by patch_size after downsampling."""
    with pytest.raises(ValueError, match="not divisible by patch_size"):
        CELLDiffNet(**{**small_config, "input_spatial_size": [10, 64, 64]})
