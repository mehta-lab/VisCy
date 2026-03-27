"""Tests for UNetViT3D deterministic virtual staining model."""

import pytest
import torch

pytest.importorskip("einops")
pytest.importorskip("diffusers")

from viscy_models.celldiff import UNetViT3D  # noqa: E402


def test_forward(small_config):
    """Default single-channel: (2, 1, 8, 64, 64) -> same shape."""
    model = UNetViT3D(in_channels=1, out_channels=1, **small_config)
    x = torch.randn(2, 1, 8, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 8, 64, 64)


def test_forward_multi_channel(small_config):
    """Multi-channel: in_channels=2, out_channels=3."""
    model = UNetViT3D(in_channels=2, out_channels=3, **small_config)
    x = torch.randn(1, 2, 8, 64, 64)
    y = model(x)
    assert y.shape == (1, 3, 8, 64, 64)


def test_num_blocks(small_config):
    """.num_blocks returns the number of downsampling stages."""
    model = UNetViT3D(**small_config)
    assert model.num_blocks == len(small_config["num_res_block"])
    assert model.num_blocks == 2


def test_no_z_downsample(small_config):
    """Z dimension is preserved through encode/decode (stride 1 on Z)."""
    for z in (8, 16):
        config = {**small_config, "input_spatial_size": [z, 64, 64]}
        model = UNetViT3D(in_channels=1, out_channels=1, **config)
        x = torch.randn(1, 1, z, 64, 64)
        y = model(x)
        assert y.shape[2] == z, f"Z mismatch: input {z}, output {y.shape[2]}"
