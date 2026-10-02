"""Tests for CELLDiffNet flow-matching backbone."""

import pytest
import torch

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
    with pytest.raises(ValueError, match="not divisible by its patch extent"):
        CELLDiffNet(**{**small_config, "input_spatial_size": [10, 64, 64]})


def test_spatial_multiple(small_config):
    """Inputs must be multiples of the Z patch and of 2**levels times the YX patch."""
    model = CELLDiffNet(in_channels=1, **small_config)
    assert model.spatial_multiple == (4, 16, 16)


def test_with_input_size_at_training_size_matches(small_config):
    """A view built for the training size reproduces the network exactly."""
    torch.manual_seed(0)
    model = CELLDiffNet(in_channels=1, **small_config).eval()
    view = model.with_input_size(small_config["input_spatial_size"])
    x, cond, t = torch.randn(1, 1, 8, 64, 64), torch.randn(1, 1, 8, 64, 64), torch.rand(1)
    with torch.no_grad():
        torch.testing.assert_close(view(x, cond, t), model(x, cond, t), rtol=0, atol=0)


def test_with_input_size_runs_larger_input_sharing_weights(small_config):
    """A larger view shares every learned weight and keeps the trained embedding on shared grid positions."""
    model = CELLDiffNet(in_channels=1, **small_config).eval()
    view = model.with_input_size([16, 96, 128])
    y = view(torch.randn(1, 1, 16, 96, 128), torch.randn(1, 1, 16, 96, 128), torch.rand(1))
    assert y.shape == (1, 1, 16, 96, 128)
    assert view.inconv.weight is model.inconv.weight
    assert view.bottleneck.blocks is model.bottleneck.blocks
    # Original network is untouched: own grid, own embedding, still rejects the new size.
    assert model.bottleneck.latent_grid_size == [2, 4, 4]
    with pytest.raises(ValueError, match="does not match expected"):
        model(torch.randn(1, 1, 16, 96, 128), torch.randn(1, 1, 16, 96, 128), torch.rand(1))
    hidden = small_config["hidden_size"]
    big = view.bottleneck.img_pos_embed.reshape(4, 6, 8, hidden)
    small = model.bottleneck.img_pos_embed.reshape(2, 4, 4, hidden)
    torch.testing.assert_close(big[:2, :4, :4], small)


def test_with_input_size_rejects_indivisible(small_config):
    """Sizes the encoder or the patch embedding cannot tile are rejected."""
    model = CELLDiffNet(in_channels=1, **small_config)
    with pytest.raises(ValueError, match="not divisible"):
        model.with_input_size([8, 72, 64])
