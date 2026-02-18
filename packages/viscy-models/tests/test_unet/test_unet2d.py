"""Tests for Unet2d model."""

import pytest
import torch

from viscy_models.unet.unet2d import Unet2d


@torch.no_grad()
def test_unet2d_default_forward():
    """Test default config forward pass with 5D input."""
    model = Unet2d(in_channels=1, out_channels=1, num_blocks=4)
    model.eval()
    x = torch.randn(1, 1, 1, 256, 256)
    out = model(x)
    assert out.shape == (1, 1, 1, 256, 256)


@pytest.mark.parametrize("num_blocks", [1, 2, 4])
@torch.no_grad()
def test_unet2d_variable_depth(num_blocks):
    """Test variable encoder/decoder depth."""
    # num_blocks=4 needs at least 2^4=16 spatial, but use 64 for safety
    spatial = 64 if num_blocks <= 2 else 256
    model = Unet2d(in_channels=1, out_channels=1, num_blocks=num_blocks)
    model.eval()
    x = torch.randn(1, 1, 1, spatial, spatial)
    out = model(x)
    assert out.shape == (1, 1, 1, spatial, spatial)


@torch.no_grad()
def test_unet2d_multichannel():
    """Test multi-channel input/output."""
    model = Unet2d(in_channels=2, out_channels=3, num_blocks=2)
    model.eval()
    x = torch.randn(1, 2, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 3, 1, 64, 64)


@pytest.mark.parametrize("residual", [True, False])
@torch.no_grad()
def test_unet2d_residual(residual):
    """Test both residual and non-residual modes produce same output shape."""
    model = Unet2d(in_channels=1, out_channels=1, num_blocks=2, residual=residual)
    model.eval()
    x = torch.randn(1, 1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)


@pytest.mark.parametrize("task", ["reg", "seg"])
@torch.no_grad()
def test_unet2d_task_mode(task):
    """Test both regression and segmentation task modes."""
    model = Unet2d(in_channels=1, out_channels=1, num_blocks=2, task=task)
    model.eval()
    x = torch.randn(1, 1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)


@torch.no_grad()
def test_unet2d_dropout():
    """Test forward pass with dropout enabled."""
    model = Unet2d(in_channels=1, out_channels=1, num_blocks=2, dropout=0.25)
    model.train()  # dropout active during training
    x = torch.randn(1, 1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)


def test_unet2d_state_dict_keys():
    """Test state dict key patterns match legacy checkpoint format."""
    model = Unet2d(in_channels=1, out_channels=1, num_blocks=2)
    keys = set(model.state_dict().keys())

    # Verify expected key prefixes exist
    prefix_checks = [
        "down_conv_block_0",
        "down_conv_block_1",
        "bottom_transition_block",
        "up_conv_block_0",
        "up_conv_block_1",
        "terminal_block",
    ]
    for prefix in prefix_checks:
        matching = [k for k in keys if k.startswith(prefix)]
        assert len(matching) > 0, f"No keys found with prefix '{prefix}'"

    # AvgPool2d has no params, so down_samp should NOT be in state dict
    down_samp_keys = [k for k in keys if k.startswith("down_samp")]
    assert len(down_samp_keys) == 0, (
        f"down_samp should not have state dict keys (AvgPool has no params), found: {down_samp_keys}"
    )

    # skip_conv_layer should NOT exist in Unet2d
    skip_keys = [k for k in keys if k.startswith("skip_conv_layer")]
    assert len(skip_keys) == 0, f"Unet2d should not have skip_conv_layer keys, found: {skip_keys}"


@torch.no_grad()
def test_unet2d_custom_num_filters():
    """Test forward pass with custom filter sizes."""
    model = Unet2d(
        in_channels=1,
        out_channels=1,
        num_blocks=2,
        num_filters=(32, 64, 128),
    )
    model.eval()
    x = torch.randn(1, 1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)
