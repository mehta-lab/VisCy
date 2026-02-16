"""Tests for Unet25d model."""

import pytest
import torch

from viscy_models.unet.unet25d import Unet25d


@torch.no_grad()
def test_unet25d_default_forward():
    """Test default Z-compression: in_stack_depth=5, out_stack_depth=1."""
    model = Unet25d(
        in_channels=1,
        out_channels=1,
        in_stack_depth=5,
        out_stack_depth=1,
        num_blocks=2,
    )
    model.eval()
    x = torch.randn(1, 1, 5, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)


@torch.no_grad()
def test_unet25d_preserved_depth():
    """Test preserved depth: in_stack_depth=5, out_stack_depth=5."""
    model = Unet25d(
        in_channels=1,
        out_channels=1,
        in_stack_depth=5,
        out_stack_depth=5,
        num_blocks=2,
    )
    model.eval()
    x = torch.randn(1, 1, 5, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 5, 64, 64)


@pytest.mark.parametrize("num_blocks", [1, 2])
@torch.no_grad()
def test_unet25d_variable_depth(num_blocks):
    """Test variable encoder/decoder depth."""
    model = Unet25d(
        in_channels=1,
        out_channels=1,
        in_stack_depth=5,
        out_stack_depth=1,
        num_blocks=num_blocks,
    )
    model.eval()
    x = torch.randn(1, 1, 5, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)


@torch.no_grad()
def test_unet25d_multichannel():
    """Test multi-channel input/output."""
    model = Unet25d(
        in_channels=2,
        out_channels=3,
        in_stack_depth=5,
        out_stack_depth=1,
        num_blocks=2,
    )
    model.eval()
    x = torch.randn(1, 2, 5, 64, 64)
    out = model(x)
    assert out.shape == (1, 3, 1, 64, 64)


@pytest.mark.parametrize("residual", [True, False])
@torch.no_grad()
def test_unet25d_residual(residual):
    """Test both residual and non-residual modes."""
    model = Unet25d(
        in_channels=1,
        out_channels=1,
        in_stack_depth=5,
        out_stack_depth=1,
        num_blocks=2,
        residual=residual,
    )
    model.eval()
    x = torch.randn(1, 1, 5, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)


@pytest.mark.parametrize("task", ["reg", "seg"])
@torch.no_grad()
def test_unet25d_task_mode(task):
    """Test both regression and segmentation task modes."""
    model = Unet25d(
        in_channels=1,
        out_channels=1,
        in_stack_depth=5,
        out_stack_depth=1,
        num_blocks=2,
        task=task,
    )
    model.eval()
    x = torch.randn(1, 1, 5, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)


def test_unet25d_state_dict_keys():
    """Test state dict key patterns match legacy checkpoint format."""
    model = Unet25d(
        in_channels=1,
        out_channels=1,
        in_stack_depth=5,
        out_stack_depth=1,
        num_blocks=2,
    )
    keys = set(model.state_dict().keys())

    # Verify expected key prefixes exist
    prefix_checks = [
        "down_conv_block_0",
        "down_conv_block_1",
        "bottom_transition_block",
        "up_conv_block_0",
        "up_conv_block_1",
        "terminal_block",
        "skip_conv_layer_0",
        "skip_conv_layer_1",
    ]
    for prefix in prefix_checks:
        matching = [k for k in keys if k.startswith(prefix)]
        assert len(matching) > 0, f"No keys found with prefix '{prefix}'"

    # AvgPool3d has no params, so down_samp should NOT be in state dict
    down_samp_keys = [k for k in keys if k.startswith("down_samp")]
    assert len(down_samp_keys) == 0, (
        f"down_samp should not have state dict keys (AvgPool has no params), "
        f"found: {down_samp_keys}"
    )


@torch.no_grad()
def test_unet25d_custom_num_filters():
    """Test forward pass with custom filter sizes."""
    model = Unet25d(
        in_channels=1,
        out_channels=1,
        in_stack_depth=5,
        out_stack_depth=1,
        num_blocks=2,
        num_filters=(32, 64, 128),
    )
    model.eval()
    x = torch.randn(1, 1, 5, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)
