"""Unit tests for Unet3d (F-Net 3D U-Net)."""

import pytest
import torch

from viscy_models.unet.unet3d import Unet3d


def test_forward_default_params():
    """Default depth=4, mult_chan=32: (2, 1, 32, 64, 64) -> same shape."""
    model = Unet3d(in_channels=1, out_channels=1, depth=4, mult_chan=32)
    x = torch.randn(2, 1, 32, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 32, 64, 64)


def test_forward_custom_params():
    """depth=2, mult_chan=16: (2, 1, 4, 16, 16) -> same shape."""
    model = Unet3d(in_channels=1, out_channels=1, depth=2, mult_chan=16)
    x = torch.randn(2, 1, 4, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 4, 16, 16)


def test_forward_multi_channel():
    """Multi-channel: in_channels=2, out_channels=3."""
    model = Unet3d(in_channels=2, out_channels=3, depth=1, mult_chan=8)
    x = torch.randn(1, 2, 4, 8, 8)
    y = model(x)
    assert y.shape == (1, 3, 4, 8, 8)


def test_z_preserved():
    """Output Z dimension matches input Z."""
    model = Unet3d(in_channels=1, out_channels=1, depth=3, mult_chan=16)
    for z in (8, 16, 32):
        x = torch.randn(1, 1, z, 16, 16)
        y = model(x)
        assert y.shape[2] == z, f"Z mismatch: input {z}, output {y.shape[2]}"


def test_weight_init():
    """Conv3d weights should have std close to 0.02 after F-Net init."""
    model = Unet3d(in_channels=1, out_channels=1, depth=2, mult_chan=16)
    for name, param in model.named_parameters():
        if "conv" in name and "weight" in name and param.dim() == 5:
            assert abs(param.std().item() - 0.02) < 0.01, f"{name} std={param.std().item():.4f}, expected ~0.02"


def test_indivisible_spatial_dims_raises():
    """Non-divisible spatial dims raise ValueError, not opaque concat error."""
    model = Unet3d(in_channels=1, out_channels=1, depth=2, mult_chan=16)
    x = torch.randn(1, 1, 5, 16, 16)  # Z=5 not divisible by 2^2=4
    with pytest.raises(ValueError, match="Spatial dim D=5 must be divisible by"):
        model(x)

    x = torch.randn(1, 1, 4, 15, 16)  # Y=15 not divisible by 4
    with pytest.raises(ValueError, match="Spatial dim H=15 must be divisible by"):
        model(x)


def test_engine_attributes():
    """Model exposes attributes required by cytoland engine."""
    model = Unet3d(depth=4, mult_chan=32, in_stack_depth=32)
    assert model.num_blocks == 4
    assert model.out_stack_depth == 32
    assert model.downsamples_z is True


def test_state_dict_keys():
    """State dict uses iterative encoder-decoder key structure."""
    model = Unet3d(in_channels=1, out_channels=1, depth=2, mult_chan=16)
    keys = set(model.state_dict().keys())

    # Check for iterative structure key prefixes
    prefix_checks = [
        "inconv",
        "_encoder_blocks",
        "_downsamples",
        "bottleneck",
        "_upsamples",
        "_decoder_blocks",
        "outconv",
    ]
    for prefix in prefix_checks:
        matching = [k for k in keys if k.startswith(prefix)]
        assert len(matching) > 0, f"No keys found with prefix '{prefix}'"
