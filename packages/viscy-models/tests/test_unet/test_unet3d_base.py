"""Tests for UNet3DBase parametrized 3D U-Net."""

import pytest
import torch

from viscy_models.unet.blocks import ConvBottleneck3D
from viscy_models.unet.unet3d_base import UNet3DBase


def _make_base(
    dims=(16, 32),
    num_res_block=(1,),
    downsample_z=False,
    residual=True,
    norm="group",
    activation="silu",
    time_embed_dim=None,
    cond_channels=None,
    in_channels=1,
    out_channels=1,
):
    """Create a small UNet3DBase with ConvBottleneck3D for testing."""
    bottleneck = ConvBottleneck3D(
        dims[-1],
        time_emb_dim=time_embed_dim,
        residual=residual,
        norm=norm,
        activation=activation,
    )
    return UNet3DBase(
        in_channels=in_channels,
        out_channels=out_channels,
        dims=list(dims),
        num_res_block=list(num_res_block),
        bottleneck=bottleneck,
        downsample_z=downsample_z,
        residual=residual,
        norm=norm,
        activation=activation,
        time_embed_dim=time_embed_dim,
        cond_channels=cond_channels,
    )


# ── Basic forward pass ──────────────────────────────────────────────────────


def test_forward_default():
    """Single level, GroupNorm + SiLU, no Z downsample."""
    model = _make_base()
    x = torch.randn(2, 1, 4, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 4, 16, 16)


def test_forward_multi_level():
    """Two levels, 2 blocks per level."""
    model = _make_base(dims=(16, 32, 64), num_res_block=(2, 2))
    x = torch.randn(1, 1, 4, 32, 32)
    y = model(x)
    assert y.shape == (1, 1, 4, 32, 32)


def test_forward_multi_channel():
    """Multiple input and output channels."""
    model = _make_base(in_channels=2, out_channels=3)
    x = torch.randn(1, 2, 4, 16, 16)
    y = model(x)
    assert y.shape == (1, 3, 4, 16, 16)


# ── Downsample Z ────────────────────────────────────────────────────────────


def test_forward_downsample_z():
    """With downsample_z=True, all three spatial dims are downsampled."""
    model = _make_base(downsample_z=True)
    # Z=4, Y=16, X=16 all divisible by 2
    x = torch.randn(1, 1, 4, 16, 16)
    y = model(x)
    assert y.shape == (1, 1, 4, 16, 16)


def test_forward_no_downsample_z():
    """With downsample_z=False, Z is preserved, only HW downsampled."""
    model = _make_base(downsample_z=False)
    x = torch.randn(1, 1, 5, 16, 16)  # Z=5 not divisible by 2, but Z not downsampled
    y = model(x)
    assert y.shape == (1, 1, 5, 16, 16)


# ── FNet configuration ──────────────────────────────────────────────────────


def test_fnet_config():
    """FNet preset: BatchNorm + ReLU, no residual, downsample Z."""
    model = _make_base(
        dims=(8, 16, 32),
        num_res_block=(1, 1),
        downsample_z=True,
        residual=False,
        norm="batch",
        activation="relu",
    )
    x = torch.randn(1, 1, 4, 16, 16)
    y = model(x)
    assert y.shape == (1, 1, 4, 16, 16)


# ── Time conditioning ──────────────────────────────────────────────────────


def test_forward_with_time():
    """Time-conditioned forward pass."""
    model = _make_base(time_embed_dim=32)
    x = torch.randn(2, 1, 4, 16, 16)
    t = torch.rand(2)
    y = model(x, t=t)
    assert y.shape == (2, 1, 4, 16, 16)


def test_forward_time_none():
    """Time-conditioned model with t=None — no error."""
    model = _make_base(time_embed_dim=32)
    x = torch.randn(2, 1, 4, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 4, 16, 16)


# ── Conditioning input ──────────────────────────────────────────────────────


def test_forward_with_cond():
    """Conditioning input added to main input projection."""
    model = _make_base(cond_channels=1)
    x = torch.randn(2, 1, 4, 16, 16)
    cond = torch.randn(2, 1, 4, 16, 16)
    y = model(x, cond=cond)
    assert y.shape == (2, 1, 4, 16, 16)


def test_forward_with_time_and_cond():
    """Full CELLDiffNet-like configuration."""
    model = _make_base(time_embed_dim=32, cond_channels=1)
    x = torch.randn(2, 1, 4, 16, 16)
    cond = torch.randn(2, 1, 4, 16, 16)
    t = torch.rand(2)
    y = model(x, cond=cond, t=t)
    assert y.shape == (2, 1, 4, 16, 16)


# ── Properties ──────────────────────────────────────────────────────────────


def test_num_blocks():
    """num_blocks equals number of encoder levels."""
    model = _make_base(dims=(16, 32, 64), num_res_block=(2, 2))
    assert model.num_blocks == 2


def test_downsamples_z_attribute():
    """downsamples_z is set as instance attribute."""
    model_z = _make_base(downsample_z=True)
    model_no_z = _make_base(downsample_z=False)
    assert model_z.downsamples_z is True
    assert model_no_z.downsamples_z is False


# ── Validation ──────────────────────────────────────────────────────────────


def test_dims_mismatch_raises():
    """len(dims) != len(num_res_block) + 1 raises ValueError."""
    with pytest.raises(ValueError, match="len\\(dims\\)"):
        _make_base(dims=(16, 32, 64), num_res_block=(1,))
