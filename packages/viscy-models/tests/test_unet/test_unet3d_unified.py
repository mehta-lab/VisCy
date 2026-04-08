"""Unified parametrized tests for all 3D U-Net variants.

Exercises Unet3d (FNet), UNetViT3D, and CELLDiffNet through shared assertions
to verify the unified UNet3DBase contract.
"""

import pytest
import torch

from viscy_models.unet.unet3d import Unet3d

diffusers = pytest.importorskip("diffusers")

from viscy_models.celldiff import CELLDiffNet, UNetViT3D  # noqa: E402

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def fnet_model():
    """Small FNet model for testing."""
    return Unet3d(in_channels=1, out_channels=1, depth=1, mult_chan=8, in_stack_depth=4)


@pytest.fixture
def vit_model():
    """Small UNetViT3D model for testing."""
    return UNetViT3D(
        input_spatial_size=[4, 16, 16],
        in_channels=1,
        out_channels=1,
        dims=[8, 16],
        num_res_block=[1],
        hidden_size=32,
        num_heads=2,
        dim_head=16,
        num_hidden_layers=1,
        patch_size=4,
    )


@pytest.fixture
def celldiff_model():
    """Small CELLDiffNet model for testing."""
    return CELLDiffNet(
        input_spatial_size=[4, 16, 16],
        in_channels=1,
        dims=[8, 16],
        num_res_block=[1],
        hidden_size=32,
        num_heads=2,
        dim_head=16,
        num_hidden_layers=1,
        patch_size=4,
    )


# ── Shared property tests ──────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ["fnet_model", "vit_model", "celldiff_model"])
def test_num_blocks(model_name, request):
    """All variants expose num_blocks as an int."""
    model = request.getfixturevalue(model_name)
    assert isinstance(model.num_blocks, int)
    assert model.num_blocks >= 1


@pytest.mark.parametrize("model_name", ["fnet_model", "vit_model", "celldiff_model"])
def test_downsamples_z(model_name, request):
    """All variants expose downsamples_z as a bool."""
    model = request.getfixturevalue(model_name)
    assert isinstance(model.downsamples_z, bool)


def test_fnet_downsamples_z(fnet_model):
    """FNet downsamples Z."""
    assert fnet_model.downsamples_z is True


def test_vit_no_downsample_z(vit_model):
    """UNetViT3D does not downsample Z."""
    assert vit_model.downsamples_z is False


def test_celldiff_no_downsample_z(celldiff_model):
    """CELLDiffNet does not downsample Z."""
    assert celldiff_model.downsamples_z is False


# ── Shared forward tests ───────────────────────────────────────────────────


def test_fnet_forward(fnet_model):
    """FNet forward pass preserves spatial shape."""
    x = torch.randn(1, 1, 4, 8, 8)
    y = fnet_model(x)
    assert y.shape == x.shape


def test_vit_forward(vit_model):
    """UNetViT3D forward pass preserves spatial shape."""
    x = torch.randn(1, 1, 4, 16, 16)
    y = vit_model(x)
    assert y.shape == x.shape


def test_celldiff_forward(celldiff_model):
    """CELLDiffNet forward pass preserves spatial shape."""
    x = torch.randn(1, 1, 4, 16, 16)
    cond = torch.randn(1, 1, 4, 16, 16)
    t = torch.rand(1)
    y = celldiff_model(x, cond, t)
    assert y.shape == x.shape


# ── UNet3DBase lineage ─────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ["fnet_model", "vit_model", "celldiff_model"])
def test_inherits_from_base(model_name, request):
    """All variants inherit from UNet3DBase."""
    from viscy_models.unet.unet3d_base import UNet3DBase

    model = request.getfixturevalue(model_name)
    assert isinstance(model, UNet3DBase)
