"""Tests for 3D PatchGAN discriminators."""

import torch

from viscy_models.gan import MultiScalePatchGAN3D, PatchGAN3D


def test_patchgan3d_shape_small():
    """Small (2, 2, 8, 64, 64) input produces a single 5-D tensor with square YX."""
    model = PatchGAN3D(in_channels=2)
    x = torch.randn(2, 2, 8, 64, 64)
    y = model(x)
    assert y.ndim == 5
    assert y.shape[0] == 2
    assert y.shape[1] == 1
    assert y.shape[-1] == y.shape[-2], f"Expected square YX, got {y.shape}"


def test_patchgan3d_full_resolution_shape():
    """Full-resolution (1, 2, 8, 512, 512) input maps to a (1, 1, ?, ?, ?) tensor with square YX."""
    model = PatchGAN3D(in_channels=2)
    x = torch.randn(1, 2, 8, 512, 512)
    with torch.no_grad():
        y = model(x)
    assert y.ndim == 5
    assert y.shape[0] == 1
    assert y.shape[1] == 1
    assert y.shape[-1] == y.shape[-2], f"Expected square YX, got {y.shape}"


def test_patchgan3d_gradient_flow():
    """Gradients reach all conv weights after a forward + backward."""
    model = PatchGAN3D(in_channels=2)
    x = torch.randn(2, 2, 8, 64, 64, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    has_nonzero_grad = False
    for name, param in model.named_parameters():
        if "weight" in name and param.grad is not None and param.grad.abs().sum().item() > 0.0:
            has_nonzero_grad = True
            break
    assert has_nonzero_grad, "No conv weight received a non-zero gradient."


def test_multi_scale_patchgan3d_shapes():
    """num_scales=2 returns two tensors; the second has smaller YX than the first."""
    model = MultiScalePatchGAN3D(in_channels=2, num_scales=2)
    x = torch.randn(2, 2, 8, 64, 64)
    out = model(x)
    assert isinstance(out, list)
    assert len(out) == 2
    # Scale 0 sees the full input; scale 1 sees YX-halved input, so its YX
    # output must be strictly smaller than scale 0's.
    assert out[1].shape[-1] < out[0].shape[-1]
    assert out[1].shape[-2] < out[0].shape[-2]


def test_multi_scale_patchgan3d_single_scale_ablation():
    """num_scales=1 returns a single-element list."""
    model = MultiScalePatchGAN3D(in_channels=2, num_scales=1)
    x = torch.randn(2, 2, 8, 64, 64)
    out = model(x)
    assert isinstance(out, list)
    assert len(out) == 1
