"""Tests for 3D PatchGAN discriminators."""

import pytest
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


def test_patchgan3d_default_raises_at_unit_depth():
    """The cubic default cannot run at Z=1 — this is what preserve_z exists for."""
    model = PatchGAN3D(in_channels=2)
    with pytest.raises(RuntimeError, match="Kernel size can't be greater than actual input size"):
        model(torch.randn(1, 2, 1, 64, 64))


def test_patchgan3d_preserve_z_runs_at_unit_depth():
    """preserve_z=True accepts Z=1 and keeps the Z extent at 1."""
    model = PatchGAN3D(in_channels=2, preserve_z=True)
    y = model(torch.randn(2, 2, 1, 64, 64))
    assert y.ndim == 5
    assert y.shape[:2] == (2, 1)
    assert y.shape[2] == 1, f"Z must pass through, got {y.shape}"
    assert y.shape[-1] == y.shape[-2], f"Expected square YX, got {y.shape}"


def test_patchgan3d_preserve_z_keeps_depth_for_a_slab() -> None:
    """Z is preserved, not merely tolerated: a 5-plane slab stays 5 planes.

    Layers 3-4 stride Z by 2 in the 3D topology, which would halve a slab
    twice. ``preserve_z`` drops that stride, so the discriminator never mixes
    planes the Z-preserving generator kept separate.
    """
    model = PatchGAN3D(in_channels=2, preserve_z=True)
    y = model(torch.randn(1, 2, 5, 64, 64))
    assert y.shape[2] == 5, f"expected Z=5 through the discriminator, got {y.shape}"


def test_patchgan3d_preserve_z_leaves_the_3d_topology_untouched() -> None:
    """preserve_z=False must reproduce the published 3D discriminator exactly.

    Guards the shared code path: the 2D arm was added by parameterizing the
    same convs the 14 pix2pix3d leaves train, so an accidental change to the
    default would silently alter every published GAN number.
    """
    torch.manual_seed(0)
    default = PatchGAN3D(in_channels=2)
    torch.manual_seed(0)
    explicit = PatchGAN3D(in_channels=2, preserve_z=False)
    for (name, a), (_, b) in zip(default.named_parameters(), explicit.named_parameters(), strict=True):
        assert torch.equal(a, b), f"{name} differs"
    for layer in (default.layer1, default.layer2):
        assert layer[0].kernel_size == (4, 4, 4)
        assert layer[0].padding == (1, 1, 1)
        assert layer[0].stride == (1, 2, 2)
    for layer in (default.layer3, default.layer4):
        assert layer[0].kernel_size == (4, 4, 4)
        assert layer[0].stride == (2, 2, 2)
    x = torch.randn(1, 2, 8, 64, 64)
    with torch.no_grad():
        assert torch.equal(default(x), explicit(x))


def test_multi_scale_patchgan3d_preserve_z_at_unit_depth() -> None:
    """The multi-scale wrapper forwards preserve_z and pools YX only."""
    model = MultiScalePatchGAN3D(in_channels=2, num_scales=2, preserve_z=True)
    out = model(torch.randn(1, 2, 1, 128, 128))
    assert len(out) == 2
    assert all(o.shape[2] == 1 for o in out), [tuple(o.shape) for o in out]
    assert out[1].shape[-1] < out[0].shape[-1]


def test_patchgan3d_preserve_z_gradient_flow() -> None:
    """Every conv weight receives a gradient at Z=1."""
    model = PatchGAN3D(in_channels=2, preserve_z=True)
    model(torch.randn(2, 2, 1, 64, 64)).sum().backward()
    missing = [
        name
        for name, param in model.named_parameters()
        if "weight" in name and (param.grad is None or param.grad.abs().sum().item() == 0.0)
    ]
    assert not missing, f"no gradient reached {missing}"
