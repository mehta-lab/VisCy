"""Unit tests for BackgroundLowPass, checked against scipy.ndimage references."""

import numpy as np
import pytest
import torch
from scipy import ndimage

from viscy_utils.losses import BackgroundLowPass


def _square_batch(shape: tuple[int, ...], lo: int, hi: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy target with a bright square; mask = the square ``[lo, hi)`` in YX of every plane."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(shape)
    mask[..., lo:hi, lo:hi] = 1.0
    target = torch.randn(shape, generator=g) + 2.0 * mask
    return target, mask


def _reference(op: BackgroundLowPass, target: torch.Tensor, mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """``(w, x')`` from scipy: square max filter, then Gaussians with ``mode="nearest"``, in float64."""
    x = target.double().numpy()
    rz, r = op.dilate_radius_z, op.dilate_radius
    dilated = ndimage.maximum_filter(
        (mask.numpy() > 0.5).astype(np.float64), size=(1, 1, 2 * rz + 1, 2 * r + 1, 2 * r + 1), mode="constant"
    )
    w = ndimage.gaussian_filter(
        dilated, sigma=(0, 0, op.sigma_feather_z, op.sigma_feather, op.sigma_feather), mode="nearest", truncate=4.0
    ).clip(0.0, 1.0)
    low = ndimage.gaussian_filter(
        x, sigma=(0, 0, op.sigma_lp_z, op.sigma_lp, op.sigma_lp), mode="nearest", truncate=4.0
    )
    return w, w * x + (1.0 - w) * low


@pytest.mark.parametrize(
    ("shape", "z_args"),
    [
        ((2, 1, 1, 64, 64), {}),
        ((2, 2, 5, 48, 40), {}),
        ((1, 1, 6, 40, 40), {"sigma_lp_z": 1.0, "sigma_feather_z": 0.5, "dilate_radius_z": 1}),
    ],
    ids=["d1", "per_plane", "with_z"],
)
def test_matches_scipy_reference(shape, z_args):
    """The blend weight and the output equal the scipy.ndimage pipeline, D=1 and D>1."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=1.5, dilate_radius=3, **z_args)
    target, mask = _square_batch(shape, 12, 28)
    mask[:, :, : shape[2] // 2] = 0.0  # planes differ, so Z handling matters
    w_ref, out_ref = _reference(op, target, mask)
    np.testing.assert_allclose(op.blend_weight(mask).numpy(), w_ref, atol=1e-6)
    np.testing.assert_allclose(op(target, mask).numpy(), out_ref, atol=1e-5)


def test_identity_where_w_is_one_and_lowpass_where_w_is_zero():
    """Inside the dilated mask shrunk by the feather radius x' = x; beyond its reach x' = G_lp(x)."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4)
    target, mask = _square_batch((2, 1, 1, 64, 64), 16, 48)
    out = op(target, mask)
    # Dilated square [12, 52); the feather kernel's radius is int(4 * 2 + 0.5) = 8.
    inner = (..., slice(20, 44), slice(20, 44))
    torch.testing.assert_close(out[inner], target[inner], rtol=0.0, atol=1e-6)
    # Rows 0-3 are > 8 rows from the dilated square: w is exactly 0 there.
    assert torch.equal(op.blend_weight(mask)[..., :4, :], torch.zeros(2, 1, 1, 4, 64))
    assert torch.equal(out[..., :4, :], op.lowpass(target)[..., :4, :])
    # ... and the low-pass is not the identity there.
    assert (out[..., :4, :] - target[..., :4, :]).abs().mean() > 0.3


def test_lowpass_keeps_the_local_mean():
    """Background = constant + checkerboard: the low-pass removes the checkerboard and keeps the constant."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4)
    yy, xx = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
    target = (3.0 + 0.5 * (-1.0) ** (yy + xx)).float().expand(1, 1, 1, 48, 48)
    out = op(target, torch.zeros_like(target))
    # Away from the border, where edge replication breaks the checkerboard's alternation.
    torch.testing.assert_close(out[..., 8:-8, 8:-8], torch.full((1, 1, 1, 32, 32), 3.0), rtol=0.0, atol=1e-4)
    # A constant passes through unchanged whatever the mask.
    const = torch.full((2, 1, 3, 32, 32), -0.7)
    _, mask = _square_batch(const.shape, 8, 20)
    torch.testing.assert_close(op(const, mask), const, rtol=0.0, atol=1e-6)


def test_empty_and_full_masks():
    """An empty (or sub-threshold) mask low-passes everything; a full mask changes nothing."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4)
    target, _ = _square_batch((2, 1, 1, 32, 32), 0, 0)
    assert torch.equal(op(target, torch.zeros_like(target)), op.lowpass(target))
    assert torch.equal(op(target, torch.full_like(target, 0.4)), op.lowpass(target))
    torch.testing.assert_close(op(target, torch.ones_like(target)), target, rtol=0.0, atol=1e-6)


def test_planes_are_independent_by_default():
    """With every ``*_z`` at 0, each Z plane is the op applied to that plane alone."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=1.0, dilate_radius=2)
    target, mask = _square_batch((2, 1, 4, 32, 32), 8, 20)
    mask[:, :, 1] = 0.0  # planes differ, so a leak across Z would show
    out = op(target, mask)
    for d in range(4):
        plane = op(target[:, :, d : d + 1], mask[:, :, d : d + 1])
        torch.testing.assert_close(out[:, :, d : d + 1], plane, rtol=0.0, atol=1e-6)


def test_dtype_and_autocast():
    """Low-precision input comes back in its dtype, computed in float32; autocast changes nothing."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4)
    target, mask = _square_batch((2, 1, 1, 32, 32), 8, 20)
    bf16 = target.bfloat16()
    out = op(bf16, mask)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, op(bf16.float(), mask).bfloat16())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_out = op(target, mask)
    assert autocast_out.dtype == torch.float32
    assert torch.equal(autocast_out, op(target, mask))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_matches_cpu():
    """The GPU path agrees with the CPU path (no convolution backend, so no TF32)."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4)
    target, mask = _square_batch((2, 1, 1, 64, 64), 16, 48)
    out = op(target.cuda(), mask.cuda())
    assert out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), op(target, mask), rtol=0.0, atol=1e-5)


def test_rejects_bad_arguments_and_shapes():
    with pytest.raises(ValueError, match="sigma_lp must be > 0"):
        BackgroundLowPass(sigma_lp=0.0, sigma_feather=2.0, dilate_radius=4)
    with pytest.raises(ValueError, match="dilate_radius must be >= 0"):
        BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=-1)
    with pytest.raises(ValueError, match="sigma_feather_z must be >= 0"):
        BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4, sigma_feather_z=-1.0)
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4)
    with pytest.raises(ValueError, match="must match target"):
        op(torch.zeros(1, 1, 1, 8, 8), torch.zeros(1, 1, 1, 8, 4))
    with pytest.raises(ValueError, match=r"\(B, C, D, H, W\)"):
        op(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
