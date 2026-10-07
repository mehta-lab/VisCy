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


def _gaussian(a: np.ndarray, sigma_z: float, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(a, sigma=(0, 0, sigma_z, sigma, sigma), mode="nearest", truncate=4.0)


def _reference(op: BackgroundLowPass, target: torch.Tensor, mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """``(w, x')`` from scipy: square max filter, Gaussians with ``mode="nearest"``, in float64."""
    x = target.double().numpy()
    rz, r = op.dilate_radius_z, op.dilate_radius
    dilated = ndimage.maximum_filter(
        (mask.numpy() > 0.5).astype(np.float64), size=(1, 1, 2 * rz + 1, 2 * r + 1, 2 * r + 1), mode="constant"
    )
    w = _gaussian(dilated, op.sigma_feather_z, op.sigma_feather).clip(0.0, 1.0)
    flat = x.copy()  # planes without background keep x
    for i, j, d in np.ndindex(*x.shape[:3]):
        if (dilated[i, j, d] == 0).any():
            flat[i, j, d] = x[i, j, d][dilated[i, j, d] == 0].mean()
    if op.sigma_lp is None:
        bg = flat
    else:
        num = _gaussian(x * (1.0 - dilated), op.sigma_lp_z, op.sigma_lp)
        den = _gaussian(1.0 - dilated, op.sigma_lp_z, op.sigma_lp)
        bg = np.where(den > 1e-6, num / np.maximum(den, 1e-6), flat)
    return w, w * x + (1.0 - w) * bg


@pytest.mark.parametrize(
    ("shape", "sigma_lp", "z_args"),
    [
        ((2, 1, 1, 64, 64), 2.0, {}),
        ((2, 2, 5, 48, 40), 2.0, {}),
        ((1, 1, 6, 40, 40), 2.0, {"sigma_lp_z": 1.0, "sigma_feather_z": 0.5, "dilate_radius_z": 1}),
        ((2, 1, 1, 64, 64), None, {}),
        ((2, 2, 5, 48, 40), None, {"sigma_feather_z": 0.5, "dilate_radius_z": 1}),
    ],
    ids=["d1", "per_plane", "with_z", "flat_d1", "flat_with_z"],
)
def test_matches_scipy_reference(shape, sigma_lp, z_args):
    """The blend weight and the output equal the scipy.ndimage pipeline, D=1 and D>1, smooth and flat."""
    op = BackgroundLowPass(sigma_lp=sigma_lp, sigma_feather=1.5, dilate_radius=3, **z_args)
    target, mask = _square_batch(shape, 12, 28)
    mask[:, :, : shape[2] // 2] = 0.0  # planes differ, so Z handling matters
    w_ref, out_ref = _reference(op, target, mask)
    np.testing.assert_allclose(op.blend_weight(op.dilated_mask(mask)).numpy(), w_ref, atol=1e-6)
    np.testing.assert_allclose(op(target, mask).numpy(), out_ref, atol=1e-5)


def test_identity_where_w_is_one_and_background_where_w_is_zero():
    """Inside the dilated mask shrunk by the feather radius x' = x; beyond its reach x' = B."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4)
    target, mask = _square_batch((2, 1, 1, 64, 64), 16, 48)
    out = op(target, mask)
    dilated = op.dilated_mask(mask)
    # Dilated square [12, 52); the feather kernel's radius is int(4 * 2 + 0.5) = 8.
    inner = (..., slice(20, 44), slice(20, 44))
    torch.testing.assert_close(out[inner], target[inner], rtol=0.0, atol=1e-6)
    # Rows 0-3 are > 8 rows from the dilated square: w is exactly 0 there.
    assert torch.equal(op.blend_weight(dilated)[..., :4, :], torch.zeros(2, 1, 1, 4, 64))
    assert torch.equal(out[..., :4, :], op.background(target, dilated)[..., :4, :])
    # ... and the background estimate is not the identity there.
    assert (out[..., :4, :] - target[..., :4, :]).abs().mean() > 0.3


def test_background_estimate_does_not_leak_the_foreground():
    """On a constant background B is that constant everywhere, even next to a bright blob a plain blur halos."""
    op = BackgroundLowPass(sigma_lp=4.0, sigma_feather=1.0, dilate_radius=2)
    mask = torch.zeros((1, 1, 1, 64, 64))
    mask[..., 16:48, 16:48] = 1.0
    target = 0.7 + 10.0 * mask
    dilated = op.dilated_mask(mask)
    ring = (..., slice(50, 54), slice(16, 48))  # just outside the dilated square [14, 50)
    assert torch.equal(dilated[ring], torch.zeros(1, 1, 1, 4, 32))
    plain = _gaussian(target.double().numpy(), 0.0, op.sigma_lp)
    assert plain[ring].max() > 2.7  # the halo a plain G_lp(x) would teach
    background = op.background(target, dilated)
    torch.testing.assert_close(background, torch.full_like(target, 0.7), rtol=0.0, atol=1e-6)
    out = op(target, mask)
    torch.testing.assert_close(out[dilated == 0], torch.full_like(out[dilated == 0], 0.7), rtol=0.0, atol=1e-6)


def test_flat_background_is_the_masked_mean_per_plane():
    """sigma_lp=None: wherever w == 0, x' is one constant per (sample, channel, plane), its background mean."""
    op = BackgroundLowPass(sigma_lp=None, sigma_feather=2.0, dilate_radius=4)
    target, mask = _square_batch((2, 2, 3, 64, 64), 16, 48)
    mask[0, :, 1] = 0.0  # planes differ
    target = target + torch.arange(3.0).reshape(3, 1, 1) + torch.linspace(-1.0, 1.0, 64)  # per-plane offset + ramp
    out = op(target, mask)
    dilated = op.dilated_mask(mask)
    zero_w = op.blend_weight(dilated) == 0
    for i, j, d in np.ndindex(2, 2, 3):
        expected = target[i, j, d][dilated[i, j, d] == 0].double().mean().item()
        values = out[i, j, d][zero_w[i, j, d]]
        assert values.numel() > 300
        torch.testing.assert_close(values, torch.full_like(values, expected), rtol=0.0, atol=1e-5)
    # An empty mask flattens every plane to its own mean.
    flat = op(target, torch.zeros_like(target))
    torch.testing.assert_close(flat, target.mean(dim=(3, 4), keepdim=True).expand_as(target), rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("sigma_lp", [None, 8.0], ids=["bgflat", "bglp"])
def test_backlight_ramp_is_removed_by_flat_and_kept_by_smooth(sigma_lp):
    """A linear backlight ramp: bgflat flattens it, bglp (even at a large sigma) keeps it."""
    op = BackgroundLowPass(sigma_lp=sigma_lp, sigma_feather=2.0, dilate_radius=4)
    slope = 0.02
    mask = torch.zeros((1, 1, 1, 96, 96))
    mask[..., 40:56, 40:56] = 1.0
    target = slope * torch.arange(96.0) + 5.0 * mask
    out = op(target, mask)
    zero_w = op.blend_weight(op.dilated_mask(mask)) == 0
    values = out[zero_w]
    if sigma_lp is None:
        assert torch.unique(values).numel() == 1
    else:
        cols = torch.arange(96.0).expand_as(target)[zero_w]
        fitted = np.polyfit(cols.numpy(), values.numpy(), 1)[0]
        assert 0.8 * slope < fitted < 1.2 * slope


def test_background_keeps_the_local_mean():
    """Background = constant + checkerboard: B removes the checkerboard and keeps the constant."""
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


def test_empty_mask_blurs_everything():
    """An empty (or sub-threshold) mask makes B a plain blur of the whole target, and x' = B."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4)
    target, _ = _square_batch((2, 1, 1, 32, 32), 0, 0)
    blurred = _gaussian(target.double().numpy(), 0.0, op.sigma_lp)
    for mask in (torch.zeros_like(target), torch.full_like(target, 0.4)):
        np.testing.assert_allclose(op(target, mask).numpy(), blurred, atol=1e-5)


@pytest.mark.parametrize("sigma_feather", [0.5, 1.0, 2.0, 3.0])
def test_all_foreground_plane_has_w_exactly_one_and_is_returned_unchanged(sigma_feather):
    """Under an all-foreground plane w == 1 exactly (not 1 - rounding), so the plane comes back bit-identical."""
    target, mask = _square_batch((2, 1, 3, 32, 32), 8, 20)
    mask[:, :, 1] = 1.0
    for sigma_lp in (2.0, None):
        op = BackgroundLowPass(sigma_lp=sigma_lp, sigma_feather=sigma_feather, dilate_radius=4)
        assert torch.equal(op.blend_weight(op.dilated_mask(mask))[:, :, 1], torch.ones(2, 1, 32, 32))
        out = op(target, mask)
        assert torch.equal(out[:, :, 1], target[:, :, 1])
        assert not torch.equal(out[:, :, 0], target[:, :, 0])


@pytest.mark.parametrize("sigma_lp", [2.0, None], ids=["smooth", "flat"])
def test_patch_without_background_is_returned_unchanged(sigma_lp):
    """Where the dilated mask covers the patch, B = x and w == 1: x' = x exactly; other patches are unaffected."""
    op = BackgroundLowPass(sigma_lp=sigma_lp, sigma_feather=2.0, dilate_radius=4)
    target, mask = _square_batch((2, 1, 1, 32, 32), 8, 20)
    mask[0] = 1.0
    mask[0, ..., 5:7, 9:11] = 0.0  # a hole the dilation fills
    dilated = op.dilated_mask(mask)
    assert torch.equal(dilated[0], torch.ones_like(dilated[0]))
    assert torch.equal(op.blend_weight(dilated)[0], torch.ones_like(dilated[0]))
    assert torch.equal(op.background(target, dilated)[0], target[0])
    out = op(target, mask)
    assert torch.equal(out[0], target[0])
    assert torch.equal(out[1:], op(target[1:], mask[1:]))


def test_thin_background_ring_falls_back_to_the_plane_mean():
    """Pixels the normalized convolution cannot reach take the plane's flat mean; nothing is NaN or Inf."""
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=1)
    target, mask = _square_batch((1, 1, 1, 48, 48), 3, 45)  # dilated: [2, 46), a 2-px background ring
    dilated = op.dilated_mask(mask)
    background = op.background(target, dilated)
    den = _gaussian(1.0 - dilated.double().numpy(), 0.0, op.sigma_lp)
    unreached = torch.from_numpy(den <= 1e-6)
    assert unreached.sum() > 100  # the kernel's support (radius 8) misses the ring from the centre
    ring_mean = target[dilated == 0].mean()
    torch.testing.assert_close(background[unreached], torch.full_like(background[unreached], ring_mean.item()))
    assert torch.isfinite(background).all()
    assert torch.isfinite(op(target, mask)).all()


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
@pytest.mark.parametrize("sigma_lp", [2.0, None], ids=["smooth", "flat"])
def test_cuda_matches_cpu(sigma_lp):
    """The GPU path agrees with the CPU path (no convolution backend, so no TF32)."""
    op = BackgroundLowPass(sigma_lp=sigma_lp, sigma_feather=2.0, dilate_radius=4)
    target, mask = _square_batch((2, 1, 1, 64, 64), 16, 48)
    out = op(target.cuda(), mask.cuda())
    assert out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), op(target, mask), rtol=0.0, atol=1e-5)


def test_rejects_bad_arguments_and_shapes():
    with pytest.raises(ValueError, match="sigma_lp must be > 0 or None"):
        BackgroundLowPass(sigma_lp=0.0, sigma_feather=2.0, dilate_radius=4)
    with pytest.raises(ValueError, match="no effect with a flat background"):
        BackgroundLowPass(sigma_lp=None, sigma_feather=2.0, dilate_radius=4, sigma_lp_z=1.0)
    assert BackgroundLowPass(sigma_lp=float("inf"), sigma_feather=2.0, dilate_radius=4).sigma_lp is None
    with pytest.raises(ValueError, match="dilate_radius must be >= 0"):
        BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=-1)
    with pytest.raises(ValueError, match="sigma_feather_z must be >= 0"):
        BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4, sigma_feather_z=-1.0)
    op = BackgroundLowPass(sigma_lp=2.0, sigma_feather=2.0, dilate_radius=4)
    with pytest.raises(ValueError, match="must match target"):
        op(torch.zeros(1, 1, 1, 8, 8), torch.zeros(1, 1, 1, 8, 4))
    with pytest.raises(ValueError, match=r"\(B, C, D, H, W\)"):
        op(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
