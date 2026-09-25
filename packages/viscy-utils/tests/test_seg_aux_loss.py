"""Unit tests for SegAuxDice."""

import pytest
import torch

from viscy_utils.losses import SegAuxDice
from viscy_utils.losses.seg_aux import _masked_median, _quantile_sorted


def _blob_batch(shape: tuple[int, ...], seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Target = bright box on a noisy background; mask = the box."""
    g = torch.Generator().manual_seed(seed)
    target = 0.1 * torch.randn(shape, generator=g)
    mask = torch.zeros(shape)
    mask[..., shape[-2] // 4 : shape[-2] // 2, shape[-1] // 4 : shape[-1] // 2] = 1.0
    target = target + 2.0 * mask
    return target, mask


@pytest.mark.parametrize("shape", [(2, 1, 1, 32, 32), (2, 2, 4, 16, 16)], ids=["2d", "3d"])
def test_perfect_prediction_beats_perturbed(shape):
    target, mask = _blob_batch(shape)
    loss_fn = SegAuxDice()
    perfect = loss_fn(target.clone(), target, mask)
    g = torch.Generator().manual_seed(1)
    perturbed = loss_fn(target + 0.5 * torch.randn(shape, generator=g), target, mask)
    assert perfect.ndim == 0
    assert perfect < perturbed
    blurred_out = loss_fn(torch.zeros(shape), target, mask)
    assert perfect < blurred_out


def test_gradient_nonzero_for_foreground_far_below_tau():
    """A foreground voxel predicted several knee widths below tau still gets a
    gradient pushing it up; v1's clamped sigmoid gave exactly zero there."""
    target, mask = _blob_batch((1, 1, 1, 32, 32))
    flat = target.flatten()
    tau = torch.quantile(flat, 1 - mask.mean())
    fg = mask.flatten() > 0.5
    s = 0.1 * (flat[fg].median() - flat[~fg].median())
    pred = target.clone()
    fg_idx = (0, 0, 0, 10, 10)
    assert mask[fg_idx] == 1
    pred[fg_idx] = tau - 8 * s
    pred.requires_grad_(True)
    SegAuxDice(c=0.1)(pred, target, mask).backward()
    assert pred.grad[fg_idx] < 0  # increasing pred there lowers the loss


@pytest.mark.parametrize("a,b", [(3.0, -5.0), (0.01, 1.0), (1.0, 0.7)])
def test_affine_invariance(a, b):
    target, mask = _blob_batch((2, 1, 4, 16, 16))
    pred = target + 0.3 * torch.randn(target.shape, generator=torch.Generator().manual_seed(2))
    loss_fn = SegAuxDice(c=0.1)
    ref = loss_fn(pred, target, mask)
    moved = loss_fn(a * pred + b, a * target + b, mask)
    torch.testing.assert_close(moved, ref, rtol=1e-4, atol=1e-5)


def test_masked_median_matches_torch_on_selection():
    g = torch.Generator().manual_seed(7)
    values = torch.randn(4, 101, generator=g)
    keep = torch.rand(4, 101, generator=g) > 0.6
    got = _masked_median(values, keep)
    want = torch.stack([torch.quantile(v[k], 0.5) for v, k in zip(values, keep)])
    torch.testing.assert_close(got, want)


def test_mask_darker_than_background_is_excluded():
    target, mask = _blob_batch((1, 1, 1, 32, 32))
    loss, comps = SegAuxDice()(target.clone(), -target, mask, return_components=True)
    assert comps["n_valid"] == 0


def test_empty_and_full_masks_are_excluded():
    target, mask = _blob_batch((3, 1, 1, 32, 32))
    mask[1] = 0.0
    mask[2] = 1.0
    pred = target + 0.3 * torch.randn(target.shape, generator=torch.Generator().manual_seed(3))
    loss_fn = SegAuxDice()
    loss, comps = loss_fn(pred, target, mask, return_components=True)
    only_valid = loss_fn(pred[:1], target[:1], mask[:1])
    assert comps["n_valid"] == 1
    torch.testing.assert_close(loss, only_valid)
    torch.testing.assert_close(comps["dice"], loss)


def test_all_invalid_is_zero_with_graph():
    target = torch.randn(2, 1, 1, 16, 16)
    pred = torch.randn(2, 1, 1, 16, 16, requires_grad=True)
    loss, comps = SegAuxDice()(pred, target, torch.zeros_like(target), return_components=True)
    assert loss.item() == 0.0
    assert comps["n_valid"] == 0
    assert loss.requires_grad
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))


def test_constant_target_is_excluded():
    target = torch.ones(1, 1, 1, 16, 16)
    mask = torch.zeros_like(target)
    mask[..., :8, :] = 1.0
    loss, comps = SegAuxDice()(torch.randn_like(target), target, mask, return_components=True)
    assert comps["n_valid"] == 0
    assert loss.item() == 0.0


def test_fractional_mask_is_binarized():
    target, mask = _blob_batch((1, 1, 1, 32, 32))
    pred = target + 0.2
    loss_fn = SegAuxDice()
    torch.testing.assert_close(loss_fn(pred, target, mask * 0.8), loss_fn(pred, target, mask))


def test_quantile_matches_torch_quantile():
    rows = torch.randn(4, 1001)
    q = torch.tensor([0.0, 0.25, 0.731, 1.0])
    got = _quantile_sorted(rows.sort(-1).values, q)
    expected = torch.stack([torch.quantile(rows[i], q[i]) for i in range(4)])
    torch.testing.assert_close(got, expected)


def test_quantile_beyond_torch_quantile_size_limit():
    """torch.quantile raises above 2**24 elements; the sort-based path does not."""
    n = 2**24 + 16
    row = torch.arange(n, dtype=torch.float32).unsqueeze(0)
    got = _quantile_sorted(row, torch.tensor([0.5]))
    torch.testing.assert_close(got, torch.tensor([(n - 1) / 2]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_computes_in_float32(dtype):
    target, mask = _blob_batch((2, 1, 4, 16, 16))
    pred = (target + 0.3).requires_grad_(True)
    ref = SegAuxDice()(pred, target, mask)
    with torch.autocast(device_type="cpu", dtype=dtype):
        low = SegAuxDice()(pred.to(dtype), target.to(dtype), mask.to(dtype))
    assert low.dtype == torch.float32
    assert torch.isfinite(low)
    torch.testing.assert_close(low, ref, rtol=0.05, atol=0.02)
    low.backward()
    assert torch.isfinite(pred.grad).all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share a shape"):
        SegAuxDice()(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 5))
