"""Tests for MixedLoss after the @torch.amp.custom_fwd(cast_inputs=fp32) removal.

Verifies the entry-point behaviour didn't change in any user-visible way:
forward returns finite scalars in fp32 inside and outside autocast, gradients
flow under autocast bf16, and the L1-only branch is bit-exact F.l1_loss.
"""

import pytest
import torch
import torch.nn.functional as F

from viscy_utils.losses import MixedLoss

# Spatial size needs to be at least 2^4 * 11 = 176 for the 5-level MS-DSSIM
# pyramid with 11x11 kernels — use 192 to match the cytoland CPU integration
# test convention.
_BATCH = (2, 1, 15, 192, 192)

_skip_no_bf16 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="CUDA + bf16 tensor-core support required",
)


def _seeded_inputs(device: str = "cuda", seed: int = 0):
    torch.manual_seed(seed)
    pred = torch.rand(*_BATCH, device=device)
    target = torch.rand(*_BATCH, device=device)
    return pred, target


@_skip_no_bf16
def test_mixed_loss_forward_finite_outside_autocast():
    """Forward returns a finite fp32 scalar outside any autocast context."""
    loss_fn = MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5)
    pred, target = _seeded_inputs()

    loss = loss_fn(pred, target)

    assert loss.dtype == torch.float32
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


@_skip_no_bf16
def test_mixed_loss_forward_under_bf16_autocast():
    """Forward under bf16 autocast returns a finite fp32 scalar.

    Drift between the no-decorator path (in autocast) and a manually
    fp32-cast path is bounded by rtol=1e-2, atol=1e-2 — within the SSIM
    helper's per-aggregate contract.
    """
    loss_fn = MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5)
    pred, target = _seeded_inputs(seed=1)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_autocast = loss_fn(pred, target)

    # Manual fp32 baseline — what the @custom_fwd(cast_inputs=fp32) decorator
    # produced. We don't actually wrap; we just call outside autocast on the
    # same fp32 inputs.
    loss_fp32 = loss_fn(pred.float(), target.float())

    assert loss_autocast.dtype == torch.float32
    assert torch.isfinite(loss_autocast).item()
    torch.testing.assert_close(loss_autocast, loss_fp32, rtol=1e-2, atol=1e-2)


@_skip_no_bf16
def test_mixed_loss_gradient_flow_under_autocast():
    """Backward through MixedLoss under autocast bf16 produces finite grads."""
    loss_fn = MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5)
    pred, target = _seeded_inputs(seed=2)
    pred = pred.detach().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = loss_fn(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all().item()


def test_mixed_loss_l1_only_matches_torch_l1():
    """ms_dssim_alpha=0 collapses MixedLoss to alpha * F.l1_loss bit-exact.

    Runs on CPU (no autocast, no SSIM). Confirms L1 branch behaviour is
    identical to torch.nn.functional.l1_loss after decorator removal.
    """
    loss_fn = MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.0)
    torch.manual_seed(3)
    pred = torch.rand(2, 1, 8, 64, 64)
    target = torch.rand(2, 1, 8, 64, 64)

    loss = loss_fn(pred, target)
    expected = F.l1_loss(pred, target) * 0.5

    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
