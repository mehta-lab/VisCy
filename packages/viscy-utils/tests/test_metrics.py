"""Tests for the bf16-precision SSIM helper in viscy_utils.evaluation.metrics.

Covers the multi-tier numerical contract:

- per-pixel SSIM equivalence on random inputs (worst-case bf16 drift)
- aggregate SSIM equivalence on random inputs (per-pixel noise averages out)
- aggregate SSIM equivalence on correlated-pair inputs (closer to training)
- gradient-flow correctness via cosine similarity and sign-flip fraction
- output dtype invariance to input dtype
- finiteness at zero data_range (the eps denominator floor)
- flat-window contrast-sensitivity ≈ 1 (the Cauchy-Schwarz covariance bound)
- finite gradients on flat-background windows (sqrt-at-zero inside that same bound)
- MS-SSIM computes outside autocast, so fp16 backward intermediates cannot overflow
  to inf and become NaN in a later multiply

The two regression tests guard the deterministic, hardware-independent halves of
the bf16 NaN fix. The remaining half — the negative-variance denominator collapse
that produced intermittent NaN loss in 2D FCMAE training — is a bf16 rounding
knife-edge (the denominator must round to exactly 0) that only surfaces over
millions of windows across many steps and varies by GPU architecture, so it is not
unit-reproducible; it is validated end-to-end (the exact failing fit ran clean
after the clamp).
"""

import pytest
import torch
import torch.nn.functional as F

from viscy_utils.evaluation import metrics as metrics_module
from viscy_utils.evaluation.metrics import _compute_ssim_and_cs_bf16, _safe_sqrt

# monai is not a hard dep of viscy-utils — skip the suite if absent rather
# than failing at import time.
_monai_regression = pytest.importorskip("monai.metrics.regression")
_monai_reference = _monai_regression.compute_ssim_and_cs

# The helper unconditionally uses bf16 convs. CUDA bf16 conv works on
# sm_80+ in tensor cores and falls back to software emulation on older
# devices, but the equivalence-vs-monai-fp32 tolerances were measured on
# Hopper — skip on hardware where bf16 emulation could push drift past
# the configured rtol/atol.
_skip_no_bf16 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="CUDA + bf16 tensor-core support required",
)

# Representative iPSC SEC61B FCMAE batch shape.
_BATCH = (2, 1, 15, 256, 256)
_KERNEL = (15, 11, 11)


def _ref(y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _monai_reference(
        y_pred,
        y,
        spatial_dims=3,
        kernel_size=_KERNEL,
        kernel_sigma=None,
        kernel_type="uniform",
        data_range=y.max(),
    )


def _bf16(y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _compute_ssim_and_cs_bf16(
        y_pred,
        y,
        kernel_size=_KERNEL,
        data_range=y.max(),
    )


@_skip_no_bf16
def test_ssim_helper_random_per_pixel_equivalence():
    """Per-pixel SSIM on random inputs, worst-case bf16 drift tier.

    Tolerance ≥2× margin over measured 0.0418 absolute drift.
    """
    torch.manual_seed(0)
    y_pred = torch.rand(*_BATCH, device="cuda")
    y = torch.rand(*_BATCH, device="cuda")

    ssim_ref, cs_ref = _ref(y_pred, y)
    ssim_helper, cs_helper = _bf16(y_pred, y)

    torch.testing.assert_close(ssim_helper, ssim_ref, rtol=5e-2, atol=1e-1)
    torch.testing.assert_close(cs_helper, cs_ref, rtol=5e-2, atol=1e-1)


@_skip_no_bf16
def test_ssim_helper_random_aggregate_equivalence():
    """Aggregate SSIM (mean over pixels) on random inputs.

    Per-pixel noise averages out across H×W ≈ 65k pixels.
    Tolerance ≥25% margin over measured 0.00776 absolute drift.
    """
    torch.manual_seed(1)
    y_pred = torch.rand(*_BATCH, device="cuda")
    y = torch.rand(*_BATCH, device="cuda")

    ssim_ref, _ = _ref(y_pred, y)
    ssim_helper, _ = _bf16(y_pred, y)

    agg_ref = ssim_ref.view(ssim_ref.shape[0], -1).mean(1)
    agg_helper = ssim_helper.view(ssim_helper.shape[0], -1).mean(1)

    torch.testing.assert_close(agg_helper, agg_ref, rtol=1e-2, atol=1e-2)


@_skip_no_bf16
def test_ssim_helper_correlated_pair_equivalence():
    """Aggregate SSIM on a correlated pair (pred = target + small noise).

    Closer to training-time inputs; SSIM lives near 1.0 so relative drift
    is much smaller than on uncorrelated random data.
    """
    torch.manual_seed(2)
    y = torch.rand(*_BATCH, device="cuda")
    y_pred = y + 0.05 * torch.randn_like(y)

    ssim_ref, _ = _ref(y_pred, y)
    ssim_helper, _ = _bf16(y_pred, y)

    agg_ref = ssim_ref.view(ssim_ref.shape[0], -1).mean(1)
    agg_helper = ssim_helper.view(ssim_helper.shape[0], -1).mean(1)

    torch.testing.assert_close(agg_helper, agg_ref, rtol=2e-3, atol=5e-3)


@_skip_no_bf16
def test_ssim_helper_gradient_flow():
    """Gradient flow contract: finite, cosine-similar, low sign-flip rate.

    Per-voxel sign equality is too brittle (~0.25% benign flips on
    non-tiny gradients). Use cosine similarity + sign-flip fraction over
    voxels above 10% of the reference grad max — relative threshold so
    the assertion is scale-invariant regardless of loss magnitude.
    """
    torch.manual_seed(3)
    y = torch.rand(*_BATCH, device="cuda")

    y_pred_ref = (y + 0.05 * torch.randn_like(y)).detach().requires_grad_(True)
    ssim_ref, _ = _ref(y_pred_ref, y)
    (1 - ssim_ref).mean().backward()
    grad_ref = y_pred_ref.grad

    y_pred_helper = y_pred_ref.detach().clone().requires_grad_(True)
    ssim_helper, _ = _bf16(y_pred_helper, y)
    (1 - ssim_helper).mean().backward()
    grad_helper = y_pred_helper.grad

    assert grad_helper is not None
    assert grad_helper.shape == grad_ref.shape
    assert torch.isfinite(grad_helper).all()

    cos_sim = F.cosine_similarity(
        grad_helper.flatten().unsqueeze(0),
        grad_ref.flatten().unsqueeze(0),
        dim=1,
    ).item()
    assert cos_sim >= 0.99, f"cosine similarity {cos_sim:.4f} below 0.99"

    # Relative threshold — observed |grad_ref| max is ~1.7e-6, so the
    # earlier absolute 1e-3 threshold was vacuous. Anchor to 10% of the
    # reference max so the assertion is scale-invariant and meaningful.
    nontiny = grad_ref.abs() > 0.1 * grad_ref.abs().max()
    assert nontiny.any(), "no non-tiny reference gradients to compare signs against"
    flip_fraction = ((grad_helper.sign() != grad_ref.sign()) & nontiny).float().sum() / nontiny.float().sum()
    assert flip_fraction.item() < 0.01, f"sign-flip fraction {flip_fraction.item():.4f} above 1%"


@_skip_no_bf16
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_ssim_helper_dtypes(input_dtype):
    """Helper returns fp32 regardless of input dtype."""
    torch.manual_seed(4)
    y_pred = torch.rand(*_BATCH, device="cuda", dtype=input_dtype)
    y = torch.rand(*_BATCH, device="cuda", dtype=input_dtype)

    ssim_helper, cs_helper = _bf16(y_pred, y)

    assert ssim_helper.dtype == torch.float32
    assert cs_helper.dtype == torch.float32


@_skip_no_bf16
def test_ssim_helper_finite_on_zero_data_range():
    """Regression: a zero ``data_range`` (→ c1=c2=0) must not yield 0/0 NaN.

    ``ms_ssim_25d`` recomputes ``data_range = target.max()`` per scale; a flat or
    all-nonpositive target zeroes ``data_range`` and with it the c1/c2 stability
    constants, so the SSIM/CS ratios become 0/0. ``clamp=True`` in the loss cannot
    repair that — ``clamp`` bounds magnitude, not NaN — so the poisoned gradient
    silently corrupts every weight. The ``_SSIM_DENOM_EPS`` denominator floor keeps
    the helper finite. This case NaNs on the unfloored implementation on every device
    (deterministic, hardware-independent).
    """
    kernel_1 = (1, 11, 11)
    shape = (2, 1, 1, 64, 64)
    y = torch.zeros(*shape, device="cuda")
    y_pred = torch.zeros_like(y)

    ssim_full, cs = _compute_ssim_and_cs_bf16(y_pred, y, kernel_size=kernel_1, data_range=y.max())

    assert torch.isfinite(ssim_full).all(), "SSIM NaN/Inf at data_range=0 (denominator not floored)"
    assert torch.isfinite(cs).all(), "contrast-sensitivity NaN/Inf at data_range=0"


@_skip_no_bf16
def test_ssim_helper_identity_holds_at_small_data_range():
    """SSIM(y, y) == 1 for every ``data_range``, including 0.

    ``_SSIM_DENOM_EPS`` must floor c1/c2 SYMMETRICALLY -- numerator and denominator.
    Flooring only the denominator keeps the helper finite (the sibling test above)
    but breaks the identity: measured SSIM(y, y) = 0.082158 at data_range 1e-3 and
    exactly 0.0 at data_range 0. Since ``ms_ssim_25d`` recomputes
    ``data_range = target.max()`` per scale from a downsampled target, an all-zero
    or masked-out crop then scored loss 1.0 for a numerically perfect prediction.
    """
    shape = (2, 1, 1, 64, 64)
    for value in (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 1.0):
        y = torch.full(shape, value, device="cuda")
        ssim_full, cs = _compute_ssim_and_cs_bf16(y, y, kernel_size=(1, 11, 11), data_range=y.max())
        assert torch.isfinite(ssim_full).all()
        assert abs(ssim_full.mean().item() - 1.0) < 1e-6, f"SSIM(y, y) != 1 at data_range={value}"
        assert abs(cs.mean().item() - 1.0) < 1e-6, f"cs(y, y) != 1 at data_range={value}"

    zeros = torch.zeros(1, 1, 5, 256, 256, device="cuda")
    assert abs(metrics_module.ms_ssim_25d(zeros, zeros, clamp=True).item() - 1.0) < 1e-6


@_skip_no_bf16
def test_ssim_helper_flat_window_contrast_is_unity():
    """Regression: near-flat windows over large-magnitude inputs keep cs≈1.

    FCMAE targets are z-scored (``NormalizeSampled``); a channel with a tiny iqr
    (e.g. iPSC/A549 nucleus, iqr~1.8) normalizes to values in the hundreds. At
    depth=1 (2D FCMAE) each SSIM window has only ``1*H_k*W_k`` samples, so the bf16
    ``mu_xx - mu_x**2`` variance estimate catastrophically cancels and rounds
    *negative* (empirically ~-100 at magnitude 150). Exact fp32 recovers cs=1 on a
    flat window because the equally-negative sigma_x/sigma_y/sigma_xy cancel in the
    ratio; clamping only the diagonal variances to >=0 breaks that cancellation and
    inflates |cs| past 10 on background windows (which dominate a VS image). The
    Cauchy-Schwarz bound on sigma_xy restores cs=1. This guards that balance — the
    unbounded-covariance variant fails it with |cs-1| > 10, deterministically.
    """
    torch.manual_seed(5)
    kernel_1 = (1, 11, 11)
    shape = (1, 1, 1, 64, 64)
    worst = 0.0
    for magnitude in (150.0, 300.0):
        y = torch.full(shape, magnitude, device="cuda") + 0.03 * torch.randn(*shape, device="cuda")
        y_pred = torch.full(shape, magnitude, device="cuda") + 0.03 * torch.randn(*shape, device="cuda")
        _, cs = _compute_ssim_and_cs_bf16(
            y_pred, y, kernel_size=kernel_1, data_range=torch.tensor(magnitude, device="cuda")
        )
        assert torch.isfinite(cs).all()
        worst = max(worst, (cs - 1.0).abs().max().item())
    assert worst < 0.5, f"near-flat contrast-sensitivity deviates from 1 by {worst:.3f} (unbounded covariance?)"


@_skip_no_bf16
def test_ssim_helper_gradient_finite_on_flat_background():
    """Regression: the Cauchy-Schwarz bound must not NaN the gradient at zero variance.

    ``sigma_xy_bound = sqrt(sigma_x * sigma_y)`` is zero wherever either clamped
    variance rounds to zero, and ``sqrt`` has an infinite derivative at zero. The
    clamp-inactive elements contribute ``0 * inf == nan``, which the conv backward
    then spreads across each window — so the forward loss stays finite while most of
    the input gradient becomes NaN. On the unguarded ``sqrt`` this fires on ordinary
    normalized batches: 41-48% of ``pred.grad`` for a z-scored field with a flat
    background at a nonzero offset, at both depth=15 and depth=1.

    Uses an offset flat region rather than ``torch.rand``: the trigger is a window
    whose mean dominates its variance, so ``mu_xx`` and ``mu_x**2`` round to the same
    bf16 value and cancel exactly.
    """
    torch.manual_seed(6)
    shape = (2, 1, 15, 192, 192)
    half = shape[-2] // 2

    def _field(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        field = torch.randn(*shape, device="cuda") * 0.1 - 0.5
        # Bottom half carries structure; the top half stays near-flat background.
        field[..., half:, :] = torch.randn(shape[0], shape[1], shape[2], shape[-2] - half, shape[-1], device="cuda")
        return field

    y_pred = _field(6).detach().requires_grad_(True)
    y = _field(7)

    ssim_full, _ = _compute_ssim_and_cs_bf16(y_pred, y, kernel_size=_KERNEL, data_range=y.max())
    (1 - ssim_full.mean()).backward()

    assert y_pred.grad is not None
    n_nan = torch.isnan(y_pred.grad).sum().item()
    assert n_nan == 0, f"{n_nan}/{y_pred.grad.numel()} NaN gradients (sqrt of zero variance not guarded)"


@_skip_no_bf16
def test_ms_ssim_runs_outside_autocast():
    """``ms_ssim_25d`` must compute in fp32 even inside a 16-mixed autocast region.

    The helper chooses its own precision deliberately -- bf16 convolutions, fp32 for
    everything after. Letting an enclosing autocast re-enter it puts *backward*
    intermediates in fp16, where the ~1e8 gradients produced by the floored SSIM
    denominator overflow to inf and a later multiply turns them into NaN. That is the
    failure that stalled three joint FCMAE resumes: ``detect_anomaly`` blamed
    ``MulBackward0`` on the Cauchy-Schwarz bound, the forward loss stayed finite, and
    every one of the model's 32.1 M parameter gradients came back NaN.

    Pins the structural invariant, because reproducing the NaN itself needs one
    specific real batch (z-scored targets ranging to ~264) far too large to ship as a
    fixture: on that batch the rate was 0.02% of the input gradient under autocast and
    exactly 0% in fp32. Hardening ``sqrt`` does not cover this and neither does
    splitting the product -- both were measured and both still NaN'd.
    """
    seen = []
    real = metrics_module._compute_ssim_and_cs_bf16

    def spy(*args, **kwargs):
        seen.append(torch.is_autocast_enabled())
        return real(*args, **kwargs)

    y = torch.rand(*_BATCH, device="cuda")
    y_pred = (y + 0.05 * torch.randn_like(y)).detach().requires_grad_(True)

    monkey = pytest.MonkeyPatch()
    monkey.setattr("viscy_utils.evaluation.metrics._compute_ssim_and_cs_bf16", spy)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            ms_ssim = metrics_module.ms_ssim_25d(y_pred, y, clamp=True)
        (1 - ms_ssim).backward()
    finally:
        monkey.undo()

    assert seen, "ms_ssim_25d did not reach the bf16 helper"
    assert not any(seen), (
        f"autocast was enabled for {sum(seen)}/{len(seen)} MS-SSIM scales; fp16 backward intermediates are the NaN path"
    )
    assert torch.isfinite(y_pred.grad).all(), "non-finite gradient under 16-mixed autocast"


def test_safe_sqrt_matches_sqrt_forward_and_zeroes_gradient_at_zero():
    """``_safe_sqrt`` is bit-exact vs ``sqrt`` forward, and finite in backward at 0.

    Runs on CPU — the sqrt-at-zero derivative is a pure autograd property with no
    bf16 or CUDA involvement. Pins both halves of the contract the SSIM bound relies
    on: identical forward values, identical gradients where the input is positive,
    and a zero (not NaN) gradient at the exact zeros.
    """
    values = torch.tensor([0.0, 1e-12, 0.25, 1.0, 4.0])

    plain = values.detach().clone().requires_grad_(True)
    plain.sqrt().sum().backward()

    safe = values.detach().clone().requires_grad_(True)
    safe_out = _safe_sqrt(safe)
    safe_out.sum().backward()

    torch.testing.assert_close(safe_out, values.sqrt(), rtol=0, atol=0)
    assert torch.isfinite(safe.grad).all(), f"safe grad not finite: {safe.grad}"
    assert safe.grad[0] == 0.0, f"expected zero gradient at x=0, got {safe.grad[0]}"
    # Unguarded sqrt is the failure being guarded against.
    assert not torch.isfinite(plain.grad[0]), "expected unguarded sqrt to blow up at x=0"
    # Everywhere the input is positive the two must agree bit-for-bit.
    torch.testing.assert_close(safe.grad[1:], plain.grad[1:], rtol=0, atol=0)
