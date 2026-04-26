"""Tests for the bf16-precision SSIM helper in viscy_utils.evaluation.metrics.

Covers the multi-tier numerical contract from the plan:

- per-pixel SSIM equivalence on random inputs (worst-case bf16 drift)
- aggregate SSIM equivalence on random inputs (per-pixel noise averages out)
- aggregate SSIM equivalence on correlated-pair inputs (closer to training)
- gradient-flow correctness via cosine similarity and sign-flip fraction
- output dtype invariance to input dtype
"""

import pytest
import torch
import torch.nn.functional as F
from monai.metrics.regression import compute_ssim_and_cs as _monai_reference

from viscy_utils.evaluation.metrics import _compute_ssim_and_cs_bf16

# Representative iPSC SEC61B FCMAE batch shape.
_BATCH = (2, 1, 15, 256, 256)
_KERNEL = (15, 11, 11)
_SPATIAL_DIMS = 3


def _ref(y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _monai_reference(
        y_pred,
        y,
        spatial_dims=_SPATIAL_DIMS,
        kernel_size=_KERNEL,
        kernel_sigma=None,
        kernel_type="uniform",
        data_range=y.max(),
    )


def _bf16(y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _compute_ssim_and_cs_bf16(
        y_pred,
        y,
        spatial_dims=_SPATIAL_DIMS,
        kernel_size=_KERNEL,
        data_range=y.max(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
def test_ssim_helper_gradient_flow():
    """Gradient flow contract: finite, cosine-similar, low sign-flip rate.

    Per-voxel sign equality is too brittle (~0.25% benign flips on
    non-tiny gradients). Use cosine similarity + sign-flip fraction over
    voxels with ``|grad_ref| > 1e-3``.
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_ssim_helper_dtypes(input_dtype):
    """Helper returns fp32 regardless of input dtype."""
    torch.manual_seed(4)
    y_pred = torch.rand(*_BATCH, device="cuda", dtype=input_dtype)
    y = torch.rand(*_BATCH, device="cuda", dtype=input_dtype)

    ssim_helper, cs_helper = _bf16(y_pred, y)

    assert ssim_helper.dtype == torch.float32
    assert cs_helper.dtype == torch.float32
