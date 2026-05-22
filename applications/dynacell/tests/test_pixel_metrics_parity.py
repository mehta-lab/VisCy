"""Numerical parity tests between the pre-migration and post-migration pixel metrics.

The golden fixture ``tests/data/pixel_metrics_golden.npz`` is generated once
(before the migration) by running:

    uv run python applications/dynacell/tests/data/_generate_pixel_metrics_golden.py

After each replacement commit (Steps 3-5 of the metric unification plan), this
suite re-runs ``compute_pixel_metrics`` on the same crop and asserts the results
match the golden within tolerance.

Tolerances (per key):
- PCC, NRMSE, PSNR    : 1e-4  (cubic uses same math; only float-reduction differences)
- SSIM                : 1e-3  (gaussian kernel differences between torch and skimage)
- Spectral_PCC, *_FSC : 1e-4  (unchanged code path)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dynacell.evaluation.metrics import compute_pixel_metrics

_GOLDEN = Path(__file__).parent / "data" / "pixel_metrics_golden.npz"

_BASE_TOLERANCES: dict[str, float] = {
    "PCC": 1e-4,
    "SSIM": 1e-3,
    "NRMSE": 1e-4,
    "PSNR": 1e-4,
}
_SPECTRAL_TOLERANCES: dict[str, float] = {
    "Spectral_PCC": 1e-4,
    "XY_FSC_Resolution": 1e-4,
    "Z_FSC_Resolution": 1e-4,
}


def _load_golden() -> dict:
    if not _GOLDEN.exists():
        pytest.skip(f"Golden fixture not generated yet — run {_GOLDEN.parent}/_generate_pixel_metrics_golden.py")
    return dict(np.load(_GOLDEN, allow_pickle=False))


def test_pixel_metrics_base_parity() -> None:
    """Base metrics (PCC/SSIM/NRMSE/PSNR) match the pre-migration golden (CPU-compatible)."""
    g = _load_golden()
    pred = torch.as_tensor(g["pred"])
    target = torch.as_tensor(g["target"])
    spacing = list(g["_spacing"])

    result = compute_pixel_metrics(
        pred, target, spacing=spacing, fsc_kwargs=None, spectral_pcc_kwargs=None, use_gpu=False
    )
    for key, tol in _BASE_TOLERANCES.items():
        expected = float(g[f"expected_{key.lower()}"])
        got = result[key]
        assert abs(got - expected) <= tol, (
            f"{key}: expected {expected:.6g}, got {got:.6g} (|Δ|={abs(got - expected):.2e} > tol={tol:.0e})"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="spectral metrics require CUDA")
def test_pixel_metrics_spectral_parity() -> None:
    """Spectral metrics match the pre-migration golden (GPU required)."""
    g = _load_golden()
    pred = torch.as_tensor(g["pred"])
    target = torch.as_tensor(g["target"])
    spacing = list(g["_spacing"])

    result = compute_pixel_metrics(pred, target, spacing=spacing, fsc_kwargs={}, spectral_pcc_kwargs={}, use_gpu=True)
    for key, tol in _SPECTRAL_TOLERANCES.items():
        expected = float(g[f"expected_{key.lower()}"])
        got = result[key]
        assert abs(got - expected) <= tol, (
            f"{key}: expected {expected:.6g}, got {got:.6g} (|Δ|={abs(got - expected):.2e} > tol={tol:.0e})"
        )
