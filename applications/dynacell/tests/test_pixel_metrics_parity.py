"""Numerical parity tests between the pre-migration and post-migration pixel metrics.

The golden fixture ``tests/data/pixel_metrics_golden.npz`` is generated once
(before the migration) by running:

    uv run python applications/dynacell/tests/data/_generate_pixel_metrics_golden.py

After each replacement commit (Steps 3-5 of the metric unification plan), this
suite re-runs ``compute_pixel_metrics`` on the same crop and asserts the results
match the golden within tolerance.

Tolerances (per key):
- PCC, NRMSE, PSNR    : 1e-4  (cubic uses same math; only float-reduction differences)
- SSIM                : 5e-2  (cubic.metrics.ssim uses skimage with reflect padding;
                                the prior in-tree implementation used replicate padding —
                                inherent boundary difference, not a bug. 5e-2 catches gross
                                errors like swapped pred/target or dropped normalization while
                                accepting implementation drift.)
- Spectral_PCC, *_FSC : 1e-4  (single cubic version; see below)

The spectral three are pinned against a *specific* cubic release, recorded in
the fixture as ``_pin_cubic_version``. They are not stable across cubic
versions and are not meant to be: 0.7.0 -> 0.9.0a1 moved ``Spectral_PCC`` by
1.9% and multiplied ``XY_FSC_Resolution`` by ``spacing_z/spacing_x`` exactly
(2.685185 here, reproduced to 2.8e-7), a deliberate rescale. The base four were
bit-identical across that same bump, which is why only the spectral values were
re-pinned. A parity failure here means "the cubic pin moved", not "the metric
broke" — check ``_pin_cubic_version`` against the pyproject pin first.

``_pin_gpu`` is recorded for the same reason: GPU reduction order can shift a
scalar by ~1e-4 across devices, which is the whole tolerance. If this fails on a
different card than the one named in the fixture, suspect that before the code.

The golden pins the *scale-sensitive* family, which is what the bare
``PSNR``/``SSIM``/``NRMSE`` names mean: the fixture predates the scale-invariant
switch. The scale-invariant family is reported alongside it under ``SI_*`` and is
pinned separately by :func:`test_scale_invariant_family_is_reported_and_distinct`
— the two must not be interchangeable, because a table that pairs one row's
scaling with another's is the specific failure this dual reporting exists to
prevent.
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
    "SSIM": 5e-2,  # replicate vs reflect padding — inherent boundary difference, not a bug
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


def test_scale_invariant_family_is_reported_and_distinct() -> None:
    """``SI_*`` must exist alongside the bare names and carry different numbers.

    On this crop the two scalings disagree by 0.147 SSIM / 1.20 dB PSNR / 0.012
    NRMSE, all in the direction the switch documented (scale-invariant scores the
    prediction as a worse structural match but a closer intensity match). If a
    future cubic release made ``normalize="min_max"`` and ``scale_invariant=True``
    agree, the SI_ columns would silently become duplicates and every "both
    scalings" table would be reporting one number twice — so assert they differ.
    """
    g = _load_golden()
    result = compute_pixel_metrics(
        torch.as_tensor(g["pred"]),
        torch.as_tensor(g["target"]),
        spacing=list(g["_spacing"]),
        fsc_kwargs=None,
        spectral_pcc_kwargs=None,
        use_gpu=False,
    )
    for key in ("SI_SSIM", "SI_NRMSE", "SI_PSNR"):
        bare = key.removeprefix("SI_")
        assert key in result, f"{key} missing — the scale-invariant family is not being reported"
        assert np.isfinite(result[key]), f"{key} is not finite: {result[key]!r}"
        assert abs(result[key] - result[bare]) > 1e-3, (
            f"{key} == {bare} ({result[key]:.6g}) — the two scalings collapsed to one number"
        )
    # PCC is affine-invariant already, so it is the anchor: it has no SI_ pair.
    assert "SI_PCC" not in result


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
