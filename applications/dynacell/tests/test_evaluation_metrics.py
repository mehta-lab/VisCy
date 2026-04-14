"""Regression tests for evaluation pixel metrics."""

import importlib
import sys
import types

import pytest
import torch


def _import_metrics_with_stubs(monkeypatch):
    """Import the metrics module with lightweight optional-dependency stubs."""
    microssim_module = types.ModuleType("microssim")
    microssim_module.MicroMS3IM = object

    cubic_module = types.ModuleType("cubic")
    cubic_cuda_module = types.ModuleType("cubic.cuda")
    cubic_cuda_module.ascupy = lambda x: x
    cubic_cuda_module.asnumpy = lambda x: x

    cubic_metrics_module = types.ModuleType("cubic.metrics")
    cubic_metrics_module.fsc_resolution = lambda *args, **kwargs: {}

    cubic_bandlimited_module = types.ModuleType("cubic.metrics.bandlimited")
    cubic_bandlimited_module.spectral_pcc = lambda *args, **kwargs: 0.0

    cubic_feature_module = types.ModuleType("cubic.feature")
    cubic_feature_voxel_module = types.ModuleType("cubic.feature.voxel")
    cubic_feature_voxel_module.regionprops_table = lambda *args, **kwargs: {}

    monkeypatch.setitem(sys.modules, "microssim", microssim_module)
    monkeypatch.setitem(sys.modules, "cubic", cubic_module)
    monkeypatch.setitem(sys.modules, "cubic.cuda", cubic_cuda_module)
    monkeypatch.setitem(sys.modules, "cubic.metrics", cubic_metrics_module)
    monkeypatch.setitem(sys.modules, "cubic.metrics.bandlimited", cubic_bandlimited_module)
    monkeypatch.setitem(sys.modules, "cubic.feature", cubic_feature_module)
    monkeypatch.setitem(sys.modules, "cubic.feature.voxel", cubic_feature_voxel_module)
    sys.modules.pop("dynacell.evaluation.metrics", None)

    return importlib.import_module("dynacell.evaluation.metrics")


def test_gain_and_offset_errors_are_not_scale_invariant(monkeypatch) -> None:
    """Shared-scale metrics should penalize intensity calibration errors."""
    metrics = _import_metrics_with_stubs(monkeypatch)

    target = torch.linspace(0.0, 1.0, steps=16 * 16).reshape(16, 16)
    prediction = target * 2.0 + 0.25
    target_range = target.max() - target.min()
    expected_rmse = torch.sqrt(torch.mean(((prediction - target) / target_range) ** 2))
    expected_psnr = -10 * torch.log10(expected_rmse**2)

    assert metrics.nrmse(target, prediction).item() == pytest.approx(expected_rmse.item())
    assert metrics.psnr(target, prediction).item() == pytest.approx(expected_psnr.item())
    assert metrics.ssim(target, prediction).item() < 0.99


def test_identical_images_still_score_perfectly(monkeypatch) -> None:
    """Shared-scale normalization should preserve perfect self-similarity."""
    metrics = _import_metrics_with_stubs(monkeypatch)

    target = torch.linspace(0.0, 1.0, steps=16 * 16).reshape(16, 16)

    assert metrics.nrmse(target, target).item() == pytest.approx(0.0)
    assert metrics.psnr(target, target).item() == float("inf")
    assert metrics.ssim(target, target).item() == pytest.approx(1.0)
