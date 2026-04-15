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


# --- corr_coef tests ---


def test_corr_coef_perfect_correlation(monkeypatch) -> None:
    """Identical signals give PCC = 1.0."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    a = torch.linspace(0.0, 1.0, 100)
    assert metrics.corr_coef(a, a).item() == pytest.approx(1.0)


def test_corr_coef_negative_correlation(monkeypatch) -> None:
    """Perfectly inverted signal gives PCC = -1.0."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    a = torch.linspace(0.0, 1.0, 100)
    assert metrics.corr_coef(a, -a).item() == pytest.approx(-1.0)


def test_corr_coef_constant_input_returns_nan(monkeypatch) -> None:
    """Zero-variance input (constant signal) returns NaN."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    a = torch.ones(100)
    b = torch.linspace(0.0, 1.0, 100)
    assert torch.isnan(metrics.corr_coef(a, b))


def test_corr_coef_shape_mismatch_raises(monkeypatch) -> None:
    """Mismatched shapes raise ValueError."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    with pytest.raises(ValueError, match="same shape"):
        metrics.corr_coef(torch.ones(10), torch.ones(20))


# --- evaluate_segmentations tests ---


def test_evaluate_segmentations_perfect_overlap() -> None:
    """Perfect overlap gives all metrics = 1.0."""
    import numpy as np

    from dynacell.evaluation.metrics import evaluate_segmentations

    mask = np.ones((8, 8), dtype=bool)
    result = evaluate_segmentations(mask, mask)
    assert result["Dice"] == pytest.approx(1.0)
    assert result["IoU"] == pytest.approx(1.0)
    assert result["Precision"] == pytest.approx(1.0)
    assert result["Recall"] == pytest.approx(1.0)
    assert result["Accuracy"] == pytest.approx(1.0)


def test_evaluate_segmentations_no_overlap() -> None:
    """No overlap gives Dice = IoU = 0."""
    import numpy as np

    from dynacell.evaluation.metrics import evaluate_segmentations

    pred = np.zeros((8, 8), dtype=bool)
    gt = np.ones((8, 8), dtype=bool)
    result = evaluate_segmentations(pred, gt)
    assert result["Dice"] == pytest.approx(0.0)
    assert result["IoU"] == pytest.approx(0.0)
    assert result["Precision"] == pytest.approx(0.0)
    assert result["Recall"] == pytest.approx(0.0)


def test_evaluate_segmentations_partial_overlap() -> None:
    """Known partial overlap gives expected values."""
    import numpy as np

    from dynacell.evaluation.metrics import evaluate_segmentations

    pred = np.zeros((4, 4), dtype=bool)
    gt = np.zeros((4, 4), dtype=bool)
    # TP: 4 pixels, FP: 2 pixels, FN: 2 pixels, TN: 8 pixels
    pred[:2, :3] = True  # 6 pixels
    gt[:2, 1:3] = True  # 4 pixels
    gt[2, :2] = True  # 2 more pixels = 6 total gt
    result = evaluate_segmentations(pred, gt)
    assert result["TP"] == 4.0
    assert result["FP"] == 2.0
    assert result["FN"] == 2.0
    assert result["TN"] == 8.0
    assert result["Dice"] == pytest.approx(2 * 4 / (2 * 4 + 2 + 2))
    assert result["Precision"] == pytest.approx(4 / 6)
    assert result["Recall"] == pytest.approx(4 / 6)


def test_evaluate_segmentations_shape_mismatch_raises() -> None:
    """Mismatched shapes raise ValueError."""
    import numpy as np

    from dynacell.evaluation.metrics import evaluate_segmentations

    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate_segmentations(np.ones((4, 4)), np.ones((4, 5)))


def test_evaluate_segmentations_both_empty() -> None:
    """Both masks empty (all background) gives Dice=0, Accuracy=1."""
    import numpy as np

    from dynacell.evaluation.metrics import evaluate_segmentations

    empty = np.zeros((4, 4), dtype=bool)
    result = evaluate_segmentations(empty, empty)
    assert result["Dice"] == pytest.approx(0.0)
    assert result["Accuracy"] == pytest.approx(1.0)
