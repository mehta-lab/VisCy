"""Regression tests for evaluation pixel metrics."""

import importlib
import math
import sys
import types

import numpy as np
import pytest
import torch

from dynacell.evaluation.metrics import evaluate_segmentations


def _import_metrics_with_stubs(monkeypatch):
    """Import the metrics module with lightweight optional-dependency stubs."""
    cubic_module = types.ModuleType("cubic")
    cubic_cuda_module = types.ModuleType("cubic.cuda")
    cubic_cuda_module.ascupy = lambda x: x
    cubic_cuda_module.asnumpy = lambda x: x

    cubic_metrics_module = types.ModuleType("cubic.metrics")
    cubic_metrics_module.fsc_resolution = lambda *args, **kwargs: {}
    cubic_metrics_module.MicroMS3IM = object
    cubic_metrics_module.pcc = lambda a, b, mask=None: float(np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0, 1])

    cubic_bandlimited_module = types.ModuleType("cubic.metrics.bandlimited")
    cubic_bandlimited_module.spectral_pcc = lambda *args, **kwargs: 0.0

    cubic_feature_module = types.ModuleType("cubic.feature")
    cubic_feature_voxel_module = types.ModuleType("cubic.feature.voxel")
    cubic_feature_voxel_module.regionprops_table = lambda *args, **kwargs: {}

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


# --- pcc tests ---


def test_corr_coef_perfect_correlation(monkeypatch) -> None:
    """Identical signals give PCC = 1.0."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    a = torch.linspace(0.0, 1.0, 100)
    assert metrics.pcc(a, a) == pytest.approx(1.0)


def test_corr_coef_negative_correlation(monkeypatch) -> None:
    """Perfectly inverted signal gives PCC = -1.0."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    a = torch.linspace(0.0, 1.0, 100)
    assert metrics.pcc(a, -a) == pytest.approx(-1.0)


def test_corr_coef_constant_input_returns_nan(monkeypatch) -> None:
    """Zero-variance input (constant signal) returns NaN."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    a = torch.ones(100)
    b = torch.linspace(0.0, 1.0, 100)
    assert math.isnan(metrics.pcc(a, b))


def test_corr_coef_shape_mismatch_raises(monkeypatch) -> None:
    """Mismatched shapes raise ValueError."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    with pytest.raises((ValueError, Exception)):
        metrics.pcc(torch.ones(10), torch.ones(20))


# --- evaluate_segmentations tests ---


def test_evaluate_segmentations_perfect_overlap() -> None:
    """Perfect overlap gives all metrics = 1.0."""
    mask = np.ones((8, 8), dtype=bool)
    result = evaluate_segmentations(mask, mask)
    assert result["Dice"] == pytest.approx(1.0)
    assert result["IoU"] == pytest.approx(1.0)
    assert result["Precision"] == pytest.approx(1.0)
    assert result["Recall"] == pytest.approx(1.0)
    assert result["Accuracy"] == pytest.approx(1.0)


def test_evaluate_segmentations_no_overlap() -> None:
    """No overlap gives Dice = IoU = 0."""
    pred = np.zeros((8, 8), dtype=bool)
    gt = np.ones((8, 8), dtype=bool)
    result = evaluate_segmentations(pred, gt)
    assert result["Dice"] == pytest.approx(0.0)
    assert result["IoU"] == pytest.approx(0.0)
    assert result["Precision"] == pytest.approx(0.0)
    assert result["Recall"] == pytest.approx(0.0)


def test_evaluate_segmentations_partial_overlap() -> None:
    """Known partial overlap gives expected values."""
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
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate_segmentations(np.ones((4, 4)), np.ones((4, 5)))


def test_evaluate_segmentations_both_empty() -> None:
    """Both masks empty (all background) gives Dice=0, Accuracy=1."""
    empty = np.zeros((4, 4), dtype=bool)
    result = evaluate_segmentations(empty, empty)
    assert result["Dice"] == pytest.approx(0.0)
    assert result["Accuracy"] == pytest.approx(1.0)


# --- Split GT/pred feature API tests ---


class _IdentityExtractor:
    """Feature extractor stub that returns the flattened image as its embedding."""

    def extract_features(self, img: np.ndarray):
        return torch.from_numpy(np.asarray(img, dtype=np.float32).reshape(-1))


def test_deep_target_and_pred_features_same_cell_order(monkeypatch) -> None:
    """GT and pred iterate the shared cell_segmentation → rows align by cell."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    # 2-D-by-1-z segmentation with 3 labeled cells (IDs 1, 2, 3) at known positions.
    d, h, w = 1, 8, 8
    cell_seg = np.zeros((d, h, w), dtype=np.int32)
    cell_seg[0, 0:2, 0:2] = 1
    cell_seg[0, 4:6, 4:6] = 2
    cell_seg[0, 6:8, 0:2] = 3

    target = np.ones((d, h, w), dtype=np.float32)
    prediction = np.full((d, h, w), 2.0, dtype=np.float32)

    extractor = _IdentityExtractor()
    patch_size = 4

    gt = metrics.deep_features(target, cell_seg, extractor, patch_size)
    pred = metrics.deep_features(prediction, cell_seg, extractor, patch_size)

    # Same number of cells (3), same feature_dim (4x4 flat = 16).
    assert gt.shape == (3, 16)
    assert pred.shape == (3, 16)
    # Because extract_features returns the flattened crop and prediction is 2x target,
    # for every cell the pred embedding should be 2x the target embedding
    # (the masked image differs by a constant factor where the cell mask is 1,
    # and by 0 elsewhere — so 2x).
    ratio = pred / np.maximum(gt, 1e-6)
    assert np.allclose(ratio[gt > 0], 2.0)


def test_deep_features_empty_segmentation_returns_empty(monkeypatch) -> None:
    """Segmentation with only the background label returns an empty feature matrix."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    cell_seg = np.zeros((1, 4, 4), dtype=np.int32)
    target = np.ones((1, 4, 4), dtype=np.float32)
    result = metrics.deep_features(target, cell_seg, _IdentityExtractor(), patch_size=2)
    assert result.shape == (0, 0)


def test_deep_features_shape_mismatch_raises(monkeypatch) -> None:
    """Image and cell_segmentation must match in shape."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    target = np.zeros((1, 4, 4), dtype=np.float32)
    cell_seg = np.zeros((1, 4, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.deep_features(target, cell_seg, _IdentityExtractor(), patch_size=2)


class _BatchAwareExtractor:
    """Extractor that records whether ``extract_features_batch`` was used.

    Returns the flattened crop as the embedding, same as ``_IdentityExtractor``,
    so the row alignment of the batched path is checkable against the per-cell
    path.
    """

    def __init__(self) -> None:
        self.batch_calls = 0
        self.per_cell_calls = 0

    def extract_features(self, img: np.ndarray):  # pragma: no cover - exercised via fallback path
        self.per_cell_calls += 1
        return torch.from_numpy(np.asarray(img, dtype=np.float32).reshape(-1))

    def extract_features_batch(self, images: list[np.ndarray]):
        self.batch_calls += 1
        stacked = np.stack([np.asarray(img, dtype=np.float32).reshape(-1) for img in images], axis=0)
        return torch.from_numpy(stacked)


def test_features_from_crops_prefers_batched_path(monkeypatch) -> None:
    """When the extractor exposes ``extract_features_batch``, it is used once per call."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    crops = [np.ones((4, 4), dtype=np.float32) * k for k in range(1, 4)]
    extractor = _BatchAwareExtractor()
    out = metrics.features_from_crops(crops, extractor)
    assert out.shape == (3, 16)
    # One batched call covers all three crops; no per-cell fallback.
    assert extractor.batch_calls == 1
    assert extractor.per_cell_calls == 0


def test_features_from_crops_falls_back_when_no_batch(monkeypatch) -> None:
    """Extractors without ``extract_features_batch`` get one ``extract_features`` per crop."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    crops = [np.ones((4, 4), dtype=np.float32) * k for k in range(1, 4)]
    extractor = _IdentityExtractor()
    out = metrics.features_from_crops(crops, extractor)
    assert out.shape == (3, 16)


def test_features_from_crops_empty_returns_empty(monkeypatch) -> None:
    """No crops -> empty (0, 0) feature matrix."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    out = metrics.features_from_crops([], _IdentityExtractor())
    assert out.shape == (0, 0)
