"""Regression tests for evaluation pixel metrics."""

import importlib
import math
import sys
import types

import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity

from dynacell.evaluation.metrics import evaluate_segmentations


def _import_metrics_with_stubs(monkeypatch):
    """Import the metrics module with lightweight optional-dependency stubs."""
    cubic_module = types.ModuleType("cubic")
    cubic_cuda_module = types.ModuleType("cubic.cuda")
    cubic_cuda_module.ascupy = lambda x: x
    cubic_cuda_module.asnumpy = lambda x: x

    cubic_metrics_module = types.ModuleType("cubic.metrics")
    cubic_metrics_module.fsc_resolution = lambda *args, **kwargs: {"mean": 2.0}
    cubic_metrics_module.frc_resolution = lambda *args, **kwargs: 3.0
    cubic_metrics_module.MicroMS3IM = object

    def _stub_pcc(a, b, mask=None):
        if a.shape != b.shape:
            raise ValueError(f"Inputs must have same shape, got {a.shape} and {b.shape}")
        return float(np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0, 1])

    cubic_metrics_module.pcc = _stub_pcc

    def _as_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    def _scale_invariant_pair(y_true, y_pred):
        """Mirror cubic's ``scale_invariant`` transform: z-score GT, LS-fit pred.

        Kept in step with ``cubic.metrics.skimage_metrics.scale_invariant`` so the
        stub tests the contract the production call relies on rather than an
        invented one: the target is standardized, the prediction is centred and
        rescaled by the least-squares gain, and ``data_range`` is derived from the
        target alone.
        """
        a, b = _as_np(y_true).astype(np.float64), _as_np(y_pred).astype(np.float64)
        gt_std = a.std()
        gt_norm = (a - a.mean()) / gt_std
        pred_zero = b - b.mean()
        alpha = (gt_norm * pred_zero).sum() / (pred_zero * pred_zero).sum()
        return gt_norm, pred_zero * alpha, float((a.max() - a.min()) / gt_std)

    def _reject_conflict(scale_invariant, normalize, normalization=None):
        # cubic raises when a normalization mode is combined with the
        # scale-invariant path, because both choose the denominator. Mirroring it
        # keeps a config that would fail in production from passing here.
        if scale_invariant and (normalize is not None or normalization is not None):
            raise ValueError("scale_invariant=True is incompatible with normalize/normalization")

    def _stub_nrmse(
        y_true, y_pred, normalization=None, normalize=None, data_range=None, mask=None, scale_invariant=False
    ):
        _reject_conflict(scale_invariant, normalize, normalization)
        if scale_invariant:
            a, b, _ = _scale_invariant_pair(y_true, y_pred)
            return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a**2)))
        a, b = _as_np(y_true), _as_np(y_pred)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-8)
        b = (b - b.min()) / max(float(b.max() - b.min()), 1e-8)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _stub_psnr(y_true, y_pred, data_range=None, normalize=None, mask=None, scale_invariant=False):
        _reject_conflict(scale_invariant, normalize)
        if scale_invariant:
            a, b, rng = _scale_invariant_pair(y_true, y_pred)
            mse = np.mean((a - b) ** 2)
            return float("inf") if mse < 1e-12 else float(10 * np.log10(rng**2 / mse))
        a, b = _as_np(y_true), _as_np(y_pred)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-8)
        b = (b - b.min()) / max(float(b.max() - b.min()), 1e-8)
        mse = np.mean((a - b) ** 2)
        return float("inf") if mse < 1e-8 else float(20 * np.log10(1.0) - 10 * np.log10(mse))

    cubic_metrics_module.nrmse = _stub_nrmse
    cubic_metrics_module.psnr = _stub_psnr

    def _stub_ssim(
        img1, img2, spatial_dims=None, data_range=None, gaussian_weights=None, scale_invariant=False, **kwargs
    ):
        _reject_conflict(scale_invariant, kwargs.get("normalize"))
        if scale_invariant:
            a, b, rng = _scale_invariant_pair(img1.squeeze(), img2.squeeze())
            return float(structural_similarity(a, b, data_range=rng))
        a = img1.numpy().squeeze()
        b = img2.numpy().squeeze()
        return float(structural_similarity(a, b, data_range=float(data_range or 1.0)))

    cubic_metrics_module.ssim = _stub_ssim

    cubic_bandlimited_module = types.ModuleType("cubic.metrics.bandlimited")
    cubic_bandlimited_module.spectral_pcc = lambda *args, **kwargs: 0.0

    cubic_feature_module = types.ModuleType("cubic.feature")
    cubic_feature_module.glcm_features = lambda *args, **kwargs: {}
    cubic_feature_voxel_module = types.ModuleType("cubic.feature.voxel")
    cubic_feature_voxel_module.regionprops_table = lambda *args, **kwargs: {}

    # cubic.scipy / cubic.skimage: the fake ``cubic`` module has no ``__path__``,
    # so these submodule imports must be stubbed explicitly too.
    cubic_scipy_module = types.ModuleType("cubic.scipy")
    cubic_scipy_ndimage_module = types.ModuleType("cubic.scipy.ndimage")
    cubic_scipy_ndimage_module.find_objects = lambda *args, **kwargs: []
    cubic_scipy_module.ndimage = cubic_scipy_ndimage_module
    cubic_skimage_module = types.ModuleType("cubic.skimage")
    cubic_skimage_filters_module = types.ModuleType("cubic.skimage.filters")
    cubic_skimage_filters_module.threshold_otsu = lambda *args, **kwargs: 0.0
    cubic_skimage_module.filters = cubic_skimage_filters_module

    monkeypatch.setitem(sys.modules, "cubic", cubic_module)
    monkeypatch.setitem(sys.modules, "cubic.cuda", cubic_cuda_module)
    monkeypatch.setitem(sys.modules, "cubic.metrics", cubic_metrics_module)
    monkeypatch.setitem(sys.modules, "cubic.metrics.bandlimited", cubic_bandlimited_module)
    monkeypatch.setitem(sys.modules, "cubic.feature", cubic_feature_module)
    monkeypatch.setitem(sys.modules, "cubic.feature.voxel", cubic_feature_voxel_module)
    monkeypatch.setitem(sys.modules, "cubic.scipy", cubic_scipy_module)
    monkeypatch.setitem(sys.modules, "cubic.scipy.ndimage", cubic_scipy_ndimage_module)
    monkeypatch.setitem(sys.modules, "cubic.skimage", cubic_skimage_module)
    monkeypatch.setitem(sys.modules, "cubic.skimage.filters", cubic_skimage_filters_module)
    sys.modules.pop("dynacell.evaluation.metrics", None)

    return importlib.import_module("dynacell.evaluation.metrics")


def test_scale_invariant_absorbs_gain_and_offset(monkeypatch) -> None:
    """An affine-transformed prediction must score as a perfect match.

    This is the property ``scale_invariant=True`` guarantees, and the reason the
    pipeline switched to it: the metric should report agreement with the target,
    not the prediction's dynamic range. The previous per-input min-max recipe also
    absorbed gain and offset, but it derived each array's scale from its own
    extremes, so a single outlier voxel moved the denominator; the scale-invariant
    form fits the prediction to the target by least squares instead.
    """
    metrics = _import_metrics_with_stubs(monkeypatch)

    target = torch.linspace(0.0, 1.0, steps=16 * 16).reshape(1, 16, 16)
    prediction = target * 2.0 + 0.25  # pure gain+offset — no information lost

    assert metrics.nrmse(target, prediction, scale_invariant=True) == pytest.approx(0.0, abs=1e-6)
    assert metrics.psnr(target, prediction, scale_invariant=True) == float("inf")
    # metrics.ssim is the module's own wrapper, which already passes
    # scale_invariant=True to cubic; it takes (H, W) / (D, H, W) input.
    assert metrics.ssim(target.squeeze(0), prediction.squeeze(0)) == pytest.approx(1.0, abs=5e-2)


def test_scale_invariant_rejects_a_normalization_mode(monkeypatch) -> None:
    """Passing both a normalization mode and scale_invariant must raise.

    Both pick the comparison's denominator, so combining them silently returned a
    different number in cubic < 0.9.0a1 -- which is how ``normalize="min_max"``
    went unnoticed on ``ssim`` (it vanished into ``**kwargs``) while quietly
    corrupting psnr and nrmse. The pipeline must fail loudly instead.
    """
    metrics = _import_metrics_with_stubs(monkeypatch)

    target = torch.linspace(0.0, 1.0, steps=16 * 16).reshape(1, 16, 16)
    with pytest.raises(ValueError, match="incompatible"):
        metrics.nrmse(target, target, normalize="min_max", scale_invariant=True)
    with pytest.raises(ValueError, match="incompatible"):
        metrics.psnr(target, target, normalize="min_max", scale_invariant=True)


def test_identical_images_still_score_perfectly(monkeypatch) -> None:
    """Scale-invariant scoring should preserve perfect self-similarity."""
    metrics = _import_metrics_with_stubs(monkeypatch)

    target = torch.linspace(0.0, 1.0, steps=16 * 16).reshape(1, 16, 16)

    assert metrics.nrmse(target, target) == pytest.approx(0.0)
    assert metrics.psnr(target, target) == float("inf")
    assert metrics.ssim(target, target) == pytest.approx(1.0)


# --- ssim / pixel-metric dimensionality (2D vs 3D) ---


def test_ssim_accepts_2d_input_and_dispatches_spatial_dims_2(monkeypatch) -> None:
    """A 2-D (H, W) input scores in-plane and passes spatial_dims=2 to cubic."""
    metrics = _import_metrics_with_stubs(monkeypatch)

    captured = {}

    def _capture_ssim(img1, img2, spatial_dims=None, data_range=None, gaussian_weights=None, **kwargs):
        captured["spatial_dims"] = spatial_dims
        captured["ndim"] = img1.ndim
        return float(structural_similarity(img1.numpy().squeeze(), img2.numpy().squeeze(), data_range=1.0))

    monkeypatch.setattr(metrics, "cubic_ssim", _capture_ssim)

    target = torch.rand(16, 16)  # 2-D (H, W)
    assert metrics.ssim(target, target) == pytest.approx(1.0)
    assert captured["spatial_dims"] == 2
    assert captured["ndim"] == 4  # (1, 1, H, W)


def test_ssim_dispatches_spatial_dims_3_for_3d(monkeypatch) -> None:
    """A 3-D (D, H, W) input keeps volumetric SSIM (spatial_dims=3)."""
    metrics = _import_metrics_with_stubs(monkeypatch)

    captured = {}

    def _capture_ssim(img1, img2, spatial_dims=None, data_range=None, gaussian_weights=None, **kwargs):
        captured["spatial_dims"] = spatial_dims
        captured["ndim"] = img1.ndim
        return 1.0  # dispatch-only stub — skimage's 3D window doesn't fit a tiny volume

    monkeypatch.setattr(metrics, "cubic_ssim", _capture_ssim)

    target = torch.rand(4, 16, 16)  # 3-D (D, H, W)
    assert metrics.ssim(target, target) == pytest.approx(1.0)
    assert captured["spatial_dims"] == 3
    assert captured["ndim"] == 5  # (1, 1, D, H, W)


def test_ssim_rejects_non_2d_3d(monkeypatch) -> None:
    """Ranks other than 2 or 3 raise a clear error."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    with pytest.raises(ValueError, match="2-D .* or 3-D"):
        metrics.ssim(torch.rand(2, 4, 16, 16), torch.rand(2, 4, 16, 16))


def test_compute_pixel_metrics_2d_uses_frc_not_fsc(monkeypatch) -> None:
    """2-D inputs report FRC_Resolution (ring) and never FSC_Resolution (shell)."""
    metrics = _import_metrics_with_stubs(monkeypatch)

    pred = torch.rand(16, 16)
    target = torch.rand(16, 16)
    out = metrics.compute_pixel_metrics(
        pred, target, spacing=(0.15, 0.15, 0.15), fsc_kwargs={"bin_delta": 5}, use_gpu=False
    )
    assert "FRC_Resolution" in out
    assert not any(k.endswith("_FSC_Resolution") for k in out)


def test_compute_pixel_metrics_3d_uses_fsc_not_frc(monkeypatch) -> None:
    """3-D inputs keep FSC_Resolution (shell) and never emit FRC_Resolution."""
    metrics = _import_metrics_with_stubs(monkeypatch)

    pred = torch.rand(8, 16, 16)  # D=8 so the stub SSIM's 3D window fits
    target = torch.rand(8, 16, 16)
    out = metrics.compute_pixel_metrics(
        pred, target, spacing=(0.5, 0.15, 0.15), fsc_kwargs={"bin_delta": 5}, use_gpu=False
    )
    assert any(k.endswith("_FSC_Resolution") for k in out)
    assert "FRC_Resolution" not in out


# --- real-cubic integration (skipped when cubic is not installed) ---


def test_ssim_real_cubic_2d_and_3d_identical_scores_one() -> None:
    """Real cubic SSIM accepts both a 2-D plane and a 3-D volume (no stubs)."""
    from dynacell.evaluation import metrics as real_metrics

    if real_metrics.cubic_ssim is None:
        pytest.skip("cubic not installed")
    img2d = torch.rand(32, 32)
    img3d = torch.rand(8, 32, 32)
    assert real_metrics.ssim(img2d, img2d) == pytest.approx(1.0, abs=1e-3)
    assert real_metrics.ssim(img3d, img3d) == pytest.approx(1.0, abs=1e-3)


def test_compute_pixel_metrics_real_cubic_2d_reports_frc() -> None:
    """Real cubic: a 2-D input yields FRC_Resolution + SSIM and no FSC key."""
    from dynacell.evaluation import metrics as real_metrics

    if real_metrics.pcc is None:
        pytest.skip("cubic not installed")
    pred = torch.rand(64, 64)
    target = torch.rand(64, 64)
    out = real_metrics.compute_pixel_metrics(
        pred, target, spacing=(0.15, 0.15, 0.15), fsc_kwargs={"bin_delta": 5}, use_gpu=False
    )
    assert "SSIM" in out and "FRC_Resolution" in out
    assert not any(k.endswith("_FSC_Resolution") for k in out)


# --- pcc tests ---


def test_pcc_perfect_correlation(monkeypatch) -> None:
    """Identical signals give PCC = 1.0."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    a = torch.linspace(0.0, 1.0, 100)
    assert metrics.pcc(a, a) == pytest.approx(1.0)


def test_pcc_negative_correlation(monkeypatch) -> None:
    """Perfectly inverted signal gives PCC = -1.0."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    a = torch.linspace(0.0, 1.0, 100)
    assert metrics.pcc(a, -a) == pytest.approx(-1.0)


def test_pcc_constant_input_returns_nan(monkeypatch) -> None:
    """Zero-variance input (constant signal) returns NaN."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    a = torch.ones(100)
    b = torch.linspace(0.0, 1.0, 100)
    assert math.isnan(metrics.pcc(a, b))


def test_pcc_shape_mismatch_raises(monkeypatch) -> None:
    """Mismatched shapes raise ValueError."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    with pytest.raises(ValueError):
        metrics.pcc(torch.ones(10), torch.ones(20))


# --- score_microssim tests ---


class _ScriptedSim:
    """Fake MicroMS3IM stub whose ``score`` method replays a scripted sequence.

    Each element of ``behavior`` is either a numeric value (returned on the
    next call) or an ``Exception`` instance (raised on the next call). This
    lets each test inject the exact mix of degenerate / valid / raising
    slices needed to exercise a single branch of ``score_microssim``.
    """

    def __init__(self, behavior):
        self.behavior = list(behavior)
        self.call_count = 0

    def score(self, target, pred):  # noqa: ARG002
        idx = self.call_count
        self.call_count += 1
        outcome = self.behavior[idx]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _three_slice_data():
    """Single-FOV microssim_data with three z-slices of trivial (4, 4) shape."""
    return [
        {
            "target": np.zeros((3, 4, 4), dtype=np.float32),
            "predict": np.zeros((3, 4, 4), dtype=np.float32),
        }
    ]


def test_score_microssim_degenerate_slice_scored_as_zero_penalizes_mean(monkeypatch) -> None:
    """A data_range ValueError scores the slice as 0 — partial collapse drags the FOV-T mean toward 0.

    This penalizes a model that collapses on a subset of slices, instead of
    silently dropping the degenerate slice from the row average (which would
    let a partially-collapsing model rank identically to one that scores well
    everywhere).
    """
    metrics = _import_metrics_with_stubs(monkeypatch)
    sim = _ScriptedSim(
        [
            ValueError("data_range must be finite and positive; got 0.0"),
            0.8,
            0.6,
        ]
    )
    out = metrics.score_microssim(_three_slice_data(), sim, use_gpu=False)
    # (0.0 + 0.8 + 0.6) / 3 — the degenerate slice contributes 0, not "absent".
    assert out[0]["MicroMS3IM"] == pytest.approx((0.0 + 0.8 + 0.6) / 3)


def test_score_microssim_all_degenerate_yields_zero(monkeypatch) -> None:
    """When every slice trips the data_range guard the FOV-T row is 0.0 — worst possible score."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    sim = _ScriptedSim([ValueError("data_range must be finite and positive; got 0.0")] * 3)
    out = metrics.score_microssim(_three_slice_data(), sim, use_gpu=False)
    assert out[0]["MicroMS3IM"] == pytest.approx(0.0)


def test_score_microssim_non_data_range_value_error_propagates(monkeypatch) -> None:
    """ValueErrors whose message does not mention data_range are real bugs — they propagate."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    sim = _ScriptedSim([ValueError("MicroSSIM was not initialized, call fit() first.")])
    with pytest.raises(ValueError, match="not initialized"):
        metrics.score_microssim(
            [
                {
                    "target": np.zeros((1, 4, 4), dtype=np.float32),
                    "predict": np.zeros((1, 4, 4), dtype=np.float32),
                }
            ],
            sim,
            use_gpu=False,
        )


def test_score_microssim_runtime_error_propagates(monkeypatch) -> None:
    """RuntimeErrors (e.g., CUDA OOM) are no longer masked by the guard."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    sim = _ScriptedSim([RuntimeError("CUDA out of memory")])
    with pytest.raises(RuntimeError, match="out of memory"):
        metrics.score_microssim(
            [
                {
                    "target": np.zeros((1, 4, 4), dtype=np.float32),
                    "predict": np.zeros((1, 4, 4), dtype=np.float32),
                }
            ],
            sim,
            use_gpu=False,
        )


def test_score_microssim_empty_fov_raises(monkeypatch) -> None:
    """An FOV with zero z-slices is an upstream stacking bug — raise rather than NaN."""
    metrics = _import_metrics_with_stubs(monkeypatch)
    sim = _ScriptedSim([])
    with pytest.raises(ValueError, match="zero z-slices"):
        metrics.score_microssim(
            [
                {
                    "target": np.zeros((0, 4, 4), dtype=np.float32),
                    "predict": np.zeros((0, 4, 4), dtype=np.float32),
                }
            ],
            sim,
            use_gpu=False,
        )


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


def test_build_crops_robust_norm_survives_hot_pixel(monkeypatch) -> None:
    """A lone hot pixel must not compress the in-cell crop toward black.

    Regression for the raw-min-max recipe: a single saturated pixel
    anywhere in the max-projection set the whole-FOV scale and flattened
    every cell's real signal to ~0, and did so asymmetrically for GT
    (hot-pixel-prone fluorescence) vs prediction (smooth) — a GT-vs-pred
    intensity-range mismatch injected straight into the deep features.
    Robust percentile normalization (``_robust_norm``, 1-99 clip) clips
    the outlier so the cell crop keeps its contrast.
    """
    metrics = _import_metrics_with_stubs(monkeypatch)
    d, h, w = 1, 12, 12  # 144 px: p99 index lands on real signal, so a lone outlier is clipped
    image = np.full((d, h, w), 5.0, dtype=np.float32)  # uniform real background
    image[0, 2:5, 2:5] = np.arange(10, 100, 10, dtype=np.float32).reshape(3, 3)  # in-cell gradient
    image[0, 10, 10] = 1.0e6  # hot pixel outside the cell
    cell_seg = np.zeros((d, h, w), dtype=np.int32)
    cell_seg[0, 2:5, 2:5] = 1

    crops = metrics.build_crops(image, cell_seg, patch_size=4)

    assert len(crops) == 1
    # float32 (not float64): _robust_norm upcasts via np.percentile, but crops
    # must match float32 model weights (DINOv3/DynaCLR feed them straight in).
    assert crops[0].dtype == np.float32
    foreground = crops[0][crops[0] > 0]
    # Raw min-max would map the 10..90 signal to ~1e-5 (max == 1e6); robust
    # norm clips the hot pixel so the in-cell values keep real spread.
    assert foreground.size == 9
    assert float(foreground.max()) > 0.5
    assert float(foreground.std()) > 0.1


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
