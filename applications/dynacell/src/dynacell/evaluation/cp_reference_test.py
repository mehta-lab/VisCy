"""Tests for the per-target CP reference and the pipeline's use of it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from dynacell.evaluation.cp_reference import (
    CP_REFERENCE_DIMENSION,
    fit_cp_reference,
    load_cp_reference,
    payload_sha256,
    resolve_cp_reference_path,
    write_cp_reference,
)
from dynacell.evaluation.feature_metrics import compute_feature_similarity_pairwise
from dynacell.evaluation.feature_select import select_features
from dynacell.evaluation.paths import DATA_ROOT, LITE_DATA_ROOT
from dynacell.evaluation.pipeline import _BackboneLists, _extend_backbone, _stage_cp_dataset_inputs

_N_FEATURES = 8
_NAMES = tuple(f"f{i}" for i in range(_N_FEATURES))


def _gt(n: int = 300, seed: int = 0) -> np.ndarray:
    """Synthetic GT CP matrix: 6 independent columns, one near-duplicate, one constant."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, _N_FEATURES)) * np.arange(1, _N_FEATURES + 1) + 10.0
    x[:, 6] = x[:, 0] * 2.0 + 1e-6 * rng.standard_normal(n)  # |r| ~ 1 with f0
    x[:, 7] = 3.0  # constant
    return x


def _reference(tmp_path: Path, gt: np.ndarray):
    """Fit, write and load a reference on ``gt``."""
    payload = fit_cp_reference(
        gt,
        target_name="er",
        dimension=CP_REFERENCE_DIMENSION,
        feature_names=_NAMES,
        cp_identity={"cp_feature_version": "test"},
        datasets=[{"dataset": "synthetic", "n_cells": gt.shape[0]}],
    )
    path = tmp_path / "er__3d.json"
    write_cp_reference(payload, path)
    return load_cp_reference(path, target_name="er", dimension=CP_REFERENCE_DIMENSION), payload


def _old_per_side_zscore(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The retired transform: each side standardized by its own mean/std."""
    return (
        (pred - pred.mean(axis=0)) / (pred.std(axis=0) + 1e-8),
        (target - target.mean(axis=0)) / (target.std(axis=0) + 1e-8),
    )


def _lists(pred: np.ndarray, gt: np.ndarray) -> _BackboneLists:
    """One-FOV ``_BackboneLists`` holding ``(pred, gt)`` as two timepoints."""
    bb = _BackboneLists()
    half = pred.shape[0] // 2
    _extend_backbone(bb, pred[:half], gt[:half], "A/1/0", 0)
    _extend_backbone(bb, pred[half:], gt[half:], "A/1/0", 1)
    return bb


def test_fit_drops_constant_and_correlated_and_records_provenance(tmp_path: Path) -> None:
    gt = _gt()
    ref, payload = _reference(tmp_path, gt)
    assert not ref.keep_mask[7]
    assert ref.keep_mask[0] ^ ref.keep_mask[6]
    np.testing.assert_allclose(ref.mean, gt[:, ref.keep_mask].mean(axis=0))
    np.testing.assert_allclose(ref.std, gt[:, ref.keep_mask].std(axis=0))
    assert payload["fit"]["n_cells"] == gt.shape[0]
    assert payload["criteria"]["fit_on"] == "gt_only"
    assert len(payload["fit"]["gt_matrix_sha256"]) == 64
    # Transformed GT is exactly standardized by the shared scaler.
    z = ref.transform(gt)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-10)


def test_hash_ignores_build_time_and_detects_edits(tmp_path: Path) -> None:
    """Identical inputs hash identically; a hand edit fails the load."""
    (tmp_path / "rebuild").mkdir()
    _, first = _reference(tmp_path, _gt())
    _, second = _reference(tmp_path / "rebuild", _gt())
    assert first["sha256"] == second["sha256"] == payload_sha256({**first, "created_at": "another time"})

    path = tmp_path / "er__3d.json"
    edited = json.loads(path.read_text())
    edited["std"][0] *= 2.0
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        load_cp_reference(path, target_name="er", dimension=CP_REFERENCE_DIMENSION)


def test_load_refuses_missing_and_other_target(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_cp_reference.py --target er --dimension 3d"):
        load_cp_reference(tmp_path / "absent.json", target_name="er", dimension=CP_REFERENCE_DIMENSION)
    _reference(tmp_path, _gt())
    with pytest.raises(ValueError, match="was fit for"):
        load_cp_reference(tmp_path / "er__3d.json", target_name="nucleus", dimension=CP_REFERENCE_DIMENSION)


def test_registry_path_is_shared_by_lite_and_full() -> None:
    """``reference_path: null`` resolves under DATA_ROOT, never LITE_DATA_ROOT."""
    config = OmegaConf.create({"target_name": "mitochondria", "feature_metrics": {"cp": {"reference_path": None}}})
    path = resolve_cp_reference_path(config)
    assert path == DATA_ROOT / "cp_reference" / "mitochondria__3d.json"
    assert LITE_DATA_ROOT not in path.parents
    config.feature_metrics.cp.reference_path = "/x/ref.json"
    assert resolve_cp_reference_path(config) == Path("/x/ref.json")


def test_two_models_share_one_mask(tmp_path: Path) -> None:
    """Two different predictions against one GT are scored on the identical feature subset.

    The retired pooled GT+pred selection gives these two models different masks,
    which is the model-dependence the reference removes.
    """
    gt = _gt()
    ref, _ = _reference(tmp_path, gt)
    rng = np.random.default_rng(1)
    pred_a = gt + 0.1 * rng.standard_normal(gt.shape)
    pred_b = gt.copy()
    pred_b[:, 6] = rng.standard_normal(gt.shape[0]) * 50.0  # decorrelates f6 from f0 in the pool
    pred_b[:, 7] = rng.standard_normal(gt.shape[0])  # a constant GT column that varies in pred

    masks = []
    for i, pred in enumerate((pred_a, pred_b)):
        save_dir = tmp_path / f"model_{i}"
        save_dir.mkdir()
        staged = _stage_cp_dataset_inputs(_lists(pred, gt), ref, save_dir)
        sidecar = json.loads((save_dir / "cp_selected_feature_mask.json").read_text())
        assert sidecar["reference_sha256"] == ref.sha256
        assert staged[1].shape[1] == staged[2].shape[1] == int(ref.keep_mask.sum())
        # Probe inputs are masked but unscaled.
        np.testing.assert_array_equal(staged[4], gt[:, ref.keep_mask])
        masks.append(sidecar["keep_mask"])
    assert masks[0] == masks[1] == [bool(b) for b in ref.keep_mask]

    old_a = select_features(gt, pred_a)[2]
    old_b = select_features(gt, pred_b)[2]
    assert not np.array_equal(old_a, old_b)


def test_over_smoothing_registers_in_the_shared_space(tmp_path: Path) -> None:
    """``pred = 0.5 * GT + c`` (compressed range + offset) now scores clearly off GT.

    The retired per-side z-score maps any per-feature affine distortion of GT back
    onto GT exactly, so it reported a perfect match (cosine 1, KID ~0). The shared
    GT scaler keeps the distortion, so the same prediction is now penalized.
    """
    gt = _gt(n=400)
    ref, _ = _reference(tmp_path, gt)
    pred = 0.5 * gt + 7.0

    old_p, old_t = _old_per_side_zscore(ref.select(pred), ref.select(gt))
    old = compute_feature_similarity_pairwise(old_p, old_t, "CP", compute_fid=False)
    new = compute_feature_similarity_pairwise(ref.transform(pred), ref.transform(gt), "CP", compute_fid=False)

    # KID of GT against itself: the estimator's floor for "indistinguishable" (not exactly 0).
    floor = compute_feature_similarity_pairwise(ref.transform(gt), ref.transform(gt), "CP", compute_fid=False)

    assert old["CP_Median_Cosine_Similarity"] == pytest.approx(1.0, abs=1e-6)
    assert old["CP_KID"] == pytest.approx(floor["CP_KID"], abs=1e-4)
    assert new["CP_Median_Cosine_Similarity"] < 0.9
    assert new["CP_KID"] > 1.0
    assert new["CP_KID"] > 100 * abs(floor["CP_KID"])
