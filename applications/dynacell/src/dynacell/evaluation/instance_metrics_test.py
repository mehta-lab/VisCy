"""Tests for the instance average-precision wrapper (real cubic, CPU)."""

import numpy as np

from dynacell.evaluation.instance_metrics import (
    DEFAULT_IOU_THRESHOLDS,
    _relabel_sequential,
    instance_average_precision,
)


def _two_squares(shape=(16, 16)) -> np.ndarray:
    """A label image with two well-separated square instances (ids 1, 2)."""
    lab = np.zeros(shape, dtype=np.uint16)
    lab[2:6, 2:6] = 1
    lab[10:14, 10:14] = 2
    return lab


def test_identical_labels_map_one():
    """Identical label images score mAP 1.0 and AP 1.0 at every threshold."""
    gt = _two_squares()
    result = instance_average_precision(gt.copy(), gt)
    assert result["mAP"] == 1.0
    assert result["n_gt"] == 2 and result["n_pred"] == 2
    for th in DEFAULT_IOU_THRESHOLDS:
        assert result[f"AP_{th:.2f}"] == 1.0
    assert result["instance_TP@0.50"] == 2.0
    assert result["instance_FP@0.50"] == 0.0
    assert result["instance_FN@0.50"] == 0.0


def test_disjoint_labels_map_zero():
    """Predicting objects nowhere near the GT scores mAP 0.0."""
    gt = _two_squares()
    pred = np.zeros_like(gt)
    pred[2:6, 10:14] = 1  # overlaps neither GT square
    result = instance_average_precision(pred, gt)
    assert result["mAP"] == 0.0


def test_one_side_empty_is_zero():
    """One empty side → AP 0.0 (not NaN); FP/FN reflect the non-empty side."""
    gt = _two_squares()
    empty = np.zeros_like(gt)
    miss = instance_average_precision(empty, gt)
    assert miss["mAP"] == 0.0
    assert miss["instance_FN@0.50"] == 2.0
    assert miss["instance_FP@0.50"] == 0.0
    extra = instance_average_precision(gt.copy(), empty)
    assert extra["mAP"] == 0.0
    assert extra["instance_FP@0.50"] == 2.0
    assert extra["instance_FN@0.50"] == 0.0


def test_both_empty_is_nan():
    """Both sides empty → AP undefined (NaN), not a spurious 0.0."""
    empty = np.zeros((16, 16), dtype=np.uint16)
    result = instance_average_precision(empty, empty)
    assert np.isnan(result["mAP"])
    assert result["n_gt"] == 0 and result["n_pred"] == 0
    for th in DEFAULT_IOU_THRESHOLDS:
        assert np.isnan(result[f"AP_{th:.2f}"])


def test_arg_swap_moves_fp_fn_not_map():
    """Swapping pred/gt leaves mAP unchanged but swaps FP and FN."""
    gt = _two_squares()
    pred = gt.copy()
    pred[10:14, 10:14] = 0  # pred has only 1 of the 2 objects
    a = instance_average_precision(pred, gt)  # 1 pred, 2 gt -> 1 FN
    b = instance_average_precision(gt, pred)  # 2 pred, 1 gt -> 1 FP
    assert a["mAP"] == b["mAP"]
    assert a["instance_FN@0.50"] == b["instance_FP@0.50"] == 1.0
    assert a["instance_FP@0.50"] == b["instance_FN@0.50"] == 0.0


def test_gapped_relabel_no_merge():
    """``_relabel_sequential`` densifies ids without merging disjoint objects."""
    lab = np.zeros((8, 8), dtype=np.uint16)
    lab[0:2, 0:2] = 5
    lab[5:7, 5:7] = 9  # gap in ids (5, 9) -> should become (1, 2)
    out = _relabel_sequential(lab)
    assert sorted(np.unique(out).tolist()) == [0, 1, 2]
    # the two original objects stay distinct (no merge)
    assert out[0, 0] != out[5, 5]


def test_relabel_preserves_object_count_for_ap():
    """AP object counts come from labels.max(); relabel keeps n correct."""
    lab = np.zeros((8, 8), dtype=np.uint16)
    lab[0:2, 0:2] = 7  # single object with a non-1 id
    result = instance_average_precision(lab.copy(), lab)
    assert result["n_gt"] == 1 and result["n_pred"] == 1
    assert result["mAP"] == 1.0
