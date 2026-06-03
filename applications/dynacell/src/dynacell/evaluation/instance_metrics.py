"""Instance-segmentation average-precision metrics (Cellpose-style).

Thin numpy wrapper over :func:`cubic.metrics.average_precision` (TP / (TP + FP +
FN) per IoU threshold, the Cellpose definition). Used for both the nucleus and
whole-cell instance-AP eval paths. The returned dict merges directly into the
per-(FOV, t) mask-metric rows, so every per-threshold AP becomes its own
``mask_metrics.csv`` column / ``.npy`` key.
"""

import numpy as np
from cubic.metrics import average_precision

DEFAULT_IOU_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
"""IoU thresholds for the AP sweep (Cellpose / StarDist standard 0.50..0.95)."""

_PRIMARY_THRESHOLD = 0.50
"""IoU threshold whose TP/FP/FN counts are persisted (the standard 0.50 operating
point); stored under ``instance_{TP,FP,FN}@0.50`` so the threshold is explicit."""


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Relabel an integer label image to a dense ``0, 1..K`` (background stays 0).

    ``cubic.average_precision`` derives object counts from ``labels.max()``, so
    labels must be gap-free for FP/FN to be correct. Uses ``np.unique`` with
    ``return_inverse`` (not connected-component relabeling — disjoint pieces that
    share an id stay one object).
    """
    labels = np.asarray(labels)
    uniq, inv = np.unique(labels, return_inverse=True)
    inv = inv.reshape(labels.shape)
    # uniq is sorted ascending; if a 0 background exists it maps to 0, real ids to
    # 1..K. If no 0 is present (no background), shift so ids become 1..K.
    return inv if uniq[0] == 0 else inv + 1


def _iou_matrix(gt: np.ndarray, pred: np.ndarray, n_gt: int, n_pred: int) -> np.ndarray:
    """Object-wise IoU matrix ``(n_gt, n_pred)`` from two sequential label images.

    Value-based (a label's IoU pools all of its pixels, connected or not), so it
    matches the "disjoint pieces that share an id stay one object" convention of
    :func:`_relabel_sequential` and ``cubic.average_precision``.
    """
    overlap = np.zeros((n_gt + 1, n_pred + 1), dtype=np.int64)
    np.add.at(overlap, (gt.ravel(), pred.ravel()), 1)
    gt_area = overlap.sum(axis=1)
    pred_area = overlap.sum(axis=0)
    inter = overlap[1:, 1:].astype(np.float64)  # drop background row/col
    union = gt_area[1:, None] + pred_area[None, 1:] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def mean_instance_dice(gt: np.ndarray, pred: np.ndarray, n_gt: int, n_pred: int) -> float:
    """Symmetric best Dice (SBD) between two sequential instance-label images.

    For every GT object take its best-overlapping pred (and vice versa), convert
    that IoU ``u`` to Dice ``2u / (1 + u)``, and average over both directions.
    Missed GT objects and spurious pred objects contribute Dice 0, so the score
    penalizes under- and over-segmentation symmetrically. Both sides empty → NaN;
    exactly one side empty → 0.0. Connectivity-agnostic (per-object, not per
    connected component), so a cytoplasm shell split by the carved nucleus is
    still scored as one object.
    """
    if n_gt == 0 and n_pred == 0:
        return float("nan")
    if n_gt == 0 or n_pred == 0:
        return 0.0
    iou = _iou_matrix(gt, pred, n_gt, n_pred)
    best = np.concatenate([iou.max(axis=1), iou.max(axis=0)])  # GT->pred and pred->GT
    return float(np.mean(2.0 * best / (1.0 + best)))


def instance_average_precision(
    labels_pred: np.ndarray,
    labels_gt: np.ndarray,
    iou_thresholds=DEFAULT_IOU_THRESHOLDS,
) -> dict:
    """Average precision of predicted vs ground-truth instance labels.

    Parameters
    ----------
    labels_pred, labels_gt : numpy.ndarray
        Predicted and ground-truth integer instance-label images (same shape).
    iou_thresholds : sequence of float
        IoU thresholds for the AP sweep.

    Returns
    -------
    dict
        ``AP_<th>`` (one per threshold), ``mAP`` (their mean), ``instance_dice``
        (symmetric best Dice; nucleus instances for the cellpose backend,
        carved-cytoplasm instances for cellpose_watershed), ``n_gt``, ``n_pred``,
        and ``instance_{TP,FP,FN}@0.50``. Both sides empty → all-NaN AP/mAP/Dice;
        exactly one side empty → AP/mAP 0.0, Dice 0.0.
    """
    thresholds = [float(t) for t in iou_thresholds]
    pred = _relabel_sequential(labels_pred)
    gt = _relabel_sequential(labels_gt)
    n_pred = int(pred.max())
    n_gt = int(gt.max())

    # cubic.average_precision returns 0 (not NaN) on empty inputs, so pre-check the
    # degenerate cases here and call it only when both sides have objects.
    if n_gt == 0 and n_pred == 0:
        ap_vals = [float("nan")] * len(thresholds)
        tp = fp = fn = float("nan")
    elif n_gt == 0 or n_pred == 0:
        ap_vals = [0.0] * len(thresholds)
        tp, fp, fn = 0.0, float(n_pred), float(n_gt)
    else:
        ap, tp_arr, fp_arr, fn_arr = average_precision(gt, pred, thresholds)
        ap_vals = [float(a) for a in np.atleast_1d(ap)]
        idx = thresholds.index(_PRIMARY_THRESHOLD) if _PRIMARY_THRESHOLD in thresholds else 0
        tp = float(np.atleast_1d(tp_arr)[idx])
        fp = float(np.atleast_1d(fp_arr)[idx])
        fn = float(np.atleast_1d(fn_arr)[idx])

    result = {"n_gt": n_gt, "n_pred": n_pred}
    for th, a in zip(thresholds, ap_vals):
        result[f"AP_{th:.2f}"] = a
    result["mAP"] = float(np.mean(ap_vals))
    result["instance_dice"] = mean_instance_dice(gt, pred, n_gt, n_pred)
    result[f"instance_TP@{_PRIMARY_THRESHOLD:.2f}"] = tp
    result[f"instance_FP@{_PRIMARY_THRESHOLD:.2f}"] = fp
    result[f"instance_FN@{_PRIMARY_THRESHOLD:.2f}"] = fn
    return result
