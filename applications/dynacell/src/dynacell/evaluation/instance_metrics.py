"""Instance-segmentation average-precision metrics (Cellpose-style).

Thin numpy wrapper over :func:`cubic.metrics.average_precision` (TP / (TP + FP +
FN) per IoU threshold, the Cellpose definition). Used for both the nucleus and
whole-cell instance-AP eval paths. The returned dict merges directly into the
per-(FOV, t) mask-metric rows, so every per-threshold AP becomes its own
``mask_metrics.csv`` column / ``.npy`` key.
"""

import numpy as np

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
        ``AP_<th>`` (one per threshold), ``mAP`` (their mean), ``n_gt``,
        ``n_pred``, and ``instance_{TP,FP,FN}@0.50``. Both sides empty → all-NaN
        AP/mAP; exactly one side empty → AP/mAP 0.0.
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
        # Imported lazily (not at module top) so importing this module for
        # DEFAULT_IOU_THRESHOLDS / _relabel_sequential — e.g. pipeline_cache pulling
        # the threshold constant — does not require the GPU-only cubic stack. The
        # actual AP computation still hard-requires cubic and fails loudly here.
        from cubic.metrics import average_precision

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
    result[f"instance_TP@{_PRIMARY_THRESHOLD:.2f}"] = tp
    result[f"instance_FP@{_PRIMARY_THRESHOLD:.2f}"] = fp
    result[f"instance_FN@{_PRIMARY_THRESHOLD:.2f}"] = fn
    return result
