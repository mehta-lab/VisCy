"""Embedding-level mean Average Precision (mAP) via copairs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_embedding_map(
    meta: pd.DataFrame,
    features: np.ndarray,
    reference_condition: str,
    target_condition: str,
    condition_col: str = "condition",
    group_col: str = "marker",
    distance: str = "cosine",
    null_size: int = 10000,
    seed: int = 0,
) -> dict | None:
    """Compute mean Average Precision for embedding-space phenotypic profiling.

    Uses ``copairs`` to compute per-cell Average Precision (AP) between a
    reference and target condition, then aggregates to mAP per group. Positive
    pairs share the same group and condition; negative pairs share only the group
    but differ in condition.

    Parameters
    ----------
    meta : pd.DataFrame
        Cell metadata, one row per cell. Must contain ``condition_col`` and
        ``group_col`` columns.
    features : np.ndarray
        Embedding matrix, shape (n_cells, n_features). Rows correspond to
        ``meta`` rows.
    reference_condition : str
        Value of ``condition_col`` for the reference/control group (``cond_a``).
    target_condition : str
        Value of ``condition_col`` for the treatment group (``cond_b``).
    condition_col : str
        Column in ``meta`` that holds condition labels. Default: ``"condition"``.
    group_col : str
        Column in ``meta`` that holds group labels (e.g. marker/organelle).
        Default: ``"marker"``.
    distance : str
        Distance metric for copairs (e.g. ``"cosine"``). Default: ``"cosine"``.
    null_size : int
        Number of null pairs for the mAP significance test. Default: 10000.
    seed : int
        Random seed. Default: 0.

    Returns
    -------
    dict or None
        ``{"mean_average_precision": float, "p_value": float,
        "n_reference": int, "n_target": int}`` or ``None`` if either condition
        has no cells.
    """
    try:
        import copairs.map
        import copairs.matching
    except ImportError as e:
        raise ImportError("copairs is required for mAP computation. Install it with: pip install copairs") from e

    mask_ref = meta[condition_col] == reference_condition
    mask_tgt = meta[condition_col] == target_condition
    mask = mask_ref | mask_tgt

    if mask_ref.sum() == 0 or mask_tgt.sum() == 0:
        return None

    sub_meta = meta[mask].reset_index(drop=True)
    sub_feats = features[mask.values]

    reference_col = "reference_index"
    sub_meta = sub_meta.copy()
    sub_meta[reference_col] = copairs.matching.assign_reference_index(
        sub_meta, reference_condition, condition_col, group_col
    )

    pos_sameby = [group_col, condition_col, reference_col]
    neg_sameby = [group_col]
    neg_diffby = [condition_col, reference_col]

    ap_df = copairs.map.average_precision(
        sub_meta,
        sub_feats,
        pos_sameby=pos_sameby,
        neg_sameby=neg_sameby,
        neg_diffby=neg_diffby,
        batch_size=20000,
        distance=distance,
    )

    target_ap = ap_df[sub_meta[condition_col] == target_condition]
    if len(target_ap) == 0:
        return None

    map_result = copairs.map.mean_average_precision(
        target_ap,
        sameby=[group_col],
        null_size=null_size,
        threshold=0.05,
        seed=seed,
    )

    if hasattr(map_result, "mean_average_precision"):
        mmap = float(map_result.mean_average_precision.iloc[0])
        pval = float(map_result.p_value.iloc[0]) if "p_value" in map_result.columns else float("nan")
    elif isinstance(map_result, dict):
        mmap = float(map_result.get("mean_average_precision", float("nan")))
        pval = float(map_result.get("p_value", float("nan")))
    else:
        return None

    return {
        "mean_average_precision": mmap,
        "p_value": pval,
        "n_reference": int(mask_ref.sum()),
        "n_target": int(mask_tgt.sum()),
    }
