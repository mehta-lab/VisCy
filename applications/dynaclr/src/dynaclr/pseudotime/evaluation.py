"""Evaluation of DTW pseudotime against ground truth annotations.

Compares DTW-derived pseudotime with annotated infection_state and
organelle_state to quantify alignment quality. Designed to run across
multiple embedding types for comparison.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

_logger = logging.getLogger(__name__)


def pseudotime_vs_annotation_auc(
    df: pd.DataFrame,
    pseudotime_col: str = "pseudotime",
    annotation_col: str = "infection_state",
    positive_value: str = "infected",
) -> float:
    """ROC-AUC of pseudotime predicting a binary annotation.

    Parameters
    ----------
    df : pd.DataFrame
        Must have pseudotime_col and annotation_col columns.
    pseudotime_col : str
        Column with DTW pseudotime values.
    annotation_col : str
        Column with ground truth annotation.
    positive_value : str
        Value in annotation_col that is the positive class.

    Returns
    -------
    float
        ROC-AUC score, or NaN if not computable.
    """
    valid = df.dropna(subset=[pseudotime_col, annotation_col])
    valid = valid[valid[annotation_col] != ""]
    if len(valid) == 0:
        return np.nan

    y_true = (valid[annotation_col] == positive_value).astype(int).to_numpy()
    y_score = valid[pseudotime_col].to_numpy()

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_score))


def onset_concordance(
    df: pd.DataFrame,
    pseudotime_col: str = "pseudotime",
    annotation_col: str = "infection_state",
    positive_value: str = "infected",
    min_track_timepoints: int = 3,
) -> tuple[float, int]:
    """Spearman correlation between DTW-derived and annotation-derived onset times.

    For each track, onset is defined as the first timepoint where the signal
    transitions to positive. Computes correlation across all tracks that have
    a detectable onset in both DTW pseudotime and annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Must have pseudotime_col, annotation_col, fov_name, track_id, t columns.
    pseudotime_col : str
        Column with DTW pseudotime values.
    annotation_col : str
        Column with ground truth annotation.
    positive_value : str
        Positive value in annotation_col.
    min_track_timepoints : int
        Minimum timepoints per track to include.

    Returns
    -------
    tuple[float, int]
        (Spearman rho, n_tracks) or (NaN, 0) if not computable.
    """
    valid = df.dropna(subset=[pseudotime_col, annotation_col])
    valid = valid[valid[annotation_col] != ""]

    dtw_onsets = []
    ann_onsets = []

    for (fov, tid), track in valid.groupby(["fov_name", "track_id"]):
        if len(track) < min_track_timepoints:
            continue
        track = track.sort_values("t")

        # Annotation onset: first timepoint with positive value
        ann_positive = track[track[annotation_col] == positive_value]
        if len(ann_positive) == 0:
            continue
        ann_onset_t = ann_positive["t"].iloc[0]

        # DTW onset: first timepoint where pseudotime exceeds median of track
        pt = track[pseudotime_col].to_numpy()
        threshold = np.median(pt)
        above = track[track[pseudotime_col] > threshold]
        if len(above) == 0:
            continue
        dtw_onset_t = above["t"].iloc[0]

        dtw_onsets.append(dtw_onset_t)
        ann_onsets.append(ann_onset_t)

    if len(dtw_onsets) < 3:
        return np.nan, len(dtw_onsets)

    rho, _ = spearmanr(dtw_onsets, ann_onsets)
    return float(rho), len(dtw_onsets)


def per_timepoint_auc(
    df: pd.DataFrame,
    pseudotime_col: str = "pseudotime",
    annotation_col: str = "infection_state",
    positive_value: str = "infected",
    time_col: str = "t",
) -> pd.DataFrame:
    """ROC-AUC of pseudotime predicting annotation at each timepoint.

    Parameters
    ----------
    df : pd.DataFrame
        Must have pseudotime_col, annotation_col, time_col columns.
    pseudotime_col : str
        Column with DTW pseudotime values.
    annotation_col : str
        Column with ground truth annotation.
    positive_value : str
        Positive value in annotation_col.
    time_col : str
        Timepoint column.

    Returns
    -------
    pd.DataFrame
        Columns: t, auc, n_cells, n_positive.
    """
    valid = df.dropna(subset=[pseudotime_col, annotation_col])
    valid = valid[valid[annotation_col] != ""]

    rows = []
    for t_val, group in valid.groupby(time_col):
        y_true = (group[annotation_col] == positive_value).astype(int).to_numpy()
        y_score = group[pseudotime_col].to_numpy()
        n_pos = int(y_true.sum())

        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(y_true, y_score))

        rows.append({"t": t_val, "auc": auc, "n_cells": len(group), "n_positive": n_pos})

    return pd.DataFrame(rows)


def _pseudotime_ap(
    df: pd.DataFrame,
    pseudotime_col: str = "pseudotime",
    annotation_col: str = "infection_state",
    positive_value: str = "infected",
) -> float:
    """Average precision (AUPRC) of pseudotime predicting a binary annotation.

    Parameters
    ----------
    df : pd.DataFrame
        Must have pseudotime_col and annotation_col columns.
    pseudotime_col : str
        Column with DTW pseudotime values.
    annotation_col : str
        Column with ground truth annotation.
    positive_value : str
        Value in annotation_col that is the positive class.

    Returns
    -------
    float
        Average precision score, or NaN if not computable.
    """
    valid = df.dropna(subset=[pseudotime_col, annotation_col])
    valid = valid[valid[annotation_col] != ""]
    if len(valid) == 0:
        return np.nan

    y_true = (valid[annotation_col] == positive_value).astype(int).to_numpy()
    y_score = valid[pseudotime_col].to_numpy()

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(average_precision_score(y_true, y_score))


def evaluate_embedding(
    alignments: pd.DataFrame,
    annotations: pd.DataFrame,
    embedding_name: str,
    dataset_id: str,
) -> dict:
    """Run full evaluation suite for one embedding × dataset.

    Parameters
    ----------
    alignments : pd.DataFrame
        Output of alignment_results_to_dataframe (has pseudotime, fov_name,
        track_id, t columns).
    annotations : pd.DataFrame
        Annotation CSV with fov_name, track_id, t, infection_state,
        organelle_state columns.
    embedding_name : str
        Name of the embedding (e.g., "sensor", "organelle", "phase").
    dataset_id : str
        Dataset identifier.

    Returns
    -------
    dict
        Summary metrics for this embedding × dataset.
    """
    # Merge alignments with annotations
    merge_keys = ["fov_name", "track_id", "t"]
    merged = alignments.merge(
        annotations[merge_keys + ["infection_state", "organelle_state"]], on=merge_keys, how="left"
    )

    result = {
        "embedding": embedding_name,
        "dataset_id": dataset_id,
        "n_cells": len(merged),
        "n_tracks": merged.groupby(["fov_name", "track_id"]).ngroup().nunique(),
    }

    # Infection state AUC + AP
    result["infection_auc"] = pseudotime_vs_annotation_auc(
        merged, pseudotime_col="pseudotime", annotation_col="infection_state", positive_value="infected"
    )
    result["infection_ap"] = _pseudotime_ap(
        merged, pseudotime_col="pseudotime", annotation_col="infection_state", positive_value="infected"
    )

    # Organelle state AUC + AP
    result["organelle_auc"] = pseudotime_vs_annotation_auc(
        merged, pseudotime_col="pseudotime", annotation_col="organelle_state", positive_value="remodel"
    )
    result["organelle_ap"] = _pseudotime_ap(
        merged, pseudotime_col="pseudotime", annotation_col="organelle_state", positive_value="remodel"
    )

    # Onset concordance (infection)
    rho, n_tracks = onset_concordance(
        merged, pseudotime_col="pseudotime", annotation_col="infection_state", positive_value="infected"
    )
    result["infection_onset_spearman"] = rho
    result["infection_onset_n_tracks"] = n_tracks

    # Onset concordance (organelle)
    rho_org, n_tracks_org = onset_concordance(
        merged, pseudotime_col="pseudotime", annotation_col="organelle_state", positive_value="remodel"
    )
    result["organelle_onset_spearman"] = rho_org
    result["organelle_onset_n_tracks"] = n_tracks_org

    # Mean DTW cost
    if "dtw_cost" in alignments.columns:
        per_track_cost = alignments.groupby(["fov_name", "track_id"])["dtw_cost"].first()
        result["mean_dtw_cost"] = float(per_track_cost.mean())
        result["median_dtw_cost"] = float(per_track_cost.median())

    _logger.info(
        "%s/%s: infection_auc=%.3f ap=%.3f, organelle_auc=%.3f ap=%.3f, onset_rho=%.3f (%d tracks)",
        embedding_name,
        dataset_id,
        result.get("infection_auc", np.nan),
        result.get("infection_ap", np.nan),
        result.get("organelle_auc", np.nan),
        result.get("organelle_ap", np.nan),
        result.get("infection_onset_spearman", np.nan),
        result.get("infection_onset_n_tracks", 0),
    )

    return result
