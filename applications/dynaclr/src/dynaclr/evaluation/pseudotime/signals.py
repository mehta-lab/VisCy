"""Per-cell signal extraction for pseudotime analysis.

Three signal extraction modes that all produce a common "signal" column:
1. Annotation-based: binary from human annotations
2. Prediction-based: binary/continuous from classifier predictions
3. Embedding distance: continuous cosine distance from baseline

Ported from:
- .ed_planning/tmp/scripts/annotation_remodling.py (annotation signal)
- .ed_planning/tmp/scripts/multi_organelle_remodeling.py (embedding distance)
- Conventions from viscy_utils/evaluation/linear_classifier.py (predictions)
"""

from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

_logger = logging.getLogger(__name__)


def extract_annotation_signal(
    df: pd.DataFrame,
    state_col: str = "organelle_state",
    positive_value: str = "remodel",
) -> pd.DataFrame:
    """Extract binary signal from human annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned dataframe with the annotation column.
    state_col : str
        Column containing the annotation state.
    positive_value : str
        Value in state_col that indicates the positive state.

    Returns
    -------
    pd.DataFrame
        Copy of df with added "signal" column (1.0 for positive, 0.0 for
        negative, NaN where state_col is NaN).
    """
    result = df.copy()
    result["signal"] = np.where(
        result[state_col].isna(),
        np.nan,
        (result[state_col] == positive_value).astype(float),
    )
    return result


def extract_prediction_signal(
    adata: ad.AnnData,
    aligned_df: pd.DataFrame,
    task: str = "organelle_state",
    positive_value: str = "remodel",
    use_probability: bool = False,
) -> pd.DataFrame:
    """Extract signal from classifier predictions stored in AnnData.

    Reads ``predicted_{task}`` from adata.obs for binary labels, or
    ``predicted_{task}_proba`` from adata.obsm for continuous probabilities.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with predictions in .obs[f"predicted_{task}"] and optionally
        probabilities in .obsm[f"predicted_{task}_proba"].
    aligned_df : pd.DataFrame
        Aligned dataframe (output of alignment.align_tracks). Must share
        index alignment with adata (fov_name, track_id, t).
    task : str
        Classification task name (used to look up predicted_{task} columns).
    positive_value : str
        Class label for the positive state.
    use_probability : bool
        If True, use prediction probability for the positive class as a
        continuous signal instead of binary predicted label.

    Returns
    -------
    pd.DataFrame
        Copy of aligned_df with added "signal" column.
    """
    pred_col = f"predicted_{task}"
    if pred_col not in adata.obs.columns:
        raise KeyError(f"Column '{pred_col}' not found in adata.obs. Run apply_linear_classifier first.")

    result = aligned_df.copy()

    # Build a lookup from adata.obs keyed by (fov_name, track_id, t)
    obs = adata.obs.copy()
    obs_key = obs.set_index(["fov_name", "track_id", "t"])

    result_key = result.set_index(["fov_name", "track_id", "t"])

    # Match rows
    common_idx = result_key.index.intersection(obs_key.index)
    _logger.info(f"Matched {len(common_idx)}/{len(result)} rows between aligned_df and adata")

    if use_probability:
        proba_key = f"predicted_{task}_proba"
        classes_key = f"predicted_{task}_classes"
        if proba_key not in adata.obsm:
            raise KeyError(f"'{proba_key}' not found in adata.obsm. Ensure classifier was run with probability output.")
        classes = adata.uns[classes_key]
        pos_idx = list(classes).index(positive_value)
        proba_matrix = adata.obsm[proba_key]

        # Map probabilities via obs index
        obs["_proba_positive"] = proba_matrix[:, pos_idx]
        obs_lookup = obs.set_index(["fov_name", "track_id", "t"])["_proba_positive"]
        result["signal"] = np.nan
        matched = result_key.index.isin(common_idx)
        result.loc[matched, "signal"] = obs_lookup.reindex(result_key.index[matched]).values
    else:
        obs_lookup = obs.set_index(["fov_name", "track_id", "t"])[pred_col]
        predictions = obs_lookup.reindex(result_key.index)
        result["signal"] = np.where(
            predictions.isna().values,
            np.nan,
            (predictions.values == positive_value).astype(float),
        )

    return result


def extract_embedding_distance(
    adata: ad.AnnData,
    aligned_df: pd.DataFrame,
    baseline_method: Literal["per_track", "control_well"] = "per_track",
    baseline_window_minutes: tuple[float, float] = (-240, -180),
    control_fov_pattern: str | None = None,
    distance_metric: str = "cosine",
    pca_n_components: int | None = None,
    min_baseline_frames: int = 2,
) -> pd.DataFrame:
    """Compute embedding distance from baseline for each cell.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with embeddings in .X.
    aligned_df : pd.DataFrame
        Aligned dataframe (output of alignment.align_tracks) with
        t_relative_minutes column.
    baseline_method : {"per_track", "control_well"}
        - "per_track": mean embedding in baseline_window per track/lineage.
        - "control_well": mean embedding from control FOV wells.
    baseline_window_minutes : tuple[float, float]
        (start, end) in minutes relative to T_perturb for per_track baseline.
    control_fov_pattern : str or None
        FOV pattern for control wells. Required when baseline_method="control_well".
    distance_metric : str
        Distance metric for scipy.spatial.distance.cdist (default: "cosine").
    pca_n_components : int or None
        If set, project embeddings to this many PCA components before computing
        distances.
    min_baseline_frames : int
        Minimum number of frames required in the baseline window per track.

    Returns
    -------
    pd.DataFrame
        Copy of aligned_df with added "signal" column (distance values).
    """
    result = aligned_df.copy()

    # Build index mapping from (fov_name, track_id, t) to adata row index
    obs = adata.obs.copy()
    obs["_adata_idx"] = np.arange(len(obs))
    obs_lookup = obs.set_index(["fov_name", "track_id", "t"])["_adata_idx"]

    result_key = result.set_index(["fov_name", "track_id", "t"])
    common_idx = result_key.index.intersection(obs_lookup.index)

    adata_indices = obs_lookup.reindex(common_idx).values.astype(int)
    result_row_mask = result_key.index.isin(common_idx)
    result_rows = np.where(result_row_mask)[0]

    _logger.info(f"Matched {len(common_idx)}/{len(result)} rows between aligned_df and adata")

    # Get embedding matrix for matched rows
    embeddings = adata.X[adata_indices]
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)

    # Get control embeddings if needed
    control_embeddings = None
    if baseline_method == "control_well" or pca_n_components is not None:
        if control_fov_pattern is not None:
            ctrl_mask = adata.obs["fov_name"].astype(str).str.contains(control_fov_pattern, regex=True)
            ctrl_emb = adata.X[ctrl_mask.values]
            if not isinstance(ctrl_emb, np.ndarray):
                ctrl_emb = np.asarray(ctrl_emb)
            if len(ctrl_emb) > 0:
                control_embeddings = ctrl_emb
                _logger.info(f"Control baseline: {len(ctrl_emb)} cells from '{control_fov_pattern}'")

    # Optional PCA projection
    if pca_n_components is not None:
        pca = PCA(n_components=pca_n_components)
        if control_embeddings is not None:
            all_emb = np.vstack([control_embeddings, embeddings])
            all_pca = pca.fit_transform(all_emb)
            control_embeddings = all_pca[: len(control_embeddings)]
            embeddings = all_pca[len(control_embeddings) :]
        else:
            embeddings = pca.fit_transform(embeddings)
        _logger.info(
            f"PCA: {pca_n_components} components, {pca.explained_variance_ratio_.sum() * 100:.1f}% variance explained"
        )

    # Build a local DataFrame for distance computation
    local_df = result.iloc[result_rows].copy()
    local_df["_emb_idx"] = np.arange(len(local_df))

    # Compute distances
    distances = np.full(len(local_df), np.nan)

    if baseline_method == "control_well":
        if control_embeddings is None:
            raise ValueError("baseline_method='control_well' requires control_fov_pattern that matches cells in adata.")
        baseline = control_embeddings.mean(axis=0, keepdims=True)
        distances = cdist(embeddings, baseline, metric=distance_metric).flatten()

    elif baseline_method == "per_track":
        for _, group in local_df.groupby(["fov_name", "track_id"]):
            group_emb_idx = group["_emb_idx"].values

            # Find baseline frames
            bl_mask = (group["t_relative_minutes"] >= baseline_window_minutes[0]) & (
                group["t_relative_minutes"] <= baseline_window_minutes[1]
            )

            if bl_mask.sum() < min_baseline_frames:
                # Fall back to control baseline if available
                if control_embeddings is not None:
                    baseline = control_embeddings.mean(axis=0, keepdims=True)
                else:
                    continue
            else:
                bl_idx = group.loc[bl_mask, "_emb_idx"].values
                baseline = embeddings[bl_idx].mean(axis=0, keepdims=True)

            track_emb = embeddings[group_emb_idx]
            track_dist = cdist(track_emb, baseline, metric=distance_metric).flatten()
            distances[group_emb_idx] = track_dist

    # Write distances back to result
    result["signal"] = np.nan
    result.iloc[result_rows, result.columns.get_loc("signal")] = distances

    n_valid = result["signal"].notna().sum()
    _logger.info(f"Computed distances for {n_valid}/{len(result)} cells")

    return result
