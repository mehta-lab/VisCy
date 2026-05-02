"""DTW-based pseudotime alignment for cellular dynamics.

Aligns cell trajectories to a template infection response using Dynamic
Time Warping (DTW). The template is built from annotated transitioning
cells via DBA (DTW Barycenter Averaging), then all cells are warped
onto it to produce pseudotime values in [0, 1].

Preprocessing pipeline: per-experiment z-score -> PCA -> L2-normalize -> DTW.
"""

from __future__ import annotations

import logging
import uuid
from typing import NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
from dtaidistance import dtw, dtw_ndim
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

_logger = logging.getLogger(__name__)

DEFAULT_POSITIVE_CLASSES: dict[str, str] = {
    "infection_state": "infected",
    "organelle_state": "remodel",
}
"""Default mapping of label column to positive class.

Used when the caller does not pass ``positive_classes`` explicitly. Keys
are ``obs`` columns whose entries are categorical strings; values are
the entry that should be treated as the positive class for downstream
binarization. Override per call when working with non-infection labels
(e.g. ``{"cell_division_state": "mitosis"}`` for ALFI).
"""
"""Default mapping of label column to positive class.

Used when the caller does not pass ``positive_classes`` explicitly. Keys
are ``obs`` columns whose entries are categorical strings; values are
the entry that should be treated as the positive class for downstream
binarization. Override per call when working with non-infection labels
(e.g. ``{"cell_division_state": "mitosis"}`` for ALFI).
"""


class TemplateResult(NamedTuple):
    """Result of building an infection response template."""

    template: np.ndarray
    template_id: str
    pca: PCA | None
    zscore_params: dict[str, tuple[np.ndarray, np.ndarray]]
    template_cell_ids: list[tuple[str, str, int]]
    n_input_tracks: int
    explained_variance: float | None
    template_labels: dict[str, np.ndarray] | None  # {col: (T,) fraction} per label column
    time_calibration: np.ndarray | None = None  # (T,) mean t_relative_minutes per template position


class AlignmentResult(NamedTuple):
    """DTW alignment result for a single cell track.

    ``length_normalized_cost`` and ``path_skew`` are scalar gating
    signals computed from the warp path. Per discussion §3.8, path_skew
    is the primary gate (rejects degenerate non-diagonal warps) and
    length_normalized_cost is the secondary gate (stereotypy filter).
    Both are surfaced in the alignment parquet so robustness checks
    can sweep gate thresholds without re-running DTW.
    """

    cell_uid: str
    dataset_id: str
    fov_name: str
    track_id: int
    timepoints: np.ndarray
    pseudotime: np.ndarray
    dtw_cost: float
    length_normalized_cost: float
    path_skew: float
    warping_path: np.ndarray
    warping_speed: np.ndarray
    propagated_labels: dict[str, np.ndarray] | None  # {col: (T,) fraction} per label column
    alignment_region: np.ndarray  # per-frame: "pre", "aligned", or "post"


def _zscore_embeddings(
    embeddings_dict: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Per-experiment z-score normalization.

    Parameters
    ----------
    embeddings_dict : dict[str, np.ndarray]
        {dataset_id: (N, D) embedding array}.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]
        Z-scored embeddings and per-experiment (mean, std) params.
    """
    zscored = {}
    params = {}
    for dataset_id, emb in embeddings_dict.items():
        mean = emb.mean(axis=0)
        std = emb.std(axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        zscored[dataset_id] = (emb - mean) / std
        params[dataset_id] = (mean, std)
    return zscored, params


def _preprocess_embeddings(
    embeddings: np.ndarray,
    pca: PCA | None = None,
) -> np.ndarray:
    """PCA transform + L2 normalize.

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) array, already z-scored.
    pca : PCA or None
        Fitted PCA model. If None, skip dimensionality reduction.

    Returns
    -------
    np.ndarray
        (N, D') L2-normalized embeddings.
    """
    if pca is not None:
        embeddings = pca.transform(embeddings)
    return normalize(embeddings, norm="l2", axis=1)


def _extract_track_trajectories(
    adata: ad.AnnData,
    df: pd.DataFrame,
    min_track_timepoints: int = 3,
    crop_window: int | None = None,
    label_cols: list[str] | None = None,
    positive_classes: dict[str, str] | None = None,
) -> list[tuple[str, int, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]]:
    """Extract per-track embedding trajectories from AnnData.

    Parameters
    ----------
    adata : ad.AnnData
        Embeddings with obs containing fov_name, track_id, t.
    df : pd.DataFrame
        Filtered tracking DataFrame (used for valid track selection).
        Must have t_perturb column if crop_window is set.
    min_track_timepoints : int
        Minimum timepoints per track (applied after cropping).
    crop_window : int or None
        If set, crop each track to [t_perturb - crop_window, t_perturb + crop_window].
        Requires t_perturb column in df. None = use full track.
    label_cols : list[str] or None
        Label columns to extract (e.g., ["infection_state", "organelle_state"]).
        Each is binarized using ``positive_classes``.
    positive_classes : dict[str, str] or None
        Mapping from label column name to its positive class value.
        Required when ``label_cols`` is provided.

    Returns
    -------
    list[tuple[str, int, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]]
        Each element: (fov_name, track_id, embeddings (T, D), timepoints (T,),
        labels {col: (T,)} or None).
    """
    valid_tracks = df.groupby(["fov_name", "track_id"]).filter(lambda x: len(x) >= min_track_timepoints)
    valid_keys = set(zip(valid_tracks["fov_name"], valid_tracks["track_id"]))

    # Build t_perturb lookup if cropping
    t_perturb_lookup: dict[tuple[str, int], int] = {}
    if crop_window is not None:
        if "t_perturb" not in df.columns:
            raise ValueError("crop_window requires t_perturb column in df")
        for (fov, tid), grp in df.groupby(["fov_name", "track_id"]):
            t_perturb_lookup[(fov, tid)] = int(grp["t_perturb"].iloc[0])

    # Build label lookups per column
    label_lookups: dict[str, dict[tuple, int]] = {}
    if label_cols:
        if positive_classes is None:
            raise ValueError("positive_classes is required when label_cols is set")
        for col in label_cols:
            if col not in df.columns:
                continue
            if col not in positive_classes:
                raise KeyError(f"positive_classes is missing entry for label column {col!r}")
            positive_val = positive_classes[col]
            lookup: dict[tuple, int] = {}
            for _, row in df.iterrows():
                val = row[col]
                if pd.notna(val) and val != "":
                    lookup[(row["fov_name"], row["track_id"], int(row["t"]))] = 1 if val == positive_val else 0
            label_lookups[col] = lookup

    obs = adata.obs.copy()
    obs["_iloc"] = np.arange(len(obs))
    trajectories = []
    for (fov_name, track_id), group in obs.groupby(["fov_name", "track_id"]):
        if (fov_name, track_id) not in valid_keys:
            continue
        sorted_group = group.sort_values("t")

        # Crop around t_perturb if requested
        if crop_window is not None and (fov_name, track_id) in t_perturb_lookup:
            tp = t_perturb_lookup[(fov_name, track_id)]
            t_vals = sorted_group["t"].to_numpy()
            mask = (t_vals >= tp - crop_window) & (t_vals <= tp + crop_window)
            sorted_group = sorted_group.iloc[mask]

        if len(sorted_group) < min_track_timepoints:
            continue

        iloc_indices = sorted_group["_iloc"].to_numpy()
        emb = adata.X[iloc_indices]
        if hasattr(emb, "toarray"):
            emb = emb.toarray()
        timepoints = sorted_group["t"].to_numpy().astype(int)

        labels = None
        if label_lookups:
            labels = {}
            for col, lookup in label_lookups.items():
                labels[col] = np.array(
                    [lookup.get((fov_name, track_id, int(t)), 0) for t in timepoints], dtype=np.float64
                )

        trajectories.append((str(fov_name), int(track_id), np.asarray(emb, dtype=np.float64), timepoints, labels))

    return trajectories


def _dba(
    sequences: list[np.ndarray],
    max_iter: int = 30,
    tol: float = 1e-5,
    init: str = "medoid",
    random_state: int = 42,
) -> np.ndarray:
    """DTW Barycenter Averaging (DBA).

    Parameters
    ----------
    sequences : list[np.ndarray]
        List of (T_i, D) sequences.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on mean absolute change.
    init : str
        Initialization method. "medoid" selects the sequence with
        lowest total DTW cost to all others.
    random_state : int
        Seed for medoid candidate subsampling. Default 42.

    Returns
    -------
    np.ndarray
        (T_avg, D) template sequence.
    """
    if len(sequences) == 0:
        raise ValueError("No sequences provided for DBA.")

    if init == "medoid":
        n = len(sequences)
        # Subsample for medoid if too many sequences (O(n²) DTW calls)
        max_medoid_candidates = 50
        if n > max_medoid_candidates:
            rng = np.random.default_rng(random_state)
            candidate_idx = rng.choice(n, max_medoid_candidates, replace=False)
            _logger.info("DBA medoid init: subsampling %d/%d candidates", max_medoid_candidates, n)
        else:
            candidate_idx = np.arange(n)
        costs = np.zeros(len(candidate_idx))
        for ci, i in enumerate(candidate_idx):
            for j in range(n):
                if i != j:
                    costs[ci] += dtw_ndim.distance(sequences[i], sequences[j])
        avg = sequences[int(candidate_idx[np.argmin(costs)])].copy()
    else:
        avg = sequences[0].copy()

    for iteration in range(max_iter):
        n_frames = avg.shape[0]
        n_dims = avg.shape[1]
        accum = np.zeros((n_frames, n_dims))
        counts = np.zeros(n_frames)

        for seq in sequences:
            _, paths = dtw_ndim.warping_paths(avg, seq)
            path = dtw.best_path(paths)
            for idx_avg, idx_seq in path:
                accum[idx_avg] += seq[idx_seq]
                counts[idx_avg] += 1

        counts = np.maximum(counts, 1)
        new_avg = accum / counts[:, np.newaxis]
        change = np.mean(np.abs(new_avg - avg))

        _logger.debug(f"DBA iteration {iteration + 1}: mean change = {change:.6f}")
        avg = new_avg

        if change < tol:
            _logger.info(f"DBA converged at iteration {iteration + 1} (change={change:.2e})")
            break

    return avg


def build_template(
    adata_dict: dict[str, ad.AnnData],
    aligned_df_dict: dict[str, pd.DataFrame],
    pca_n_components: int | None = 20,
    pca_variance_threshold: float | None = None,
    dba_max_iter: int = 30,
    dba_tol: float = 1e-5,
    dba_init: str = "medoid",
    control_adata_dict: dict[str, ad.AnnData] | None = None,
    crop_window: int | dict[str, int] | None = None,
    positive_classes: dict[str, str] | None = None,
    random_state: int = 42,
) -> TemplateResult:
    """Build a DTW pseudotime template from annotated single-cell trajectories.

    Generic over the underlying biology — what was previously called
    ``build_infection_template`` works for any anchored event (infection
    onset, mitotic entry, immune activation) provided the caller supplies
    the appropriate label-to-positive-class mapping via ``positive_classes``.

    Parameters
    ----------
    adata_dict : dict[str, ad.AnnData]
        {dataset_id: adata} with embeddings for the cells used to build
        the template.
    aligned_df_dict : dict[str, pd.DataFrame]
        {dataset_id: aligned_df} with t_perturb assigned.
    pca_n_components : int or None
        Number of PCA components. Ignored if pca_variance_threshold is set.
    pca_variance_threshold : float or None
        If set, auto-select components to explain this variance fraction.
    dba_max_iter : int
        Max DBA iterations.
    dba_tol : float
        DBA convergence tolerance.
    dba_init : str
        DBA initialization ("medoid").
    control_adata_dict : dict[str, ad.AnnData] | None
        Control embeddings per dataset, included in PCA fitting.
    crop_window : int or dict[str, int] or None
        If set, crop each track to [t_perturb - crop_window, t_perturb + crop_window]
        before DBA. Produces a shorter template centered on the anchored
        event. Pass a dict to use per-dataset crop windows (e.g. when
        datasets have different frame intervals and crop_window was
        derived from a fixed duration in minutes). None = use full
        tracks (variable length).
    positive_classes : dict[str, str] or None
        Mapping from label column name to its positive-class value.
        Defaults to :data:`DEFAULT_POSITIVE_CLASSES` (infection labels).
        Pass ``{"cell_division_state": "mitosis"}`` for ALFI division
        templates.
    random_state : int
        Seed for reproducible PCA / medoid subsampling. Default 42.

    Returns
    -------
    TemplateResult
        Template array, PCA model, z-score params, and metadata.
    """
    if positive_classes is None:
        positive_classes = DEFAULT_POSITIVE_CLASSES
    raw_embeddings = {}
    for dataset_id, adata in adata_dict.items():
        emb = adata.X
        if hasattr(emb, "toarray"):
            emb = emb.toarray()
        raw_embeddings[dataset_id] = np.asarray(emb, dtype=np.float64)

    if control_adata_dict is not None:
        for dataset_id, adata in control_adata_dict.items():
            ctrl_key = f"{dataset_id}__control"
            emb = adata.X
            if hasattr(emb, "toarray"):
                emb = emb.toarray()
            raw_embeddings[ctrl_key] = np.asarray(emb, dtype=np.float64)

    zscored, zscore_params = _zscore_embeddings(raw_embeddings)

    all_zscored = np.concatenate(list(zscored.values()), axis=0)
    use_pca = pca_n_components is not None or pca_variance_threshold is not None
    pca = None
    explained_variance = None

    if use_pca:
        if pca_variance_threshold is not None:
            pca = PCA(n_components=pca_variance_threshold, svd_solver="full", random_state=random_state)
        else:
            n_comp = min(pca_n_components, all_zscored.shape[1], all_zscored.shape[0])
            pca = PCA(n_components=n_comp, random_state=random_state)
        pca.fit(all_zscored)
        explained_variance = float(np.sum(pca.explained_variance_ratio_))
        _logger.info(f"PCA: {pca.n_components_} components explain {explained_variance:.1%} variance")

    clean_zscore_params = {k: v for k, v in zscore_params.items() if "__control" not in k}

    trajectories = []
    track_labels: list[dict[str, np.ndarray] | None] = []
    track_t_rels: list[np.ndarray] = []
    cell_ids: list[tuple[str, str, int]] = []

    # Detect which label columns are available across all datasets
    label_cols = [col for col in positive_classes if any(col in df.columns for df in aligned_df_dict.values())]
    label_cols_or_none = label_cols if label_cols else None

    for dataset_id, adata in adata_dict.items():
        df = aligned_df_dict[dataset_id]
        ds_zscored_emb = zscored[dataset_id]

        zscored_adata = ad.AnnData(X=ds_zscored_emb, obs=adata.obs.copy())
        zscored_adata.obs.index = adata.obs.index

        # Build t_relative_minutes lookup for this dataset
        t_rel_lookup: dict[tuple[str, int, int], float] = {}
        if "t_relative_minutes" in df.columns:
            for _, row in df.iterrows():
                t_rel_lookup[(str(row["fov_name"]), int(row["track_id"]), int(row["t"]))] = float(
                    row["t_relative_minutes"]
                )

        ds_crop_window = crop_window[dataset_id] if isinstance(crop_window, dict) else crop_window
        tracks = _extract_track_trajectories(
            zscored_adata,
            df,
            min_track_timepoints=1,
            crop_window=ds_crop_window,
            label_cols=label_cols_or_none,
            positive_classes=positive_classes,
        )
        for fov_name, track_id, emb, timepoints, labels in tracks:
            processed = _preprocess_embeddings(emb, pca=pca)
            trajectories.append(processed)
            track_labels.append(labels)
            cell_ids.append((dataset_id, fov_name, track_id))
            t_rel = np.array([t_rel_lookup.get((fov_name, track_id, int(t)), np.nan) for t in timepoints])
            track_t_rels.append(t_rel)

    if len(trajectories) == 0:
        raise ValueError("No valid trajectories found for template building.")

    _logger.info(f"Building template from {len(trajectories)} trajectories")
    template = _dba(trajectories, max_iter=dba_max_iter, tol=dba_tol, init=dba_init, random_state=random_state)
    template = normalize(template, norm="l2", axis=1)

    # Compute template labels and time calibration via DTW alignment back to template.
    # One DTW path per track; labels and t_relative_minutes mapped through the same path.
    n_template = template.shape[0]
    template_labels = None
    time_calibration = None

    has_labels = label_cols and all(lb is not None for lb in track_labels)
    has_t_rel = any(np.any(np.isfinite(t)) for t in track_t_rels)

    if has_labels or has_t_rel:
        label_sums = {col: np.zeros(n_template) for col in label_cols} if has_labels else {}
        label_counts = {col: np.zeros(n_template) for col in label_cols} if has_labels else {}
        time_sums = np.zeros(n_template)
        time_counts = np.zeros(n_template)

        for seq, labels_dict, t_rel_arr in zip(trajectories, track_labels, track_t_rels):
            _, paths = dtw_ndim.warping_paths(template, seq)
            path = dtw.best_path(paths)
            if has_labels and labels_dict is not None:
                for col in label_cols:
                    if col not in labels_dict:
                        continue
                    col_labels = labels_dict[col]
                    for idx_template, idx_seq in path:
                        if idx_seq < len(col_labels):
                            label_sums[col][idx_template] += col_labels[idx_seq]
                            label_counts[col][idx_template] += 1
            for idx_template, idx_seq in path:
                if idx_seq < len(t_rel_arr) and np.isfinite(t_rel_arr[idx_seq]):
                    time_sums[idx_template] += t_rel_arr[idx_seq]
                    time_counts[idx_template] += 1

        if has_labels:
            template_labels = {}
            for col in label_cols:
                counts = np.maximum(label_counts[col], 1)
                template_labels[col] = label_sums[col] / counts
                _logger.info(
                    "Template labels [%s]: %d positions, fraction range [%.2f, %.2f]",
                    col,
                    n_template,
                    template_labels[col].min(),
                    template_labels[col].max(),
                )

        if has_t_rel and time_counts.sum() > 0:
            raw_cal = np.where(time_counts > 0, time_sums / np.maximum(time_counts, 1), np.nan)
            # Interpolate any gaps linearly
            positions = np.arange(n_template)
            valid_mask = np.isfinite(raw_cal)
            if valid_mask.sum() >= 2:
                time_calibration = np.interp(positions, positions[valid_mask], raw_cal[valid_mask])
            elif valid_mask.sum() == 1:
                time_calibration = np.full(n_template, raw_cal[valid_mask][0])
            _logger.info(
                "Time calibration: %d positions, range [%.1f, %.1f] min",
                n_template,
                time_calibration.min(),
                time_calibration.max(),
            )

    return TemplateResult(
        template=template,
        template_id=str(uuid.uuid4()),
        pca=pca,
        zscore_params=clean_zscore_params,
        template_cell_ids=cell_ids,
        n_input_tracks=len(trajectories),
        explained_variance=explained_variance,
        template_labels=template_labels,
        time_calibration=time_calibration,
    )


def dtw_align_tracks(
    adata: ad.AnnData,
    df: pd.DataFrame,
    template_result: TemplateResult,
    dataset_id: str,
    min_track_timepoints: int = 3,
    psi: int | None = None,
    subsequence: bool = False,
) -> list[AlignmentResult]:
    """Align cell tracks to a template using DTW.

    Parameters
    ----------
    adata : ad.AnnData
        Embeddings with obs containing fov_name, track_id, t.
    df : pd.DataFrame
        Tracking DataFrame (optionally with t_perturb).
    template_result : TemplateResult
        Template from :func:`build_template`.
    dataset_id : str
        Identifier for this dataset.
    min_track_timepoints : int
        Minimum timepoints per track.
    psi : int or None
        Psi relaxation for DTW. If None, auto-computed:
        - subsequence=True: psi = max(track_len - template_len, 0)
        - subsequence=False: psi = template_len // 2
    subsequence : bool
        If True, use subsequence DTW: sweep the (short) template across
        the (long) cell track to find the best-matching segment.
        Frames before the matched region get pseudotime=0,
        frames after get pseudotime=1.
        Use this when the template was built with crop_window.

    Returns
    -------
    list[AlignmentResult]
        One result per aligned track.
    """
    emb = adata.X
    if hasattr(emb, "toarray"):
        emb = emb.toarray()
    emb = np.asarray(emb, dtype=np.float64)

    if dataset_id in template_result.zscore_params:
        mean, std = template_result.zscore_params[dataset_id]
    else:
        mean = emb.mean(axis=0)
        std = emb.std(axis=0)
        std = np.where(std < 1e-10, 1.0, std)
    emb_zscored = (emb - mean) / std

    zscored_adata = ad.AnnData(X=emb_zscored, obs=adata.obs.copy())
    zscored_adata.obs.index = adata.obs.index

    tracks = _extract_track_trajectories(zscored_adata, df, min_track_timepoints)
    template = template_result.template
    t_template = template.shape[0]

    results = []
    for fov_name, track_id, track_emb, timepoints, _labels in tracks:
        processed = _preprocess_embeddings(track_emb, pca=template_result.pca)
        n_track = len(processed)

        # Compute psi (must be < min(template_len, track_len))
        max_psi = min(n_track - 1, t_template - 1)
        if psi is not None:
            track_psi = min(psi, max_psi)
        elif subsequence:
            # Allow template to float anywhere within the track
            track_psi = max_psi
        else:
            track_psi = min(t_template // 2, max_psi)

        _, paths = dtw_ndim.warping_paths(template, processed, psi=track_psi)
        path = dtw.best_path(paths)
        path_arr = np.array(path)

        cost = paths[path_arr[-1, 0], path_arr[-1, 1]]

        # length-normalized cost: divide raw DTW cost by the warp path
        # length so longer matches don't accumulate more cost simply by
        # having more steps. This is the standard ranking signal for
        # subsequence DTW.
        if len(path_arr) > 0 and np.isfinite(cost):
            length_normalized_cost = float(cost) / float(len(path_arr))
        else:
            length_normalized_cost = float("inf")

        # Path skew: mean per-step deviation from the ideal diagonal in
        # the warp path's own coordinates. The ideal warp from
        # (0, 0) to (T-1, n-1) has slope (n-1)/(T-1); the diagonal at
        # warp-path step k is (template_step, query_step) =
        # (k * (T-1)/(K-1), k * (n-1)/(K-1)). Skew is the mean L1
        # normalized distance from each warp-path point to that ideal
        # diagonal point, divided by max(T, n) for [0, 1] scaling.
        if len(path_arr) >= 2 and t_template > 1 and n_track > 1:
            K = len(path_arr)
            ideal_t = np.linspace(path_arr[0, 0], path_arr[-1, 0], K)
            ideal_q = np.linspace(path_arr[0, 1], path_arr[-1, 1], K)
            dev = np.abs(path_arr[:, 0] - ideal_t) + np.abs(path_arr[:, 1] - ideal_q)
            denom = max(t_template, n_track)
            path_skew = float(dev.mean() / denom)
        else:
            path_skew = 0.0

        pseudotime = np.zeros(n_track)
        speed = np.zeros(n_track)
        alignment_region = np.full(n_track, "aligned", dtype=object)

        # Map each query frame to its template position
        # DTW path: (idx_template, idx_query) pairs
        # A query frame may appear multiple times; keep the last (highest) template position
        matched_template_pos = np.full(n_track, -1.0)
        for idx_template, idx_query in path:
            if idx_query < n_track:
                matched_template_pos[idx_query] = idx_template

        if subsequence and t_template > 1:
            # Find the matched region (query frames that got a template assignment)
            matched_mask = matched_template_pos >= 0
            if matched_mask.any():
                first_matched = np.argmax(matched_mask)
                last_matched = n_track - 1 - np.argmax(matched_mask[::-1])

                # Within matched region: pseudotime from template position
                for i in range(first_matched, last_matched + 1):
                    if matched_template_pos[i] >= 0:
                        pseudotime[i] = matched_template_pos[i] / (t_template - 1)

                # Forward-fill any gaps within the matched region
                for i in range(first_matched + 1, last_matched + 1):
                    if matched_template_pos[i] < 0:
                        pseudotime[i] = pseudotime[i - 1]

                # Before matched region: pseudotime = 0
                pseudotime[:first_matched] = 0.0
                # After matched region: pseudotime = 1
                pseudotime[last_matched + 1 :] = 1.0
                alignment_region[:first_matched] = "pre"
                alignment_region[last_matched + 1 :] = "post"
            else:
                pseudotime[:] = 0.0
                alignment_region[:] = "pre"
        elif t_template > 1:
            # Standard DTW: template position / (template_length - 1)
            template_positions = np.zeros(n_track)
            for idx_template, idx_query in path:
                if idx_query < n_track:
                    template_positions[idx_query] = idx_template
            pseudotime = template_positions / (t_template - 1)

        # Propagate template labels to cell frames via warping path
        propagated_labels = None
        if template_result.template_labels is not None:
            propagated_labels = {}
            for col, tl in template_result.template_labels.items():
                col_propagated = np.full(n_track, np.nan)
                for idx_template, idx_query in path:
                    if idx_query < n_track and idx_template < len(tl):
                        col_propagated[idx_query] = tl[idx_template]

                if subsequence:
                    matched_mask_lbl = matched_template_pos >= 0
                    if matched_mask_lbl.any():
                        first_m = np.argmax(matched_mask_lbl)
                        last_m = n_track - 1 - np.argmax(matched_mask_lbl[::-1])
                        for i in range(first_m + 1, last_m + 1):
                            if np.isnan(col_propagated[i]):
                                col_propagated[i] = col_propagated[i - 1]
                        col_propagated[:first_m] = 0.0
                        col_propagated[last_m + 1 :] = 1.0

                propagated_labels[col] = col_propagated

        # Compute warping speed (discrete derivative of pseudotime)
        for i in range(n_track):
            if i == 0:
                speed[i] = pseudotime[1] - pseudotime[0] if n_track > 1 else 0.0
            elif i == n_track - 1:
                speed[i] = pseudotime[i] - pseudotime[i - 1]
            else:
                speed[i] = (pseudotime[i + 1] - pseudotime[i - 1]) / 2

        cell_uid = f"{dataset_id}/{fov_name}/{track_id}"
        results.append(
            AlignmentResult(
                cell_uid=cell_uid,
                dataset_id=dataset_id,
                fov_name=fov_name,
                track_id=track_id,
                timepoints=timepoints,
                pseudotime=pseudotime,
                length_normalized_cost=length_normalized_cost,
                path_skew=path_skew,
                dtw_cost=float(cost),
                warping_path=path_arr,
                warping_speed=speed,
                propagated_labels=propagated_labels,
                alignment_region=alignment_region,
            )
        )

    _logger.info(f"Aligned {len(results)} tracks for dataset {dataset_id}")
    return results


def classify_response_groups(
    alignment_results: list[AlignmentResult] | pd.DataFrame,
    cost_percentile_threshold: float = 75.0,
    speed_clustering_method: str = "quantile",
    speed_quantile: float = 0.5,
) -> pd.DataFrame:
    """Classify aligned cells into response groups.

    Groups:
    - non_responder: DTW cost above percentile threshold
    - early_responder: responders with above-median mean warping speed
    - late_responder: responders with below-median mean warping speed

    Parameters
    ----------
    alignment_results : list[AlignmentResult] or pd.DataFrame
        Alignment results. If DataFrame, must have columns:
        cell_uid, dtw_cost, mean_warping_speed (or warping_speed).
    cost_percentile_threshold : float
        Percentile of DTW cost above which cells are non-responders.
    speed_clustering_method : str
        "quantile" or "kmeans" for splitting early/late.
    speed_quantile : float
        Quantile threshold for speed split (used when method="quantile").

    Returns
    -------
    pd.DataFrame
        One row per cell with columns: cell_uid, dataset_id,
        response_group, dtw_cost, mean_warping_speed.
    """
    if isinstance(alignment_results, pd.DataFrame):
        df = alignment_results.copy()
        if "mean_warping_speed" not in df.columns and "warping_speed" in df.columns:
            df["mean_warping_speed"] = df.groupby("cell_uid")["warping_speed"].transform("mean")
        per_cell = df.groupby("cell_uid").first().reset_index()
        records = []
        for _, row in per_cell.iterrows():
            records.append(
                {
                    "cell_uid": row["cell_uid"],
                    "dataset_id": row.get("dataset_id", ""),
                    "dtw_cost": row["dtw_cost"],
                    "mean_warping_speed": row["mean_warping_speed"],
                }
            )
    else:
        records = []
        for r in alignment_results:
            records.append(
                {
                    "cell_uid": r.cell_uid,
                    "dataset_id": r.dataset_id,
                    "dtw_cost": r.dtw_cost,
                    "mean_warping_speed": float(np.mean(np.abs(r.warping_speed))),
                }
            )

    df = pd.DataFrame(records)
    if len(df) == 0:
        df["response_group"] = pd.Series(dtype=str)
        return df

    cost_threshold = np.percentile(df["dtw_cost"], cost_percentile_threshold)
    df["response_group"] = "non_responder"

    responder_mask = df["dtw_cost"] <= cost_threshold
    responders = df[responder_mask]

    if len(responders) > 0:
        if speed_clustering_method == "quantile":
            speed_threshold = responders["mean_warping_speed"].quantile(speed_quantile)
            df.loc[responder_mask & (df["mean_warping_speed"] >= speed_threshold), "response_group"] = "early_responder"
            df.loc[responder_mask & (df["mean_warping_speed"] < speed_threshold), "response_group"] = "late_responder"
        elif speed_clustering_method == "kmeans":
            from sklearn.cluster import KMeans

            speeds = responders["mean_warping_speed"].to_numpy().reshape(-1, 1)
            if len(speeds) >= 2:
                km = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = km.fit_predict(speeds)
                cluster_means = [speeds[labels == c].mean() for c in range(2)]
                fast_cluster = int(np.argmax(cluster_means))
                resp_indices = responders.index
                for idx, label in zip(resp_indices, labels):
                    if label == fast_cluster:
                        df.loc[idx, "response_group"] = "early_responder"
                    else:
                        df.loc[idx, "response_group"] = "late_responder"
            else:
                df.loc[responder_mask, "response_group"] = "early_responder"

    _logger.info(
        f"Classification: {(df['response_group'] == 'early_responder').sum()} early, "
        f"{(df['response_group'] == 'late_responder').sum()} late, "
        f"{(df['response_group'] == 'non_responder').sum()} non-responder"
    )

    return df[["cell_uid", "dataset_id", "response_group", "dtw_cost", "mean_warping_speed"]]


def alignment_results_to_dataframe(
    results: list[AlignmentResult],
    template_id: str,
    time_calibration: np.ndarray | None = None,
) -> pd.DataFrame:
    """Flatten alignment results into a DataFrame (one row per timepoint).

    Parameters
    ----------
    results : list[AlignmentResult]
        Output of dtw_align_tracks.
    template_id : str
        Template UUID to attach.
    time_calibration : np.ndarray or None
        (T_template,) array mapping template position to mean t_relative_minutes.
        If provided, adds an ``estimated_t_rel_minutes`` column.

    Returns
    -------
    pd.DataFrame
        Columns: cell_uid, dataset_id, fov_name, track_id, t,
        pseudotime, dtw_cost, warping_speed, template_id,
        plus propagated_{label}_label for each label column,
        plus estimated_t_rel_minutes if time_calibration is provided.
    """
    rows = []
    for r in results:
        for i, t in enumerate(r.timepoints):
            row = {
                "cell_uid": r.cell_uid,
                "dataset_id": r.dataset_id,
                "fov_name": r.fov_name,
                "track_id": r.track_id,
                "t": int(t),
                "pseudotime": float(r.pseudotime[i]),
                "dtw_cost": r.dtw_cost,
                "length_normalized_cost": float(r.length_normalized_cost),
                "path_skew": float(r.path_skew),
                "warping_speed": float(r.warping_speed[i]),
                "alignment_region": r.alignment_region[i],
                "template_id": template_id,
            }
            if r.propagated_labels is not None:
                for col, arr in r.propagated_labels.items():
                    col_clean = col.replace("_state", "")
                    row[f"propagated_{col_clean}_label"] = float(arr[i])
            rows.append(row)
    df = pd.DataFrame(rows)
    if time_calibration is not None and len(df) > 0:
        T = len(time_calibration)
        df["estimated_t_rel_minutes"] = np.interp(
            df["pseudotime"].to_numpy() * (T - 1),
            np.arange(T),
            time_calibration,
        )
    return df


def extract_dtw_pseudotime(
    adata: ad.AnnData,
    df: pd.DataFrame,
    template_result: TemplateResult,
    dataset_id: str,
    min_track_timepoints: int = 3,
    cost_percentile_threshold: float = 75.0,
    speed_clustering_method: str = "quantile",
    speed_quantile: float = 0.5,
    psi: int | None = None,
) -> pd.DataFrame:
    """Run align + classify + flatten in one call (convenience wrapper).

    Parameters
    ----------
    adata : ad.AnnData
        Embeddings AnnData.
    df : pd.DataFrame
        Tracking DataFrame.
    template_result : TemplateResult
        Built template.
    dataset_id : str
        Dataset identifier.
    min_track_timepoints : int
        Minimum timepoints per track.
    cost_percentile_threshold : float
        Non-responder cost threshold percentile.
    speed_clustering_method : str
        "quantile" or "kmeans".
    speed_quantile : float
        Speed split quantile.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with pseudotime renamed to "signal" for metrics
        compatibility, plus dtw_cost, warping_speed, response_group columns.
    """
    results = dtw_align_tracks(adata, df, template_result, dataset_id, min_track_timepoints, psi=psi)
    flat = alignment_results_to_dataframe(
        results, template_result.template_id, time_calibration=template_result.time_calibration
    )
    classifications = classify_response_groups(
        results,
        cost_percentile_threshold=cost_percentile_threshold,
        speed_clustering_method=speed_clustering_method,
        speed_quantile=speed_quantile,
    )
    merged = flat.merge(classifications[["cell_uid", "response_group"]], on="cell_uid", how="left")
    merged = merged.rename(columns={"pseudotime": "signal"})
    return merged
