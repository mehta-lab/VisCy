"""Shared helpers for per-organelle Stage 3 readouts.

Each per-organelle script (readout_sec61, readout_g3bp1, readout_phase)
is a thin wrapper that:

1. Loads its alignment parquet (Path A-anno, A-LC, or B output).
2. Loads the matching organelle channel embedding zarr per dataset.
3. Computes per-cell cosine distance from a per-cell pre-baseline.
4. (G3BP1) Computes oscillation-aware metrics on the post-window.
5. Aggregates across cells per cohort with FOV-stratified mock as null.

The FOV-stratified mock null follows discussion §3.7 and the round 2
ML-engineer critique: pooled mock distributions inflate the 95th
percentile with FOV-to-FOV variance, not per-cell variance. We match
each productive cell to mocks from the same FOV and use that local
distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from dynaclr.pseudotime import date_prefix_from_dataset_id, find_embedding_zarr

_logger = logging.getLogger(__name__)

ALIGNMENT_DIRS = {
    "A-anno": "A-anno",
    "A-LC": "A-LC",
    "B": "B",
}


def load_alignment_parquet(
    align_root: Path,
    track: str,
    candidate_set: str,
) -> pd.DataFrame:
    """Load a Stage 2 alignment parquet for one track.

    Path B parquets carry a ``{template}_{flavor}_on_{candidate_set}.parquet``
    name; Path A parquets are ``{candidate_set}.parquet``. The caller can
    pass the full filename via ``--alignment-parquet`` to bypass this
    lookup.
    """
    if track not in ALIGNMENT_DIRS:
        raise KeyError(f"Unknown track {track!r}; expected one of {list(ALIGNMENT_DIRS)}")
    track_dir = align_root / ALIGNMENT_DIRS[track] / "alignments"
    if track in ("A-anno", "A-LC"):
        path = track_dir / f"{candidate_set}.parquet"
    else:
        # Path B: glob for any template/flavor matching this candidate set.
        matches = list(track_dir.glob(f"*_on_{candidate_set}.parquet"))
        if not matches:
            raise FileNotFoundError(f"No Path B parquet under {track_dir} for {candidate_set!r}")
        if len(matches) > 1:
            _logger.warning(f"Multiple Path B parquets for {candidate_set!r}; using {matches[0].name}")
        path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"Alignment parquet not found: {path}")
    return pd.read_parquet(path)


def load_organelle_embeddings(
    dataset_cfgs: dict[str, dict],
    datasets_in_use: list[str],
    embedding_pattern: str,
) -> dict[str, ad.AnnData]:
    """Load the organelle channel zarr per dataset, date-matched.

    Re-uses :func:`find_embedding_zarr` from the library. Returns one
    AnnData per dataset; keys missing here will skip downstream cells.
    """
    out: dict[str, ad.AnnData] = {}
    for ds_id in datasets_in_use:
        if ds_id not in dataset_cfgs:
            _logger.warning(f"dataset_id {ds_id!r} missing from datasets.yaml; skipping")
            continue
        ds_cfg = dataset_cfgs[ds_id]
        prefix = date_prefix_from_dataset_id(ds_id)
        try:
            zarr_path = find_embedding_zarr(ds_cfg["pred_dir"], prefix + embedding_pattern)
        except FileNotFoundError as exc:
            _logger.warning(f"[{ds_id}] no embedding zarr matched {prefix + embedding_pattern}: {exc}")
            continue
        adata = ad.read_zarr(zarr_path)
        adata.obs_names_make_unique()
        out[ds_id] = adata
        _logger.info(f"[{ds_id}] loaded {Path(zarr_path).name} ({adata.n_obs} cells)")
    return out


def per_cell_baseline_distance(
    align_df: pd.DataFrame,
    adata_by_dataset: dict[str, ad.AnnData],
    baseline_window_minutes: tuple[float, float] = (-240, -60),
    metric: str = "cosine",
    min_baseline_frames: int = 2,
) -> pd.DataFrame:
    """Compute per-cell cosine distance from a per-cell pre-baseline.

    For each ``(dataset_id, fov_name, track_id)`` in the alignment
    parquet, fetches the matching frames from the organelle AnnData,
    computes a per-track baseline as the mean of pre-window embeddings,
    then writes ``signal = cosine_distance(embedding, baseline)`` per
    frame.

    Cells missing from the embedding zarr or with fewer than
    ``min_baseline_frames`` baseline frames produce NaN signal.

    Returns a copy of ``align_df`` with a new ``signal`` column.
    """
    from scipy.spatial.distance import cdist

    out = align_df.copy()
    out["signal"] = np.nan

    for ds_id, ds_group in out.groupby("dataset_id"):
        if ds_id not in adata_by_dataset:
            continue
        adata = adata_by_dataset[ds_id]
        obs = adata.obs[["fov_name", "track_id", "t"]].copy()
        obs["fov_name"] = obs["fov_name"].astype(str)
        obs["track_id"] = obs["track_id"].astype(int)
        obs["t"] = obs["t"].astype(int)
        obs["_idx"] = np.arange(len(obs))

        # Per-track loop. Vectorising would require careful indexing for
        # the per-track baseline computation; the explicit loop is
        # readable and fast enough for our cohort sizes.
        for (fov, tid), track_rows in ds_group.groupby(["fov_name", "track_id"]):
            track_obs = obs[(obs["fov_name"] == str(fov)) & (obs["track_id"] == int(tid))]
            if track_obs.empty:
                continue
            t_to_idx = dict(zip(track_obs["t"].astype(int), track_obs["_idx"].astype(int)))

            # Baseline frames: pre-window in t_rel_minutes.
            bl_mask = (track_rows["t_rel_minutes"] >= baseline_window_minutes[0]) & (
                track_rows["t_rel_minutes"] <= baseline_window_minutes[1]
            )
            bl_t = track_rows.loc[bl_mask, "t"].astype(int).tolist()
            bl_indices = [t_to_idx[t] for t in bl_t if t in t_to_idx]
            if len(bl_indices) < min_baseline_frames:
                continue
            X = adata.X
            baseline = np.asarray(X[bl_indices]).mean(axis=0, keepdims=True)

            # Compute distance for every frame in track_rows.
            row_t = track_rows["t"].astype(int).tolist()
            row_indices = [t_to_idx.get(t, -1) for t in row_t]
            valid = np.array([idx >= 0 for idx in row_indices])
            if not valid.any():
                continue
            present = np.array([idx for idx, ok in zip(row_indices, valid) if ok])
            embeddings = np.asarray(X[present])
            distances = cdist(embeddings, baseline, metric=metric).flatten()

            row_idx_array = track_rows.index.to_numpy()[valid]
            out.loc[row_idx_array, "signal"] = distances

    return out


def fov_stratified_threshold(
    productive_signal: pd.DataFrame,
    mock_signal: pd.DataFrame,
    percentile: float = 95.0,
) -> pd.DataFrame:
    """Per-FOV threshold from mock cells.

    For each FOV present in the productive set, compute the
    ``percentile``th of mock-cell ``signal`` values from the same FOV.
    Returns a frame with ``(fov_name, threshold)``.

    Falls back to the global mock percentile for FOVs without enough
    mock data (< 30 frames). Per discussion §3.8 #8 the per-FOV check
    is mandatory; the fallback is documented.
    """
    if mock_signal.empty:
        raise RuntimeError("Mock cohort empty; cannot compute FOV-stratified threshold")
    mock_valid = mock_signal[mock_signal["signal"].notna()]
    if mock_valid.empty:
        raise RuntimeError("Mock cohort has no finite signal values")

    global_threshold = float(np.percentile(mock_valid["signal"].to_numpy(), percentile))

    rows = []
    productive_fovs = sorted(productive_signal["fov_name"].astype(str).unique())
    for fov in productive_fovs:
        sub = mock_valid[mock_valid["fov_name"].astype(str) == fov]
        if len(sub) < 30:
            rows.append({"fov_name": fov, "threshold": global_threshold, "n_mock": len(sub), "fallback": True})
            continue
        rows.append(
            {
                "fov_name": fov,
                "threshold": float(np.percentile(sub["signal"].to_numpy(), percentile)),
                "n_mock": len(sub),
                "fallback": False,
            }
        )
    return pd.DataFrame(rows)


def oscillation_metrics_per_cell(
    cohort_signal: pd.DataFrame,
    threshold_by_fov: pd.DataFrame,
    post_window_minutes: tuple[float, float] = (180, 540),
) -> pd.DataFrame:
    """Per-cell oscillation statistics for G3BP1 (post-window only).

    Real-time post-window only — never warped (per discussion §3.6).
    Stress granule kinetics are minute-scale; warping by NS3 is meaningless.

    Per-cell scalars:
      - excursion_count: number of zero-crossings of (signal > threshold).
      - dwell_time_minutes: total time above threshold.
      - largest_excursion_amplitude: peak signal − threshold across the window.
      - largest_excursion_duration_minutes: longest contiguous span above threshold.

    Cells with no post-window frames or no FOV threshold get NaN.
    """
    threshold_lookup = dict(zip(threshold_by_fov["fov_name"].astype(str), threshold_by_fov["threshold"]))
    rows = []
    for (ds_id, fov, tid), g in cohort_signal.groupby(["dataset_id", "fov_name", "track_id"]):
        g = g.sort_values("t")
        post = g[(g["t_rel_minutes"] >= post_window_minutes[0]) & (g["t_rel_minutes"] <= post_window_minutes[1])]
        if post.empty or post["signal"].notna().sum() == 0:
            continue
        threshold = threshold_lookup.get(str(fov))
        if threshold is None:
            continue
        signal = post["signal"].to_numpy()
        t_minutes = post["t_rel_minutes"].to_numpy()
        above = signal > threshold
        # Excursion count: count rising edges (False → True) in the above mask.
        rising = np.diff(above.astype(int)) == 1
        excursion_count = int(rising.sum()) + (1 if above[0] else 0)
        # Dwell time: estimate via successive frame intervals where above is True.
        if len(t_minutes) >= 2:
            frame_intervals = np.diff(t_minutes)
            dwell_intervals = frame_intervals * above[:-1].astype(float)
            dwell_time = float(dwell_intervals.sum())
        else:
            dwell_time = 0.0 if not above.any() else float(t_minutes[-1] - t_minutes[0])
        # Largest excursion amplitude.
        amp_above = signal[above]
        largest_amp = float(amp_above.max() - threshold) if amp_above.size else float("nan")
        # Largest excursion duration: longest contiguous run of above==True.
        if above.any():
            run_lens = []
            cur_start = None
            for i, v in enumerate(above):
                if v and cur_start is None:
                    cur_start = i
                elif not v and cur_start is not None:
                    run_lens.append((cur_start, i - 1))
                    cur_start = None
            if cur_start is not None:
                run_lens.append((cur_start, len(above) - 1))
            run_durations = [float(t_minutes[end] - t_minutes[start]) for start, end in run_lens]
            largest_dur = float(max(run_durations)) if run_durations else 0.0
        else:
            largest_dur = 0.0

        rows.append(
            {
                "dataset_id": ds_id,
                "fov_name": str(fov),
                "track_id": int(tid),
                "lineage_id": int(g["lineage_id"].iloc[0]),
                "cohort": str(g["cohort"].iloc[0]),
                "threshold": float(threshold),
                "excursion_count": excursion_count,
                "dwell_time_minutes": dwell_time,
                "largest_excursion_amplitude": largest_amp,
                "largest_excursion_duration_minutes": largest_dur,
            }
        )
    return pd.DataFrame(rows)
