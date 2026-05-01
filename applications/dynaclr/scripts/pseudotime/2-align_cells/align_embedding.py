"""Path B alignment: subsequence DTW on the NS3 channel embedding.

Stage 2 of the pseudotime pipeline (Path B track). Takes a template
built by ``1-build_template/build_template.py`` and scans it across
every track in a *query set*. Subsequence DTW finds, per query track,
the time window where the template best matches — i.e. when that cell
traversed the event encoded by the template. Frames inside the matched
window are mapped to template-relative minutes via the template's
``time_calibration``; frames outside are labelled ``pre`` / ``post``.

Preprocessing re-uses the build-time z-score + PCA + L2 stored in the
template zarr. Never refit at alignment time.

Output parquet matches the unified Stage 2 schema (per DAG §7.4) so
Path A and Path B parquets can be compared directly in Stage 4.
``length_normalized_cost`` and ``path_skew`` are surfaced as columns
(not just used as filters) so downstream sweeps can re-gate without
re-running DTW. Path-skew is the primary gate (rejects degenerate
non-diagonal warps); cost is the secondary gate (stereotypy filter).

Usage::

    cd applications/dynaclr/scripts/pseudotime/2-align_cells
    uv run python align_embedding.py \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/align_cells.yaml \
        --template infection_nondividing_sensor \
        --flavor raw \
        --query-set sensor_all_transitioning
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from dynaclr.pseudotime import (
    AlignmentResult,
    alignment_results_to_dataframe,
    date_prefix_from_dataset_id,
    dtw_align_tracks,
    find_embedding_zarr,
    load_template_flavor,
    read_template_attrs,
)
from dynaclr.pseudotime.alignment import filter_tracks

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = SCRIPT_DIR.parent / "1-build_template" / "templates"
CANDIDATES_DIR = SCRIPT_DIR.parent / "0-select_candidates" / "candidates"
OUTPUT_ALIGNMENTS_DIR = SCRIPT_DIR / "B" / "alignments"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _load_query_embeddings(
    query_cfg: dict,
    dataset_cfgs: dict[str, dict],
    embedding_pattern: str,
    min_track_timepoints: int,
    template_len_frames: int,
) -> tuple[dict[str, ad.AnnData], dict[str, pd.DataFrame]]:
    """Load query embedding zarrs and build per-dataset filtered track dfs.

    Parameters
    ----------
    query_cfg : dict
        Query-set config entry from ``config['query_sets'][name]``. Must
        include ``datasets`` (list) and may include ``track_filter``
        (dict of ``obs``-column → required value) and
        ``min_track_minutes`` (float).
    dataset_cfgs : dict[str, dict]
        Map ``{dataset_id: dataset_cfg}`` from ``config['datasets']``.
    embedding_pattern : str
        Glob pattern for the channel's zarr file
        (e.g. ``"*_viral_sensor_*.zarr"``).
    min_track_timepoints : int
        CLI-level floor on track length, passed to :func:`filter_tracks`.

    Returns
    -------
    tuple[dict[str, ad.AnnData], dict[str, pd.DataFrame]]
        ``(adata_dict, df_dict)`` keyed by ``dataset_id``. The df has
        at least ``fov_name``, ``track_id``, ``t`` — columns used by
        ``dtw_align_tracks`` for valid-track selection.
    """
    adata_dict: dict[str, ad.AnnData] = {}
    df_dict: dict[str, pd.DataFrame] = {}

    track_filter = query_cfg.get("track_filter", {}) or {}
    min_track_minutes = query_cfg.get("min_track_minutes")
    min_pre_minutes = float(query_cfg.get("min_pre_minutes", 0))
    min_post_minutes = float(query_cfg.get("min_post_minutes", 0))

    for ds_entry in query_cfg["datasets"]:
        dataset_id = ds_entry["dataset_id"]
        ds_cfg = dataset_cfgs[dataset_id]
        fov_pattern = ds_cfg.get("fov_pattern")
        frame_interval = ds_cfg["frame_interval_minutes"]

        prefix = date_prefix_from_dataset_id(dataset_id)
        zarr_path = find_embedding_zarr(ds_cfg["pred_dir"], prefix + embedding_pattern)

        adata = ad.read_zarr(zarr_path)
        adata.obs_names_make_unique()

        # FOV restriction from the dataset config (e.g. "C/2") — keeps us
        # out of control wells unless the user explicitly wants them.
        if fov_pattern is not None:
            fov_mask = adata.obs["fov_name"].astype(str).str.contains(fov_pattern, regex=False)
            adata = adata[fov_mask.to_numpy()].copy()

        # Track-level filters from query_cfg['track_filter']: each entry
        # requires the obs column to equal the provided value on every
        # frame of the track. Applied in adata space so the dropped rows
        # don't make it into the DTW solver.
        for col, required_value in track_filter.items():
            if col not in adata.obs.columns:
                raise KeyError(f"track_filter column {col!r} not in adata.obs for {dataset_id}")
            mask = adata.obs[col].astype(str) == str(required_value)
            adata = adata[mask.to_numpy()].copy()

        # Build a tracking df for the stage-1 filter helper.
        df = adata.obs[["fov_name", "track_id", "t"]].copy()
        df["fov_name"] = df["fov_name"].astype(str)
        df["track_id"] = df["track_id"].astype(int)
        df["t"] = df["t"].astype(int)

        # Pass 1 (necessary condition): track must be long enough to hold at least
        # the template itself plus any required pre/post headroom. The sufficient
        # condition — that the match actually lands inside the track with that
        # headroom on either side — is enforced post-DTW in main().
        if min_track_minutes is not None:
            min_frames = int(np.ceil(float(min_track_minutes) / frame_interval))
        else:
            min_frames = min_track_timepoints
        headroom_frames = int(np.ceil((min_pre_minutes + min_post_minutes) / frame_interval))
        min_frames = max(min_frames, min_track_timepoints, template_len_frames + headroom_frames)
        df = filter_tracks(df, min_timepoints=min_frames)

        # Subset adata to surviving (fov, track, t) rows.
        keep_keys = set(zip(df["fov_name"], df["track_id"], df["t"]))
        keep_mask = [
            (str(f), int(tid), int(t)) in keep_keys
            for f, tid, t in zip(adata.obs["fov_name"], adata.obs["track_id"], adata.obs["t"])
        ]
        adata = adata[np.asarray(keep_mask)].copy()

        if len(adata) == 0:
            _logger.warning(f"[{dataset_id}] no tracks survived filters; skipping")
            continue

        adata_dict[dataset_id] = adata
        df_dict[dataset_id] = df
        _logger.info(
            f"[{dataset_id}] {adata.n_obs} rows, "
            f"{df.groupby(['fov_name', 'track_id']).ngroups} tracks "
            f"(min {min_frames} frames = {min_frames * frame_interval} min)"
        )

    return adata_dict, df_dict


def _enrich_with_cohort_metadata(
    flat: pd.DataFrame,
    candidate_set: str,
    frame_interval_minutes: dict[str, float],
) -> pd.DataFrame:
    """Join Path B alignment frame with Stage 0 cohort metadata.

    Reads the productive-cohort CSV produced by :mod:`select_candidates`
    and merges in ``lineage_id``, ``cohort``, ``divides``, and
    ``t_zero`` per ``(dataset_id, fov_name, track_id)`` so the Path B
    parquet matches the unified Stage 2 schema (per DAG §7.4). Computes
    real-time ``t_rel_minutes`` per row.

    Path B alignment runs only on the productive cohort. Cells not
    matched in the cohort CSV get ``cohort="productive"`` (default for
    Path B input), ``lineage_id=""`` (orphan sentinel), ``t_zero=NaN``.
    """
    flat = flat.copy()
    cand_csv = CANDIDATES_DIR / f"{candidate_set}_productive.csv"
    if not cand_csv.exists():
        _logger.warning(
            f"Productive cohort CSV {cand_csv} not found; lineage_id, cohort, divides, t_zero will be missing."
        )
        flat["lineage_id"] = ""
        flat["cohort"] = "productive"
        flat["divides"] = "none"
        flat["t_zero"] = pd.NA
        flat["t_rel_minutes"] = np.nan
        return flat

    productive = pd.read_csv(cand_csv)
    # Reduce to per-track metadata: lineage_id, divides are per-lineage.
    per_track = (
        productive.groupby(["dataset_id", "fov_name", "track_id"])
        .agg(
            lineage_id=("lineage_id", "first"),
            divides=("divides", "first"),
        )
        .reset_index()
    )
    flat["fov_name"] = flat["fov_name"].astype(str)
    flat["track_id"] = flat["track_id"].astype(int)
    per_track["fov_name"] = per_track["fov_name"].astype(str)
    per_track["track_id"] = per_track["track_id"].astype(int)
    flat = flat.merge(per_track, on=["dataset_id", "fov_name", "track_id"], how="left")
    flat["lineage_id"] = flat["lineage_id"].fillna("").astype(str)
    flat["divides"] = flat["divides"].fillna("none")
    flat["cohort"] = "productive"

    # t_zero: per-lineage first frame where infection_state == "infected".
    productive_pos = productive[productive["infection_state"] == "infected"]
    t_zero_lookup = productive_pos.groupby("lineage_id")["t"].min().to_dict() if not productive_pos.empty else {}
    flat["t_zero"] = flat["lineage_id"].map(t_zero_lookup)

    # t_rel_minutes = (t - t_zero) * frame_interval. NaN when no anchor.
    fi = flat["dataset_id"].map(frame_interval_minutes)
    has_anchor = flat["t_zero"].notna()
    flat["t_rel_minutes"] = np.where(
        has_anchor,
        (flat["t"] - flat["t_zero"].fillna(0)) * fi,
        np.nan,
    )
    return flat


def _per_track_match_metadata(
    results: list[AlignmentResult],
    frame_interval_minutes: dict[str, float],
) -> pd.DataFrame:
    """Derive per-track subsequence-match bounds from alignment results.

    ``match_q_start`` / ``match_q_end`` are the absolute query frames
    bounding the first / last template-matched frame (``alignment_region
    == "aligned"``). ``match_duration_minutes`` is the real-time span of
    the match using the dataset's frame interval.

    Parameters
    ----------
    results : list[AlignmentResult]
        Output of :func:`dtw_align_tracks`.
    frame_interval_minutes : dict[str, float]
        Map ``{dataset_id: frame_interval_minutes}``.

    Returns
    -------
    pd.DataFrame
        One row per cell with ``dataset_id``, ``fov_name``,
        ``track_id``, ``match_q_start``, ``match_q_end``,
        ``match_duration_minutes``.
    """
    rows = []
    for r in results:
        aligned_mask = r.alignment_region == "aligned"
        if not aligned_mask.any():
            # Shouldn't happen when subsequence=True with any match, but
            # keep the row so downstream joins don't drop the cell.
            q_start, q_end, duration = np.nan, np.nan, np.nan
        else:
            aligned_times = r.timepoints[aligned_mask]
            q_start = int(aligned_times.min())
            q_end = int(aligned_times.max())
            fi = frame_interval_minutes[r.dataset_id]
            duration = float(q_end - q_start) * fi
        rows.append(
            {
                "dataset_id": r.dataset_id,
                "fov_name": r.fov_name,
                "track_id": r.track_id,
                "match_q_start": q_start,
                "match_q_end": q_end,
                "match_duration_minutes": duration,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point for Stage 2 subsequence alignment."""
    parser = argparse.ArgumentParser(description="Subsequence-DTW-align query tracks to a template (Stage 2)")
    parser.add_argument("--datasets", required=True, help="Path to datasets.yaml (shared infra config)")
    parser.add_argument("--config", required=True, help="Path to align_cells.yaml")
    parser.add_argument("--template", required=True, help="Template name under config['templates']")
    parser.add_argument(
        "--flavor",
        choices=["raw", "pca"],
        default="raw",
        help="Which template flavor to align against (default: raw)",
    )
    parser.add_argument(
        "--candidate-set",
        default=None,
        help=(
            "Candidate-set name from candidates.yaml; used to join cohort + lineage "
            "metadata into the unified Stage 2 parquet. Defaults to --query-set if omitted."
        ),
    )
    parser.add_argument(
        "--query-set",
        required=True,
        help="Query-set name under config['query_sets']",
    )
    parser.add_argument(
        "--min-track-timepoints",
        type=int,
        default=3,
        help="Minimum timepoints per track (default: 3). Overridden by query_set.min_track_minutes when larger.",
    )
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of template length that the matched window must cover "
            "(default: 0.5). Rejects degenerate subsequence DTW matches where the solver "
            "collapses the template onto a few query frames. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-skew",
        type=float,
        default=0.8,
        help=(
            "Maximum allowed path skewness in [0, 1], where 0 = perfectly diagonal. "
            "Rejects L-shaped or heavily non-diagonal warps that slip past the psi cap. "
            "Default mirrors the old find_best_match_dtw_bernd_clifford default. Set to 1 to disable."
        ),
    )
    parser.add_argument(
        "--min-match-minutes",
        type=float,
        default=None,
        help=(
            "Minimum real-time duration of the matched window. Supersedes --min-match-ratio "
            "when set. Per-track frame threshold = ceil(min_match_minutes / frame_interval_minutes). "
            "Use this to apply a single wall-clock bar across query datasets with different "
            "frame intervals."
        ),
    )
    args = parser.parse_args()

    config = load_stage_config(args.datasets, args.config)

    query_cfg = config.get("query_sets", {}).get(args.query_set)
    if query_cfg is None:
        raise KeyError(f"Query set {args.query_set!r} not in config['query_sets']")

    template_path = TEMPLATES_DIR / f"template_{args.template}.zarr"
    if not template_path.exists():
        raise FileNotFoundError(f"Template zarr not found: {template_path}")

    # Template channel is recorded in the zarr's config_snapshot attrs by build_template.py.
    # Reading from the zarr (rather than requiring the build-template config) keeps Stage 2
    # self-contained: a template zarr is all you need to know what it aligns to.
    snapshot = read_template_attrs(template_path).get("config_snapshot", {})
    template_entry = snapshot.get("templates", {}).get(args.template, {})
    template_channel = template_entry.get("channel")
    if template_channel is None:
        raise ValueError(
            f"Template zarr {template_path} has no recorded channel in config_snapshot; "
            f"was it built by the current build_template.py?"
        )

    query_channel = query_cfg.get("channel", template_channel)
    if query_channel != template_channel:
        raise ValueError(
            f"Query set channel {query_channel!r} does not match template channel "
            f"{template_channel!r}. Alignment must happen in the template's embedding space."
        )

    _logger.info(f"Loading template {template_path} (flavor={args.flavor})")
    template_result, _attrs = load_template_flavor(template_path, args.flavor)
    _logger.info(f"  template shape {template_result.template.shape}, {template_result.n_input_tracks} input tracks")

    embedding_pattern = config["embeddings"][query_channel]
    dataset_cfgs = {d["dataset_id"]: d for d in config["datasets"]}

    _logger.info(f"Loading query set {args.query_set!r} ({len(query_cfg['datasets'])} datasets)")
    adata_dict, df_dict = _load_query_embeddings(
        query_cfg,
        dataset_cfgs,
        embedding_pattern,
        args.min_track_timepoints,
        template_len_frames=template_result.template.shape[0],
    )
    if not adata_dict:
        raise RuntimeError(f"Query set {args.query_set!r} produced no usable tracks")

    all_results: list[AlignmentResult] = []
    frame_interval_by_ds = {d["dataset_id"]: float(d["frame_interval_minutes"]) for d in config["datasets"]}

    # psi is a TEMPLATE-axis budget (frames of the cost matrix), not a query-time
    # budget. The frame-unit default (t_template // 2) inside dtw_align_tracks is
    # the right value regardless of query frame rate. We don't scale psi by the
    # query's frame interval — see dtw_align_tracks' inline note.
    for dataset_id, adata in adata_dict.items():
        _logger.info(f"Aligning {dataset_id} (subsequence DTW)")
        results = dtw_align_tracks(
            adata=adata,
            df=df_dict[dataset_id],
            template_result=template_result,
            dataset_id=dataset_id,
            min_track_timepoints=args.min_track_timepoints,
            subsequence=True,
        )
        all_results.extend(results)

    if not all_results:
        raise RuntimeError("No alignment results produced")

    drop_log: dict[str, int] = {"n_input_tracks": len(all_results)}

    # Drop tracks whose DTW solver could not find a valid path — these show up
    # as `length_normalized_cost == inf` (dtaidistance returns an unreachable
    # endpoint when psi overflows the cost band on very short tracks). They
    # carry no ranking signal and only pollute downstream plots.
    n_before = len(all_results)
    all_results = [r for r in all_results if np.isfinite(r.length_normalized_cost)]
    n_dropped = n_before - len(all_results)
    drop_log["n_dropped_non_finite_cost"] = n_dropped
    if n_dropped:
        _logger.warning(
            f"Dropped {n_dropped}/{n_before} tracks with non-finite DTW cost "
            f"(likely too short relative to template length {template_result.template.shape[0]})"
        )
    if not all_results:
        raise RuntimeError("No tracks produced a finite DTW cost; check min_track_minutes vs template length")

    # Skew filter — primary gate per discussion §3.8 #2. Rejects degenerate
    # non-diagonal warps (L-shape, cliff, etc.) without rejecting biological
    # rate variance. Run BEFORE the cost / min-match filters so cost gating
    # operates on a population of valid warps only.
    if args.max_skew < 1.0:
        n_before = len(all_results)
        all_results = [r for r in all_results if r.path_skew <= args.max_skew]
        n_skewed = n_before - len(all_results)
        drop_log["n_dropped_max_skew"] = n_skewed
        if n_skewed:
            _logger.warning(
                f"Dropped {n_skewed}/{n_before} tracks with path_skew > {args.max_skew:.2f} "
                "(non-diagonal warps; relax --max-skew to keep them)"
            )
        if not all_results:
            raise RuntimeError(
                f"No tracks survived skew filter (max_skew={args.max_skew}); relax or disable with --max-skew 1"
            )

    # Drop tracks whose matched window is shorter than --min-match-ratio of the
    # template length. Subsequence DTW with psi relaxation can collapse the
    # template onto a 1-frame query window (every template position warped to
    # the same query frame) — cost is near-zero but the match is meaningless.
    # A window shorter than ~half the template can't plausibly represent the
    # event the template encodes.
    t_template = template_result.template.shape[0]

    # Minute-based filter takes precedence — frame-rate invariant across query datasets.
    # Threshold is computed per-track using that dataset's frame_interval_minutes so
    # a 10 min/frame track needs more frames than a 30 min/frame track for the same
    # real-time match duration.
    if args.min_match_minutes is not None and args.min_match_minutes > 0:
        n_before = len(all_results)
        kept = []
        for r in all_results:
            fi = frame_interval_by_ds[r.dataset_id]
            min_aligned_track = max(2, int(np.ceil(args.min_match_minutes / fi)))
            if int((r.alignment_region == "aligned").sum()) >= min_aligned_track:
                kept.append(r)
        all_results = kept
        n_short = n_before - len(all_results)
        drop_log["n_dropped_min_match_minutes"] = n_short
        if n_short:
            _logger.warning(
                f"Dropped {n_short}/{n_before} tracks with matched window shorter than "
                f"{args.min_match_minutes:.0f} min (threshold is per-track: ceil(minutes / frame_interval))"
            )
        if not all_results:
            raise RuntimeError(
                f"No tracks matched at least {args.min_match_minutes:.0f} min; "
                "lower --min-match-minutes or rebuild the template"
            )
    elif args.min_match_ratio > 0:
        # Legacy frame-based path; fine when all query datasets share a frame interval
        # with the template build set, but not cross-dataset safe.
        min_aligned = max(2, int(np.ceil(args.min_match_ratio * t_template)))
        n_before = len(all_results)
        all_results = [r for r in all_results if int((r.alignment_region == "aligned").sum()) >= min_aligned]
        n_short = n_before - len(all_results)
        drop_log["n_dropped_min_match_ratio"] = n_short
        if n_short:
            _logger.warning(
                f"Dropped {n_short}/{n_before} tracks with matched window < {min_aligned} frames "
                f"({args.min_match_ratio:.0%} of template length {t_template}) — likely degenerate collapses"
            )
        if not all_results:
            raise RuntimeError(
                f"No tracks matched at least {min_aligned} frames; lower --min-match-ratio or rebuild the template"
            )

    # Pre/post headroom filter: the match must leave at least min_pre_minutes
    # before match_q_start and min_post_minutes after match_q_end, both relative
    # to the cell's own track start/end. This is the sufficient condition that
    # pairs with pass 1's necessary condition in _load_query_embeddings.
    min_pre_minutes = float(query_cfg.get("min_pre_minutes", 0))
    min_post_minutes = float(query_cfg.get("min_post_minutes", 0))
    if min_pre_minutes > 0 or min_post_minutes > 0:
        filtered = []
        for r in all_results:
            fi = frame_interval_by_ds[r.dataset_id]
            pre_needed = int(np.ceil(min_pre_minutes / fi))
            post_needed = int(np.ceil(min_post_minutes / fi))
            aligned_mask = r.alignment_region == "aligned"
            if not aligned_mask.any():
                continue
            aligned_times = r.timepoints[aligned_mask]
            q_start, q_end = int(aligned_times.min()), int(aligned_times.max())
            track_min, track_max = int(r.timepoints.min()), int(r.timepoints.max())
            if (q_start - track_min) >= pre_needed and (track_max - q_end) >= post_needed:
                filtered.append(r)
        n_before = len(all_results)
        all_results = filtered
        n_cut = n_before - len(all_results)
        drop_log["n_dropped_pre_post_headroom"] = n_cut
        if n_cut:
            _logger.warning(
                f"Dropped {n_cut}/{n_before} tracks without required pre/post headroom "
                f"(need ≥ {min_pre_minutes:.0f} min before + {min_post_minutes:.0f} min after the match)"
            )
        if not all_results:
            raise RuntimeError(
                "No tracks have the required pre/post headroom; "
                "loosen min_pre_minutes/min_post_minutes in the query-set YAML"
            )

    flat = alignment_results_to_dataframe(
        all_results,
        template_id=template_result.template_id,
        time_calibration=template_result.time_calibration,
    )

    match_meta = _per_track_match_metadata(all_results, frame_interval_by_ds)
    flat = flat.merge(match_meta, on=["dataset_id", "fov_name", "track_id"], how="left")

    # estimated_t_rel_minutes (now renamed t_rel_minutes_warped per DAG §7.4)
    # is only meaningful inside the aligned window. Pre/post frames get NaN
    # here; their real-time t_rel_minutes is computed below from t_zero.
    if "estimated_t_rel_minutes" in flat.columns:
        outside = flat["alignment_region"].isin(["pre", "post"])
        flat.loc[outside, "estimated_t_rel_minutes"] = np.nan
        flat = flat.rename(columns={"estimated_t_rel_minutes": "t_rel_minutes_warped"})
    else:
        flat["t_rel_minutes_warped"] = np.nan

    # Unified Stage 2 schema (per DAG §7.4): join Path A's per-lineage
    # cohort + lineage_id + t_zero by reading the candidate-set CSVs
    # from Stage 0. The candidate-set name is taken from --candidate-set
    # if provided, else inferred from the query_set name.
    candidate_set = getattr(args, "candidate_set", None) or args.query_set
    flat = _enrich_with_cohort_metadata(flat, candidate_set, frame_interval_by_ds)
    flat["track_path"] = "B"

    OUTPUT_ALIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ALIGNMENTS_DIR / f"{args.template}_{args.flavor}_on_{args.query_set}.parquet"
    flat.to_parquet(out_path, index=False)
    n_tracks = flat.groupby(["dataset_id", "fov_name", "track_id"]).ngroups
    _logger.info(
        f"Wrote {out_path} ({len(flat)} rows, {n_tracks} tracks, "
        f"{(flat['alignment_region'] == 'aligned').sum()} aligned frames)"
    )

    # Sidecar JSON capturing per-filter drop counts so a reviewer can see how many
    # tracks made it through each guard without grepping stderr.
    drop_log["n_kept"] = n_tracks
    drop_log_path = out_path.with_suffix(".drop_log.json")
    with open(drop_log_path, "w") as f:
        json.dump(drop_log, f, indent=2)
    _logger.info(f"Wrote drop log {drop_log_path}: {drop_log}")


if __name__ == "__main__":
    main()
