r"""Auto-select candidates and tag cohorts for Stage 0 of the new DAG.

Reads per-dataset annotation CSVs and produces four cohort-tagged
annotation CSVs per candidate set:

- ``{candidate_set}_productive.csv`` — transitioning lineages with a
  manual ``t_key_event``-equivalent anchor (first ``infected`` frame).
- ``{candidate_set}_bystander.csv`` — lineages in infected wells whose
  LC prediction stays mostly uninfected.
- ``{candidate_set}_abortive.csv`` — lineages in infected wells with a
  brief positive run shorter than ``min_run`` and no sustained rise.
- ``{candidate_set}_mock.csv`` — lineages from uninfected control wells.

Per discussion §3.2 and the execution plan: the cohort definitions are
operational and LC-derived. ``parent_track_id`` is preserved through the
pipeline so ``identify_lineages()`` can reconnect mother + daughter into
shared ``lineage_id``.

Usage::

    cd applications/dynaclr/scripts/pseudotime/0-select_candidates
    uv run python select_candidates.py \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/candidates.yaml \
        --candidate-set zikv_productive_07_24
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATES_DIR = SCRIPT_DIR / "candidates"

sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))
from lineage_utils import (  # noqa: E402
    assign_lineage_ids,
    classify_divides,
    tag_cohort_for_lineage,
)
from utils import load_stage_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

LABEL_VALUES: dict[str, tuple[str, str]] = {
    "infection_state": ("infected", "uninfected"),
    "organelle_state": ("remodeled", "noremodeled"),
    "cell_division_state": ("mitosis", "interphase"),
}

# Output schema columns (per DAG §11).
OUTPUT_COLUMNS = [
    "dataset_id",
    "fov_name",
    "lineage_id",
    "track_id",
    "parent_track_id",
    "t",
    "cohort",
    "divides",
    "infection_state",
    "organelle_state",
    "cell_division_state",
]


def _select_productive_tracks(
    ann_df: pd.DataFrame,
    dataset_id: str,
    fov_pattern: str,
    filter_cfg: dict,
    frame_interval_minutes: float,
) -> pd.DataFrame:
    """Pick transitioning tracks from one dataset's annotations.

    Preserves ``parent_track_id`` (the previous version dropped it).

    Parameters
    ----------
    ann_df : pd.DataFrame
        Raw annotations CSV.
    dataset_id, fov_pattern : str
        Dataset id + FOV substring filter.
    filter_cfg : dict
        Per-candidate-set ``productive_filter`` dict.
    frame_interval_minutes : float
        Dataset-specific frame interval for minute → frame conversions.

    Returns
    -------
    pd.DataFrame
        Per-frame rows with productive cohort tag, lineage_id assigned later.
    """
    anchor_label = filter_cfg["anchor_label"]
    anchor_positive = filter_cfg["anchor_positive"]
    anchor_negative = filter_cfg.get("anchor_negative", "uninfected")
    min_pre = float(filter_cfg.get("min_pre_minutes", 0))
    min_post = float(filter_cfg.get("min_post_minutes", 0))
    crop_window_minutes = filter_cfg["crop_window_minutes"]

    pre_frames = int(round(min_pre / frame_interval_minutes))
    post_frames = int(round(min_post / frame_interval_minutes))
    crop_half = int(round(float(crop_window_minutes) / frame_interval_minutes))

    sub = ann_df[ann_df["fov_name"].astype(str).str.contains(fov_pattern, regex=False)].copy()
    if sub.empty:
        _logger.warning(f"[{dataset_id}] no rows match fov_pattern {fov_pattern!r}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows: list[dict] = []
    for (fov, tid), g in sub.groupby(["fov_name", "track_id"]):
        g = g.sort_values("t")
        states = set(g[anchor_label].dropna())
        if anchor_positive not in states or anchor_negative not in states:
            continue

        t_onset = int(g[g[anchor_label] == anchor_positive]["t"].min())
        t_min, t_max = int(g["t"].min()), int(g["t"].max())
        if (t_onset - t_min) < pre_frames or (t_max - t_onset) < post_frames:
            continue

        t_before = max(t_min, t_onset - crop_half)
        t_after = min(t_max, t_onset + crop_half)

        parent_id = int(g.iloc[0].get("parent_track_id", -1))

        in_window = g[(g["t"] >= t_before) & (g["t"] <= t_after)]
        for _, r in in_window.iterrows():
            row = {
                "dataset_id": dataset_id,
                "fov_name": str(fov),
                "track_id": int(tid),
                "parent_track_id": parent_id,
                "t": int(r["t"]),
            }
            for label_col in LABEL_VALUES:
                if label_col in r and pd.notna(r[label_col]) and r[label_col] != "":
                    row[label_col] = r[label_col]
                else:
                    row[label_col] = ""
            rows.append(row)

    return pd.DataFrame(rows)


def _select_well_tracks(
    ann_df: pd.DataFrame,
    dataset_id: str,
    fov_pattern: str,
    min_track_minutes: float,
    frame_interval_minutes: float,
) -> pd.DataFrame:
    """Pull every track from a well, no anchor filter.

    Used for bystander/abortive (infected well) and mock (uninfected well)
    cohorts where we keep every long-enough track.

    Parameters
    ----------
    min_track_minutes : float
        Drop tracks shorter than this many real minutes.
    """
    min_frames = int(round(min_track_minutes / frame_interval_minutes))

    sub = ann_df[ann_df["fov_name"].astype(str).str.contains(fov_pattern, regex=False)].copy()
    if sub.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows: list[dict] = []
    for (fov, tid), g in sub.groupby(["fov_name", "track_id"]):
        if len(g) < min_frames:
            continue
        g = g.sort_values("t")
        parent_id = int(g.iloc[0].get("parent_track_id", -1))
        for _, r in g.iterrows():
            row = {
                "dataset_id": dataset_id,
                "fov_name": str(fov),
                "track_id": int(tid),
                "parent_track_id": parent_id,
                "t": int(r["t"]),
            }
            for label_col in LABEL_VALUES:
                if label_col in r and pd.notna(r[label_col]) and r[label_col] != "":
                    row[label_col] = r[label_col]
                else:
                    row[label_col] = ""
            rows.append(row)

    return pd.DataFrame(rows)


def _load_lc_predictions(
    dataset_cfg: dict,
    fov_pattern: str,
    embedding_pattern: str,
    pred_column: str = "predicted_infection_state",
) -> pd.DataFrame:
    """Pull LC predictions from the embedding zarr's obs.

    Returns per (fov_name, track_id, t) rows with ``pred_column``. Only
    needed for bystander/abortive cohort tagging; productive and mock
    cohorts don't depend on LC.

    Returns empty DataFrame if no embedding zarr matches the pattern.
    """
    pred_dir = Path(dataset_cfg["pred_dir"])
    matches = list(pred_dir.glob(embedding_pattern))
    if not matches:
        _logger.warning(
            f"No embedding zarr matched {embedding_pattern} in {pred_dir}; "
            f"LC-derived cohorts will fall back to bystander default"
        )
        return pd.DataFrame()

    adata = ad.read_zarr(matches[0])
    if pred_column not in adata.obs.columns:
        _logger.warning(f"{pred_column} not in {matches[0]} .obs; LC fallback")
        return pd.DataFrame()

    obs = adata.obs[["fov_name", "track_id", "t", pred_column]].copy()
    obs = obs[obs["fov_name"].astype(str).str.contains(fov_pattern, regex=False)]
    obs["fov_name"] = obs["fov_name"].astype(str)
    obs["track_id"] = obs["track_id"].astype(int)
    obs["t"] = obs["t"].astype(int)
    return obs


def _t_zero_per_lineage(
    productive_df: pd.DataFrame,
    anchor_label: str,
    anchor_positive: str,
) -> dict[int, int]:
    """First frame where ``anchor_label == anchor_positive`` per lineage."""
    out: dict[int, int] = {}
    for lineage_id, g in productive_df.groupby("lineage_id"):
        if lineage_id < 0:
            continue
        positive = g[g[anchor_label] == anchor_positive]
        if positive.empty:
            continue
        out[int(lineage_id)] = int(positive["t"].min())
    return out


def _well_is_uninfected(fov_name: str, mock_well_patterns: list[str]) -> bool:
    return any(p in str(fov_name) for p in mock_well_patterns)


def _emit_cohort(
    df: pd.DataFrame,
    cohort: str,
    t_zero_lookup: dict[int, int],
    window_frames_by_dataset: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    """Add cohort + divides columns and order the output schema.

    ``window_frames_by_dataset`` maps ``dataset_id`` to ``(k_pre_frames,
    k_post_frames)``. Each lineage lives in exactly one dataset, so we
    look up its dataset's frame conversions to classify ``divides``.
    """
    df = df.copy()
    df["cohort"] = cohort

    divides_per_lineage: dict[int, str] = {}
    for lineage_id, g in df.groupby("lineage_id"):
        if lineage_id < 0:
            divides_per_lineage[lineage_id] = "none"
            continue
        t_zero = t_zero_lookup.get(int(lineage_id))
        if t_zero is None:
            divides_per_lineage[lineage_id] = "none"
            continue
        dataset_id = str(g["dataset_id"].iloc[0])
        k_pre_frames, k_post_frames = window_frames_by_dataset[dataset_id]
        divides_per_lineage[lineage_id] = classify_divides(g, t_zero, k_pre_frames, k_post_frames)

    df["divides"] = df["lineage_id"].map(divides_per_lineage).fillna("none")

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[OUTPUT_COLUMNS]


def _build_dataset_cohorts(
    dataset_id: str,
    dataset_cfg: dict,
    cand_cfg: dict,
    embedding_pattern: str,
) -> dict[str, pd.DataFrame]:
    """Produce productive / bystander / abortive / mock dataframes for one dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset key from ``datasets.yaml``.
    dataset_cfg : dict
        Single dataset entry from ``datasets.yaml``.
    cand_cfg : dict
        Candidate-set entry from ``candidates.yaml``.
    embedding_pattern : str
        Glob pattern for the LC-prediction embedding zarr (e.g. sensor channel).
    """
    fov_pattern = dataset_cfg.get("fov_pattern", "")
    frame_interval = float(dataset_cfg["frame_interval_minutes"])
    ann_path = Path(dataset_cfg["annotations_path"])
    _logger.info(f"[{dataset_id}] reading {ann_path}")
    ann_df = pd.read_csv(ann_path)

    productive_filter = cand_cfg["productive_filter"]
    cohort_rules = cand_cfg.get("cohort_rules", {})
    lineage_rules = cand_cfg.get("lineage_rules", {})

    mock_patterns: list[str] = list(cohort_rules.get("mock_well_patterns", []))
    bystander_fraction = float(cohort_rules.get("bystander_uninfected_fraction", 0.8))
    abortive_min_run = int(cohort_rules.get("abortive_min_run", 3))

    k_pre = int(round(float(lineage_rules.get("transition_window_k_pre_minutes", 60)) / frame_interval))
    k_post = int(round(float(lineage_rules.get("transition_window_k_post_minutes", 120)) / frame_interval))

    # 1) Productive cohort from the infected well.
    productive_df = _select_productive_tracks(
        ann_df,
        dataset_id=dataset_id,
        fov_pattern=fov_pattern,
        filter_cfg=productive_filter,
        frame_interval_minutes=frame_interval,
    )

    # 2) Mock cohort from uninfected control wells.
    mock_parts: list[pd.DataFrame] = []
    for pat in mock_patterns:
        ctrl_df = _select_well_tracks(
            ann_df,
            dataset_id=dataset_id,
            fov_pattern=pat,
            min_track_minutes=float(productive_filter.get("min_pre_minutes", 0))
            + float(productive_filter.get("min_post_minutes", 0)),
            frame_interval_minutes=frame_interval,
        )
        if not ctrl_df.empty:
            mock_parts.append(ctrl_df)
    mock_df = pd.concat(mock_parts, ignore_index=True) if mock_parts else pd.DataFrame(columns=OUTPUT_COLUMNS)

    # 3) Bystander + abortive: every track in the infected well that's
    #    not in the productive cohort. LC predictions discriminate.
    well_df = _select_well_tracks(
        ann_df,
        dataset_id=dataset_id,
        fov_pattern=fov_pattern,
        min_track_minutes=float(productive_filter.get("min_pre_minutes", 0))
        + float(productive_filter.get("min_post_minutes", 0)),
        frame_interval_minutes=frame_interval,
    )

    productive_track_keys = set(zip(productive_df["fov_name"].astype(str), productive_df["track_id"].astype(int)))
    well_non_productive_df = well_df[
        ~well_df.apply(
            lambda r: (str(r["fov_name"]), int(r["track_id"])) in productive_track_keys,
            axis=1,
        )
    ].copy()

    lc_obs = _load_lc_predictions(dataset_cfg, fov_pattern=fov_pattern, embedding_pattern=embedding_pattern)

    return {
        "productive": productive_df,
        "well_non_productive": well_non_productive_df,  # to be split into bystander/abortive
        "mock": mock_df,
        "_meta": {
            "fov_pattern": fov_pattern,
            "frame_interval": frame_interval,
            "mock_patterns": mock_patterns,
            "bystander_fraction": bystander_fraction,
            "abortive_min_run": abortive_min_run,
            "k_pre": k_pre,
            "k_post": k_post,
            "lc_obs": lc_obs,
        },
    }


def _split_bystander_abortive(
    well_df: pd.DataFrame,
    lc_obs: pd.DataFrame,
    bystander_fraction: float,
    abortive_min_run: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split non-productive tracks into bystander and abortive."""
    if well_df.empty:
        return well_df.copy(), well_df.copy()

    # Per-track LC prediction sequences.
    if lc_obs is None or len(lc_obs) == 0:
        # Without LC, all non-productive tracks get tagged bystander
        # (the conservative default per lineage_utils.tag_cohort_for_lineage).
        return well_df.copy(), well_df.iloc[0:0].copy()

    bystander_keys: set[tuple[str, int]] = set()
    abortive_keys: set[tuple[str, int]] = set()
    for (fov, tid), g in lc_obs.groupby(["fov_name", "track_id"]):
        g = g.sort_values("t")
        cohort = tag_cohort_for_lineage(
            df=g,
            well_is_uninfected=False,
            has_manual_t_zero=False,
            lc_predictions=g["predicted_infection_state"],
            min_run=abortive_min_run,
            bystander_uninfected_fraction=bystander_fraction,
        )
        if cohort == "bystander":
            bystander_keys.add((str(fov), int(tid)))
        elif cohort == "abortive":
            abortive_keys.add((str(fov), int(tid)))

    bystander_df = well_df[
        well_df.apply(lambda r: (str(r["fov_name"]), int(r["track_id"])) in bystander_keys, axis=1)
    ].copy()
    abortive_df = well_df[
        well_df.apply(lambda r: (str(r["fov_name"]), int(r["track_id"])) in abortive_keys, axis=1)
    ].copy()
    return bystander_df, abortive_df


def _funnel_report(
    candidate_set: str,
    cohort_dfs: dict[str, pd.DataFrame],
) -> str:
    """Render a markdown funnel report (lineage and frame counts per cohort)."""
    lines = [f"# Funnel — {candidate_set}", ""]
    lines.append("| cohort | n_lineages | n_tracks | n_frames | divides=none | pre | during | post |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cohort, df in cohort_dfs.items():
        if df.empty:
            lines.append(f"| {cohort} | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
            continue
        n_lineages = df["lineage_id"].nunique()
        n_tracks = df.groupby(["dataset_id", "fov_name", "track_id"]).ngroups
        n_frames = len(df)
        divides_counts = df.drop_duplicates(["dataset_id", "fov_name", "lineage_id"])["divides"].value_counts()
        lines.append(
            f"| {cohort} | {n_lineages} | {n_tracks} | {n_frames} | "
            f"{divides_counts.get('none', 0)} | {divides_counts.get('pre', 0)} | "
            f"{divides_counts.get('during', 0)} | {divides_counts.get('post', 0)} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Write per-cohort annotation CSVs for one candidate set."""
    parser = argparse.ArgumentParser(description="Auto-select candidates and tag cohorts (Stage 0).")
    parser.add_argument("--datasets", required=True, help="Path to datasets.yaml")
    parser.add_argument("--config", required=True, help="Path to candidates.yaml")
    parser.add_argument("--candidate-set", required=True, help="Name under config['candidate_sets']")
    args = parser.parse_args()

    config = load_stage_config(args.datasets, args.config)
    cand_sets = config.get("candidate_sets", {})
    if args.candidate_set not in cand_sets:
        raise KeyError(
            f"Candidate set {args.candidate_set!r} not in config['candidate_sets']. Known: {sorted(cand_sets)}"
        )
    cand_cfg = cand_sets[args.candidate_set]

    if cand_cfg.get("source") == "manual":
        raise SystemExit(
            f"Candidate set {args.candidate_set!r} declares source=manual; use manual_candidates.py instead."
        )

    dataset_ids = cand_cfg["datasets"]
    dataset_cfgs = {d["dataset_id"]: d for d in config["datasets"]}

    # Embedding pattern for LC predictions (sensor channel by convention).
    embedding_pattern = config.get("embeddings", {}).get("sensor", "*_viral_sensor_*.zarr")

    productive_parts: list[pd.DataFrame] = []
    bystander_parts: list[pd.DataFrame] = []
    abortive_parts: list[pd.DataFrame] = []
    mock_parts: list[pd.DataFrame] = []

    for ds_id in dataset_ids:
        if ds_id not in dataset_cfgs:
            raise KeyError(f"dataset_id {ds_id!r} not in datasets.yaml")
        ds_cfg = dataset_cfgs[ds_id]
        results = _build_dataset_cohorts(
            dataset_id=ds_id,
            dataset_cfg=ds_cfg,
            cand_cfg=cand_cfg,
            embedding_pattern=embedding_pattern,
        )
        meta = results["_meta"]
        bystander_df, abortive_df = _split_bystander_abortive(
            results["well_non_productive"],
            meta["lc_obs"],
            meta["bystander_fraction"],
            meta["abortive_min_run"],
        )
        if not results["productive"].empty:
            productive_parts.append(results["productive"])
        if not bystander_df.empty:
            bystander_parts.append(bystander_df)
        if not abortive_df.empty:
            abortive_parts.append(abortive_df)
        if not results["mock"].empty:
            mock_parts.append(results["mock"])

    if not productive_parts:
        raise RuntimeError(f"Candidate set {args.candidate_set!r} produced no productive lineages.")

    productive_df = pd.concat(productive_parts, ignore_index=True)
    bystander_df = (
        pd.concat(bystander_parts, ignore_index=True) if bystander_parts else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    abortive_df = (
        pd.concat(abortive_parts, ignore_index=True) if abortive_parts else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    mock_df = pd.concat(mock_parts, ignore_index=True) if mock_parts else pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Lineage reconnection — operates per (dataset, fov) group inside identify_lineages().
    productive_df = assign_lineage_ids(productive_df, return_both_branches=True)
    bystander_df = (
        assign_lineage_ids(bystander_df, return_both_branches=True) if not bystander_df.empty else bystander_df
    )
    abortive_df = assign_lineage_ids(abortive_df, return_both_branches=True) if not abortive_df.empty else abortive_df
    mock_df = assign_lineage_ids(mock_df, return_both_branches=True) if not mock_df.empty else mock_df

    # Cap productive lineages to max_lineages by length.
    max_lineages = cand_cfg.get("max_lineages")
    if max_lineages is not None:
        lineage_lengths = (
            productive_df.groupby(["dataset_id", "fov_name", "lineage_id"]).size().sort_values(ascending=False)
        )
        keep = set(lineage_lengths.head(max_lineages).index)
        mask = productive_df.apply(lambda r: (r["dataset_id"], r["fov_name"], int(r["lineage_id"])) in keep, axis=1)
        n_before = productive_df.groupby(["dataset_id", "fov_name", "lineage_id"]).ngroups
        productive_df = productive_df[mask].reset_index(drop=True)
        n_after = productive_df.groupby(["dataset_id", "fov_name", "lineage_id"]).ngroups
        _logger.info(f"Capped productive lineages from {n_before} to {n_after} (max {max_lineages})")

    # Compute t_zero per productive lineage and build the divides classifier.
    filter_cfg = cand_cfg["productive_filter"]
    t_zero_lookup = _t_zero_per_lineage(
        productive_df,
        anchor_label=filter_cfg["anchor_label"],
        anchor_positive=filter_cfg["anchor_positive"],
    )

    # Per-dataset window-frame conversion. Each lineage lives in one
    # dataset, so divides classification picks the frame interval from
    # its dataset.
    lineage_rules = cand_cfg.get("lineage_rules", {})
    k_pre_minutes = float(lineage_rules.get("transition_window_k_pre_minutes", 60))
    k_post_minutes = float(lineage_rules.get("transition_window_k_post_minutes", 120))
    window_frames_by_dataset: dict[str, tuple[int, int]] = {}
    for ds_id in dataset_ids:
        frame_interval = float(dataset_cfgs[ds_id]["frame_interval_minutes"])
        window_frames_by_dataset[ds_id] = (
            int(round(k_pre_minutes / frame_interval)),
            int(round(k_post_minutes / frame_interval)),
        )

    productive_out = _emit_cohort(productive_df, "productive", t_zero_lookup, window_frames_by_dataset)
    bystander_out = _emit_cohort(bystander_df, "bystander", t_zero_lookup, window_frames_by_dataset)
    abortive_out = _emit_cohort(abortive_df, "abortive", t_zero_lookup, window_frames_by_dataset)
    mock_out = _emit_cohort(mock_df, "mock", t_zero_lookup, window_frames_by_dataset)

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    cohort_dfs = {
        "productive": productive_out,
        "bystander": bystander_out,
        "abortive": abortive_out,
        "mock": mock_out,
    }
    for cohort, df in cohort_dfs.items():
        out_path = CANDIDATES_DIR / f"{args.candidate_set}_{cohort}.csv"
        df.to_csv(out_path, index=False)
        _logger.info(f"Wrote {out_path} ({len(df)} rows)")

    funnel_path = CANDIDATES_DIR / f"{args.candidate_set}_funnel.md"
    funnel_path.write_text(_funnel_report(args.candidate_set, cohort_dfs))
    _logger.info(f"Wrote {funnel_path}")


if __name__ == "__main__":
    main()
