"""Stage 1: Build multiple DTW templates with track filtering.

Builds separate DBA templates for different biological programs:
- infection_nondividing: cleanest infection signal
- infection_dividing: infection + division
- division_uninfected: pure cell cycle

Each template filters tracks by division state and infection state
before running DBA.

Usage::

    uv run python \
        applications/dynaclr/scripts/pseudotime/0-build_templates/build_templates.py \
        --config applications/dynaclr/configs/pseudotime/multi_template.yaml
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
import zarr

from dynaclr.evaluation.pseudotime.alignment import align_tracks
from dynaclr.evaluation.pseudotime.dtw_alignment import build_infection_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _find_zarr(pred_dir: str, pattern: str) -> str:
    """Find a single zarr matching pattern in pred_dir."""
    matches = glob.glob(str(Path(pred_dir) / pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No zarr matching {pattern} in {pred_dir}")
    return matches[0]


def _load_annotations_with_tracking(annotations_path: str, adata: ad.AnnData) -> pd.DataFrame:
    """Load annotations and merge with adata obs."""
    annotations = pd.read_csv(annotations_path)
    merge_cols = ["fov_name", "track_id", "t"]
    return adata.obs.merge(annotations, on=merge_cols, how="left", suffixes=("", "_ann"))


def _division_timing(df: pd.DataFrame) -> pd.Series:
    """For each track, return when division occurs relative to infection onset.

    Returns a Series indexed by (fov_name, track_id) with values:

    - ``"before"``: division happens before first infected timepoint
    - ``"after"``: division happens after first infected timepoint
    - ``"no_division"``: track does not divide
    - ``"no_infection_onset"``: divides but no uninfected->infected transition visible
    """
    parent_set: set[tuple] = set()
    if "parent_track_id" in df.columns:
        for _, row in df[df["parent_track_id"] != -1][["fov_name", "parent_track_id"]].drop_duplicates().iterrows():
            parent_set.add((row["fov_name"], row["parent_track_id"]))

    records = []
    for (fov, tid), track in df.groupby(["fov_name", "track_id"]):
        has_parent = "parent_track_id" in track.columns and track["parent_track_id"].iloc[0] != -1
        has_children = (fov, tid) in parent_set
        divides = has_parent or has_children

        if not divides:
            records.append({"fov_name": fov, "track_id": tid, "division_timing": "no_division"})
            continue

        if "infection_state" not in track.columns:
            records.append({"fov_name": fov, "track_id": tid, "division_timing": "no_infection_onset"})
            continue

        infected_tps = track[track["infection_state"] == "infected"]["t"]
        uninfected_tps = track[track["infection_state"] == "uninfected"]["t"]
        if len(infected_tps) == 0 or len(uninfected_tps) == 0:
            records.append({"fov_name": fov, "track_id": tid, "division_timing": "no_infection_onset"})
            continue
        onset_t = int(infected_tps.min())

        if has_parent:
            div_t = int(track["t"].min())
        else:
            children_rows = df[(df["fov_name"] == fov) & (df["parent_track_id"] == tid)]
            if len(children_rows) == 0:
                records.append({"fov_name": fov, "track_id": tid, "division_timing": "no_infection_onset"})
                continue
            div_t = int(children_rows["t"].min())

        timing = "before" if div_t <= onset_t else "after"
        records.append({"fov_name": fov, "track_id": tid, "division_timing": timing})

    return pd.DataFrame(records).set_index(["fov_name", "track_id"])["division_timing"]


def _classify_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Add division and infection classification columns per track.

    Adds columns:

    - ``divides``: bool (track has parent or children)
    - ``infection_class``: ``"transitioning"`` | ``"infected_only"`` | ``"uninfected_only"`` | ``"unknown"``
    - ``division_timing``: ``"before"`` | ``"after"`` | ``"no_division"`` | ``"no_infection_onset"``
    """
    parent_set: set[tuple] = set()
    if "parent_track_id" in df.columns:
        children = df[df["parent_track_id"] != -1]
        for _, row in children[["fov_name", "parent_track_id"]].drop_duplicates().iterrows():
            parent_set.add((row["fov_name"], row["parent_track_id"]))

    track_classifications = []
    for (fov, tid), track in df.groupby(["fov_name", "track_id"]):
        has_parent = "parent_track_id" in track.columns and track["parent_track_id"].iloc[0] != -1
        has_children = (fov, tid) in parent_set
        divides = has_parent or has_children

        states = set(track["infection_state"].dropna().unique()) if "infection_state" in track.columns else set()
        infected = "infected" in states
        uninfected = "uninfected" in states

        if infected and uninfected:
            infection_class = "transitioning"
        elif infected:
            infection_class = "infected_only"
        elif uninfected:
            infection_class = "uninfected_only"
        else:
            infection_class = "unknown"

        for idx in track.index:
            track_classifications.append({"_idx": idx, "divides": divides, "infection_class": infection_class})

    class_df = pd.DataFrame(track_classifications).set_index("_idx")
    classified = df.join(class_df)

    timing = _division_timing(classified)
    # Expand Series back to per-row by joining on (fov_name, track_id)
    classified = classified.join(timing, on=["fov_name", "track_id"])
    return classified


def _filter_tracks_by_criteria(df: pd.DataFrame, track_filter: dict) -> pd.DataFrame:
    """Filter tracks based on template criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'divides', 'infection_class', and 'division_timing' columns
        from _classify_tracks.
    track_filter : dict
        Keys:

        - ``infection_state``: ``"transitioning"``, ``"uninfected_only"``, etc.
        - ``divides``: bool
        - ``division_timing``: ``"before"`` | ``"after"`` | ``"no_division"`` | ``"no_infection_onset"``
    """
    result = df.copy()

    infection_state = track_filter.get("infection_state")
    if infection_state is not None:
        result = result[result["infection_class"] == infection_state]

    divides = track_filter.get("divides")
    if divides is not None:
        result = result[result["divides"] == divides]

    division_timing = track_filter.get("division_timing")
    if division_timing is not None:
        result = result[result["division_timing"] == division_timing]

    return result


def _save_template(
    template_result,
    path: Path,
    config: dict,
    template_name: str,
    track_counts: dict | None = None,
) -> None:
    """Save template to zarr."""
    store = zarr.open(str(path), mode="w")
    store.create_array("template", data=template_result.template)

    attrs = {
        "template_id": template_result.template_id,
        "template_name": template_name,
        "n_input_tracks": template_result.n_input_tracks,
        "template_cell_ids": [list(c) for c in template_result.template_cell_ids],
    }

    if track_counts is not None:
        attrs["track_counts_per_dataset"] = track_counts

    if template_result.pca is not None:
        pca = template_result.pca
        store.create_array("pca_components", data=pca.components_)
        store.create_array("pca_mean", data=pca.mean_)
        store.create_array("pca_explained_variance_ratio", data=pca.explained_variance_ratio_)
        store.create_array("pca_explained_variance", data=pca.explained_variance_)
        attrs["pca_n_components"] = int(pca.n_components_)
        attrs["pca_n_features_in"] = int(pca.n_features_in_)
        attrs["pca_n_samples_seen"] = int(pca.n_samples_)

    if template_result.explained_variance is not None:
        attrs["explained_variance"] = template_result.explained_variance

    zscore_group = store.create_group("zscore_params")
    for dataset_id, (mean, std) in template_result.zscore_params.items():
        ds_group = zscore_group.create_group(dataset_id)
        ds_group.create_array("mean", data=mean)
        ds_group.create_array("std", data=std)

    if template_result.template_labels is not None:
        labels_group = store.create_group("template_labels")
        for col_name, col_arr in template_result.template_labels.items():
            labels_group.create_array(col_name, data=col_arr)

    if template_result.time_calibration is not None:
        store.create_array("time_calibration", data=template_result.time_calibration)

    # Store crop_window_minutes so downstream steps know to use subsequence DTW
    template_cfg = config.get("templates", {}).get(template_name, {})
    crop_window_minutes = template_cfg.get("crop_window_minutes")
    if crop_window_minutes is not None:
        attrs["crop_window_minutes"] = int(crop_window_minutes)

    attrs["config_snapshot"] = config
    store.attrs.update(attrs)


def main() -> None:
    """Build multiple templates from annotated datasets."""
    parser = argparse.ArgumentParser(description="Build multiple DTW templates (Stage 1)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "templates"
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_patterns = config["embeddings"]

    # Build each template
    for template_name, template_cfg in config["templates"].items():
        _logger.info("=" * 60)
        _logger.info(f"Building template: {template_name}")
        _logger.info(f"  {template_cfg.get('description', '')}")

        emb_pattern = emb_patterns[template_cfg["embedding"]]
        track_filter = template_cfg.get("track_filter", {})
        min_track_minutes = template_cfg.get("min_track_minutes")

        adata_dict: dict[str, ad.AnnData] = {}
        aligned_df_dict: dict[str, pd.DataFrame] = {}
        control_adata_dict: dict[str, ad.AnnData] = {}
        track_counts: dict[str, dict] = {}

        for ds in template_cfg["datasets"]:
            dataset_id = ds["dataset_id"]
            _logger.info(f"  Loading dataset: {dataset_id}")

            frame_interval = ds["frame_interval_minutes"]
            min_track_tp = int(min_track_minutes / frame_interval) if min_track_minutes is not None else 10
            _logger.info(f"    min_track_tp = {min_track_tp} frames ({min_track_minutes} min)")

            zarr_path = _find_zarr(ds["pred_dir"], emb_pattern)
            adata = ad.read_zarr(zarr_path)
            annotations = _load_annotations_with_tracking(ds["annotations_path"], adata)

            # Classify tracks by division and infection state
            classified = _classify_tracks(annotations)

            # Filter to desired tracks
            filtered = _filter_tracks_by_criteria(classified, track_filter)
            n_annotated = classified.groupby(["fov_name", "track_id"]).ngroups
            n_after_filter = filtered.groupby(["fov_name", "track_id"]).ngroups
            _logger.info(f"    Track filter: {n_annotated} -> {n_after_filter} tracks")

            if len(filtered) == 0:
                _logger.warning(f"    No tracks after filtering for {dataset_id}")
                continue

            # Align (compute t_perturb) — only for infection templates
            if track_filter.get("infection_state") in (
                "transitioning",
                "infected_only",
            ):
                aligned = align_tracks(
                    filtered,
                    frame_interval_minutes=ds["frame_interval_minutes"],
                    fov_pattern=ds.get("fov_pattern"),
                    min_track_timepoints=min_track_tp,
                )
            else:
                # For uninfected templates, no t_perturb — use raw time
                aligned = filtered.copy()
                track_lengths = aligned.groupby(["fov_name", "track_id"])["t"].transform("nunique")
                aligned = aligned[track_lengths >= min_track_tp].copy()
                aligned["t_perturb"] = 0
                aligned["t_relative_minutes"] = aligned["t"] * ds["frame_interval_minutes"]

            if len(aligned) == 0:
                _logger.warning(f"    No tracks after alignment for {dataset_id}")
                continue

            n_after_align = aligned.groupby(["fov_name", "track_id"]).ngroups
            track_counts[dataset_id] = {
                "n_annotated": n_annotated,
                "n_after_class_filter": n_after_filter,
                "n_after_min_timepoints": n_after_align,
            }

            adata_dict[dataset_id] = adata
            aligned_df_dict[dataset_id] = aligned

            # Control cells for PCA
            control_pattern = ds.get("control_fov_pattern")
            if control_pattern:
                ctrl_mask = adata.obs["fov_name"].astype(str).str.contains(control_pattern, regex=True).to_numpy()
                n_ctrl = int(ctrl_mask.sum())
                if n_ctrl > 0:
                    ctrl_X = adata.X[ctrl_mask]
                    if hasattr(ctrl_X, "toarray"):
                        ctrl_X = ctrl_X.toarray()
                    ctrl_obs = adata.obs.iloc[np.where(ctrl_mask)[0]].copy().reset_index(drop=True)
                    control_adata_dict[dataset_id] = ad.AnnData(X=np.asarray(ctrl_X), obs=ctrl_obs)
                    _logger.info(f"    Control cells for PCA: {n_ctrl}")

        if len(adata_dict) == 0:
            _logger.warning(f"  No data for template {template_name}, skipping")
            continue

        # Apply total track cap across all datasets (random sample, reproducible)
        max_tracks = template_cfg.get("max_tracks")
        if max_tracks is not None:
            all_track_ids = [
                (ds_id, fov, tid)
                for ds_id, df in aligned_df_dict.items()
                for (fov, tid) in df.groupby(["fov_name", "track_id"]).groups
            ]
            n_total = len(all_track_ids)
            if n_total > max_tracks:
                rng = np.random.default_rng(seed=0)
                keep = set(map(tuple, rng.choice(len(all_track_ids), size=max_tracks, replace=False).tolist()))
                keep_ids = {(all_track_ids[i][0], all_track_ids[i][1], all_track_ids[i][2]) for i in keep}
                aligned_df_dict = {
                    ds_id: df[df.apply(lambda r: (ds_id, r["fov_name"], r["track_id"]) in keep_ids, axis=1)]
                    for ds_id, df in aligned_df_dict.items()
                }
                _logger.info(f"  max_tracks cap: {n_total} -> {max_tracks} tracks (seed=0)")

        crop_window_minutes = template_cfg.get("crop_window_minutes")
        crop_window: dict[str, int] | None = None
        if crop_window_minutes is not None:
            crop_window = {
                ds["dataset_id"]: int(crop_window_minutes / ds["frame_interval_minutes"])
                for ds in template_cfg["datasets"]
                if ds["dataset_id"] in adata_dict
            }
            for ds_id, cw in crop_window.items():
                _logger.info(f"  [{ds_id}] crop_window = {cw} frames ({crop_window_minutes} min)")

        template_result = build_infection_template(
            adata_dict=adata_dict,
            aligned_df_dict=aligned_df_dict,
            pca_n_components=template_cfg.get("pca_n_components", 20),
            pca_variance_threshold=template_cfg.get("pca_variance_threshold"),
            dba_max_iter=template_cfg.get("dba_max_iter", 30),
            dba_tol=template_cfg.get("dba_tol", 1e-5),
            dba_init=template_cfg.get("dba_init", "medoid"),
            control_adata_dict=control_adata_dict if control_adata_dict else None,
            crop_window=crop_window,
        )

        template_path = output_dir / f"template_{template_name}.zarr"
        _save_template(template_result, template_path, config, template_name, track_counts)
        _logger.info(f"  Saved: {template_path}")
        _logger.info(f"  Shape: {template_result.template.shape}, from {template_result.n_input_tracks} tracks")


if __name__ == "__main__":
    main()
