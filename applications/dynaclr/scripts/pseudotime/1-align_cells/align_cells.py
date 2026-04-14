"""Stage 2: DTW-align cells to infection template.

Loads a pre-built template and aligns cell trajectories from one or more
datasets. Annotations are optional -- when not provided, raw frame times
are used instead of annotation-derived t_perturb.

Usage::

    uv run python align_cells.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
import zarr
from sklearn.decomposition import PCA

from dynaclr.evaluation.pseudotime.alignment import align_tracks
from dynaclr.evaluation.pseudotime.dtw_alignment import (
    TemplateResult,
    alignment_results_to_dataframe,
    dtw_align_tracks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _find_zarr(pred_dir: str, pattern: str) -> str:
    """Find a single zarr matching pattern in pred_dir."""
    import glob

    matches = glob.glob(str(Path(pred_dir) / pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No zarr matching {pattern} in {pred_dir}")
    return matches[0]


def _resolve_embeddings_path(ds: dict, config: dict) -> str:
    """Resolve embeddings path from either direct path or pred_dir + pattern."""
    if "embeddings_path" in ds:
        return ds["embeddings_path"]
    # Multi-template config: resolve from pred_dir + embedding pattern
    emb_patterns = config.get("embeddings", {})
    template_name = config.get("alignment", {}).get("template", "infection_nondividing")
    template_cfg = config.get("templates", {}).get(template_name, {})
    emb_key = template_cfg.get("embedding", "sensor")
    pattern = emb_patterns.get(emb_key, "timeaware_sensor_*.zarr")
    return _find_zarr(ds["pred_dir"], pattern)


def main() -> None:
    """Align cell tracks to template using DTW."""
    parser = argparse.ArgumentParser(description="DTW-align cells to template (Stage 2)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    script_dir = Path(__file__).resolve().parent
    pseudotime_dir = script_dir.parent
    output_dir = script_dir / "alignments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template from step 0
    alignment_cfg = config["alignment"]
    template_name = alignment_cfg.get("template", None)
    if template_name:
        template_path = pseudotime_dir / "0-build_templates" / "templates" / f"template_{template_name}.zarr"
    else:
        template_path = pseudotime_dir / "0-build_templates" / "templates" / "template.zarr"
    template_result, template_attrs = _load_template(template_path)
    use_subsequence = "crop_window_minutes" in template_attrs
    _logger.info(
        f"Loaded template from {template_path}, shape={template_result.template.shape}"
        f", subsequence={use_subsequence}"
        + (f", crop_window_minutes={template_attrs['crop_window_minutes']}" if use_subsequence else "")
    )

    min_track_minutes = alignment_cfg.get("min_track_minutes")

    template_name_safe = (template_name or "default").replace("/", "_")
    all_dfs = []
    for ds in alignment_cfg["datasets"]:
        dataset_id = ds["dataset_id"]
        _logger.info(f"Aligning dataset: {dataset_id}")

        emb_path = _resolve_embeddings_path(ds, config)
        adata = ad.read_zarr(emb_path)
        frame_interval = ds["frame_interval_minutes"]
        min_track_tp = int(min_track_minutes / frame_interval) if min_track_minutes is not None else 3
        _logger.info(f"  min_track_tp = {min_track_tp} frames ({min_track_minutes} min)")
        fov_pattern = ds.get("fov_pattern")

        annotations_path = ds.get("annotations_path")
        aligned = None

        # Try annotation-based alignment first
        if annotations_path is not None:
            annotations = _load_annotations(annotations_path, adata)
            aligned = align_tracks(
                annotations,
                frame_interval_minutes=frame_interval,
                fov_pattern=fov_pattern,
                min_track_timepoints=min_track_tp,
            )
            if len(aligned) > 0:
                _logger.info(f"  Aligned from annotations: {aligned.groupby(['fov_name', 'track_id']).ngroups} tracks")

        # Fall back to predictions if annotations gave nothing
        if (aligned is None or len(aligned) == 0) and "predicted_infection_state" in adata.obs.columns:
            _logger.info(f"  Falling back to predicted_infection_state for {dataset_id}")
            obs = adata.obs.copy()
            obs["infection_state"] = obs["predicted_infection_state"]
            if "parent_track_id" not in obs.columns:
                obs["parent_track_id"] = -1
            aligned = align_tracks(
                obs,
                frame_interval_minutes=frame_interval,
                fov_pattern=fov_pattern,
                min_track_timepoints=min_track_tp,
            )

        # Last resort: raw frame times
        if aligned is None or len(aligned) == 0:
            _logger.info(f"  No annotations/predictions for {dataset_id}, using raw frame times")
            obs = adata.obs.copy()
            if fov_pattern is not None:
                obs = obs[obs["fov_name"].str.contains(fov_pattern)]
            track_lengths = obs.groupby(["fov_name", "track_id"])["t"].transform("nunique")
            obs = obs[track_lengths >= min_track_tp].reset_index(drop=True)
            obs["t_perturb"] = 0
            obs["t_relative_minutes"] = obs["t"] * frame_interval
            aligned = obs

        valid_keys = set(zip(aligned["fov_name"], aligned["track_id"], aligned["t"]))
        mask = [(row["fov_name"], row["track_id"], row["t"]) in valid_keys for _, row in adata.obs.iterrows()]
        adata_filtered = adata[mask].copy()

        results = dtw_align_tracks(
            adata_filtered,
            aligned,
            template_result,
            dataset_id,
            min_track_timepoints=min_track_tp,
            subsequence=use_subsequence,
        )
        flat = alignment_results_to_dataframe(
            results, template_result.template_id, time_calibration=template_result.time_calibration
        )

        t_rel_map = aligned.set_index(["fov_name", "track_id", "t"])["t_relative_minutes"].to_dict()
        flat["t_relative_minutes"] = flat.apply(
            lambda row: t_rel_map.get((row["fov_name"], row["track_id"], row["t"]), np.nan),
            axis=1,
        )

        all_dfs.append(flat)
        _logger.info(f"  Aligned {len(results)} tracks, {len(flat)} timepoints")

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = output_dir / f"alignments_{template_name_safe}.parquet"
    combined.to_parquet(out_path, index=False)
    _logger.info(f"Saved {len(combined)} rows to {out_path}")


def _load_template(path: Path) -> tuple[TemplateResult, dict]:
    """Load TemplateResult from template.zarr.

    Returns
    -------
    tuple[TemplateResult, dict]
        The template result and the raw zarr attrs dict.
    """
    store = zarr.open(str(path), mode="r")

    template = np.array(store["template"])
    template_id = store.attrs["template_id"]
    n_input_tracks = store.attrs["n_input_tracks"]
    cell_ids = [tuple(c) for c in store.attrs["template_cell_ids"]]

    pca = None
    explained_variance = None
    if "pca_components" in store:
        n_comp = store.attrs["pca_n_components"]
        pca = PCA(n_components=n_comp)
        pca.components_ = np.array(store["pca_components"])
        pca.mean_ = np.array(store["pca_mean"])
        pca.explained_variance_ratio_ = np.array(store["pca_explained_variance_ratio"])
        pca.explained_variance_ = np.array(store["pca_explained_variance"])
        pca.n_components_ = n_comp
        pca.n_features_in_ = store.attrs.get("pca_n_features_in", pca.components_.shape[1])
        pca.n_samples_ = store.attrs.get("pca_n_samples_seen", 0)
        explained_variance = store.attrs.get("explained_variance")

    zscore_params = {}
    if "zscore_params" in store:
        for dataset_id in store["zscore_params"]:
            mean = np.array(store["zscore_params"][dataset_id]["mean"])
            std = np.array(store["zscore_params"][dataset_id]["std"])
            zscore_params[dataset_id] = (mean, std)

    template_labels = None
    if "template_labels" in store:
        node = store["template_labels"]
        if isinstance(node, zarr.Array):
            # Old single-array format → wrap as infection_state
            template_labels = {"infection_state": np.array(node)}
        else:
            # New group format: one array per label column
            template_labels = {col: np.array(node[col]) for col in node}

    time_calibration = None
    if "time_calibration" in store:
        time_calibration = np.array(store["time_calibration"])

    result = TemplateResult(
        template=template,
        template_id=template_id,
        pca=pca,
        zscore_params=zscore_params,
        template_cell_ids=cell_ids,
        n_input_tracks=n_input_tracks,
        explained_variance=explained_variance,
        template_labels=template_labels,
        time_calibration=time_calibration,
    )
    return result, dict(store.attrs)


def _load_annotations(annotations_path: str, adata: ad.AnnData) -> pd.DataFrame:
    """Load annotations CSV and merge with adata obs."""
    annotations = pd.read_csv(annotations_path)
    obs_cols = set(adata.obs.columns)
    ann_cols = set(annotations.columns)

    merge_cols = list({"fov_name", "track_id", "t"} & obs_cols & ann_cols)
    if merge_cols:
        return adata.obs.merge(annotations, on=merge_cols, how="left", suffixes=("", "_ann"))

    return annotations


if __name__ == "__main__":
    main()
