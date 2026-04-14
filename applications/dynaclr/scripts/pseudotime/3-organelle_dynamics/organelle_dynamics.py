"""Measure per-organelle embedding dynamics along infection pseudotime.

Uses the infection pseudotime from sensor DTW alignment, then loads
each organelle's embeddings and computes how they change relative
to a baseline (low-pseudotime cells).

This reveals the temporal ordering of organelle remodeling:
which organelle's embedding starts diverging first?

Usage::

    uv run python organelle_dynamics.py --config multi_template.yaml
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from dynaclr.evaluation.pseudotime.metrics import (
    aggregate_population,
    compute_track_timing,
    find_half_max_time,
    find_onset_time,
    find_peak_metrics,
    run_statistical_tests,
)
from dynaclr.evaluation.pseudotime.plotting import (
    plot_onset_comparison,
    plot_response_curves,
    plot_timing_distributions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _find_zarr(pred_dir: str, pattern: str) -> str:
    """Find a single zarr matching pattern in pred_dir."""
    matches = glob.glob(str(Path(pred_dir) / pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No zarr matching {pattern} in {pred_dir}")
    return matches[0]


def compute_organelle_distance(
    adata: ad.AnnData,
    aligned_cells: pd.DataFrame,
    baseline_pseudotime_range: tuple[float, float] = (0.0, 0.2),
    distance_metric: str = "cosine",
    pca_n_components: int = 20,
) -> pd.DataFrame:
    """Compute per-cell organelle embedding distance from baseline.

    Baseline is defined as cells with pseudotime in the specified range
    (i.e., cells at the start of the infection trajectory = uninfected-like).

    Parameters
    ----------
    adata : ad.AnnData
        Organelle embeddings.
    aligned_cells : pd.DataFrame
        Must have fov_name, track_id, t, pseudotime columns.
    baseline_pseudotime_range : tuple[float, float]
        Pseudotime range defining the baseline population.
    distance_metric : str
        Distance metric for scipy cdist.
    pca_n_components : int
        PCA components for organelle embeddings before distance.

    Returns
    -------
    pd.DataFrame
        aligned_cells with added 'organelle_distance' column.
    """
    result = aligned_cells.copy()

    # Build index: (fov_name, track_id, t) -> adata row
    obs = adata.obs.copy()
    obs["_iloc"] = np.arange(len(obs))
    obs_lookup = obs.set_index(["fov_name", "track_id", "t"])["_iloc"]

    # Match aligned cells to adata
    result_key = list(zip(result["fov_name"], result["track_id"], result["t"]))
    result_multi = pd.MultiIndex.from_tuples(result_key, names=["fov_name", "track_id", "t"])

    common = result_multi.intersection(obs_lookup.index)
    if len(common) == 0:
        result["organelle_distance"] = np.nan
        return result

    adata_idx = obs_lookup.reindex(common).to_numpy().astype(int)
    result_mask = result_multi.isin(common)
    result_rows = np.where(result_mask)[0]

    emb = adata.X[adata_idx]
    if hasattr(emb, "toarray"):
        emb = emb.toarray()
    emb = np.asarray(emb, dtype=np.float64)

    _logger.info(f"  Matched {len(common)} cells, PCA {emb.shape[1]} -> {pca_n_components}")

    # PCA
    pca = PCA(n_components=min(pca_n_components, emb.shape[1], emb.shape[0]))
    emb_pca = pca.fit_transform(emb)

    # Identify baseline cells (low pseudotime)
    pt_values = result.iloc[result_rows]["pseudotime"].to_numpy()
    bl_mask = (pt_values >= baseline_pseudotime_range[0]) & (pt_values <= baseline_pseudotime_range[1])
    n_baseline = bl_mask.sum()

    if n_baseline < 2:
        _logger.warning(f"  Only {n_baseline} baseline cells, using global mean")
        baseline = emb_pca.mean(axis=0, keepdims=True)
    else:
        baseline = emb_pca[bl_mask].mean(axis=0, keepdims=True)
        _logger.info(f"  Baseline: {n_baseline} cells (pseudotime {baseline_pseudotime_range})")

    # Compute distance from baseline
    distances = cdist(emb_pca, baseline, metric=distance_metric).flatten()

    result["organelle_distance"] = np.nan
    result.iloc[result_rows, result.columns.get_loc("organelle_distance")] = distances

    return result


def normalize_distance(
    df: pd.DataFrame,
    baseline_pseudotime_range: tuple[float, float] = (0.0, 0.2),
    signal_col: str = "organelle_distance",
) -> pd.DataFrame:
    """Z-score normalize distances relative to the baseline population.

    After normalization, baseline cells have mean ~0, std ~1.
    Positive values = more different from baseline than typical baseline variation.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'pseudotime' and signal_col columns.
    baseline_pseudotime_range : tuple[float, float]
        Pseudotime range defining baseline.
    signal_col : str
        Column to normalize.

    Returns
    -------
    pd.DataFrame
        Copy with added '{signal_col}_zscore' column.
    """
    result = df.copy()
    valid = result.dropna(subset=["pseudotime", signal_col])
    bl = valid[
        (valid["pseudotime"] >= baseline_pseudotime_range[0]) & (valid["pseudotime"] <= baseline_pseudotime_range[1])
    ]

    if len(bl) < 2:
        result[f"{signal_col}_zscore"] = np.nan
        return result

    bl_mean = bl[signal_col].mean()
    bl_std = bl[signal_col].std()
    if bl_std < 1e-10:
        bl_std = 1.0

    result[f"{signal_col}_zscore"] = (result[signal_col] - bl_mean) / bl_std
    return result


def main() -> None:
    """Compute per-organelle dynamics along infection pseudotime."""
    parser = argparse.ArgumentParser(description="Organelle dynamics along infection pseudotime")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--alignments", type=str, default=None, help="Path to alignments parquet file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    script_dir = Path(__file__).resolve().parent
    pseudotime_dir = script_dir.parent
    dynamics_dir = script_dir / "organelle_dynamics"
    dynamics_dir.mkdir(parents=True, exist_ok=True)

    emb_patterns = config["embeddings"]
    org_cfg = config["organelle_dynamics"]
    baseline_range = tuple(org_cfg["baseline_pseudotime_range"])
    n_bins_pseudotime = org_cfg.get("time_bins_pseudotime", 20)
    distance_metric = org_cfg.get("distance_metric", "cosine")

    # Load infection pseudotime alignments from step 1
    alignments_path = (
        Path(args.alignments)
        if args.alignments
        else pseudotime_dir / "1-align_cells" / "alignments" / "alignments.parquet"
    )
    if not alignments_path.exists():
        raise FileNotFoundError(
            f"{alignments_path} not found. Run align_cells.py first "
            f"(or build_templates.py + align_cells.py for multi-template)."
        )
    alignments = pd.read_parquet(alignments_path)
    _logger.info(f"Loaded {len(alignments)} alignment rows from {alignments_path}")

    # Determine time column for real-time analysis
    if "estimated_t_rel_minutes" in alignments.columns:
        time_col = "estimated_t_rel_minutes"
        _logger.info("Using estimated_t_rel_minutes for real-time analysis")
    elif "t_relative_minutes" in alignments.columns:
        time_col = "t_relative_minutes"
        _logger.info("Using t_relative_minutes for real-time analysis (no template calibration)")
    else:
        time_col = None
        _logger.info("No real-time column found; producing pseudotime-only outputs")

    # Per-organelle analysis
    all_organelle_data: list[pd.DataFrame] = []

    # Build dataset lookup from config
    ds_lookup = {ds["dataset_id"]: ds for ds in config["datasets"]}

    for org_name, org_settings in org_cfg["organelles"].items():
        _logger.info(f"=== {org_name}: {org_settings['label']} ===")
        emb_key = org_settings["embedding"]
        emb_pattern = emb_patterns[emb_key]

        # Which dataset_ids contain this organelle?
        org_dataset_ids = org_settings.get("dataset_ids", list(ds_lookup.keys()))

        all_ds_results = []

        for dataset_id in org_dataset_ids:
            ds = ds_lookup.get(dataset_id)
            if ds is None:
                _logger.warning(f"  Dataset {dataset_id} not found in config, skipping")
                continue

            ds_alignments = alignments[alignments["dataset_id"] == dataset_id]
            if len(ds_alignments) == 0:
                _logger.info(f"  No alignments for {dataset_id}, skipping")
                continue

            try:
                zarr_path = _find_zarr(ds["pred_dir"], emb_pattern)
            except FileNotFoundError:
                _logger.warning(f"  Skipping {org_name}/{dataset_id} — zarr not found")
                continue

            _logger.info(f"  Loading {org_name} embeddings for {dataset_id}")
            adata = ad.read_zarr(zarr_path)

            ds_result = compute_organelle_distance(
                adata,
                ds_alignments,
                baseline_pseudotime_range=baseline_range,
                distance_metric=distance_metric,
            )
            ds_result["organelle"] = org_name
            ds_result["dataset_id"] = dataset_id
            all_ds_results.append(ds_result)

        if len(all_ds_results) == 0:
            _logger.warning(f"  No data for {org_name}")
            continue

        combined = pd.concat(all_ds_results, ignore_index=True)
        combined = normalize_distance(combined, baseline_pseudotime_range=baseline_range)

        n_valid = combined["organelle_distance"].notna().sum()
        _logger.info(f"  {org_name}: {n_valid} cells with distance values")

        all_organelle_data.append(combined)

    if not all_organelle_data:
        _logger.warning("No organelle data computed. Exiting.")
        plt.close("all")
        return

    all_data = pd.concat(all_organelle_data, ignore_index=True)

    # Save per-cell data
    all_data.to_parquet(dynamics_dir / "organelle_distances.parquet", index=False)
    _logger.info(f"Saved per-cell data to {dynamics_dir / 'organelle_distances.parquet'}")

    organelle_configs = {name: cfg for name, cfg in org_cfg["organelles"].items()}

    # --- Secondary: pseudotime-binned aggregation (preserved from original) ---
    organelle_curves_pseudotime: dict[str, pd.DataFrame] = {}
    for org_name in organelle_configs:
        org_data = all_data[all_data["organelle"] == org_name]
        if len(org_data) == 0:
            continue
        bins = np.linspace(0, 1, n_bins_pseudotime + 1)
        org_data = org_data.copy()
        org_data["t_relative_minutes"] = org_data["pseudotime"]  # borrow column for aggregate_population
        pop_df = aggregate_population(
            org_data,
            time_bins=bins,
            signal_col="organelle_distance_zscore",
            signal_type="continuous",
        )
        # Rename time_minutes back to pseudotime_bin for secondary output
        pop_df = pop_df.rename(columns={"time_minutes": "pseudotime_bin"})
        # Rescale pseudotime_bin to [0,1] if needed (aggregate_population uses bin centers)
        if pop_df["pseudotime_bin"].max() > 1.0:
            pop_df["pseudotime_bin"] = pop_df["pseudotime_bin"] / pop_df["pseudotime_bin"].max()
        organelle_curves_pseudotime[org_name] = pop_df

    if organelle_curves_pseudotime:
        curves_list = []
        for org_name, curve in organelle_curves_pseudotime.items():
            c = curve.copy()
            c["organelle"] = org_name
            curves_list.append(c)
        pd.concat(curves_list, ignore_index=True).to_parquet(
            dynamics_dir / "aggregated_curves_pseudotime.parquet", index=False
        )

    # --- Primary: real-time analysis ---
    if time_col is None:
        _logger.info("Skipping real-time analysis (no time column).")
        plt.close("all")
        return

    # Build real-time bins: crop_window_minutes * 2 range or default ±600 min
    time_range_min = float(all_data[time_col].min())
    time_range_max = float(all_data[time_col].max())
    _logger.info(f"Real-time range: [{time_range_min:.0f}, {time_range_max:.0f}] min")
    time_bins = np.arange(
        np.floor(time_range_min / 30) * 30,
        np.ceil(time_range_max / 30) * 30 + 30,
        30,
    )

    organelle_curves_realtime: dict[str, pd.DataFrame] = {}
    timing_rows: list[dict] = []
    per_org_track_timing: list[pd.DataFrame] = []

    for org_name in organelle_configs:
        org_data = all_data[all_data["organelle"] == org_name].copy()
        if len(org_data) == 0:
            continue

        org_data["t_relative_minutes"] = org_data[time_col]
        org_data["signal"] = org_data["organelle_distance_zscore"]

        pop_df = aggregate_population(org_data, time_bins, signal_col="signal", signal_type="continuous")
        organelle_curves_realtime[org_name] = pop_df

        onset_minutes, threshold, bl_mean, bl_std = find_onset_time(
            pop_df, baseline_window=(-600, -60), sigma_threshold=2.0, signal_col="mean"
        )
        t50 = find_half_max_time(pop_df, signal_col="mean")
        peak_metrics = find_peak_metrics(pop_df, signal_col="mean")

        timing_rows.append(
            {
                "organelle": org_name,
                "T_onset_minutes": onset_minutes,
                "T_50_minutes": t50,
                **peak_metrics,
                "baseline_mean": bl_mean,
                "baseline_std": bl_std,
                "threshold": threshold,
                "n_tracks": org_data["cell_uid"].nunique() if "cell_uid" in org_data.columns else np.nan,
            }
        )

        org_data["marker"] = org_name
        track_timing = compute_track_timing(org_data, signal_col="signal", signal_type="continuous")
        track_timing["organelle"] = org_name
        per_org_track_timing.append(track_timing)

    # Save real-time aggregated curves
    if organelle_curves_realtime:
        curves_list = []
        for org_name, curve in organelle_curves_realtime.items():
            c = curve.copy()
            c["organelle"] = org_name
            curves_list.append(c)
        pd.concat(curves_list, ignore_index=True).to_parquet(
            dynamics_dir / "aggregated_curves_realtime.parquet", index=False
        )

    # Save timing summary
    if timing_rows:
        timing_df = pd.DataFrame(timing_rows).sort_values("T_onset_minutes")
        timing_df.to_parquet(dynamics_dir / "timing_summary.parquet", index=False)
        timing_df.to_csv(dynamics_dir / "timing_summary.csv", index=False)
        _logger.info("\n=== Organelle Timing Summary ===\n%s", timing_df.to_string(index=False))

    # Save per-track timing
    if per_org_track_timing:
        track_timing_df = pd.concat(per_org_track_timing, ignore_index=True)
        track_timing_df.to_parquet(dynamics_dir / "track_timing.parquet", index=False)

    # Statistical tests
    if per_org_track_timing and len(per_org_track_timing) >= 2:
        track_timing_df = pd.concat(per_org_track_timing, ignore_index=True)
        organelle_results = {
            org_name: {"combined_df": all_data[all_data["organelle"] == org_name].copy()}
            for org_name in organelle_configs
            if len(all_data[all_data["organelle"] == org_name]) > 0
        }
        try:
            stats = run_statistical_tests(organelle_results, track_timing_df)
            stats.to_parquet(dynamics_dir / "statistical_tests.parquet", index=False)
            stats.to_csv(dynamics_dir / "statistical_tests.csv", index=False)
            _logger.info("\n=== Statistical Tests ===\n%s", stats.to_string(index=False))
        except Exception as e:
            _logger.warning(f"Statistical tests failed: {e}")

    # Plots
    if organelle_curves_realtime:
        plot_response_curves(
            organelle_curves_realtime,
            organelle_configs,
            dynamics_dir,
            signal_type="continuous",
            title="Organelle remodeling — estimated real time",
            filename_prefix="organelle_dynamics_realtime",
        )
        _logger.info(f"Real-time response curves saved to {dynamics_dir}")

    if per_org_track_timing:
        track_timing_df = pd.concat(per_org_track_timing, ignore_index=True)
        plot_timing_distributions(track_timing_df, organelle_configs, dynamics_dir)
        _logger.info(f"Timing distributions saved to {dynamics_dir}")

    if timing_rows:
        timing_df = pd.DataFrame(timing_rows)
        timing_df["marker"] = timing_df["organelle"]
        # Add color from organelle_configs
        timing_df["color"] = timing_df["organelle"].map(
            {name: cfg.get("color", "#888888") for name, cfg in organelle_configs.items()}
        )
        plot_onset_comparison(timing_df, dynamics_dir)
        _logger.info(f"Onset comparison saved to {dynamics_dir}")

    plt.close("all")


if __name__ == "__main__":
    main()
