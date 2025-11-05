# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import read_zarr
from iohub import open_ome_zarr

from viscy.data.triplet import TripletDataset
from viscy.representation.pseudotime import (
    CytoDtw,
    get_aligned_image_sequences,
)

# %%
logger = logging.getLogger("viscy")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configuration
NAPARI = True
if NAPARI:
    import os

    import napari

    os.environ["DISPLAY"] = ":1"
    viewer = napari.Viewer()

# %%
# File paths and configuration
output_root = Path(
    "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/infection_state/output/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
)

ALIGN_TYPE = "infection_apoptotic"  # Options: "cell_division" or "infection_state" or "apoptosis"
ALIGNMENT_CHANNEL = "sensor"  # sensor, phase, organelle

NORMALIZE_N_TIMEPOINTS = 3
# FOV filtering configuration
# Modify these for different datasets/experimental conditions
INFECTED_FOV_PATTERN = (
    "B/2"  # Pattern to match infected FOVs (e.g., "B/2", "infected", "treatment")
)
UNINFECTED_FOV_PATTERN = (
    "B/1"  # Pattern to match uninfected/control FOVs (e.g., "B/1", "control")
)
INFECTED_LABEL = "Infected"  # Label for infected condition in plots
UNINFECTED_LABEL = "Uninfected"  # Label for uninfected/control condition in plots

# Data paths
data_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
segmentation_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/train_test_mito_seg_2.zarr"  # Segmentation masks
features_path_sensor = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/anndata_predictions/sensor_160patch_104ckpt_ver3max.zarr"
features_path_phase = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/anndata_predictions/phase_160patch_104ckpt_ver3max.zarr"
features_path_organelle = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/anndata_predictions/organelle_160patch_104ckpt_ver3max.zarr"

# %%
# Load master features dataframe
master_features_path = (
    output_root / f"master_features_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}.csv"
)
master_df = pd.read_csv(master_features_path)
logger.info(f"Loaded master features from {master_features_path}")
logger.info(f"Shape: {master_df.shape}")
logger.info(f"Columns: {list(master_df.columns)}")

# Load alignment metadata
import pickle

metadata_path = output_root / f"alignment_metadata_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}.pkl"
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

consensus_lineage = metadata["consensus_pattern"]
consensus_annotations = metadata["consensus_annotations"]
consensus_metadata = metadata["consensus_metadata"]
reference_cell_info = metadata.get("reference_cell_info")
raw_infection_timepoint = metadata.get("raw_infection_timepoint")
aligned_region_bounds = metadata.get("aligned_region_bounds")

# Use raw infection timepoint from metadata (mapped from top-1 cell)
# This is the actual timepoint in the data, not the consensus index
infection_timepoint = raw_infection_timepoint

if infection_timepoint is not None:
    logger.info(f"Using raw infection timepoint from metadata: t={infection_timepoint}")
    if reference_cell_info is not None:
        logger.info(
            f"Reference cell: {reference_cell_info['fov_name']}, "
            f"track_ids={reference_cell_info['track_ids']}, "
            f"DTW distance={reference_cell_info.get('dtw_distance', 'N/A'):.3f}"
        )
else:
    # Fallback: compute from consensus annotations (legacy, likely incorrect for raw data)
    if consensus_annotations is not None and "infected" in consensus_annotations:
        infection_timepoint = consensus_annotations.index("infected")
        logger.warning(
            f"WARNING: Using consensus index as infection timepoint (t={infection_timepoint}). "
            f"This is likely INCORRECT for raw data! Re-run compute_alignment.py to fix."
        )

# Compute aligned region bounds from the consensus pattern length
if aligned_region_bounds is None:
    # Compute aligned region in RAW time space using reference cell info
    if reference_cell_info is not None and raw_infection_timepoint is not None:
        # The aligned region spans the consensus length centered around infection
        consensus_half_length = len(consensus_lineage) // 2
        aligned_region_bounds = (
            raw_infection_timepoint - consensus_half_length,
            raw_infection_timepoint + (len(consensus_lineage) - consensus_half_length),
        )
        logger.info(
            f"Computed aligned region in raw time space: {aligned_region_bounds} "
            f"(centered around infection t={raw_infection_timepoint}, consensus_length={len(consensus_lineage)})"
        )
    else:
        # Fallback: no meaningful bounds
        aligned_region_bounds = None
        logger.warning(
            "Could not compute aligned region bounds - missing reference cell info or raw infection timepoint"
        )

logger.info(f"Loaded alignment metadata from {metadata_path}")
logger.info(f"Infection timepoint: {infection_timepoint}")
logger.info(f"Aligned region: {aligned_region_bounds}")

# Load AnnData for CytoDtw methods (needed for plotting)
ad_features_alignment = read_zarr(
    features_path_sensor
    if ALIGNMENT_CHANNEL == "sensor"
    else (
        features_path_phase if ALIGNMENT_CHANNEL == "phase" else features_path_organelle
    )
)
cytodtw = CytoDtw(ad_features_alignment)

# Set the consensus pattern in CytoDtw instance for plotting methods
cytodtw.consensus_data = {
    "pattern": consensus_lineage,
    "annotations": consensus_annotations,
    "metadata": consensus_metadata,
}
logger.info("Set consensus pattern in CytoDtw instance for plotting")


# %%
# Baseline normalization utilities
def compute_baseline_from_aggregated_trajectory(
    df: pd.DataFrame,
    feature_columns: list,
    baseline_timepoints: tuple = None,
    n_baseline_timepoints: int = 10,
) -> dict:
    """
    Compute baseline values from early timepoints of an aggregated trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated trajectory dataframe with 't' and '{feature}_median' columns
    feature_columns : list
        Raw feature names (will look for '{feature}_median' columns)
    baseline_timepoints : tuple, optional
        (start, end) timepoint range for baseline. If None, uses first n timepoints
    n_baseline_timepoints : int
        Number of initial timepoints to use as baseline (only if baseline_timepoints is None)

    Returns
    -------
    dict
        Baseline values for each feature: {feature: baseline_value}
    """
    baseline_values = {}

    if baseline_timepoints is not None:
        baseline_mask = df["t"].between(baseline_timepoints[0], baseline_timepoints[1])
    else:
        sorted_times = sorted(df["t"].unique())
        baseline_times = sorted_times[:n_baseline_timepoints]
        baseline_mask = df["t"].isin(baseline_times)

    for feature in feature_columns:
        median_col = f"{feature}_median"

        if median_col not in df.columns:
            baseline_values[feature] = None
            continue

        baseline_vals = df.loc[baseline_mask, median_col].dropna()

        if len(baseline_vals) > 0:
            baseline_values[feature] = baseline_vals.mean()
        else:
            baseline_values[feature] = None

    return baseline_values


def normalize_aggregated_trajectory(
    df: pd.DataFrame,
    feature_columns: list,
    baseline_timepoints: tuple = None,
    n_baseline_timepoints: int = 10,
    suffix: str = "_normalized",
) -> pd.DataFrame:
    """
    Add baseline-normalized features to an aggregated trajectory dataframe.

    Modular function that operates on a single dataframe. Computes baselines from
    the dataframe itself and adds normalized columns.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated trajectory dataframe with 't' and '{feature}_median' columns
    feature_columns : list
        Raw feature names to normalize
    baseline_timepoints : tuple, optional
        (start, end) timepoint range for baseline. If None, uses first n timepoints
    n_baseline_timepoints : int
        Number of initial timepoints to use as baseline (only if baseline_timepoints is None)
    suffix : str
        Suffix for normalized column names (default: "_normalized")

    Returns
    -------
    pd.DataFrame
        Dataframe with normalized feature columns added (e.g., '{feature}_median_normalized')
    """
    df = df.copy()

    # Compute baselines from this dataframe
    baselines = compute_baseline_from_aggregated_trajectory(
        df, feature_columns, baseline_timepoints, n_baseline_timepoints
    )

    # Add normalized columns for each feature
    for feature in feature_columns:
        median_col = f"{feature}_median"
        baseline = baselines.get(feature)

        # Skip CV/SEM features (already relative metrics)
        if feature.endswith("_cv") or feature.endswith("_sem"):
            continue

        # Skip pre-normalized features
        if feature.startswith("normalized_"):
            continue

        if median_col in df.columns and baseline is not None:
            df[f"{median_col}{suffix}"] = (df[median_col] - baseline) / (
                np.abs(baseline) + 1e-6
            )

            # Also normalize q25 and q75 if present
            q25_col = f"{feature}_q25"
            q75_col = f"{feature}_q75"
            if q25_col in df.columns:
                df[f"{q25_col}{suffix}"] = (df[q25_col] - baseline) / (
                    np.abs(baseline) + 1e-6
                )
            if q75_col in df.columns:
                df[f"{q75_col}{suffix}"] = (df[q75_col] - baseline) / (
                    np.abs(baseline) + 1e-6
                )

    return df


# %%
# Unified trajectory aggregation function
def aggregate_trajectory_by_time(
    df: pd.DataFrame,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Unified function to aggregate trajectory data by timepoint.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-filtered dataframe with features and 't' column
    feature_columns : list
        Features to aggregate

    Returns
    -------
    pd.DataFrame
        Aggregated trajectory with columns: t, n_cells, {feature}_{stat}
        where stat is one of: mean, median, std, q25, q75
    """
    aggregated_data = []

    for t in sorted(df["t"].unique()):
        time_slice = df[df["t"] == t]

        row_data = {
            "t": t,
            "n_cells": len(time_slice),
        }

        # Compute statistics for each feature
        for feature in feature_columns:
            if feature not in time_slice.columns:
                # Feature doesn't exist - set all stats to NaN
                row_data[f"{feature}_median"] = np.nan
                row_data[f"{feature}_mean"] = np.nan
                row_data[f"{feature}_std"] = np.nan
                row_data[f"{feature}_q25"] = np.nan
                row_data[f"{feature}_q75"] = np.nan
                continue

            values = time_slice[feature].dropna()

            if len(values) == 0:
                # No valid values - set all stats to NaN
                row_data[f"{feature}_median"] = np.nan
                row_data[f"{feature}_mean"] = np.nan
                row_data[f"{feature}_std"] = np.nan
                row_data[f"{feature}_q25"] = np.nan
                row_data[f"{feature}_q75"] = np.nan
                continue

            # Compute all statistics
            row_data[f"{feature}_mean"] = values.mean()
            row_data[f"{feature}_median"] = values.median()
            row_data[f"{feature}_std"] = values.std()
            row_data[f"{feature}_q25"] = values.quantile(0.25)
            row_data[f"{feature}_q75"] = values.quantile(0.75)

        aggregated_data.append(row_data)

    result_df = pd.DataFrame(aggregated_data)

    logger.info(
        f"Aggregated {len(result_df)} timepoints from {len(df)} total observations"
    )

    return result_df


# %%
# Data filtering and preparation
min_track_length = 20

# Filter to infected cells using configured pattern
filtered_infected_df = master_df[
    master_df["fov_name"].str.contains(INFECTED_FOV_PATTERN)
].copy()

# Filter by track length
track_lengths = filtered_infected_df.groupby(["fov_name", "track_id"]).size()
valid_tracks = track_lengths[track_lengths >= min_track_length].index
filtered_infected_df = filtered_infected_df[
    filtered_infected_df.set_index(["fov_name", "track_id"]).index.isin(valid_tracks)
].reset_index(drop=True)

logger.info(
    f"Filtered {INFECTED_LABEL} cells ({INFECTED_FOV_PATTERN}): "
    f"{len(valid_tracks)} tracks with >= {min_track_length} timepoints"
)

# Filter to uninfected cells using configured pattern
uninfected_filtered_df = master_df[
    master_df["fov_name"].str.contains(UNINFECTED_FOV_PATTERN)
].copy()

# Filter by track length
track_lengths = uninfected_filtered_df.groupby(["fov_name", "track_id"]).size()
valid_tracks = track_lengths[track_lengths >= min_track_length].index
uninfected_filtered_df = uninfected_filtered_df[
    uninfected_filtered_df.set_index(["fov_name", "track_id"]).index.isin(valid_tracks)
].reset_index(drop=True)

logger.info(
    f"Filtered {UNINFECTED_LABEL} cells ({UNINFECTED_FOV_PATTERN}): "
    f"{len(valid_tracks)} tracks with >= {min_track_length} timepoints"
)

# %%
# Select features for analysis
common_response_features = [
    "organelle_PC1",
    "organelle_PC2",
    "organelle_PC3",
    "phase_PC1",
    "phase_PC2",
    "phase_PC3",
    "edge_density",
    "organelle_volume",
    "homogeneity",
    "contrast",
    "segs_count",
    "segs_mean_area",
    "segs_mean_eccentricity",
    "segs_mean_frangi_mean",
    "segs_circularity_mean",
    "segs_circularity_cv",
    "segs_eccentricity_cv",
    "segs_area_cv",
]

# %%
# Compute aggregated trajectories

# 1. Common response from top N aligned cells
top_n_cells = 10
alignment_col = f"dtw_{ALIGN_TYPE}_aligned"

# Get aligned cells only
aligned_cells = filtered_infected_df[filtered_infected_df[alignment_col]].copy()

# Select top N lineages by distance
if "distance" in aligned_cells.columns and "lineage_id" in aligned_cells.columns:
    top_lineages = (
        aligned_cells.drop_duplicates("lineage_id")
        .nsmallest(top_n_cells, "distance")["lineage_id"]
        .tolist()
    )
    logger.info(
        f"Selected top {len(top_lineages)} lineages by DTW distance: {top_lineages}"
    )
else:
    top_lineages = aligned_cells["lineage_id"].unique()[:top_n_cells].tolist()
    logger.info(f"Selected {len(top_lineages)} lineages (no distance info)")

# Filter to top lineages - include ALL timepoints (aligned and unaligned)
top_cells_df = filtered_infected_df[
    filtered_infected_df["lineage_id"].isin(top_lineages)
].copy()

logger.info(
    f"Filtered to {len(top_cells_df)} observations from {len(top_lineages)} lineages"
)

# Aggregate common response
common_response_df = aggregate_trajectory_by_time(
    top_cells_df,
    feature_columns=common_response_features,
)

# 2. Uninfected baseline
uninfected_baseline_df = aggregate_trajectory_by_time(
    uninfected_filtered_df,
    feature_columns=common_response_features,
)

# 3. Global infected average (all B/2 cells)
global_infected_df = aggregate_trajectory_by_time(
    filtered_infected_df,
    feature_columns=common_response_features,
)

logger.info(f"Common response shape: {common_response_df.shape}")
logger.info(f"Uninfected baseline shape: {uninfected_baseline_df.shape}")
logger.info(f"Global infected shape: {global_infected_df.shape}")

# Apply baseline normalization to each dataframe independently
logger.info("Normalizing each trajectory to its own baseline (first 10 timepoints)")
common_response_df = normalize_aggregated_trajectory(
    common_response_df,
    feature_columns=common_response_features,
    n_baseline_timepoints=NORMALIZE_N_TIMEPOINTS,
)
uninfected_baseline_df = normalize_aggregated_trajectory(
    uninfected_baseline_df,
    feature_columns=common_response_features,
    n_baseline_timepoints=NORMALIZE_N_TIMEPOINTS,
)
global_infected_df = normalize_aggregated_trajectory(
    global_infected_df,
    feature_columns=common_response_features,
    n_baseline_timepoints=NORMALIZE_N_TIMEPOINTS,
)


# %%
# Legacy function for backward compatibility with plotting functions
# NOTE: This function is deprecated. Use normalize_aggregated_trajectory() instead,
# which adds normalized columns directly to dataframes (per-dataframe approach).
def compute_baseline_normalization_values(
    infected_df: pd.DataFrame,
    uninfected_df: pd.DataFrame,
    global_infected_df: pd.DataFrame,
    feature_columns: list,
    n_baseline_timepoints: int = 10,
):
    """
    [DEPRECATED] Compute baseline normalization values from the first n timepoints of each trajectory.

    This function returns a nested dict for backward compatibility with plotting functions.
    For new code, use normalize_aggregated_trajectory() which operates on individual dataframes.

    Parameters
    ----------
    infected_df : pd.DataFrame
        Infected common response aggregated dataframe with 't' column
    uninfected_df : pd.DataFrame
        Uninfected baseline aggregated dataframe with 't' column
    global_infected_df : pd.DataFrame
        Global infected average aggregated dataframe with 't' column
    feature_columns : list
        Features to compute baselines for
    n_baseline_timepoints : int
        Number of initial timepoints to use as baseline (default: 10)

    Returns
    -------
    dict
        Baseline values for each feature and trajectory type
    """
    baseline_values = {}

    logger.info(
        f"Computing baseline from first {n_baseline_timepoints} timepoints of each trajectory"
    )

    for feature in feature_columns:
        median_col = f"{feature}_median"
        baseline_values[feature] = {}

        # Infected trajectory baseline
        if median_col in infected_df.columns:
            sorted_times = sorted(infected_df["t"].unique())
            baseline_times = sorted_times[:n_baseline_timepoints]
            baseline_mask = infected_df["t"].isin(baseline_times)
            baseline_vals = infected_df.loc[baseline_mask, median_col].dropna()
            if len(baseline_vals) > 0:
                baseline_values[feature]["infected"] = baseline_vals.mean()
            else:
                baseline_values[feature]["infected"] = None

        # Uninfected trajectory baseline
        if median_col in uninfected_df.columns:
            sorted_times = sorted(uninfected_df["t"].unique())
            baseline_times = sorted_times[:n_baseline_timepoints]
            baseline_mask = uninfected_df["t"].isin(baseline_times)
            baseline_vals = uninfected_df.loc[baseline_mask, median_col].dropna()
            if len(baseline_vals) > 0:
                baseline_values[feature]["uninfected"] = baseline_vals.mean()
            else:
                baseline_values[feature]["uninfected"] = None

        # Global infected trajectory baseline
        if global_infected_df is not None and median_col in global_infected_df.columns:
            sorted_times = sorted(global_infected_df["t"].unique())
            baseline_times = sorted_times[:n_baseline_timepoints]
            baseline_mask = global_infected_df["t"].isin(baseline_times)
            baseline_vals = global_infected_df.loc[baseline_mask, median_col].dropna()
            if len(baseline_vals) > 0:
                baseline_values[feature]["global"] = baseline_vals.mean()
            else:
                baseline_values[feature]["global"] = None

    return baseline_values


baseline_normalization_values = compute_baseline_normalization_values(
    common_response_df,
    uninfected_baseline_df,
    global_infected_df,
    feature_columns=common_response_features,
    n_baseline_timepoints=NORMALIZE_N_TIMEPOINTS,  # Match baseline timepoints from compute_alignment.py (1-10)
)

logger.info(
    f"Computed baseline values for {len(baseline_normalization_values)} features"
)


# %%
def plot_binned_period_comparison(
    infected_df: pd.DataFrame,
    uninfected_df: pd.DataFrame,
    feature_columns: list,
    periods: dict,
    baseline_period_name: str = None,
    infection_time: int = None,
    baseline_values: dict = None,
    global_infected_df: pd.DataFrame = None,
    output_root: Path = None,
    figsize=(18, 14),
    plot_type: str = "line",
    add_stats: bool = True,
    infected_label: str = "Infected",
    uninfected_label: str = "Uninfected",
):
    """
    Plot binned period comparison showing fold-change across biological phases.

    Parameters
    ----------
    infected_df : pd.DataFrame
        Infected common response aggregated dataframe
    uninfected_df : pd.DataFrame
        Uninfected baseline aggregated dataframe
    feature_columns : list
        Features to plot
    periods : dict
        Dictionary defining time periods for binning.
        Keys are period labels (str), values are tuples (start_time, end_time).
        Example: {"Baseline": (0, 10), "Early": (10, 20), "Late": (20, 30)}
    baseline_period_name : str, optional
        Name of the period to use as baseline for normalization.
        If None, uses the first period in the periods dict.
    infection_time : int, optional
        Infection timepoint (only used to mark infection event on plot)
    baseline_values : dict, optional
        Pre-computed baseline normalization values
    global_infected_df : pd.DataFrame, optional
        Global average of all infected cells
    output_root : Path, optional
        Directory to save output figure
    figsize : tuple
        Figure size
    plot_type : str
        'line' for connected line plots or 'bar' for grouped bar plots (default: 'line')
    add_stats : bool
        If True, perform statistical testing and mark significant differences (default: True)
    infected_label : str
        Label for infected condition in plots
    uninfected_label : str
        Label for uninfected/control condition in plots
    """
    from scipy.stats import ttest_ind

    # Determine baseline period for normalization
    if baseline_period_name is None:
        # Use first period as baseline
        baseline_period_name = list(periods.keys())[0]

    if baseline_period_name not in periods:
        raise ValueError(
            f"Baseline period '{baseline_period_name}' not found in periods dict. "
            f"Available periods: {list(periods.keys())}"
        )

    period_names = list(periods.keys())
    n_periods = len(periods)

    n_features = len(feature_columns)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    # Colorblind-friendly palette
    colors = {
        "uninfected": "#1f77b4",  # blue
        "infected": "#ff7f0e",  # orange
        "global": "#2ca02c",  # green
    }

    # Store statistical results for logging
    stats_results = {}

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        median_col = f"{feature}_median"

        # Check if feature exists
        if (
            median_col not in infected_df.columns
            or median_col not in uninfected_df.columns
        ):
            ax.text(0.5, 0.5, f"{feature}\nno data", ha="center", va="center")
            ax.set_title(feature)
            continue

        # Check if CV/SEM feature or pre-normalized feature (no additional normalization)
        is_cv_feature = feature.endswith("_cv") or feature.endswith("_sem")
        is_prenormalized = feature.startswith("normalized_")

        # Compute values for each period
        period_values = {"uninfected": [], "infected": [], "global": []}
        period_errors = {"uninfected": [], "infected": [], "global": []}

        # Baseline period values for normalization
        baseline_period = periods[baseline_period_name]

        def compute_baseline_value(df, feature_col):
            """Compute baseline value from baseline period."""
            mask = (df["t"] >= baseline_period[0]) & (df["t"] <= baseline_period[1])
            values = df.loc[mask, feature_col].dropna()
            return values.mean() if len(values) > 0 else None

        # Compute baseline for this feature from each trajectory
        uninfected_baseline = compute_baseline_value(uninfected_df, median_col)
        infected_baseline = compute_baseline_value(infected_df, median_col)
        global_baseline = None
        if global_infected_df is not None and median_col in global_infected_df.columns:
            global_baseline = compute_baseline_value(global_infected_df, median_col)

        # For each period, compute aggregate value and normalize
        for period_name, (t_start, t_end) in periods.items():
            # Uninfected
            mask = (uninfected_df["t"] >= t_start) & (uninfected_df["t"] <= t_end)
            values = uninfected_df.loc[mask, median_col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()

                # Normalize to baseline if not CV feature and not pre-normalized
                if (
                    not is_cv_feature
                    and not is_prenormalized
                    and uninfected_baseline is not None
                ):
                    mean_val = mean_val / uninfected_baseline
                    std_val = std_val / (np.abs(uninfected_baseline) + 1e-6)

                period_values["uninfected"].append(mean_val)
                period_errors["uninfected"].append(std_val)
            else:
                period_values["uninfected"].append(np.nan)
                period_errors["uninfected"].append(np.nan)

            # Infected
            mask = (infected_df["t"] >= t_start) & (infected_df["t"] <= t_end)
            values = infected_df.loc[mask, median_col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()

                if (
                    not is_cv_feature
                    and not is_prenormalized
                    and infected_baseline is not None
                ):
                    mean_val = mean_val / infected_baseline
                    std_val = std_val / (np.abs(infected_baseline) + 1e-6)

                period_values["infected"].append(mean_val)
                period_errors["infected"].append(std_val)
            else:
                period_values["infected"].append(np.nan)
                period_errors["infected"].append(np.nan)

            # Global infected
            if (
                global_infected_df is not None
                and median_col in global_infected_df.columns
            ):
                mask = (global_infected_df["t"] >= t_start) & (
                    global_infected_df["t"] <= t_end
                )
                values = global_infected_df.loc[mask, median_col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()

                    if (
                        not is_cv_feature
                        and not is_prenormalized
                        and global_baseline is not None
                    ):
                        mean_val = mean_val / global_baseline
                        std_val = std_val / (np.abs(global_baseline) + 1e-6)

                    period_values["global"].append(mean_val)
                    period_errors["global"].append(std_val)
                else:
                    period_values["global"].append(np.nan)
                    period_errors["global"].append(np.nan)

        # Statistical testing between infected and uninfected at each period
        p_values = []
        if add_stats:
            stats_results[feature] = {}
            for period_name, (t_start, t_end) in periods.items():
                # Get raw values for statistical testing
                uninfected_mask = (uninfected_df["t"] >= t_start) & (
                    uninfected_df["t"] <= t_end
                )
                infected_mask = (infected_df["t"] >= t_start) & (
                    infected_df["t"] <= t_end
                )

                uninfected_vals = uninfected_df.loc[
                    uninfected_mask, median_col
                ].dropna()
                infected_vals = infected_df.loc[infected_mask, median_col].dropna()

                if len(uninfected_vals) >= 3 and len(infected_vals) >= 3:
                    _, p_val = ttest_ind(uninfected_vals, infected_vals)
                    p_values.append(p_val)
                    stats_results[feature][period_name] = p_val
                else:
                    p_values.append(np.nan)
                    stats_results[feature][period_name] = np.nan

        x = np.arange(n_periods)

        if plot_type == "line":
            # Line plot with error bars
            ax.errorbar(
                x,
                period_values["uninfected"],
                yerr=period_errors["uninfected"],
                label=f"{uninfected_label}",
                color=colors["uninfected"],
                marker="o",
                markersize=8,
                linewidth=2.5,
                capsize=4,
                capthick=2,
            )
            ax.errorbar(
                x,
                period_values["infected"],
                yerr=period_errors["infected"],
                label=f"{infected_label} (top-N)",
                color=colors["infected"],
                marker="s",
                markersize=8,
                linewidth=2.5,
                capsize=4,
                capthick=2,
            )

            if global_infected_df is not None:
                ax.errorbar(
                    x,
                    period_values["global"],
                    yerr=period_errors["global"],
                    label=f"All {infected_label}",
                    color=colors["global"],
                    marker="^",
                    markersize=7,
                    linewidth=2,
                    linestyle="--",
                    capsize=4,
                    capthick=1.5,
                    alpha=0.8,
                )

            # Mark significant differences with asterisks
            if add_stats and len(p_values) > 0:
                # Get valid (non-NaN) values for computing y_max
                valid_uninfected = [
                    v for v in period_values["uninfected"] if not np.isnan(v)
                ]
                valid_infected = [
                    v for v in period_values["infected"] if not np.isnan(v)
                ]

                # Only add markers if we have valid values
                if len(valid_uninfected) > 0 and len(valid_infected) > 0:
                    y_max = max(max(valid_uninfected), max(valid_infected))
                    y_offset = 0.1 * (y_max - 1.0) if not is_cv_feature else 0.1 * y_max

                    for i, p_val in enumerate(p_values):
                        if not np.isnan(p_val):
                            # Determine significance level
                            if p_val < 0.001:
                                marker = "***"
                            elif p_val < 0.01:
                                marker = "**"
                            elif p_val < 0.05:
                                marker = "*"
                            else:
                                marker = "ns"

                            if marker != "ns":
                                # Position text above the higher of the two values
                                # Skip if either value is NaN
                                if not np.isnan(
                                    period_values["uninfected"][i]
                                ) and not np.isnan(period_values["infected"][i]):
                                    max_val = max(
                                        period_values["uninfected"][i],
                                        period_values["infected"][i],
                                    )
                                    ax.text(
                                        x[i],
                                        max_val + y_offset,
                                        marker,
                                        ha="center",
                                        va="bottom",
                                        fontsize=10,
                                        fontweight="bold",
                                        color="black",
                                    )

        else:  # bar plot
            width = 0.25

            ax.bar(
                x - width,
                period_values["uninfected"],
                width,
                label=f"{uninfected_label}",
                color=colors["uninfected"],
                yerr=period_errors["uninfected"],
                capsize=3,
            )
            ax.bar(
                x,
                period_values["infected"],
                width,
                label=f"{infected_label} (top-N)",
                color=colors["infected"],
                yerr=period_errors["infected"],
                capsize=3,
            )

            if global_infected_df is not None:
                ax.bar(
                    x + width,
                    period_values["global"],
                    width,
                    label=f"All {infected_label}",
                    color=colors["global"],
                    yerr=period_errors["global"],
                    capsize=3,
                    alpha=0.8,
                )

        # Add horizontal line at 1.0 (no change from baseline)
        if not is_cv_feature:
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Period")
        if is_cv_feature:
            ax.set_ylabel(f"{feature}\n(raw value)")
        elif is_prenormalized:
            ax.set_ylabel(f"{feature}\n(pre-normalized, baseline t=1-10)")
        else:
            ax.set_ylabel(f"{feature}\n(fold-change from baseline)")
        ax.set_title(feature)
        ax.set_xticks(x)
        ax.set_xticklabels(period_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    # Create title with statistical note
    title = "Binned Period Comparison: Fold-Change Across Infection Phases"
    if add_stats:
        title += "\n(* p<0.05, ** p<0.01, *** p<0.001)"

    plt.suptitle(
        title,
        fontsize=14,
        y=1.00,
    )
    plt.tight_layout()

    if output_root is not None:
        save_path = output_root / f"binned_period_comparison_{ALIGN_TYPE}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved binned period comparison to {save_path}")

    plt.show()

    # Log summary in markdown format
    logger.info("\n## Binned Period Comparison Summary")
    logger.info(f"**Infection timepoint:** {infection_time}")
    logger.info("\n### Period Definitions")
    for period_name, (t_start, t_end) in periods.items():
        logger.info(f"- **{period_name}:** t={t_start} to t={t_end}")

    if add_stats and len(stats_results) > 0:
        logger.info("\n### Statistical Significance (t-tests)")
        logger.info(
            "Comparing infected top-N vs uninfected at each period. Significance levels: * p<0.05, ** p<0.01, *** p<0.001\n"
        )

        # Create markdown table
        logger.info(f"| Feature | {' | '.join(period_names)} |")
        logger.info(f"|---------|{'---------|-' * (len(period_names) - 1)}---------|")

        for feature, period_results in stats_results.items():
            sig_markers = []
            for period_name in period_names:
                p_val = period_results.get(period_name, np.nan)
                if np.isnan(p_val):
                    sig_markers.append("N/A")
                elif p_val < 0.001:
                    sig_markers.append(f"***({p_val:.3e})")
                elif p_val < 0.01:
                    sig_markers.append(f"**({p_val:.3f})")
                elif p_val < 0.05:
                    sig_markers.append(f"*({p_val:.3f})")
                else:
                    sig_markers.append(f"ns({p_val:.3f})")

            logger.info(f"| {feature} | {' | '.join(sig_markers)} |")

        logger.info("\nns = not significant (p >= 0.05)")


# %%
def plot_infected_vs_uninfected_comparison(
    infected_df: pd.DataFrame,
    uninfected_df: pd.DataFrame,
    feature_columns: list,
    baseline_values: dict = None,
    infection_time: int = None,
    aligned_region: tuple = None,
    figsize=(18, 14),
    n_consecutive_divergence: int = 5,
    global_infected_df: pd.DataFrame = None,
    normalize_to_baseline: bool = True,
    infected_label: str = "Infected",
    uninfected_label: str = "Uninfected",
):
    """
    Plot comparison of infected (DTW-aligned) vs uninfected (baseline) trajectories.

    Shows where infected cells diverge from normal behavior (crossover points).

    This function preferentially uses pre-computed normalized columns from dataframes
    (e.g., '{feature}_median_normalized') if they exist. Otherwise, it falls back to
    computing normalization on-the-fly using baseline_values.

    NOTE: Features ending in '_cv' or '_sem' are plotted as raw values without baseline
    normalization, since CV and SEM are already relative/uncertainty metrics.

    Parameters
    ----------
    infected_df : pd.DataFrame
        Infected common response with 't' column and optional '{feature}_*_normalized' columns
    uninfected_df : pd.DataFrame
        Uninfected baseline with optional '{feature}_*_normalized' columns
    feature_columns : list
        Features to plot
    baseline_values : dict, optional
        Pre-computed baseline normalization values for each feature and trajectory.
        Only used if normalized columns don't exist in dataframes (legacy fallback).
    infection_time : int, optional
        Pre-computed infection timepoint in raw time
    aligned_region : tuple, optional
        Pre-computed aligned region boundaries (start, end)
    figsize : tuple
        Figure size
    n_consecutive_divergence : int
        Number of consecutive timepoints required to confirm divergence (default: 5)
    global_infected_df : pd.DataFrame, optional
        Global average of ALL infected cells without alignment, with optional normalized columns
    normalize_to_baseline : bool
        If True, use normalized columns or compute normalization to show fold-change
    infected_label : str
        Label for infected condition in plots
    uninfected_label : str
        Label for uninfected/control condition in plots
    """
    from scipy.interpolate import interp1d

    n_features = len(feature_columns)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    # Colorblind-friendly palette
    uninfected_color = "#1f77b4"  # blue
    infected_color = "#ff7f0e"  # orange

    # Use pre-computed baseline values or initialize empty dict
    if baseline_values is None:
        baseline_values = {}
        logger.warning("No baseline_values provided - plots will not be normalized")

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        median_col = f"{feature}_median"
        q25_col = f"{feature}_q25"
        q75_col = f"{feature}_q75"

        # Check if data exists
        if (
            median_col not in infected_df.columns
            or median_col not in uninfected_df.columns
        ):
            ax.text(0.5, 0.5, f"{feature}\nno data", ha="center", va="center")
            ax.set_title(feature)
            continue

        # Highlight aligned region first (background layer)
        if aligned_region is not None:
            ax.axvspan(
                aligned_region[0],
                aligned_region[1],
                alpha=0.1,
                color="gray",
                label="Aligned region",
                zorder=0,
            )

        # Plot uninfected baseline
        uninfected_time = uninfected_df["t"].values

        # Check if this is a CV/SEM or pre-normalized feature
        is_cv_feature = feature.endswith("_cv") or feature.endswith("_sem")
        is_prenormalized = feature.startswith("normalized_")

        # Check if normalized columns exist in dataframe (from normalize_aggregated_trajectory)
        normalized_median_col = f"{median_col}_normalized"
        normalized_q25_col = f"{q25_col}_normalized"
        normalized_q75_col = f"{q75_col}_normalized"
        has_normalized_columns = (
            normalized_median_col in uninfected_df.columns
            and normalized_q25_col in uninfected_df.columns
            and normalized_q75_col in uninfected_df.columns
        )

        # Use pre-computed normalized columns if available and requested
        if (
            normalize_to_baseline
            and not is_cv_feature
            and not is_prenormalized
            and has_normalized_columns
        ):
            uninfected_median = uninfected_df[normalized_median_col].values
            uninfected_q25 = uninfected_df[normalized_q25_col].values
            uninfected_q75 = uninfected_df[normalized_q75_col].values
        else:
            # Fall back to raw values
            uninfected_median = uninfected_df[median_col].values
            uninfected_q25 = uninfected_df[q25_col].values
            uninfected_q75 = uninfected_df[q75_col].values

            # Apply baseline normalization on-the-fly if needed (legacy approach)
            if (
                normalize_to_baseline
                and not is_cv_feature
                and not is_prenormalized
                and feature in baseline_values
                and baseline_values[feature]["uninfected"] is not None
            ):
                baseline = baseline_values[feature]["uninfected"]
                uninfected_median = (uninfected_median - baseline) / (
                    np.abs(baseline) + 1e-6
                )
                uninfected_q25 = (uninfected_q25 - baseline) / (np.abs(baseline) + 1e-6)
                uninfected_q75 = (uninfected_q75 - baseline) / (np.abs(baseline) + 1e-6)

        ax.plot(
            uninfected_time,
            uninfected_median,
            color=uninfected_color,
            linewidth=2.5,
            label=f"{uninfected_label}",
            linestyle="-",
        )
        ax.fill_between(
            uninfected_time,
            uninfected_q25,
            uninfected_q75,
            color=uninfected_color,
            alpha=0.2,
        )

        # Plot infected aligned response
        infected_time = infected_df["t"].values

        # Check if normalized columns exist for infected trajectory
        has_normalized_columns_infected = (
            normalized_median_col in infected_df.columns
            and normalized_q25_col in infected_df.columns
            and normalized_q75_col in infected_df.columns
        )

        # Use pre-computed normalized columns if available and requested
        if (
            normalize_to_baseline
            and not is_cv_feature
            and not is_prenormalized
            and has_normalized_columns_infected
        ):
            infected_median = infected_df[normalized_median_col].values
            infected_q25 = infected_df[normalized_q25_col].values
            infected_q75 = infected_df[normalized_q75_col].values
        else:
            # Fall back to raw values
            infected_median = infected_df[median_col].values
            infected_q25 = infected_df[q25_col].values
            infected_q75 = infected_df[q75_col].values

            # Apply baseline normalization on-the-fly if needed (legacy approach)
            if (
                normalize_to_baseline
                and not is_cv_feature
                and not is_prenormalized
                and feature in baseline_values
                and baseline_values[feature]["infected"] is not None
            ):
                baseline = baseline_values[feature]["infected"]
                infected_median = (infected_median - baseline) / (
                    np.abs(baseline) + 1e-6
                )
                infected_q25 = (infected_q25 - baseline) / (np.abs(baseline) + 1e-6)
                infected_q75 = (infected_q75 - baseline) / (np.abs(baseline) + 1e-6)

        ax.plot(
            infected_time,
            infected_median,
            color=infected_color,
            linewidth=2.5,
            label=f"{infected_label} (top-N aligned)",
            linestyle="-",
        )
        ax.fill_between(
            infected_time,
            infected_q25,
            infected_q75,
            color=infected_color,
            alpha=0.2,
        )

        # Plot global infected average
        if global_infected_df is not None and median_col in global_infected_df.columns:
            global_time = global_infected_df["t"].values

            # Check if normalized columns exist for global trajectory
            has_normalized_columns_global = (
                normalized_median_col in global_infected_df.columns
                and normalized_q25_col in global_infected_df.columns
                and normalized_q75_col in global_infected_df.columns
            )

            # Use pre-computed normalized columns if available and requested
            if (
                normalize_to_baseline
                and not is_cv_feature
                and not is_prenormalized
                and has_normalized_columns_global
            ):
                global_median = global_infected_df[normalized_median_col].values
                global_q25 = global_infected_df[normalized_q25_col].values
                global_q75 = global_infected_df[normalized_q75_col].values
            else:
                # Fall back to raw values
                global_median = global_infected_df[median_col].values
                global_q25 = (
                    global_infected_df[q25_col].values
                    if q25_col in global_infected_df.columns
                    else None
                )
                global_q75 = (
                    global_infected_df[q75_col].values
                    if q75_col in global_infected_df.columns
                    else None
                )

                # Apply baseline normalization on-the-fly if needed (legacy approach)
                if (
                    normalize_to_baseline
                    and not is_cv_feature
                    and not is_prenormalized
                    and feature in baseline_values
                    and baseline_values[feature]["global"] is not None
                ):
                    baseline = baseline_values[feature]["global"]
                    global_median = (global_median - baseline) / (
                        np.abs(baseline) + 1e-6
                    )
                    if global_q25 is not None:
                        global_q25 = (global_q25 - baseline) / (np.abs(baseline) + 1e-6)
                    if global_q75 is not None:
                        global_q75 = (global_q75 - baseline) / (np.abs(baseline) + 1e-6)

            ax.plot(
                global_time,
                global_median,
                color="#15ba10",  # green
                linewidth=2,
                label=f"All {infected_label} (no alignment)",
                linestyle="--",
                alpha=0.8,
            )
            if global_q25 is not None and global_q75 is not None:
                ax.fill_between(
                    global_time,
                    global_q25,
                    global_q75,
                    color="#15ba10",
                    alpha=0.15,
                )

        # Mark infection timepoint
        if infection_time is not None:
            ax.axvline(
                infection_time,
                color="red",
                linestyle="-",
                alpha=0.8,
                linewidth=2.5,
                label="Infection event",
                zorder=5,
            )

        # Find consecutive divergence points
        if len(uninfected_median) > 0 and n_consecutive_divergence > 0:
            uninfected_std = np.nanstd(uninfected_median)

            # Interpolate uninfected to match infected timepoints
            if len(uninfected_time) > 1 and len(infected_time) > 1:
                min_t = max(uninfected_time.min(), infected_time.min())
                max_t = min(uninfected_time.max(), infected_time.max())

                if min_t < max_t:
                    interp_func = interp1d(
                        uninfected_time,
                        uninfected_median,
                        kind="linear",
                        fill_value="extrapolate",
                    )

                    # Find timepoints where infected is significantly different
                    overlap_mask = (infected_time >= min_t) & (infected_time <= max_t)
                    if infection_time is not None:
                        overlap_mask = overlap_mask & (infected_time >= infection_time)

                    overlap_times = infected_time[overlap_mask]
                    overlap_infected = infected_median[overlap_mask]
                    overlap_uninfected = interp_func(overlap_times)

                    divergence = np.abs(overlap_infected - overlap_uninfected)
                    threshold = 1.5 * uninfected_std

                    divergent_mask = divergence > threshold

                    # Find consecutive divergence streaks
                    if np.any(divergent_mask):
                        consecutive_start = None
                        consecutive_count = 0

                        for i, is_divergent in enumerate(divergent_mask):
                            if is_divergent:
                                if consecutive_start is None:
                                    consecutive_start = i
                                consecutive_count += 1

                                if consecutive_count >= n_consecutive_divergence:
                                    first_divergence = overlap_times[consecutive_start]
                                    ax.axvline(
                                        first_divergence,
                                        color="red",
                                        linestyle="--",
                                        alpha=0.6,
                                        linewidth=2,
                                        label=f"Divergence (t={first_divergence:.0f})",
                                        zorder=4,
                                    )
                                    break
                            else:
                                consecutive_start = None
                                consecutive_count = 0

        ax.set_xlabel("Time")
        # Update y-axis label
        if feature.endswith("_cv"):
            ax.set_ylabel(f"{feature}\n(raw CV)")
        elif feature.endswith("_sem"):
            ax.set_ylabel(f"{feature}\n(raw SEM)")
        elif is_prenormalized:
            ax.set_ylabel(f"{feature}\n(pre-normalized, baseline t=1-10)")
        elif normalize_to_baseline and feature in baseline_values:
            ax.set_ylabel(f"{feature}\n(fold-change from baseline)")
        else:
            ax.set_ylabel(feature)
        ax.set_title(feature)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        output_root / f"infected_vs_uninfected_comparison_{ALIGN_TYPE}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


# %%
# Plot individual lineages
# Filter to FOVs of interest AND only include rows with valid lineage_id
# Important: Include "consensus" to ensure plotting methods have the reference pattern
alignment_df_for_plotting = master_df[
    (
        master_df["fov_name"].str.contains(INFECTED_FOV_PATTERN)
        | (master_df["fov_name"] == "consensus")
    )
    & ~master_df["lineage_id"].isna()
].copy()

logger.info(
    f"Filtered plotting dataframe ({INFECTED_FOV_PATTERN}): {len(alignment_df_for_plotting)} rows, "
    f"{alignment_df_for_plotting['lineage_id'].nunique()} unique lineages"
)
logger.info(
    f"Includes consensus: {(alignment_df_for_plotting['fov_name'] == 'consensus').any()}"
)

fig = cytodtw.plot_individual_lineages(
    alignment_df_for_plotting,
    alignment_name=ALIGN_TYPE,
    feature_columns=[
        "sensor_PC1",
        "homogeneity",
        "contrast",
        "edge_density",
        "segs_count",
        "segs_total_area",
        "segs_mean_area",
    ],
    max_lineages=8,
    aligned_linewidth=2.5,
    unaligned_linewidth=1.4,
    y_offset_step=0.0,
)

# %%
# Heatmap showing all tracks
fig = cytodtw.plot_global_trends(
    alignment_df_for_plotting,
    alignment_name=ALIGN_TYPE,
    plot_type="heatmap",
    cmap="RdBu",
    figsize=(12, 12),
    feature_columns=[
        "organelle_PC1",
        "organelle_PC2",
        "organelle_PC3",
        "edge_density",
        "segs_count",
        "segs_total_area",
        "segs_mean_area",
        "segs_circularity_mean",
        "segs_mean_frangi_mean",
    ],
    max_lineages=8,
)

# %%
# Binned period comparison
if infection_timepoint is not None:
    # Define biologically meaningful periods relative to infection
    periods = {
        "Baseline": (0, NORMALIZE_N_TIMEPOINTS),
        "Pre-infection": (infection_timepoint - 5, infection_timepoint),
        "Infection": (infection_timepoint - 1, infection_timepoint + 1),
        "post_infection": (infection_timepoint + 5, infection_timepoint + 10),
    }

    plot_binned_period_comparison(
        common_response_df,
        uninfected_baseline_df,
        common_response_features,
        periods=periods,
        baseline_period_name="Baseline",
        infection_time=infection_timepoint,
        baseline_values=baseline_normalization_values,
        global_infected_df=global_infected_df,
        output_root=output_root,
        plot_type="line",
        add_stats=True,
        infected_label=INFECTED_LABEL,
        uninfected_label=UNINFECTED_LABEL,
        figsize=(20, 20),
    )

# %%
# Infected vs uninfected comparison
plot_infected_vs_uninfected_comparison(
    common_response_df,
    uninfected_baseline_df,
    common_response_features,
    baseline_values=baseline_normalization_values,
    infection_time=infection_timepoint,
    aligned_region=aligned_region_bounds,
    figsize=(18, 14),
    n_consecutive_divergence=5,
    global_infected_df=global_infected_df,
    normalize_to_baseline=True,
    infected_label=INFECTED_LABEL,
    uninfected_label=UNINFECTED_LABEL,
)

# %%
# Napari visualization
if NAPARI:
    # Load matches to get top lineages for visualization
    matches_path = (
        output_root
        / f"consensus_lineage_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}_matching_lineages_cosine.csv"
    )
    matches = pd.read_csv(matches_path)
    top_n = 5
    top_matches = matches.head(top_n)

    z_range = slice(0, 1)
    initial_yx_patch_size = (192, 192)

    positions = []
    seg_positions = []
    tracks_tables = []
    images_plate = open_ome_zarr(data_path)
    segmentation_plate = open_ome_zarr(segmentation_path)

    # Load matching positions
    print(f"Loading positions for {len(top_matches)} FOV matches...")
    matches_found = 0
    for _, pos in images_plate.positions():
        pos_name = pos.zgroup.name
        pos_normalized = pos_name.lstrip("/")

        if pos_normalized in top_matches["fov_name"].values:
            positions.append(pos)
            matches_found += 1

            # Get ALL tracks for this FOV
            tracks_df = cytodtw.adata.obs[
                cytodtw.adata.obs["fov_name"] == pos_normalized
            ].copy()

            if len(tracks_df) > 0:
                tracks_df = tracks_df.dropna(subset=["x", "y"])
                tracks_df["x"] = tracks_df["x"].astype(int)
                tracks_df["y"] = tracks_df["y"].astype(int)
                tracks_tables.append(tracks_df)

                if matches_found == 1:
                    processing_channels = pos.channel_names

    print(
        f"Loaded {matches_found} positions with {sum(len(df) for df in tracks_tables)} total tracks"
    )

    # Load matching segmentation positions
    # We need to match both positions and tracks_tables in the same order
    print(f"Loading segmentation positions for {len(top_matches)} FOV matches...")
    seg_tracks_tables = []
    seg_matches_found = 0

    # Create a mapping from fov_name to tracks for quick lookup
    fov_to_tracks = {}
    for i, pos in enumerate(positions):
        fov_name = pos.zgroup.name.lstrip("/")
        fov_to_tracks[fov_name] = tracks_tables[i]

    for _, seg_pos in segmentation_plate.positions():
        seg_pos_name = seg_pos.zgroup.name
        seg_pos_normalized = seg_pos_name.lstrip("/")

        if (
            seg_pos_normalized in top_matches["fov_name"].values
            and seg_pos_normalized in fov_to_tracks
        ):
            seg_positions.append(seg_pos)
            seg_tracks_tables.append(fov_to_tracks[seg_pos_normalized])
            seg_matches_found += 1

    print(f"Loaded {seg_matches_found} segmentation positions with matching tracks")

    # Create TripletDataset
    if len(positions) > 0 and len(tracks_tables) > 0:
        if "processing_channels" not in locals():
            processing_channels = positions[0].channel_names

        selected_channels = processing_channels
        print(
            f"Creating TripletDataset with {len(selected_channels)} channels: {selected_channels}"
        )

        dataset = TripletDataset(
            positions=positions,
            tracks_tables=tracks_tables,
            channel_names=selected_channels,
            initial_yx_patch_size=initial_yx_patch_size,
            z_range=z_range,
            fit=False,
            predict_cells=False,
            include_fov_names=None,
            include_track_ids=None,
            time_interval=1,
            return_negative=False,
        )
        print(f"TripletDataset created with {len(dataset.valid_anchors)} valid anchors")

        # Create segmentation dataset
        if len(seg_positions) > 0:
            # Get actual channel names from segmentation zarr
            seg_channel_names = seg_positions[0].channel_names
            print(f"Segmentation channel names: {seg_channel_names}")

            segmentation_dataset = TripletDataset(
                positions=seg_positions,
                tracks_tables=seg_tracks_tables,  # Matching tracks for segmentation positions
                channel_names=seg_channel_names,  # Use actual channel names from segmentation zarr
                initial_yx_patch_size=initial_yx_patch_size,
                z_range=z_range,
                fit=False,
                predict_cells=False,
                include_fov_names=None,
                include_track_ids=None,
                time_interval=1,
                return_negative=False,
            )
            print(
                f"Segmentation TripletDataset created with {len(segmentation_dataset.valid_anchors)} valid anchors"
            )

        # Get aligned sequences
        def load_images_from_triplet_dataset(fov_name, track_ids):
            """Load images from TripletDataset for given FOV and track IDs."""
            matching_indices = []
            for dataset_idx in range(len(dataset.valid_anchors)):
                anchor_row = dataset.valid_anchors.iloc[dataset_idx]
                if (
                    anchor_row["fov_name"] == fov_name
                    and anchor_row["track_id"] in track_ids
                ):
                    matching_indices.append(dataset_idx)

            if not matching_indices:
                logger.warning(
                    f"No matching indices found for FOV {fov_name}, tracks {track_ids}"
                )
                return {}

            batch_data = dataset.__getitems__(matching_indices)
            images = []
            for i in range(len(matching_indices)):
                img_data = {
                    "anchor": batch_data["anchor"][i],
                    "index": batch_data["index"][i],
                }
                images.append(img_data)

            images.sort(key=lambda x: x["index"]["t"])
            return {img["index"]["t"]: img for img in images}

        # Segmentation loader function
        def load_segmentations_from_zarr(fov_name, track_ids):
            """Load segmentations from segmentation TripletDataset for given FOV and track IDs."""
            if len(seg_positions) == 0:
                return {}

            matching_indices = []
            for dataset_idx in range(len(segmentation_dataset.valid_anchors)):
                anchor_row = segmentation_dataset.valid_anchors.iloc[dataset_idx]
                if (
                    anchor_row["fov_name"] == fov_name
                    and anchor_row["track_id"] in track_ids
                ):
                    matching_indices.append(dataset_idx)

            if not matching_indices:
                logger.warning(
                    f"No matching segmentation indices found for FOV {fov_name}, tracks {track_ids}"
                )
                return {}

            batch_data = segmentation_dataset.__getitems__(matching_indices)
            segmentations = []
            for i in range(len(matching_indices)):
                seg_data = {
                    "anchor": batch_data["anchor"][i],
                    "index": batch_data["index"][i],
                }
                segmentations.append(seg_data)

            segmentations.sort(key=lambda x: x["index"]["t"])
            return {seg["index"]["t"]: seg for seg in segmentations}

        # Filter alignment dataframe for napari
        # Include consensus for get_aligned_image_sequences to work properly
        filtered_for_napari = master_df[
            master_df["fov_name"].str.contains(INFECTED_FOV_PATTERN)
            | (master_df["fov_name"] == "consensus")
        ]

        concatenated_image_sequences = get_aligned_image_sequences(
            cytodtw_instance=cytodtw,
            df=filtered_for_napari,
            alignment_name=ALIGN_TYPE,
            image_loader_fn=load_images_from_triplet_dataset,
            max_lineages=30,
        )

        # Load segmentation sequences
        concatenated_segmentation_sequences = {}
        if len(seg_positions) > 0:
            concatenated_segmentation_sequences = get_aligned_image_sequences(
                cytodtw_instance=cytodtw,
                df=filtered_for_napari,
                alignment_name=ALIGN_TYPE,
                image_loader_fn=load_segmentations_from_zarr,
                max_lineages=30,
            )
            print(
                f"Loaded segmentation sequences for {len(concatenated_segmentation_sequences)} lineages"
            )

        # Compute maximum unaligned_before_length for padding
        max_before_length = max(
            seq_data["unaligned_before_length"]
            for seq_data in concatenated_image_sequences.values()
        )
        logger.info(f"Maximum unaligned_before_length: {max_before_length}")

        # Load into napari with padding
        for lineage_id, seq_data in concatenated_image_sequences.items():
            concatenated_images = seq_data["concatenated_images"]
            meta = seq_data["metadata"]
            unaligned_before_length = seq_data["unaligned_before_length"]
            aligned_length = seq_data["aligned_length"]
            unaligned_after_length = seq_data["unaligned_after_length"]

            if len(concatenated_images) == 0:
                continue

            # Calculate padding needed
            padding_needed = max_before_length - unaligned_before_length
            logger.info(
                f"Lineage {lineage_id}: before={unaligned_before_length}, "
                f"aligned={aligned_length}, after={unaligned_after_length}, padding={padding_needed}"
            )

            # Stack images into time series
            image_stack = []
            for img_sample in concatenated_images:
                if img_sample is not None:
                    img_tensor = img_sample["anchor"]
                    img_np = img_tensor.cpu().numpy()
                    image_stack.append(img_np)

            if len(image_stack) > 0:
                time_series = np.stack(image_stack, axis=0)
                n_channels = time_series.shape[1]

                # Add padding frames at the beginning (zeros)
                if padding_needed > 0:
                    padding_shape = (padding_needed,) + time_series.shape[1:]
                    padding_frames = np.zeros(padding_shape, dtype=time_series.dtype)
                    time_series = np.concatenate([padding_frames, time_series], axis=0)
                    logger.info(
                        f"Added {padding_needed} padding frames. New shape: {time_series.shape}"
                    )

                # Set up colormap
                if n_channels == 2:
                    colormap = ["green", "magenta"]
                elif n_channels == 3:
                    colormap = ["gray", "green", "magenta"]
                else:
                    colormap = ["gray"] * n_channels

                # Add each channel as a separate layer
                for channel_idx in range(n_channels):
                    channel_data = time_series[:, channel_idx, :, :, :]
                    channel_name = (
                        processing_channels[channel_idx]
                        if channel_idx < len(processing_channels)
                        else f"ch{channel_idx}"
                    )
                    layer_name = f"FULL_track_id_{meta['track_ids'][0]}_FOV_{meta['fov_name']}_dist_{meta['dtw_distance']:.3f}_{channel_name}"

                    viewer.add_image(
                        channel_data,
                        name=layer_name,
                        contrast_limits=(channel_data.min(), channel_data.max()),
                        colormap=colormap[channel_idx],
                        blending="additive",
                    )
                    logger.debug(
                        f"Added {channel_name} channel for lineage {lineage_id}"
                    )

                # Create shape layer with circle marker for aligned frames
                # The circle appears only during aligned frames
                aligned_start_frame = (
                    max_before_length  # After padding, all aligned regions start here
                )
                aligned_end_frame = aligned_start_frame + aligned_length

                # Create circle coordinates for top-left corner (approximately 10% from edges)
                img_height = time_series.shape[3]  # y dimension
                img_width = time_series.shape[4]  # x dimension
                circle_center_y = img_height * 0.1
                circle_center_x = img_width * 0.1
                circle_radius = (
                    min(img_height, img_width) * 0.05
                )  # 5% of smaller dimension

                # Create circles for each aligned frame
                # For napari 4D ellipses: need 4 corner vertices defining the bounding box
                shapes_data = []
                for frame_idx in range(aligned_start_frame, aligned_end_frame):
                    # Ellipse is defined by 4 corners of bounding box in order: top-left, top-right, bottom-right, bottom-left
                    ellipse = np.array(
                        [
                            [
                                frame_idx,
                                0,
                                circle_center_y - circle_radius,
                                circle_center_x - circle_radius,
                            ],  # top-left
                            [
                                frame_idx,
                                0,
                                circle_center_y - circle_radius,
                                circle_center_x + circle_radius,
                            ],  # top-right
                            [
                                frame_idx,
                                0,
                                circle_center_y + circle_radius,
                                circle_center_x + circle_radius,
                            ],  # bottom-right
                            [
                                frame_idx,
                                0,
                                circle_center_y + circle_radius,
                                circle_center_x - circle_radius,
                            ],  # bottom-left
                        ]
                    )
                    shapes_data.append(ellipse)

                if len(shapes_data) > 0:
                    # Add shapes layer with circles marking aligned frames
                    viewer.add_shapes(
                        shapes_data,
                        shape_type="ellipse",
                        edge_width=3,
                        edge_color="orange",
                        face_color="transparent",
                        name=f"ALIGNED_MARKER_track_id_{meta['track_ids'][0]}_FOV_{meta['fov_name']}",
                        ndim=4,
                    )
                    logger.info(
                        f"Added alignment marker for lineage {lineage_id} "
                        f"(frames {aligned_start_frame} to {aligned_end_frame})"
                    )

                # Add segmentation labels layer if available
                if (
                    lineage_id in concatenated_segmentation_sequences
                    and len(
                        concatenated_segmentation_sequences[lineage_id][
                            "concatenated_images"
                        ]
                    )
                    > 0
                ):
                    seg_images = concatenated_segmentation_sequences[lineage_id][
                        "concatenated_images"
                    ]
                    seg_stack = []
                    for seg_sample in seg_images:
                        if seg_sample is not None:
                            seg_tensor = seg_sample["anchor"]
                            seg_np = seg_tensor.cpu().numpy()
                            # Segmentation is typically channel 0 only, squeeze out channel dim
                            seg_np_squeezed = seg_np[0, 0, :, :]  # (z, y, x) -> (y, x)
                            seg_stack.append(seg_np_squeezed)

                    if len(seg_stack) > 0:
                        seg_time_series = np.stack(seg_stack, axis=0)  # (t, y, x)

                        # Add padding to segmentation to match image padding
                        if padding_needed > 0:
                            seg_padding_shape = (
                                padding_needed,
                            ) + seg_time_series.shape[1:]
                            seg_padding_frames = np.zeros(
                                seg_padding_shape, dtype=seg_time_series.dtype
                            )
                            seg_time_series = np.concatenate(
                                [seg_padding_frames, seg_time_series], axis=0
                            )
                            logger.info(
                                f"Added {padding_needed} padding frames to segmentation. New shape: {seg_time_series.shape}"
                            )

                        seg_time_series = seg_time_series[
                            :, np.newaxis, :, :
                        ]  # (t, 1, y, x)
                        layer_name = f"FULL_track_id_{meta['track_ids'][0]}_FOV_{meta['fov_name']}_dist_{meta['dtw_distance']:.3f}_segmentation"
                        viewer.add_labels(
                            seg_time_series.astype(np.uint16),
                            name=layer_name,
                        )
                        logger.debug(
                            f"Added segmentation labels for lineage {lineage_id}"
                        )

# %%
from cmap import Colormap
from skimage.exposure import adjust_gamma, rescale_intensity

figure_output_path = output_root / "figure_parts"
figure_output_path.mkdir(exist_ok=True, parents=True)

green_cmap = Colormap("green")
magenta_cmap = Colormap("magenta")

seq_values = list(concatenated_image_sequences.keys())

# Taking the first lineage for example
lineage_id = seq_values[len(seq_values) - 6]

concatenated_images = concatenated_image_sequences[lineage_id]["concatenated_images"]

# Stack images into time series
image_stack = []
for img_sample in concatenated_images:
    if img_sample is not None:
        img_tensor = img_sample["anchor"]
        img_np = img_tensor.cpu().numpy()
        image_stack.append(img_np)

    if len(image_stack) > 0:
        time_series = np.stack(image_stack, axis=0)
        n_channels = time_series.shape[1]

tidx_figures = [
    min(0, infection_timepoint - 5),
    infection_timepoint,
    min(infection_timepoint + 10, time_series.shape[0] - 1),
    min(infection_timepoint + 20, time_series.shape[0] - 1),
]

organelle_clims = (104, 383)
sensor_clims = (102, 165)
phase_clims = (-0.79, 0.6)
for tidx in tidx_figures:
    # FIXME: hardcoded channel order the current dataset
    img_phase = time_series[tidx, 0, 0]
    img_organelle = time_series[tidx, 1, 0]
    img_sensor = time_series[tidx, 2, 0]

    # Apply gamma correction first (optional)
    img_sensor = adjust_gamma(img_sensor, gamma=1)
    img_organelle = adjust_gamma(img_organelle, gamma=1)
    img_phase = rescale_intensity(img_phase, in_range=phase_clims, out_range=(0, 1))

    # Use in_range to specify your contrast limits
    img_sensor = rescale_intensity(img_sensor, in_range=sensor_clims, out_range=(0, 1))
    img_organelle = rescale_intensity(
        img_organelle, in_range=organelle_clims, out_range=(0, 1)
    )

    # Apply colormaps
    img_sensor = magenta_cmap(img_sensor)
    img_organelle = green_cmap(img_organelle)
    img_rgb = np.clip(img_sensor[..., :3] + img_organelle[..., :3], 0, 1)

    # Fluorescence only
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        figure_output_path / f"lineage_{lineage_id}_t{tidx}_fluor.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Phase only
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(img_phase, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        figure_output_path / f"lineage_{lineage_id}_t{tidx}_phase.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
# %%
