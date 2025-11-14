# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import read_zarr
from divergence_utils import quantify_divergence
from iohub import open_ome_zarr

from viscy.data.triplet import TripletDataset
from viscy.representation.pseudotime import (
    CytoDtw,
    filter_tracks_by_fov_and_length,
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

# Cropping configuration
CROP_TO_ABSOLUTE_BOUNDS = (
    True  # If True, crop before/after regions to stay within [0, max_absolute_time]
)

NORMALIZE_N_TIMEPOINTS = 5
NORMALIZE_N_CELLS_FOR_BASELINE = 5

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
metadata_path = output_root / f"alignment_metadata_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}.pkl"

# %%
# Load master features dataframe
master_features_path = (
    output_root / f"master_features_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}.csv"
)
master_df = pd.read_csv(master_features_path)
logger.info(f"Loaded master features from {master_features_path}")
logger.info(f"Shape: {master_df.shape}")
logger.info(f"Columns: {list(master_df.columns)}")

# Load AnnData for CytoDtw methods (needed for plotting)
ad_features_alignment = read_zarr(
    features_path_sensor
    if ALIGNMENT_CHANNEL == "sensor"
    else (
        features_path_phase if ALIGNMENT_CHANNEL == "phase" else features_path_organelle
    )
)

cytodtw = CytoDtw(ad_features_alignment)
cytodtw.load_consensus(metadata_path)

# TODO: we should get rid of this redundancy later
metadata = cytodtw.consensus_data
consensus_lineage = metadata.get("consensus_pattern")
consensus_annotations = metadata.get("consensus_annotations")
consensus_metadata = metadata.get("consensus_metadata")
reference_cell_info = metadata.get("reference_cell_info")
alignment_infection_timepoint = metadata.get("raw_infection_timepoint")
aligned_region_bounds = metadata.get("aligned_region_bounds")

# Get infection timepoint in absolute time coordinates (mapped from reference cell)
# NOTE: This is the actual 't' value where infection occurs, NOT the consensus index
absolute_infection_timepoint = alignment_infection_timepoint

# Also get consensus infection index for validation
consensus_infection_idx = (
    consensus_annotations.index("infected")
    if consensus_annotations and "infected" in consensus_annotations
    else None
)

# Log both values to help debug coordinate system alignment
logger.info("Infection timepoint mapping:")
logger.info(
    f"  - Consensus infection index (within consensus window): {consensus_infection_idx}"
)
logger.info(
    f"  - Absolute infection timepoint (reference cell's t value): {absolute_infection_timepoint}"
)
if consensus_infection_idx is not None and absolute_infection_timepoint is not None:
    logger.info(
        f"  - Difference: {abs(absolute_infection_timepoint - consensus_infection_idx)} timepoints"
    )
    logger.info(
        f"  - Using absolute timepoint ({absolute_infection_timepoint}) for coordinate alignment"
    )

# %%
# Add warped coordinates if not already present
warped_col = f"dtw_{ALIGN_TYPE}_warped_t"
if warped_col not in master_df.columns:
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Adding warped coordinates for {ALIGN_TYPE} alignment...")
    logger.info(f"{'=' * 70}")
    master_df = cytodtw.add_warped_coordinates(master_df, alignment_name=ALIGN_TYPE)
    # Save augmented master_df
    master_df.to_csv(master_features_path, index=False)
    logger.info(f"Saved master_df with warped coordinates to {master_features_path}")
else:
    logger.info(f"\nWarped coordinates already present: {warped_col}")
    # Load warped metadata from consensus_data if available
    if cytodtw.consensus_data and "warped_metadata" in cytodtw.consensus_data:
        warped_meta = cytodtw.consensus_data["warped_metadata"]
        logger.info(f"Warped metadata: {warped_meta}")

# %%
# Data filtering and preparation
min_track_length = 20

# Filter to infected and uninfected cells using utility function
filtered_infected_df = filter_tracks_by_fov_and_length(
    master_df,
    fov_pattern=INFECTED_FOV_PATTERN,
    min_timepoints=min_track_length,
)
logger.info(
    f"Filtered {INFECTED_LABEL} cells: "
    f"{filtered_infected_df.groupby(['fov_name', 'track_id']).ngroups} tracks"
)

uninfected_filtered_df = filter_tracks_by_fov_and_length(
    master_df,
    fov_pattern=UNINFECTED_FOV_PATTERN,
    min_timepoints=min_track_length,
)
logger.info(
    f"Filtered {UNINFECTED_LABEL} cells: "
    f"{uninfected_filtered_df.groupby(['fov_name', 'track_id']).ngroups} tracks"
)

consensus_df = master_df[master_df["lineage_id"] == -1].copy()

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
# Compute aggregated trajectories of the common response from top N aligned cells
top_n_cells = 30

# Get aligned cells only
aligned_cells = filtered_infected_df[
    filtered_infected_df[f"dtw_{ALIGN_TYPE}_aligned"].fillna(False)
].copy()

# Select top N lineages by DTW distance
top_lineages_df = aligned_cells.drop_duplicates(["fov_name", "lineage_id"]).nsmallest(
    top_n_cells, f"dtw_{ALIGN_TYPE}_distance"
)[["fov_name", "lineage_id"]]

logger.info(
    f"Selected top {len(top_lineages_df)} lineages by DTW distance from fovs {top_lineages_df['fov_name'].unique()}"
)

# Filter using merge
top_cells_df = filtered_infected_df.merge(
    top_lineages_df, on=["fov_name", "lineage_id"], how="inner"
).copy()
top_cells_df = top_cells_df.sort_values(
    [f"dtw_{ALIGN_TYPE}_distance", "fov_name", "lineage_id", "t"]
)


# %%
# Helper function to aggregate trajectories
def aggregate_trajectory(
    df: pd.DataFrame,
    feature_columns: list,
    baseline_n_timepoints: int = 3,
    time_column: str = "t",
    min_cell_count_for_baseline: int = 5,
) -> pd.DataFrame:
    """
    Aggregate trajectories across cells by computing median and IQR at each timepoint.

    Also normalizes features to baseline (window with sufficient cell coverage).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with feature columns and time column
    feature_columns : list
        List of feature columns to aggregate
    baseline_n_timepoints : int
        Number of consecutive timepoints to use as baseline for normalization
    time_column : str
        Name of the time column to group by (default: "t" for absolute time,
        can be "dtw_{alignment}_warped_t" for warped time)
    min_cell_count_for_baseline : int
        Minimum number of cells required per timepoint to be included in baseline window

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with columns: {time_column}, {feature}_median, {feature}_q25, {feature}_q75,
        and normalized versions {feature}_median_normalized, etc.
    """
    # Group by timepoint
    grouped = df.groupby(time_column)

    agg_data = []
    for t, group in grouped:
        row = {time_column: t}
        for feature in feature_columns:
            if feature in group.columns:
                values = group[feature].dropna()
                if len(values) > 0:
                    row[f"{feature}_median"] = values.median()
                    row[f"{feature}_q25"] = values.quantile(0.25)
                    row[f"{feature}_q75"] = values.quantile(0.75)
                    row[f"{feature}_count"] = len(values)
                else:
                    row[f"{feature}_median"] = np.nan
                    row[f"{feature}_q25"] = np.nan
                    row[f"{feature}_q75"] = np.nan
                    row[f"{feature}_count"] = 0
        agg_data.append(row)

    agg_df = pd.DataFrame(agg_data).sort_values(time_column).reset_index(drop=True)

    # Normalize to baseline (window with sufficient cell coverage)
    baseline_mask = None
    baseline_timepoints = None

    if baseline_n_timepoints > 0:
        # Get cell counts per timepoint (use any feature's count column)
        count_col = f"{feature_columns[0]}_count"
        if count_col in agg_df.columns:
            # Find timepoints with sufficient cell coverage
            sufficient_cells_mask = agg_df[count_col] >= min_cell_count_for_baseline
            sufficient_timepoints = agg_df[sufficient_cells_mask][time_column].values

            if len(sufficient_timepoints) >= baseline_n_timepoints:
                # Find consecutive window of N timepoints with sufficient cells
                best_window_start = None
                for i in range(len(agg_df) - baseline_n_timepoints + 1):
                    window_slice = agg_df.iloc[i : i + baseline_n_timepoints]
                    if all(window_slice[count_col] >= min_cell_count_for_baseline):
                        best_window_start = i
                        break

                if best_window_start is not None:
                    # Use this window
                    baseline_mask = pd.Series(False, index=agg_df.index)
                    baseline_mask.iloc[
                        best_window_start : best_window_start + baseline_n_timepoints
                    ] = True
                    baseline_timepoints = agg_df.loc[baseline_mask, time_column].values
                    logger.info(
                        f"  Baseline window: t={baseline_timepoints[0]:.1f}-{baseline_timepoints[-1]:.1f}, "
                        f"cell counts: {agg_df.loc[baseline_mask, count_col].min():.0f}-{agg_df.loc[baseline_mask, count_col].max():.0f}"
                    )
                else:
                    # Fallback: use all timepoints with sufficient cells
                    baseline_mask = sufficient_cells_mask
                    baseline_timepoints = sufficient_timepoints
                    logger.warning(
                        f"  Could not find {baseline_n_timepoints} consecutive timepoints with ≥{min_cell_count_for_baseline} cells. "
                        f"Using all {len(baseline_timepoints)} timepoints with sufficient coverage."
                    )
            else:
                # Not enough timepoints with sufficient cells - use all available
                logger.warning(
                    f"  Insufficient timepoints with ≥{min_cell_count_for_baseline} cells "
                    f"(found {len(sufficient_timepoints)}, need {baseline_n_timepoints}). "
                    f"Using all timepoints for baseline."
                )
                baseline_mask = pd.Series(True, index=agg_df.index)
                baseline_timepoints = agg_df[time_column].values
        else:
            # Fallback to original logic if count column not found
            logger.warning(
                "  Cell count column not found, using first N timepoints for baseline"
            )
            baseline_mask = agg_df[time_column] <= (
                agg_df[time_column].min() + baseline_n_timepoints - 1
            )
            baseline_timepoints = agg_df.loc[baseline_mask, time_column].values

        # Normalize using selected baseline
        if baseline_mask is not None:
            for feature in feature_columns:
                median_col = f"{feature}_median"
                q25_col = f"{feature}_q25"
                q75_col = f"{feature}_q75"

                # Skip CV/SEM features, PC features, and already normalized features
                # PCs are relative measures and should not be baseline-normalized
                if (
                    feature.endswith("_cv")
                    or feature.endswith("_sem")
                    or feature.startswith("normalized_")
                    or "_PC" in feature  # Skip PC1, PC2, PC3, etc.
                ):
                    continue

                if median_col in agg_df.columns:
                    # Compute baseline mean from median values
                    baseline_values = agg_df.loc[baseline_mask, median_col].dropna()
                    if len(baseline_values) > 0:
                        baseline_mean = baseline_values.mean()
                        if not np.isnan(baseline_mean) and baseline_mean != 0:
                            # Normalize
                            agg_df[f"{median_col}_normalized"] = (
                                agg_df[median_col] / baseline_mean
                            )
                            agg_df[f"{q25_col}_normalized"] = (
                                agg_df[q25_col] / baseline_mean
                            )
                            agg_df[f"{q75_col}_normalized"] = (
                                agg_df[q75_col] / baseline_mean
                            )

    return agg_df


# %%
logger.info("\n" + "=" * 70)
logger.info("Computing aggregated trajectories in ABSOLUTE TIME")
logger.info("=" * 70)

# Aggregate top N infected cells (common response)
logger.info("Aggregating top-N infected cells (common response):")
common_response_df = aggregate_trajectory(
    top_cells_df,
    common_response_features,
    baseline_n_timepoints=NORMALIZE_N_TIMEPOINTS,
    min_cell_count_for_baseline=NORMALIZE_N_CELLS_FOR_BASELINE,
)
logger.info(
    f"Common response (top-{top_n_cells}): {len(common_response_df)} timepoints"
)

# Aggregate uninfected baseline
logger.info("\nAggregating uninfected baseline:")
uninfected_baseline_df = aggregate_trajectory(
    uninfected_filtered_df,
    common_response_features,
    baseline_n_timepoints=NORMALIZE_N_TIMEPOINTS,
    min_cell_count_for_baseline=NORMALIZE_N_CELLS_FOR_BASELINE,
)
logger.info(f"Uninfected baseline: {len(uninfected_baseline_df)} timepoints")

# Aggregate all infected cells (global average)
logger.info("\nAggregating all infected cells (global average):")
global_infected_df = aggregate_trajectory(
    filtered_infected_df,
    common_response_features,
    baseline_n_timepoints=NORMALIZE_N_TIMEPOINTS,
    min_cell_count_for_baseline=5,
)
logger.info(f"Global infected average: {len(global_infected_df)} timepoints")

# Get anchor metadata from consensus
anchor_metadata = {
    "anchor_start": (
        aligned_region_bounds[0] if aligned_region_bounds is not None else None
    ),
    "anchor_end": (
        aligned_region_bounds[1] if aligned_region_bounds is not None else None
    ),
    "window_start": common_response_df["t"].min(),
    "window_end": common_response_df["t"].max(),
}
logger.info(f"Anchor metadata: {anchor_metadata}")

# %%
logger.info("\n" + "=" * 70)
logger.info("Computing aggregated trajectories in WARPED TIME")
logger.info("=" * 70)

# Filter to cells with valid warped coordinates (includes aligned + unaligned before/after)
warped_col = f"dtw_{ALIGN_TYPE}_warped_t"
top_cells_warped_df = top_cells_df[~top_cells_df[warped_col].isna()].copy()

logger.info(
    f"Cells with warped coordinates (full concatenated sequence): {len(top_cells_warped_df)} rows, "
    f"{top_cells_warped_df.groupby(['fov_name', 'lineage_id']).ngroups} lineages"
)

# Aggregate top N infected cells in warped time
logger.info("\nAggregating top-N infected cells in warped time:")
common_response_warped_df = aggregate_trajectory(
    top_cells_warped_df,
    common_response_features,
    baseline_n_timepoints=NORMALIZE_N_TIMEPOINTS,
    time_column=warped_col,
    min_cell_count_for_baseline=NORMALIZE_N_CELLS_FOR_BASELINE,
)
logger.info(
    f"Common response in warped time (top-{top_n_cells}): {len(common_response_warped_df)} timepoints"
)

# Rename warped_t column to "t" for plotting compatibility
common_response_warped_df = common_response_warped_df.rename(columns={warped_col: "t"})

# Aggregate all infected cells in warped time (no alignment requirement)
# This shows what the average infected trajectory looks like WITHOUT DTW synchronization
filtered_infected_warped_df = filtered_infected_df[
    ~filtered_infected_df[warped_col].isna()
].copy()

logger.info(
    f"All infected cells with warped coordinates: {len(filtered_infected_warped_df)} rows, "
    f"{filtered_infected_warped_df.groupby(['fov_name', 'lineage_id']).ngroups} lineages"
)

logger.info("\nAggregating all infected cells in warped time (no alignment):")
global_infected_warped_df = aggregate_trajectory(
    filtered_infected_warped_df,
    common_response_features,
    baseline_n_timepoints=NORMALIZE_N_TIMEPOINTS,
    time_column=warped_col,
    min_cell_count_for_baseline=NORMALIZE_N_CELLS_FOR_BASELINE,
)
logger.info(
    f"Global infected in warped time (no alignment): {len(global_infected_warped_df)} timepoints"
)

# Rename warped_t column to "t" for plotting compatibility
global_infected_warped_df = global_infected_warped_df.rename(columns={warped_col: "t"})

# Get warped metadata for period definitions
if cytodtw.consensus_data and "warped_metadata" in cytodtw.consensus_data:
    warped_meta = cytodtw.consensus_data["warped_metadata"]
    logger.info(f"Warped metadata: {warped_meta}")


# %%
def plot_binned_period_comparison(
    infected_df: pd.DataFrame,
    uninfected_df: pd.DataFrame,
    feature_columns: list,
    periods: dict,
    baseline_period_name: str = None,
    infection_time: int = None,
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

        # Check if CV/SEM, PC, or pre-normalized feature (no additional normalization)
        is_cv_feature = feature.endswith("_cv") or feature.endswith("_sem")
        is_pc_feature = "_PC" in feature  # PC features should not be normalized
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

                # Normalize to baseline if not CV/PC feature and not pre-normalized
                if (
                    not is_cv_feature
                    and not is_pc_feature
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
                    and not is_pc_feature
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
                        and not is_pc_feature
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

        # Add horizontal line at 1.0 (no change from baseline) - but not for PCs or CVs
        if not is_cv_feature and not is_pc_feature:
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Period")
        if is_cv_feature:
            ax.set_ylabel(f"{feature}\n(raw value)")
        elif is_pc_feature:
            ax.set_ylabel(f"{feature}\n(PC units, not normalized)")
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

    # Create a single shared legend for the entire figure
    # Get handles and labels from the first subplot (they're all the same)
    handles, labels = axes[0].get_legend_handles_labels()
    # Place legend on the right side of the figure
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        fontsize=9,
        frameon=True,
    )

    # Create title with statistical note
    title = "Binned Period Comparison: Fold-Change Across Infection Phases"
    if add_stats:
        title += "\n(* p<0.05, ** p<0.01, *** p<0.001)"

    plt.suptitle(
        title,
        fontsize=14,
        y=1.00,
    )
    # Use tight_layout with extra space on the right for the shared legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if output_root is not None:
        # Detect if this is warped time by checking the label
        is_warped = "DTW-aligned" in infected_label
        suffix = "warped" if is_warped else "absolute"
        save_path = output_root / f"binned_period_comparison_{ALIGN_TYPE}_{suffix}.png"
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
    warped_metadata: dict,
    anchor_metadata: dict = None,
    figsize=(18, 14),
    n_consecutive_divergence: int = 5,
    global_infected_df: pd.DataFrame = None,
    normalize_to_baseline: bool = True,
    infected_label: str = "Infected",
    uninfected_label: str = "Uninfected",
    output_root: Path = None,
    plot_cell_counts: bool = True,
    min_cell_count_threshold: int = 10,
):
    """
    Plot comparison of infected vs uninfected trajectories in warped/pseudotime.

    Compares trajectories in synchronized biological time (warped/pseudotime coordinates)
    where infected cells' aligned regions are synchronized. Shows where infected cells
    diverge from normal behavior (crossover points).

    Uses pre-computed normalized columns from dataframes (e.g., '{feature}_median_normalized')
    added by aggregate_trajectory().

    NOTE: Features ending in '_cv' or '_sem' are plotted as raw values without baseline
    normalization, since CV and SEM are already relative/uncertainty metrics.

    Parameters
    ----------
    infected_df : pd.DataFrame
        Infected common response aggregated in warped time with 't' column (warped coordinates)
        and '{feature}_*_normalized' columns
    uninfected_df : pd.DataFrame
        Uninfected baseline shifted to warped time coordinates with '{feature}_*_normalized' columns
    feature_columns : list
        Features to plot
    warped_metadata : dict
        Metadata from warped coordinate system containing:
        - max_unaligned_before: warped time where aligned region starts
        - consensus_aligned_length: length of the synchronized aligned region
        - total_warped_length: total length of warped time axis
    anchor_metadata : dict, optional
        Metadata for highlighting the aligned region bounds:
        - anchor_start: warped time where aligned region starts
        - anchor_end: warped time where aligned region ends
        - window_start: warped time where aggregated data starts
        - window_end: warped time where aggregated data ends
    figsize : tuple
        Figure size
    n_consecutive_divergence : int
        Number of consecutive timepoints required to confirm divergence (default: 5)
    global_infected_df : pd.DataFrame, optional
        Global average of ALL infected cells in warped time, with normalized columns
    normalize_to_baseline : bool
        If True, use normalized columns to show fold-change
    infected_label : str
        Label for infected condition in plots
    uninfected_label : str
        Label for uninfected/control condition in plots
    output_root : Path, optional
        Directory to save output figure
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

        # Highlight aligned/anchor region in warped time (background layer)
        if anchor_metadata is not None:
            anchor_start = anchor_metadata.get("anchor_start")
            anchor_end = anchor_metadata.get("anchor_end")
            if anchor_start is not None and anchor_end is not None:
                ax.axvspan(
                    anchor_start,
                    anchor_end,
                    alpha=0.15,
                    color="gray",
                    label="Synchronized aligned region",
                    zorder=0,
                )

        # Filter timepoints based on cell count threshold
        count_col = f"{feature}_count"

        # Plot uninfected baseline
        uninfected_time = uninfected_df["t"].values

        # Check if this is a CV/SEM, PC, or pre-normalized feature
        is_cv_feature = feature.endswith("_cv") or feature.endswith("_sem")
        is_pc_feature = "_PC" in feature  # PC features should not be normalized
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

        # Use normalized columns if available and requested (but not for PCs)
        if (
            normalize_to_baseline
            and not is_cv_feature
            and not is_pc_feature
            and not is_prenormalized
            and has_normalized_columns
        ):
            uninfected_median = uninfected_df[normalized_median_col].values
            uninfected_q25 = uninfected_df[normalized_q25_col].values
            uninfected_q75 = uninfected_df[normalized_q75_col].values
        else:
            # Use raw values
            uninfected_median = uninfected_df[median_col].values
            uninfected_q25 = uninfected_df[q25_col].values
            uninfected_q75 = uninfected_df[q75_col].values

        # Filter by cell count threshold
        if count_col in uninfected_df.columns:
            valid_mask = uninfected_df[count_col].values >= min_cell_count_threshold
            uninfected_time_filtered = uninfected_time[valid_mask]
            uninfected_median_filtered = uninfected_median[valid_mask]
            uninfected_q25_filtered = uninfected_q25[valid_mask]
            uninfected_q75_filtered = uninfected_q75[valid_mask]
        else:
            uninfected_time_filtered = uninfected_time
            uninfected_median_filtered = uninfected_median
            uninfected_q25_filtered = uninfected_q25
            uninfected_q75_filtered = uninfected_q75

        ax.plot(
            uninfected_time_filtered,
            uninfected_median_filtered,
            color=uninfected_color,
            linewidth=2.5,
            label=f"{uninfected_label}",
            linestyle="-",
        )
        ax.fill_between(
            uninfected_time_filtered,
            uninfected_q25_filtered,
            uninfected_q75_filtered,
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

        # Use normalized columns if available and requested (but not for PCs)
        if (
            normalize_to_baseline
            and not is_cv_feature
            and not is_pc_feature
            and not is_prenormalized
            and has_normalized_columns_infected
        ):
            infected_median = infected_df[normalized_median_col].values
            infected_q25 = infected_df[normalized_q25_col].values
            infected_q75 = infected_df[normalized_q75_col].values
        else:
            # Use raw values
            infected_median = infected_df[median_col].values
            infected_q25 = infected_df[q25_col].values
            infected_q75 = infected_df[q75_col].values

        # Filter by cell count threshold
        if count_col in infected_df.columns:
            valid_mask = infected_df[count_col].values >= min_cell_count_threshold
            infected_time_filtered = infected_time[valid_mask]
            infected_median_filtered = infected_median[valid_mask]
            infected_q25_filtered = infected_q25[valid_mask]
            infected_q75_filtered = infected_q75[valid_mask]
        else:
            infected_time_filtered = infected_time
            infected_median_filtered = infected_median
            infected_q25_filtered = infected_q25
            infected_q75_filtered = infected_q75

        ax.plot(
            infected_time_filtered,
            infected_median_filtered,
            color=infected_color,
            linewidth=2.5,
            label=f"{infected_label} (top-N aligned)",
            linestyle="-",
        )
        ax.fill_between(
            infected_time_filtered,
            infected_q25_filtered,
            infected_q75_filtered,
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

            # Use normalized columns if available and requested (but not for PCs)
            if (
                normalize_to_baseline
                and not is_cv_feature
                and not is_pc_feature
                and not is_prenormalized
                and has_normalized_columns_global
            ):
                global_median = global_infected_df[normalized_median_col].values
                global_q25 = global_infected_df[normalized_q25_col].values
                global_q75 = global_infected_df[normalized_q75_col].values
            else:
                # Use raw values
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

            # Filter by cell count threshold
            if count_col in global_infected_df.columns:
                valid_mask = (
                    global_infected_df[count_col].values >= min_cell_count_threshold
                )
                global_time_filtered = global_time[valid_mask]
                global_median_filtered = global_median[valid_mask]
                global_q25_filtered = (
                    global_q25[valid_mask] if global_q25 is not None else None
                )
                global_q75_filtered = (
                    global_q75[valid_mask] if global_q75 is not None else None
                )
            else:
                global_time_filtered = global_time
                global_median_filtered = global_median
                global_q25_filtered = global_q25
                global_q75_filtered = global_q75

            ax.plot(
                global_time_filtered,
                global_median_filtered,
                color="#15ba10",  # green
                linewidth=2,
                label=f"All {infected_label} (no alignment)",
                linestyle="--",
                alpha=0.8,
            )
            if global_q25_filtered is not None and global_q75_filtered is not None:
                ax.fill_between(
                    global_time_filtered,
                    global_q25_filtered,
                    global_q75_filtered,
                    color="#15ba10",
                    alpha=0.15,
                )

        # Mark alignment start (where synchronized biological response begins)
        if warped_metadata is not None:
            alignment_start = warped_metadata.get("max_unaligned_before")
            if alignment_start is not None:
                ax.axvline(
                    alignment_start,
                    color="red",
                    linestyle="-",
                    alpha=0.8,
                    linewidth=2.5,
                    label="Alignment start (infection)",
                    zorder=5,
                )

        # Find consecutive divergence points (use filtered data)
        if len(uninfected_median_filtered) > 0 and n_consecutive_divergence > 0:
            uninfected_std = np.nanstd(uninfected_median_filtered)

            # Interpolate uninfected to match infected timepoints (use filtered data)
            if len(uninfected_time_filtered) > 1 and len(infected_time_filtered) > 1:
                min_t = max(
                    uninfected_time_filtered.min(), infected_time_filtered.min()
                )
                max_t = min(
                    uninfected_time_filtered.max(), infected_time_filtered.max()
                )

                if min_t < max_t:
                    interp_func = interp1d(
                        uninfected_time_filtered,
                        uninfected_median_filtered,
                        kind="linear",
                        fill_value="extrapolate",
                    )

                    # Find timepoints where infected is significantly different
                    # Allow divergence detection across all timepoints (including before infection)
                    overlap_mask = (infected_time_filtered >= min_t) & (
                        infected_time_filtered <= max_t
                    )

                    overlap_times = infected_time_filtered[overlap_mask]
                    overlap_infected = infected_median_filtered[overlap_mask]
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

        ax.set_xlabel("Warped Pseudotime")
        # Update y-axis label
        if feature.endswith("_cv"):
            ax.set_ylabel(f"{feature}\n(raw CV)")
        elif feature.endswith("_sem"):
            ax.set_ylabel(f"{feature}\n(raw SEM)")
        elif is_pc_feature:
            ax.set_ylabel(f"{feature}\n(PC units, not normalized)")
        elif is_prenormalized:
            ax.set_ylabel(f"{feature}\n(pre-normalized, baseline t=1-10)")
        elif normalize_to_baseline and has_normalized_columns:
            ax.set_ylabel(f"{feature}\n(fold-change from baseline)")
        else:
            ax.set_ylabel(feature)
        ax.set_title(feature)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    # Create a single shared legend for the entire figure
    # Get handles and labels from the first subplot (they're all the same)
    handles, labels = axes[0].get_legend_handles_labels()
    # Place legend on the right side of the figure
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        fontsize=9,
        frameon=True,
    )

    # Use tight_layout with extra space on the right for the shared legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save with warped time suffix
    if output_root is not None:
        save_path = (
            output_root / f"infected_vs_uninfected_comparison_{ALIGN_TYPE}_warped.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved warped time comparison plot to {save_path}")

    plt.show()


# %%
# Plot individual lineages
# Use top_cells_df which has BOTH min_track_length filter AND top-N by DTW distance
# Important: Include "consensus" to ensure plotting methods have the reference pattern
consensus_df = master_df[master_df["lineage_id"] == -1].copy()

alignment_df_for_plotting = pd.concat([top_cells_df, consensus_df], ignore_index=True)

logger.info(
    f"Filtered plotting dataframe ({INFECTED_FOV_PATTERN}): {len(alignment_df_for_plotting)} rows, "
    f"{alignment_df_for_plotting['lineage_id'].nunique()} unique lineages"
)
logger.info(
    f"Includes consensus: {(alignment_df_for_plotting['fov_name'] == 'consensus').any()}"
)
logger.info(
    f"All lineages have minimum {min_track_length} timepoints and are top-{top_n_cells} by DTW distance (except consensus)"
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
    max_lineages=10,
)
# %%
# Infected vs uninfected comparison - WARPED TIME
logger.info("\n" + "=" * 70)
logger.info("WARPED TIME: Infected vs Uninfected Comparison")
logger.info("=" * 70)

# Get warped metadata
if cytodtw.consensus_data and "warped_metadata" in cytodtw.consensus_data:
    warped_meta = cytodtw.consensus_data["warped_metadata"]
    max_unaligned_before = warped_meta["max_unaligned_before"]
    consensus_aligned_length = warped_meta["consensus_aligned_length"]

    # Shift uninfected trajectory to align with warped time
    # Strategy: align infection time in uninfected with start of aligned region in warped time
    uninfected_shifted_df = uninfected_baseline_df.copy()
    time_shift = max_unaligned_before - absolute_infection_timepoint
    uninfected_shifted_df["t"] = uninfected_shifted_df["t"] + time_shift

    # Shift global infected trajectory to align with warped time (same shift)
    # This shows all infected cells without alignment in ABSOLUTE time, shifted to warped coordinates for comparison
    global_infected_shifted_df = global_infected_df.copy()
    global_infected_shifted_df["t"] = global_infected_df["t"] + time_shift

    # Create warped anchor metadata
    warped_anchor_metadata = {
        "anchor_start": max_unaligned_before,
        "anchor_end": max_unaligned_before + consensus_aligned_length - 1,
        "window_start": common_response_warped_df["t"].min(),
        "window_end": common_response_warped_df["t"].max(),
    }

    logger.info(f"Warped anchor metadata: {warped_anchor_metadata}")
    logger.info(
        f"Shifted uninfected time by {time_shift} frames to align with warped time"
    )
    logger.info(
        f"Shifted global infected (no alignment) by same {time_shift} frames to align with warped time"
    )

    # Plot warped time comparison
    plot_infected_vs_uninfected_comparison(
        common_response_warped_df,  # Infected top-N in warped time (aligned)
        uninfected_shifted_df,  # Uninfected shifted to warped time
        common_response_features,
        warped_metadata=warped_meta,
        anchor_metadata=warped_anchor_metadata,
        figsize=(18, 14),
        n_consecutive_divergence=5,
        global_infected_df=global_infected_shifted_df,  # All infected in warped time (no alignment)
        normalize_to_baseline=True,
        infected_label=INFECTED_LABEL,
        uninfected_label=UNINFECTED_LABEL,
        output_root=output_root,
        min_cell_count_threshold=NORMALIZE_N_CELLS_FOR_BASELINE,
    )

    # Divergence quantification analysis
    logger.info("\n" + "=" * 70)
    logger.info("DIVERGENCE TIMING ANALYSIS: Quantifying Organelle Remodeling")
    logger.info("=" * 70)

    # Configuration for divergence detection
    n_consecutive = 5
    threshold_multiplier = 1.5

    # Collect divergence results
    divergence_results = []

    for feature in common_response_features:
        logger.info(f"\nAnalyzing divergence for: {feature}")

        # Comparison 1: Aligned infected vs Uninfected (conserved response)
        result_aligned = quantify_divergence(
            test_df=common_response_warped_df,
            reference_df=uninfected_shifted_df,
            feature=f"{feature}_median",
            n_consecutive=n_consecutive,
            threshold_std_multiplier=threshold_multiplier,
            normalize_to_baseline=False,  # Already normalized in dataframes
        )
        result_aligned["feature"] = feature
        result_aligned["comparison"] = "aligned_vs_uninfected"
        divergence_results.append(result_aligned)

        # Comparison 2: Unaligned infected vs Uninfected (population average)
        result_unaligned = quantify_divergence(
            test_df=global_infected_shifted_df,
            reference_df=uninfected_shifted_df,
            feature=f"{feature}_median",
            n_consecutive=n_consecutive,
            threshold_std_multiplier=threshold_multiplier,
            normalize_to_baseline=False,
        )
        result_unaligned["feature"] = feature
        result_unaligned["comparison"] = "unaligned_vs_uninfected"
        divergence_results.append(result_unaligned)

        # Comparison 3: Aligned vs Unaligned infected (effect of synchronization)
        result_aligned_vs_unaligned = quantify_divergence(
            test_df=common_response_warped_df,
            reference_df=global_infected_shifted_df,
            feature=f"{feature}_median",
            n_consecutive=n_consecutive,
            threshold_std_multiplier=threshold_multiplier,
            normalize_to_baseline=False,
        )
        result_aligned_vs_unaligned["feature"] = feature
        result_aligned_vs_unaligned["comparison"] = "aligned_vs_unaligned"
        divergence_results.append(result_aligned_vs_unaligned)

    # Create results dataframe
    divergence_df = pd.DataFrame(divergence_results)

    # Save to CSV
    divergence_csv_path = output_root / f"divergence_analysis_{ALIGN_TYPE}.csv"
    divergence_df.to_csv(divergence_csv_path, index=False)
    logger.info(f"\nSaved divergence analysis to: {divergence_csv_path}")

    # Log results in markdown format
    logger.info("\n## Divergence Timing Analysis Results")
    logger.info("**Analysis**: Organelle remodeling timing during infection")
    logger.info(
        "**Method**: DTW-synchronized (aligned) vs unsynchronized (unaligned) populations"
    )
    logger.info(
        f"**Detection**: {n_consecutive} consecutive timepoints above {threshold_multiplier}x reference IQR\n"
    )

    # Summary statistics by comparison type
    for comparison in [
        "aligned_vs_uninfected",
        "unaligned_vs_uninfected",
        "aligned_vs_unaligned",
    ]:
        comparison_data = divergence_df[divergence_df["comparison"] == comparison]

        if comparison == "aligned_vs_uninfected":
            comp_label = "**DTW-Aligned Infected vs Uninfected Control**"
            description = "Conserved response timing in synchronized cells"
        elif comparison == "unaligned_vs_uninfected":
            comp_label = "**Unaligned Infected vs Uninfected Control**"
            description = "Population average without synchronization"
        else:
            comp_label = "**DTW-Aligned vs Unaligned Infected**"
            description = "Effect of DTW synchronization"

        logger.info(f"\n### {comp_label}")
        logger.info(f"_{description}_\n")

        # Table header
        logger.info(
            "| Feature | Divergence Time | Time from Start | Magnitude | Detected |"
        )
        logger.info(
            "|---------|----------------|-----------------|-----------|----------|"
        )

        for _, row in comparison_data.iterrows():
            divergence_time = (
                f"{row['divergence_time']:.1f}" if row["has_divergence"] else "N/A"
            )
            time_from_start = (
                f"{row['time_from_start']:.1f}" if row["has_divergence"] else "N/A"
            )
            magnitude = f"{row['divergence_magnitude']:.3f}"
            detected = "✓" if row["has_divergence"] else "✗"

            logger.info(
                f"| {row['feature']} | {divergence_time} | {time_from_start} | "
                f"{magnitude} | {detected} |"
            )

    # Summary insights
    logger.info("\n### Key Insights")

    # Which features diverge earliest in aligned cells?
    aligned_divergent = divergence_df[
        (divergence_df["comparison"] == "aligned_vs_uninfected")
        & (divergence_df["has_divergence"])
    ].sort_values("divergence_time")

    if len(aligned_divergent) > 0:
        earliest_features = aligned_divergent.head(3)
        logger.info("\n**Earliest remodeling (DTW-aligned):**")
        for _, row in earliest_features.iterrows():
            logger.info(
                f"- **{row['feature']}**: t={row['divergence_time']:.1f} (Δt={row['time_from_start']:.1f})"
            )

    # Does synchronization help reveal timing?
    features_with_both = []
    for feature in common_response_features:
        aligned_div = divergence_df[
            (divergence_df["feature"] == feature)
            & (divergence_df["comparison"] == "aligned_vs_uninfected")
        ].iloc[0]
        unaligned_div = divergence_df[
            (divergence_df["feature"] == feature)
            & (divergence_df["comparison"] == "unaligned_vs_uninfected")
        ].iloc[0]

        if aligned_div["has_divergence"] and unaligned_div["has_divergence"]:
            time_diff = (
                aligned_div["divergence_time"] - unaligned_div["divergence_time"]
            )
            features_with_both.append(
                {
                    "feature": feature,
                    "time_diff": time_diff,
                    "aligned_time": aligned_div["divergence_time"],
                    "unaligned_time": unaligned_div["divergence_time"],
                }
            )

    if len(features_with_both) > 0:
        logger.info(
            "\n**Impact of DTW synchronization (features with divergence in both):**"
        )
        for item in sorted(
            features_with_both, key=lambda x: abs(x["time_diff"]), reverse=True
        )[:3]:
            direction = "earlier" if item["time_diff"] < 0 else "later"
            logger.info(
                f"- **{item['feature']}**: Aligned diverges {abs(item['time_diff']):.1f} "
                f"timepoints {direction} (t={item['aligned_time']:.1f} vs t={item['unaligned_time']:.1f})"
            )

    logger.info("\n" + "=" * 70)
else:
    logger.warning("Warped metadata not available, skipping warped time comparison")


# %%
from cmap import Colormap
from skimage.exposure import adjust_gamma, rescale_intensity

z_range = slice(0, 1)
initial_yx_patch_size = (192, 192)
# Top matches should be unique fov_name and lineage_id combinations
matches_path = (
    output_root
    / f"consensus_lineage_{ALIGN_TYPE}_{ALIGNMENT_CHANNEL}_matching_lineages_cosine.csv"
)
matches = pd.read_csv(matches_path)
top_matches = matches.head(top_n_cells)

positions = []
tracks_tables = []
images_plate = open_ome_zarr(data_path)
selected_channels = images_plate.channel_names
# Load matching positions
print(f"Loading positions for {len(top_matches)} FOV matches...")
matches_found = 0
for _, pos in images_plate.positions():
    pos_name = pos.zgroup.name
    pos_normalized = pos_name.lstrip("/")

    if pos_normalized in top_matches["fov_name"].values:
        positions.append(pos)
        matches_found += 1

        # Get ALL tracks for this FOV to ensure TripletDataset has complete access
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


def load_images_from_triplet_dataset(fov_name, track_ids):
    """Load images from TripletDataset for given FOV and track IDs."""
    matching_indices = []
    for dataset_idx in range(len(dataset.valid_anchors)):
        anchor_row = dataset.valid_anchors.iloc[dataset_idx]
        if anchor_row["fov_name"] == fov_name and anchor_row["track_id"] in track_ids:
            matching_indices.append(dataset_idx)

    if not matching_indices:
        logger.warning(
            f"No matching indices found for FOV {fov_name}, tracks {track_ids}"
        )
        return {}

    # Get images and create time mapping
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


# Filter alignment_df_for_plotting to only aligned rows for loading just the aligned region
alignment_col = f"dtw_{ALIGN_TYPE}_aligned"
aligned_only_df = alignment_df_for_plotting[
    alignment_df_for_plotting[alignment_col]
].copy()

concatenated_image_sequences = get_aligned_image_sequences(
    cytodtw_instance=cytodtw,
    df=aligned_only_df,
    alignment_name=ALIGN_TYPE,
    image_loader_fn=load_images_from_triplet_dataset,
    max_lineages=30,
)

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

infection_timepoint = (
    absolute_infection_timepoint  # Use the computed value from line 103
)
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
