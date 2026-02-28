# %%
"""
Prediction-based organelle remodeling analysis.

Measures remodeling timing using classifier predictions
(predicted_organelle_state in AnnData) instead of human annotations.

Pipeline: alignment → prediction signal → aggregation → metrics → plotting

Usage: Run as a Jupyter-compatible script (# %% cell markers).
"""

import glob
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from applications.dynaclr.evaluation.pseudotime.alignment import align_tracks
from applications.dynaclr.evaluation.pseudotime.metrics import (
    aggregate_population,
    compute_track_timing,
    find_half_max_time,
    find_onset_time,
    find_peak_metrics,
    run_statistical_tests,
)
from applications.dynaclr.evaluation.pseudotime.plotting import (
    plot_cell_heatmap,
    plot_onset_comparison,
    plot_response_curves,
    plot_timing_distributions,
)
from applications.dynaclr.evaluation.pseudotime.signals import (
    extract_prediction_signal,
)

# %%
# ===========================================================================
# Dataset configuration
# ===========================================================================

ANNOTATIONS_ROOT = Path("/hpc/projects/organelle_phenotyping/datasets/annotations")
EMBEDDINGS_ROOT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics"
)

ORGANELLE_CONFIG = {
    "G3BP1": {
        "experiments": [
            {
                "embeddings_path": EMBEDDINGS_ROOT
                / "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "4-phenotyping/predictions/DynaCLR-2D-BagOfChannels-timeaware/v3",
                "embeddings_pattern": "*organelle*.zarr",
                "annotations_path": ANNOTATIONS_ROOT
                / "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV_combined_annotations.csv",
                "fov_pattern": "C/2",
                "frame_interval_minutes": 30,
                "task": "organelle_state",
                "label": "2025_07_22 ZIKV",
            },
        ],
        "controls": [],
        "label": "G3BP1 (Stress Granule)",
        "color": "#1f77b4",
    },
    "SEC61B": {
        "experiments": [
            {
                "embeddings_path": EMBEDDINGS_ROOT
                / "2024_11_07_A549_SEC61_DENV"
                / "4-phenotyping/2-predictions/DynaCLR-2D-BagOfChannels-timeaware/v3",
                "embeddings_pattern": "*organelle*.zarr",
                "annotations_path": ANNOTATIONS_ROOT
                / "2024_11_07_A549_SEC61B_DENV"
                / "2024_11_07_A549_SEC61B_DENV_combined_annotations.csv",
                "fov_pattern": "C/2",
                "frame_interval_minutes": 10,
                "task": "organelle_state",
                "label": "2024_11_07 DENV",
            },
        ],
        "controls": [],
        "label": "SEC61B (ER)",
        "color": "#ff7f0e",
    },
}

# Analysis parameters
T_PERTURB_SOURCE = "annotation"  # Default: use human annotations for T_perturb
USE_PROBABILITY = False  # Set True to use continuous probability instead of binary
TIME_BINS_MINUTES = np.arange(-600, 901, 30)
MIN_CELLS_PER_BIN = 5
MIN_TRACK_TIMEPOINTS = 3
ONSET_THRESHOLD_SIGMA = 2

RESULTS_DIR = Path(__file__).parent / "results" / "prediction_remodeling"

# %%
# ===========================================================================
# Step 1 + 2: Load data, alignment, and signal extraction
# ===========================================================================

organelle_results = {}

for organelle, config in ORGANELLE_CONFIG.items():
    print(f"\n{'=' * 60}")
    print(f"Processing {organelle}")
    print(f"{'=' * 60}")

    all_experiment_dfs = []

    for exp in config["experiments"]:
        print(f"\n  Experiment: {exp['label']}")

        # Load embeddings (AnnData with predictions)
        emb_files = glob.glob(
            str(exp["embeddings_path"] / exp["embeddings_pattern"])
        )
        if not emb_files:
            print(f"    No embeddings found matching: {exp['embeddings_pattern']}")
            continue

        adata = ad.read_zarr(emb_files[0])
        print(f"    Loaded {adata.shape[0]:,} embeddings")

        # Check predictions exist
        task = exp.get("task", "organelle_state")
        pred_col = f"predicted_{task}"
        if pred_col not in adata.obs.columns:
            print(f"    WARNING: '{pred_col}' not in adata.obs — skipping")
            continue

        # Load annotations for infection state alignment
        ann_df = pd.read_csv(exp["annotations_path"])
        if "parent_track_id" not in ann_df.columns:
            ann_df["parent_track_id"] = -1

        # Step 1: Alignment (using annotations for T_perturb)
        aligned = align_tracks(
            ann_df,
            frame_interval_minutes=exp["frame_interval_minutes"],
            source=T_PERTURB_SOURCE,
            fov_pattern=exp["fov_pattern"],
            min_track_timepoints=MIN_TRACK_TIMEPOINTS,
        )

        # Step 2: Signal extraction (prediction-based)
        aligned = extract_prediction_signal(
            adata,
            aligned,
            task=task,
            positive_value="remodel",
            use_probability=USE_PROBABILITY,
        )
        aligned["experiment"] = exp["label"]
        aligned["organelle"] = organelle
        all_experiment_dfs.append(aligned)

    if not all_experiment_dfs:
        print(f"  No data for {organelle}, skipping")
        continue

    combined = pd.concat(all_experiment_dfs, ignore_index=True)

    # Step 3: Aggregate
    signal_type = "continuous" if USE_PROBABILITY else "fraction"
    population_df = aggregate_population(
        combined, TIME_BINS_MINUTES, signal_type=signal_type
    )

    n_tracks = combined.groupby(["fov_name", "track_id", "experiment"]).ngroups
    organelle_results[organelle] = {
        "combined_df": combined,
        "population_df": population_df,
        "config": config,
        "n_tracks": n_tracks,
        "n_experiments": len(config["experiments"]),
        "n_frames": len(combined),
    }

    print(
        f"\n  **{organelle} summary**: {n_tracks} tracks, "
        f"{len(config['experiments'])} experiments, {len(combined):,} total frames"
    )

# %%
# ===========================================================================
# Step 4: Timing metrics
# ===========================================================================

timing_rows = []
for organelle, res in organelle_results.items():
    pop_df = res["population_df"]

    t_onset, threshold, bl_mean, bl_std = find_onset_time(
        pop_df,
        sigma_threshold=ONSET_THRESHOLD_SIGMA,
        min_cells_per_bin=MIN_CELLS_PER_BIN,
    )
    t_50 = find_half_max_time(pop_df)
    peak = find_peak_metrics(pop_df)

    timing_rows.append(
        {
            "organelle": organelle,
            "T_onset_minutes": t_onset,
            "T_50_minutes": t_50,
            "T_peak_minutes": peak["T_peak_minutes"],
            "peak_amplitude": peak["peak_amplitude"],
            "T_return_minutes": peak["T_return_minutes"],
            "pulse_duration_minutes": peak["pulse_duration_minutes"],
            "auc": peak["auc"],
            "baseline_mean": bl_mean,
            "baseline_std": bl_std,
            "n_tracks": res["n_tracks"],
            "n_experiments": res["n_experiments"],
        }
    )

timing_df = pd.DataFrame(timing_rows)
print("\n## Prediction-based Timing Metrics\n")
print(timing_df.to_string(index=False))

# Per-track timing
signal_type = "continuous" if USE_PROBABILITY else "fraction"
all_track_timing = []
for organelle, res in organelle_results.items():
    track_timing = compute_track_timing(
        res["combined_df"], signal_type=signal_type
    )
    track_timing["organelle"] = organelle
    all_track_timing.append(track_timing)

track_timing_df = pd.concat(all_track_timing, ignore_index=True)

# %%
# ===========================================================================
# Step 5: Plotting
# ===========================================================================

organelle_curves = {
    org: res["population_df"] for org, res in organelle_results.items()
}
organelle_configs = {org: res["config"] for org, res in organelle_results.items()}

plot_response_curves(
    organelle_curves,
    organelle_configs,
    RESULTS_DIR,
    signal_type=signal_type,
    min_cells_per_bin=MIN_CELLS_PER_BIN,
    title="Prediction-based organelle remodeling after infection",
    filename_prefix="prediction_remodeling_comparison",
)

for organelle, res in organelle_results.items():
    plot_cell_heatmap(
        res["combined_df"],
        TIME_BINS_MINUTES,
        signal_type=signal_type,
        organelle_label=res["config"]["label"],
        output_dir=RESULTS_DIR,
        filename_prefix=f"{organelle}_prediction_heatmap",
    )

if len(track_timing_df) > 0:
    plot_timing_distributions(
        track_timing_df,
        organelle_configs,
        RESULTS_DIR,
        filename_prefix="per_track_onset_duration",
    )

    plot_onset_comparison(
        timing_df,
        RESULTS_DIR,
        filename_prefix="onset_comparison",
    )

# %%
# ===========================================================================
# Step 6: Statistical tests
# ===========================================================================

if len(organelle_results) > 1 and len(track_timing_df) > 0:
    stats_df = run_statistical_tests(organelle_results, track_timing_df)
    print("\n## Statistical Tests\n")
    print(stats_df.to_string(index=False))
    stats_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

# %%
# ===========================================================================
# Step 7: Save CSVs
# ===========================================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timing_df.to_csv(RESULTS_DIR / "timing_metrics.csv", index=False)
track_timing_df.to_csv(RESULTS_DIR / "per_track_timing.csv", index=False)

for organelle, res in organelle_results.items():
    curve_path = RESULTS_DIR / f"{organelle}_population_curve.csv"
    res["population_df"].to_csv(curve_path, index=False)

print(f"\nResults saved to {RESULTS_DIR}")

# %%
