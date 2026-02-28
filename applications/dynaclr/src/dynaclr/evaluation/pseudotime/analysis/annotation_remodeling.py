# %%
"""
Annotation-based organelle remodeling analysis.

Measures remodeling timing using human annotations (organelle_state column)
directly from annotation CSVs — no model predictions required.

Pipeline: alignment → annotation signal → aggregation → metrics → plotting

Usage: Run as a Jupyter-compatible script (# %% cell markers).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from dynaclr.evaluation.pseudotime.src.alignment import align_tracks
from dynaclr.evaluation.pseudotime.src.metrics import (
    aggregate_population,
    compute_track_timing,
    find_half_max_time,
    find_onset_time,
    find_peak_metrics,
    run_statistical_tests,
)
from dynaclr.evaluation.pseudotime.src.plotting import (
    plot_cell_heatmap,
    plot_onset_comparison,
    plot_response_curves,
    plot_timing_distributions,
)
from dynaclr.evaluation.pseudotime.src.signals import (
    extract_annotation_signal,
)

# %%
# ===========================================================================
# Dataset configuration
# ===========================================================================

ANNOTATIONS_ROOT = Path("/hpc/projects/organelle_phenotyping/datasets/annotations")

ORGANELLE_CONFIG = {
    "G3BP1": {
        "experiments": [
            {
                "csv_path": ANNOTATIONS_ROOT
                / "2025_01_24_A549_G3BP1_DENV"
                / "2025_01_24_A549_G3BP1_DENV_combined_annotations.csv",
                "fov_pattern": "C/2",
                "frame_interval_minutes": 30,
                "label": "2025_01_24 DENV",
            },
            {
                "csv_path": ANNOTATIONS_ROOT
                / "2025_01_28_A549_G3BP1_ZIKV_DENV"
                / "2025_01_28_A549_G3BP1_ZIKV_DENV_combined_annotations.csv",
                "fov_pattern": "C/4",
                "frame_interval_minutes": 30,
                "label": "2025_01_28 ZIKV/DENV",
            },
            {
                "csv_path": ANNOTATIONS_ROOT
                / "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV_combined_annotations.csv",
                "fov_pattern": "C/2",
                "frame_interval_minutes": 10,
                "label": "2025_07_22 ZIKV",
            },
            {
                "csv_path": ANNOTATIONS_ROOT
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_combined_annotations.csv",
                "fov_pattern": "C/2",
                "frame_interval_minutes": 30,
                "label": "2025_07_24 ZIKV",
            },
        ],
        "controls": [
            {
                "csv_path": ANNOTATIONS_ROOT
                / "2025_01_28_A549_G3BP1_ZIKV_DENV"
                / "2025_01_28_A549_G3BP1_ZIKV_DENV_combined_annotations.csv",
                "fov_pattern": "B/4",
                "frame_interval_minutes": 30,
                "label": "2025_01_28 control (B/4)",
            },
            {
                "csv_path": ANNOTATIONS_ROOT
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_combined_annotations.csv",
                "fov_pattern": "C/1",
                "frame_interval_minutes": 30,
                "label": "2025_07_24 control (C/1)",
            },
        ],
        "label": "G3BP1 (Stress Granule)",
        "color": "#1f77b4",
    },
    "SEC61B": {
        "experiments": [
            {
                "csv_path": ANNOTATIONS_ROOT
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_combined_annotations.csv",
                "fov_pattern": "A/2",
                "frame_interval_minutes": 30,
                "label": "2025_07_24 ZIKV (SEC61B)",
            },
        ],
        "controls": [],
        "label": "SEC61B (ER)",
        "color": "#ff7f0e",
    },
}

# Analysis parameters
T_PERTURB_SOURCE = "annotation"
TIME_BINS_MINUTES = np.arange(-600, 901, 30)
MIN_CELLS_PER_BIN = 5
MIN_TRACK_TIMEPOINTS = 3
ONSET_THRESHOLD_SIGMA = 2

RESULTS_DIR = Path(__file__).parent / "results" / "annotation_remodeling"

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
        df = pd.read_csv(exp["csv_path"])
        print(f"    Loaded {len(df):,} annotations, t range: {df['t'].min()}-{df['t'].max()}")

        # Ensure parent_track_id exists
        if "parent_track_id" not in df.columns:
            df["parent_track_id"] = -1

        # Step 1: Alignment
        aligned = align_tracks(
            df,
            frame_interval_minutes=exp["frame_interval_minutes"],
            source=T_PERTURB_SOURCE,
            fov_pattern=exp["fov_pattern"],
            min_track_timepoints=MIN_TRACK_TIMEPOINTS,
        )

        # Step 2: Signal extraction (annotation-based)
        aligned = extract_annotation_signal(aligned, state_col="organelle_state", positive_value="remodel")
        aligned["experiment"] = exp["label"]
        aligned["organelle"] = organelle
        all_experiment_dfs.append(aligned)

    if not all_experiment_dfs:
        print(f"  No data for {organelle}, skipping")
        continue

    combined = pd.concat(all_experiment_dfs, ignore_index=True)

    # Step 3: Aggregate
    fraction_df = aggregate_population(combined, TIME_BINS_MINUTES, signal_type="fraction")

    n_tracks = combined.groupby(["fov_name", "track_id", "experiment"]).ngroups
    organelle_results[organelle] = {
        "combined_df": combined,
        "fraction_df": fraction_df,
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
# Process controls
# ===========================================================================

control_results = {}
for organelle, config in ORGANELLE_CONFIG.items():
    if not config.get("controls"):
        continue
    ctrl_dfs = []
    for ctrl in config["controls"]:
        df = pd.read_csv(ctrl["csv_path"])
        df = df[df["fov_name"].str.startswith(ctrl["fov_pattern"])].copy()
        ctrl_dfs.append(df)
    if ctrl_dfs:
        control_combined = pd.concat(ctrl_dfs, ignore_index=True)
        n_total = len(control_combined.dropna(subset=["organelle_state"]))
        n_remodel = (control_combined["organelle_state"] == "remodel").sum()
        fraction = n_remodel / n_total if n_total > 0 else 0
        control_results[organelle] = {
            "n_total": n_total,
            "n_remodel": n_remodel,
            "fraction": fraction,
        }
        print(f"  {organelle} control: {n_remodel}/{n_total} = {fraction:.4f}")

# %%
# ===========================================================================
# Step 4: Timing metrics
# ===========================================================================

timing_rows = []
for organelle, res in organelle_results.items():
    frac_df = res["fraction_df"]

    t_onset, threshold, bl_mean, bl_std = find_onset_time(
        frac_df,
        sigma_threshold=ONSET_THRESHOLD_SIGMA,
        min_cells_per_bin=MIN_CELLS_PER_BIN,
    )
    t_50 = find_half_max_time(frac_df)
    peak = find_peak_metrics(frac_df)

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
print("\n## Remodeling Timing Metrics\n")
print(timing_df.to_string(index=False))

# Per-track timing
all_track_timing = []
for organelle, res in organelle_results.items():
    track_timing = compute_track_timing(res["combined_df"], signal_type="fraction")
    track_timing["organelle"] = organelle
    all_track_timing.append(track_timing)

track_timing_df = pd.concat(all_track_timing, ignore_index=True)

# %%
# ===========================================================================
# Step 5: Plotting
# ===========================================================================

organelle_curves = {org: res["fraction_df"] for org, res in organelle_results.items()}
organelle_configs = {org: res["config"] for org, res in organelle_results.items()}

plot_response_curves(
    organelle_curves,
    organelle_configs,
    RESULTS_DIR,
    signal_type="fraction",
    min_cells_per_bin=MIN_CELLS_PER_BIN,
    title="Annotation-based organelle remodeling after infection",
    filename_prefix="annotation_remodeling_comparison",
)

for organelle, res in organelle_results.items():
    plot_cell_heatmap(
        res["combined_df"],
        TIME_BINS_MINUTES,
        signal_type="fraction",
        organelle_label=res["config"]["label"],
        output_dir=RESULTS_DIR,
        filename_prefix=f"{organelle}_annotation_heatmap",
    )

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

if len(organelle_results) > 1:
    stats_df = run_statistical_tests(organelle_results, track_timing_df, control_results or None)
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
    frac_path = RESULTS_DIR / f"{organelle}_fraction_curve.csv"
    res["fraction_df"].to_csv(frac_path, index=False)

print(f"\nResults saved to {RESULTS_DIR}")

# %%
