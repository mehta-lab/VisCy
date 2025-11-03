# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.distance import (
    compute_track_displacement,
)

# Paths to datasets
feature_paths = {
    "7 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_7mins.zarr",
    "14 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_14mins.zarr",
    "28 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_28mins.zarr",
    "56 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_56mins.zarr",
    "91 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_91mins.zarr",
}


cmap = plt.get_cmap("tab10")  # or use "Set2", "tab20", etc.
labels = list(feature_paths.keys())
interval_colors = {label: cmap(i % cmap.N) for i, label in enumerate(labels)}

# Print and check each path
for label, path in feature_paths.items():
    print(f"{label} color: {interval_colors[label]}")
    assert Path(path).exists(), f"Path {path} does not exist"

# %% Compute MSD for each dataset
results = {}
raw_displacements = {}

DISTANCE_METRIC = "cosine"
for label, path in feature_paths.items():
    results[label] = {}
    print(f"\nProcessing {label}...")
    embedding_dataset = read_embedding_dataset(Path(path))

    # Compute displacements
    displacements_per_tau = compute_track_displacement(
        embedding_dataset=embedding_dataset,
        distance_metric=DISTANCE_METRIC,
    )

    # Store displacements with conditional normalization
    if DISTANCE_METRIC == "cosine":
        # Cosine distance is already scale-invariant, no normalization needed
        for tau, displacements in displacements_per_tau.items():
            results[label][tau] = displacements
    else:
        # Normalize by embeddings variance for euclidean distance
        embeddings_variance = np.var(embedding_dataset["features"].values)
        for tau, displacements in displacements_per_tau.items():
            results[label][tau] = [disp / embeddings_variance for disp in displacements]


# %% Plot MSD vs time (linear scale)
show_power_law_fits = True
log_scale = True
title = "Mean Track Displacement vs Time Shift"

fig, ax = plt.subplots(figsize=(10, 7))

for model_type, msd_data in results.items():
    time_lags = sorted(msd_data.keys())
    msd_means = []
    msd_stds = []

    # Compute mean and std of MSD for each time lag
    for tau in time_lags:
        displacements = np.array(msd_data[tau])
        msd_means.append(np.mean(displacements))
        msd_stds.append(np.std(displacements) / np.sqrt(len(displacements)))

    time_lags = np.array(time_lags)
    msd_means = np.array(msd_means)
    msd_stds = np.array(msd_stds)

    # Plot with error bars
    color = interval_colors.get(model_type, "#1f77b4")
    ax.errorbar(
        time_lags,
        msd_means,
        yerr=msd_stds,
        marker="o",
        label=f"{model_type.replace('_', ' ').title()}",
        color=color,
        capsize=3,
        capthick=1,
        linewidth=2,
        markersize=6,
    )
    # Fit power law if requested
    if show_power_law_fits and len(time_lags) > 3:
        valid_mask = (time_lags > 0) & (msd_means > 0)
        if np.sum(valid_mask) > 3:
            log_tau = np.log(time_lags[valid_mask])
            log_msd = np.log(msd_means[valid_mask])

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_tau, log_msd
            )

            # Plot fit line
            tau_fit = np.linspace(
                time_lags[valid_mask][0], time_lags[valid_mask][-1], 50
            )
            msd_fit = np.exp(intercept) * tau_fit**slope

            ax.plot(
                tau_fit,
                msd_fit,
                "--",
                color=color,
                alpha=0.7,
                label=f"{model_type}: α={slope:.2f} (R²={r_value**2:.3f})",
            )

    ax.set_xlabel("Time Lag (τ)", fontsize=12)
    ax.set_ylabel("Mean Track Displacement", fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    ax.legend()
    plt.tight_layout()
plt.savefig(f"msd_vs_time_shift_{DISTANCE_METRIC}.png", dpi=300)
# %%
# Step size analysis


def extract_step_sizes(embedding_dataset: xr.Dataset):
    """Extract step sizes with simple coordinate access."""

    unique_tracks_df = (
        embedding_dataset[["fov_name", "track_id"]].to_dataframe().drop_duplicates()
    )
    all_step_sizes = []

    for fov_name, track_id in zip(
        unique_tracks_df["fov_name"], unique_tracks_df["track_id"]
    ):
        track_data = embedding_dataset.where(
            (embedding_dataset["fov_name"] == fov_name)
            & (embedding_dataset["track_id"] == track_id),
            drop=True,
        )
        time_order = np.argsort(track_data["t"].values)
        times = track_data["t"].values[time_order]
        track_embeddings = track_data["features"].values[time_order]
        if len(times) != len(np.unique(times)):
            print(f"Duplicates found in FOV {fov_name}, track {track_id}")

        if len(track_embeddings) > 1:
            steps = np.diff(track_embeddings, axis=0)
            step_sizes = np.linalg.norm(steps, axis=1)
            all_step_sizes.extend(step_sizes)

    return np.array(all_step_sizes)


all_step_data = {}
cv_values = []
labels = []

for label, path in feature_paths.items():
    print(f"\nProcessing {label}...")
    embedding_dataset = read_embedding_dataset(Path(path))
    steps = extract_step_sizes(embedding_dataset)
    all_step_data[label] = steps

    # Calculate coefficient of variation
    cv = np.std(steps) / np.mean(steps)
    cv_values.append(cv)
    labels.append(label.replace("_", " ").title())

# %%
# Plot histograms
ax1, ax2 = plt.subplots(1, 2, figsize=(15, 6))[1]

for model_type, steps in all_step_data.items():
    color = interval_colors.get(model_type, "#1f77b4")
    ax1.hist(
        steps,
        bins=50,
        alpha=0.7,
        color=color,
        label=f"{model_type.replace('_', ' ').title()} (n={len(steps)}, μ={np.mean(steps):.3f}, σ={np.std(steps):.3f})",
    )

ax1.set_xlabel("Step Size")
ax1.set_ylabel("Frequency")
ax1.set_title("Step Size Distributions")
ax1.legend()

# Plot coefficient of variation
bar_colors = [
    interval_colors.get(model_type, "#1f77b4") for model_type in results.keys()
]
bars = ax2.bar(labels, cv_values, color=bar_colors, alpha=0.7)
ax2.set_ylabel("Coefficient of Variation (σ/μ)")
ax2.set_title("Step Size Variability")
ax2.tick_params(axis="x", rotation=45)
plt.tight_layout()
# plt.show()
plt.savefig(f"step_size_distributions_{DISTANCE_METRIC}.png", dpi=300)

# %%
