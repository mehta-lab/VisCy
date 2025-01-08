# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.distance import (
    compute_displacement,
    compute_displacement_statistics,
)

# Paths to datasets
feature_paths = {
    "7 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_7mins.zarr",
    "21 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_21mins.zarr",
    "28 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_updated_28mins.zarr",
    "56 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_56mins.zarr",
    "Cell Aware": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_cellaware.zarr",
    "Classical": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_classical.zarr",
}

# Different normalization strategies to test
norm_strategies = [None, "per_feature", "per_embedding", "per_dataset"]
labels = {
    None: "Raw",
    "per_feature": "Per-feature z-score",
    "per_embedding": "Unit norm",
    "per_dataset": "Dataset z-score",
}

# Colors for different time intervals
interval_colors = {
    "7 min interval": "blue",
    "21 min interval": "red",
    "28 min interval": "green",
    "56 min interval": "purple",
    "Cell Aware": "orange",
    "Classical": "gray",
}

# %% Compute MSD for each dataset
results = {}
raw_displacements = {}

for label, path in feature_paths.items():
    print(f"\nProcessing {label}...")
    embedding_dataset = read_embedding_dataset(Path(path))

    for norm in norm_strategies:
        # Compute displacements with different normalization strategies
        displacements = compute_displacement(
            embedding_dataset=embedding_dataset,
            distance_metric="euclidean_squared",
            normalize=norm,
        )
        means, stds = compute_displacement_statistics(displacements)
        results[f"{label} ({labels[norm]})"] = (means, stds)
        raw_displacements[f"{label} ({labels[norm]})"] = displacements

        # Print some statistics
        taus = sorted(means.keys())
        print(f"\n{labels[norm]}:")
        print(f"  Number of different τ values: {len(taus)}")
        print(f"  τ range: {min(taus)} to {max(taus)}")
        print(f"  MSD at τ=1: {means[1]:.4f} ± {stds[1]:.4f}")

# %% Plot MSD vs time - one plot per normalization strategy (linear scale)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, norm in enumerate(norm_strategies):
    ax = axes[i]

    # Plot each time interval for this normalization strategy
    for interval_label, path in feature_paths.items():
        result_label = f"{interval_label} ({labels[norm]})"
        means, stds = results[result_label]

        # Sort by tau for plotting
        taus = sorted(means.keys())
        mean_values = [means[tau] for tau in taus]
        std_values = [stds[tau] for tau in taus]

        ax.plot(
            taus,
            mean_values,
            "-",
            color=interval_colors[interval_label],
            alpha=0.5,
            zorder=1,
        )
        ax.scatter(
            taus,
            mean_values,
            color=interval_colors[interval_label],
            s=20,
            label=f"{interval_label}",
            zorder=2,
        )

    ax.set_xlabel("Time Shift (τ)")
    ax.set_ylabel("Mean Square Displacement")
    ax.set_title(f"MSD vs Time Shift\n({labels[norm]})")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()

# %% Plot MSD vs time - one plot per normalization strategy (log-log scale with slopes)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, norm in enumerate(norm_strategies):
    ax = axes[i]

    # Plot each time interval for this normalization strategy
    for interval_label, path in feature_paths.items():
        result_label = f"{interval_label} ({labels[norm]})"
        means, stds = results[result_label]

        # Sort by tau for plotting
        taus = sorted(means.keys())
        mean_values = [means[tau] for tau in taus]
        std_values = [stds[tau] for tau in taus]

        # Filter out non-positive values for log scale
        valid_mask = np.array(mean_values) > 0
        valid_taus = np.array(taus)[valid_mask]
        valid_means = np.array(mean_values)[valid_mask]

        # Calculate slope using linear regression on log-log values
        log_taus = np.log(valid_taus)
        log_means = np.log(valid_means)
        slope, intercept = np.polyfit(log_taus, log_means, 1)

        ax.plot(
            valid_taus,
            valid_means,
            "-",
            color=interval_colors[interval_label],
            alpha=0.5,
            zorder=1,
        )
        ax.scatter(
            valid_taus,
            valid_means,
            color=interval_colors[interval_label],
            s=20,
            label=f"{interval_label} (α={slope:.2f})",
            zorder=2,
        )

        # Plot fitted line
        fit_line = np.exp(intercept + slope * log_taus)
        ax.plot(
            valid_taus,
            fit_line,
            "--",
            color=interval_colors[interval_label],
            alpha=0.3,
            zorder=1,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time Shift (τ)")
    ax.set_ylabel("Mean Square Displacement")
    ax.set_title(f"MSD vs Time Shift (log-log)\n({labels[norm]})")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(title="α = slope in log-log space")

plt.tight_layout()
plt.show()

# %% Print detailed slope analysis
print("\nSlope Analysis (α):")
print("α = 1: Normal diffusion")
print("α < 1: Sub-diffusion (confined/hindered)")
print("α > 1: Super-diffusion (directed/active)\n")

for norm in norm_strategies:
    print(f"\n{labels[norm]}:")
    for interval_label in feature_paths.keys():
        result_label = f"{interval_label} ({labels[norm]})"
        means, _ = results[result_label]

        # Calculate slope
        taus = np.array(sorted(means.keys()))
        mean_values = np.array([means[tau] for tau in taus])
        valid_mask = mean_values > 0

        if np.sum(valid_mask) > 1:
            log_taus = np.log(taus[valid_mask])
            log_means = np.log(mean_values[valid_mask])
            slope, _ = np.polyfit(log_taus, log_means, 1)

            motion_type = (
                "normal diffusion"
                if abs(slope - 1) < 0.1
                else "sub-diffusion" if slope < 1 else "super-diffusion"
            )

            print(f"  {interval_label}: α = {slope:.2f} ({motion_type})")

# %% Plot slopes analysis
slopes_data = []
intervals = []
norm_types = []

for norm in norm_strategies:
    for interval_label in feature_paths.keys():
        result_label = f"{interval_label} ({labels[norm]})"
        means, _ = results[result_label]

        # Calculate slope
        taus = np.array(sorted(means.keys()))
        mean_values = np.array([means[tau] for tau in taus])
        valid_mask = mean_values > 0

        if np.sum(valid_mask) > 1:  # Need at least 2 points for slope
            log_taus = np.log(taus[valid_mask])
            log_means = np.log(mean_values[valid_mask])
            slope, _ = np.polyfit(log_taus, log_means, 1)

            slopes_data.append(slope)
            intervals.append(interval_label)
            norm_types.append(labels[norm])

# Create bar plot
plt.figure(figsize=(12, 6))

# Set up positions for grouped bars
unique_intervals = list(feature_paths.keys())
unique_norms = [labels[n] for n in norm_strategies]
x = np.arange(len(unique_intervals))
width = 0.8 / len(norm_strategies)  # Width of bars

for i, norm_label in enumerate(unique_norms):
    mask = np.array(norm_types) == norm_label
    norm_slopes = np.array(slopes_data)[mask]
    norm_intervals = np.array(intervals)[mask]

    positions = x + (i - len(norm_strategies) / 2 + 0.5) * width

    plt.bar(positions, norm_slopes, width, label=norm_label, alpha=0.7)

# Add reference lines
plt.axhline(y=1, color="k", linestyle="--", alpha=0.3, label="Normal diffusion (α=1)")
plt.axhline(y=0, color="k", linestyle="-", alpha=0.2)

plt.xlabel("Time Interval")
plt.ylabel("Slope (α)")
plt.title("MSD Slopes by Time Interval and Normalization Strategy")
plt.xticks(x, unique_intervals, rotation=45)
plt.legend(title="Normalization", bbox_to_anchor=(1.05, 1), loc="upper left")

# Add annotations for diffusion regimes
plt.text(
    plt.xlim()[1] * 1.2, 1.5, "Super-diffusion", rotation=90, verticalalignment="center"
)
plt.text(
    plt.xlim()[1] * 1.2, 0.5, "Sub-diffusion", rotation=90, verticalalignment="center"
)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
