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

# %% Plot MSD vs time - one plot per normalization strategy
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

        # Plot MSD with confidence band
        ax.plot(
            taus,
            mean_values,
            "-",
            color=interval_colors[interval_label],
            label=f"{interval_label}",
        )
        ax.fill_between(
            taus,
            np.array(mean_values) - np.array(std_values),
            np.array(mean_values) + np.array(std_values),
            alpha=0.3,
            color=interval_colors[interval_label],
        )

    ax.set_xlabel("Time Shift (τ)")
    ax.set_ylabel("Mean Square Displacement")
    ax.set_title(f"MSD vs Time Shift\n({labels[norm]})")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
# %%
