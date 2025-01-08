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
}

# %% Compute MSD for each dataset
results = {}
raw_displacements = {}
max_tau = 200

# Different normalization strategies to test
norm_strategies = [None, "per_feature", "per_embedding", "per_dataset"]
colors = {
    None: "blue",
    "per_feature": "red",
    "per_embedding": "green",
    "per_dataset": "purple",
}
labels = {
    None: "Raw",
    "per_feature": "Per-feature z-score",
    "per_embedding": "Unit norm",
    "per_dataset": "Dataset z-score",
}

for label, path in feature_paths.items():
    print(f"\nProcessing {label}...")
    embedding_dataset = read_embedding_dataset(Path(path))

    for norm in norm_strategies:
        # Compute displacements with different normalization strategies
        displacements = compute_displacement(
            embedding_dataset=embedding_dataset,
            max_tau=max_tau,
            distance_metric="euclidean_squared",
            normalize=norm,
        )
        means, stds = compute_displacement_statistics(displacements)
        results[f"{label} ({labels[norm]})"] = (means, stds)
        raw_displacements[f"{label} ({labels[norm]})"] = displacements

        print(f"{labels[norm]} MSD at tau=1: {means[1]:.4f} ± {stds[1]:.4f}")

# %% Plot results with sample sizes
plt.figure(figsize=(12, 8))

for label, (means, stds) in results.items():
    taus = list(means.keys())
    mean_values = list(means.values())
    std_values = list(stds.values())

    # Get the normalization strategy from the label
    norm = next(n for n in norm_strategies if labels[n] in label)
    color = colors[norm]

    # Plot MSD with confidence band
    plt.plot(taus, mean_values, "o-", color=color, label=f"{label} (mean)")
    plt.fill_between(
        taus,
        np.array(mean_values) - np.array(std_values),
        np.array(mean_values) + np.array(std_values),
        alpha=0.3,
        color=color,
        label=f"{label} (±1σ)",
    )

plt.xlabel("Time Shift (τ)")
plt.ylabel("Mean Square Displacement")
plt.title(
    "Mean Square Displacement vs Time Shift\n(Comparing Normalization Strategies)"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %% Plot displacement distributions for different taus
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, norm in enumerate(norm_strategies):
    label = f"7 min interval ({labels[norm]})"
    displacements = raw_displacements[label]

    # Plot distributions for a few selected taus
    selected_taus = [1, 5, max_tau]
    for tau in selected_taus:
        values = displacements[tau]
        axes[i].hist(values, bins=50, alpha=0.3, density=True, label=f"τ = {tau}")

    axes[i].set_xlabel("Square Displacement")
    axes[i].set_ylabel("Density")
    axes[i].set_title(f"Distribution of Square Displacements\n({labels[norm]})")
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# %%
