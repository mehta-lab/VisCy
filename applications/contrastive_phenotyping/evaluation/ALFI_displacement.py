# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.distance import (
    calculate_normalized_euclidean_distance_cell,
    compute_displacement,
    compute_dynamic_range,
    compute_rms_per_track,
)
from collections import defaultdict
from tabulate import tabulate

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict

# %% function

# Removed redundant compute_displacement_mean_std_full function
# Removed redundant compute_dynamic_range and compute_rms_per_track functions


def plot_rms_histogram(rms_values, label, bins=30):
    """
    Plot histogram of RMS values across tracks.

    Parameters:
    rms_values : list
        List of RMS values, one for each track.
    label : str
        Label for the dataset (used in the title).
    bins : int, optional
        Number of bins for the histogram. Default is 30.

    Returns:
    None: Displays the histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(rms_values, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    plt.title(f"Histogram of RMS Values Across Tracks ({label})", fontsize=16)
    plt.xlabel("RMS of Time Derivative", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)
    plt.show()


def plot_displacement(
    mean_displacement, std_displacement, label, metrics_no_track=None
):
    """
    Plot embedding displacement over time with mean and standard deviation.

    Parameters:
    mean_displacement : dict
        Mean displacement for each tau.
    std_displacement : dict
        Standard deviation of displacement for each tau.
    label : str
        Label for the dataset.
    metrics_no_track : dict, optional
        Metrics for the "Classical Contrastive (No Tracking)" dataset to compare against.

    Returns:
    None: Displays the plot.
    """
    plt.figure(figsize=(10, 6))
    taus = list(mean_displacement.keys())
    mean_values = list(mean_displacement.values())
    std_values = list(std_displacement.values())

    plt.plot(taus, mean_values, marker="o", label=f"{label}", color="green")
    plt.fill_between(
        taus,
        np.array(mean_values) - np.array(std_values),
        np.array(mean_values) + np.array(std_values),
        color="green",
        alpha=0.3,
        label=f"Std Dev ({label})",
    )

    if metrics_no_track:
        mean_values_no_track = list(metrics_no_track["mean_displacement"].values())
        std_values_no_track = list(metrics_no_track["std_displacement"].values())

        plt.plot(
            taus,
            mean_values_no_track,
            marker="o",
            label="Classical Contrastive (No Tracking)",
            color="blue",
        )
        plt.fill_between(
            taus,
            np.array(mean_values_no_track) - np.array(std_values_no_track),
            np.array(mean_values_no_track) + np.array(std_values_no_track),
            color="blue",
            alpha=0.3,
            label="Std Dev (No Tracking)",
        )

    plt.xlabel("Time Shift (τ)", fontsize=14)
    plt.ylabel("Euclidean Distance", fontsize=14)
    plt.title(f"Embedding Displacement Over Time ({label})", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()


def plot_overlay_displacement(overlay_displacement_data):
    """
    Plot embedding displacement over time for all datasets in one plot.

    Parameters:
    overlay_displacement_data : dict
        A dictionary containing mean displacement per tau for all datasets.

    Returns:
    None: Displays the plot.
    """
    plt.figure(figsize=(12, 8))
    for label, mean_displacement in overlay_displacement_data.items():
        taus = list(mean_displacement.keys())
        mean_values = list(mean_displacement.values())
        plt.plot(taus, mean_values, marker="o", label=label)

    plt.xlabel("Time Shift (τ)", fontsize=14)
    plt.ylabel("Euclidean Distance", fontsize=14)
    plt.title("Overlayed Embedding Displacement Over Time", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()


# %% hist stats
def plot_boxplot_rms_across_models(datasets_rms):
    """
    Plot a boxplot for the distribution of RMS values across models.

    Parameters:
    datasets_rms : dict
        A dictionary where keys are dataset names and values are lists of RMS values.

    Returns:
    None: Displays the boxplot.
    """
    plt.figure(figsize=(12, 6))
    labels = list(datasets_rms.keys())
    data = list(datasets_rms.values())
    print(labels)
    print(data)
    # Plot the boxplot
    plt.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)

    plt.title(
        "Distribution of RMS of Rate of Change of Embedding Across Models", fontsize=16
    )
    plt.ylabel("RMS of Time Derivative", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_histogram_absolute_differences(datasets_abs_diff):
    """
    Plot histograms of absolute differences across embeddings for all models.

    Parameters:
    datasets_abs_diff : dict
        A dictionary where keys are dataset names and values are lists of absolute differences.

    Returns:
    None: Displays the histograms.
    """
    plt.figure(figsize=(12, 6))
    for label, abs_diff in datasets_abs_diff.items():
        plt.hist(abs_diff, bins=50, alpha=0.5, label=label, density=True)

    plt.title("Histograms of Absolute Differences Across Models", fontsize=16)
    plt.xlabel("Absolute Difference", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.7)
    plt.tight_layout()
    plt.show()


# %% Paths to datasets
feature_paths = {
    "7 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_7mins.zarr",
    "21 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_21mins.zarr",
    "28 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_updated_28mins.zarr",
    "56 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_56mins.zarr",
    "Cell Aware": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_cellaware.zarr",
}

no_track_path = "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_classical.zarr"

# %% Process Datasets
max_tau = 69
metrics = {}

overlay_displacement_data = {}
datasets_rms = {}
datasets_abs_diff = {}

# Process "No Tracking" dataset
features_path_no_track = Path(no_track_path)
embedding_dataset_no_track = read_embedding_dataset(features_path_no_track)

mean_displacement_no_track, std_displacement_no_track = compute_displacement(
    embedding_dataset_no_track, max_tau=max_tau, return_mean_std=True
)
dynamic_range_no_track = compute_dynamic_range(mean_displacement_no_track)
metrics["No Tracking"] = {
    "dynamic_range": dynamic_range_no_track,
    "mean_displacement": mean_displacement_no_track,
    "std_displacement": std_displacement_no_track,
}

overlay_displacement_data["No Tracking"] = mean_displacement_no_track

print("\nProcessing No Tracking dataset...")
print(f"Dynamic Range for No Tracking: {dynamic_range_no_track}")

plot_displacement(mean_displacement_no_track, std_displacement_no_track, "No Tracking")

rms_values_no_track = compute_rms_per_track(embedding_dataset_no_track)
datasets_rms["No Tracking"] = rms_values_no_track

print(f"Plotting histogram of RMS values for No Tracking dataset...")
plot_rms_histogram(rms_values_no_track, "No Tracking", bins=30)

# Compute absolute differences for "No Tracking"
abs_diff_no_track = np.concatenate(
    [
        np.linalg.norm(
            np.diff(embedding_dataset_no_track["features"].values[indices], axis=0),
            axis=-1,
        )
        for indices in np.split(
            np.arange(len(embedding_dataset_no_track["track_id"])),
            np.where(np.diff(embedding_dataset_no_track["track_id"]) != 0)[0] + 1,
        )
    ]
)
datasets_abs_diff["No Tracking"] = abs_diff_no_track

# Process other datasets
for label, path in feature_paths.items():
    print(f"\nProcessing {label} dataset...")

    features_path = Path(path)
    embedding_dataset = read_embedding_dataset(features_path)

    mean_displacement, std_displacement = compute_displacement(
        embedding_dataset, max_tau=max_tau, return_mean_std=True
    )
    dynamic_range = compute_dynamic_range(mean_displacement)
    metrics[label] = {
        "dynamic_range": dynamic_range,
        "mean_displacement": mean_displacement,
        "std_displacement": std_displacement,
    }

    overlay_displacement_data[label] = mean_displacement

    print(f"Dynamic Range for {label}: {dynamic_range}")

    plot_displacement(
        mean_displacement,
        std_displacement,
        label,
        metrics_no_track=metrics.get("No Tracking", None),
    )

    rms_values = compute_rms_per_track(embedding_dataset)
    datasets_rms[label] = rms_values

    print(f"Plotting histogram of RMS values for {label}...")
    plot_rms_histogram(rms_values, label, bins=30)

    abs_diff = np.concatenate(
        [
            np.linalg.norm(
                np.diff(embedding_dataset["features"].values[indices], axis=0), axis=-1
            )
            for indices in np.split(
                np.arange(len(embedding_dataset["track_id"])),
                np.where(np.diff(embedding_dataset["track_id"]) != 0)[0] + 1,
            )
        ]
    )
    datasets_abs_diff[label] = abs_diff

print("\nPlotting overlayed displacement for all datasets...")
plot_overlay_displacement(overlay_displacement_data)

print("\nSummary of Dynamic Ranges:")
for label, metric in metrics.items():
    print(f"{label}: Dynamic Range = {metric['dynamic_range']}")

print("\nPlotting RMS boxplot across models...")
plot_boxplot_rms_across_models(datasets_rms)

print("\nPlotting histograms of absolute differences across models...")
plot_histogram_absolute_differences(datasets_abs_diff)


# %%
