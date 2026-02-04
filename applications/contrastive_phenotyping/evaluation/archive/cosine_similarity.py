# %%
# Import necessary libraries, try euclidean distance for both features and
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.distance import (
    calculate_cosine_similarity_cell,
    compute_displacement,
    compute_displacement_mean_std,
)

# %% Paths and parameters.


features_path_30_min = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)


feature_path_no_track = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr"
)


features_path_any_time = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_2chan_128patch_32projDim/2chan_128patch_56ckpt_FebTest.zarr"
)


data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)


tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
)


# %% Load embedding datasets for all three sampling
fov_name = "/B/4/6"
track_id = 52

embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)
embedding_dataset_any_time = read_embedding_dataset(features_path_any_time)

# Calculate cosine similarities for each sampling
time_points_30_min, cosine_similarities_30_min = calculate_cosine_similarity_cell(
    embedding_dataset_30_min, fov_name, track_id
)
time_points_no_track, cosine_similarities_no_track = calculate_cosine_similarity_cell(
    embedding_dataset_no_track, fov_name, track_id
)
time_points_any_time, cosine_similarities_any_time = calculate_cosine_similarity_cell(
    embedding_dataset_any_time, fov_name, track_id
)

# %% Plot cosine similarities over time for all three conditions

plt.figure(figsize=(10, 6))

plt.plot(
    time_points_no_track,
    cosine_similarities_no_track,
    marker="o",
    label="classical contrastive (no tracking)",
)
plt.plot(
    time_points_any_time, cosine_similarities_any_time, marker="o", label="cell aware"
)
plt.plot(
    time_points_30_min,
    cosine_similarities_30_min,
    marker="o",
    label="cell & time aware (interval 30 min)",
)

plt.xlabel("Time Delay (t)")
plt.ylabel("Cosine Similarity with First Time Point")
plt.title("Cosine Similarity Over Time for Infected Cell")

# plt.savefig('infected_cell_example.pdf', format='pdf')


plt.grid(True)

plt.legend()

plt.savefig("new_example_cell.svg", format="svg")


plt.show()
# %%


# %% import statements


# %% Paths to datasets
features_path_30_min = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)
feature_path_no_track = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr"
)
# features_path_any_time = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_1chan_128patch_32projDim/1chan_128patch_63ckpt_FebTest.zarr")


# %% Read embedding datasets
embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)
# embedding_dataset_any_time = read_embedding_dataset(features_path_any_time)


# %% Compute displacements for both datasets (using Euclidean distance and Cosine similarity)
max_tau = 10  # Maximum time shift (tau) to compute displacements


# mean_displacement_30_min, std_displacement_30_min = compute_displacement_mean_std(embedding_dataset_30_min, max_tau, use_cosine=False, use_dissimilarity=False)
# mean_displacement_no_track, std_displacement_no_track = compute_displacement_mean_std(embedding_dataset_no_track, max_tau, use_cosine=False, use_dissimilarity=False)
# mean_displacement_any_time, std_displacement_any_time = compute_displacement_mean_std(embedding_dataset_any_time, max_tau, use_cosine=False)


mean_displacement_30_min_cosine, std_displacement_30_min_cosine = (
    compute_displacement_mean_std(
        embedding_dataset_30_min, max_tau, use_cosine=True, use_dissimilarity=False
    )
)
mean_displacement_no_track_cosine, std_displacement_no_track_cosine = (
    compute_displacement_mean_std(
        embedding_dataset_no_track, max_tau, use_cosine=True, use_dissimilarity=False
    )
)
# mean_displacement_any_time_cosine, std_displacement_any_time_cosine = compute_displacement_mean_std(embedding_dataset_any_time, max_tau, use_cosine=True)
# %% Plot 1: Euclidean Displacements
plt.figure(figsize=(10, 6))


taus = list(mean_displacement_30_min_cosine.keys())
mean_values_30_min = list(mean_displacement_30_min_cosine.values())
std_values_30_min = list(std_displacement_30_min_cosine.values())


mean_values_no_track = list(mean_displacement_no_track_cosine.values())
std_values_no_track = list(std_displacement_no_track_cosine.values())


# mean_values_any_time = list(mean_displacement_any_time.values())
# std_values_any_time = list(std_displacement_any_time.values())


# Plotting Euclidean displacements
plt.plot(
    taus, mean_values_30_min, marker="o", label="Cell & Time Aware (30 min interval)"
)
plt.fill_between(
    taus,
    np.array(mean_values_30_min) - np.array(std_values_30_min),
    np.array(mean_values_30_min) + np.array(std_values_30_min),
    color="gray",
    alpha=0.3,
    label="Std Dev (30 min interval)",
)


plt.plot(
    taus, mean_values_no_track, marker="o", label="Classical Contrastive (No Tracking)"
)
plt.fill_between(
    taus,
    np.array(mean_values_no_track) - np.array(std_values_no_track),
    np.array(mean_values_no_track) + np.array(std_values_no_track),
    color="blue",
    alpha=0.3,
    label="Std Dev (No Tracking)",
)


plt.xlabel("Time Shift (τ)")
plt.ylabel("Displacement")
plt.title("Embedding Displacement Over Time")
plt.grid(True)
plt.legend()


# plt.savefig('embedding_displacement_euclidean.svg', format='svg')
# plt.savefig('embedding_displacement_euclidean.pdf', format='pdf')


# Show the Euclidean plot
plt.show()


# %% Plot 2: Cosine Displacements
plt.figure(figsize=(10, 6))

taus = list(mean_displacement_30_min_cosine.keys())

# Plotting Cosine displacements
mean_values_30_min_cosine = list(mean_displacement_30_min_cosine.values())
std_values_30_min_cosine = list(std_displacement_30_min_cosine.values())


mean_values_no_track_cosine = list(mean_displacement_no_track_cosine.values())
std_values_no_track_cosine = list(std_displacement_no_track_cosine.values())


plt.plot(
    taus,
    mean_values_30_min_cosine,
    marker="o",
    label="Cell & Time Aware (30 min interval)",
)
plt.fill_between(
    taus,
    np.array(mean_values_30_min_cosine) - np.array(std_values_30_min_cosine),
    np.array(mean_values_30_min_cosine) + np.array(std_values_30_min_cosine),
    color="gray",
    alpha=0.3,
    label="Std Dev (30 min interval)",
)


plt.plot(
    taus,
    mean_values_no_track_cosine,
    marker="o",
    label="Classical Contrastive (No Tracking)",
)
plt.fill_between(
    taus,
    np.array(mean_values_no_track_cosine) - np.array(std_values_no_track_cosine),
    np.array(mean_values_no_track_cosine) + np.array(std_values_no_track_cosine),
    color="blue",
    alpha=0.3,
    label="Std Dev (No Tracking)",
)


plt.xlabel("Time Shift (τ)")
plt.ylabel("Cosine Similarity")
plt.title("Embedding Displacement Over Time")


plt.grid(True)
plt.legend()
plt.savefig("1_std_cosine_plot.svg", format="svg")

# Show the Cosine plot
plt.show()
# %%


# %% Paths to datasets
features_path_30_min = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)
feature_path_no_track = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr"
)


# %% Read embedding datasets
embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)


# %% Compute displacements for both datasets (using Cosine similarity)
max_tau = 10  # Maximum time shift (tau) to compute displacements


# Compute displacements for Cell & Time Aware (30 min interval) using Cosine similarity
displacement_per_tau_aware_cosine = compute_displacement(
    embedding_dataset_30_min,
    max_tau,
    use_cosine=True,
    use_dissimilarity=False,
    use_umap=False,
)


# Compute displacements for Classical Contrastive (No Tracking) using Cosine similarity
displacement_per_tau_contrastive_cosine = compute_displacement(
    embedding_dataset_no_track,
    max_tau,
    use_cosine=True,
    use_dissimilarity=False,
    use_umap=False,
)


# %% Prepare data for violin plot
def prepare_violin_data(taus, displacement_aware, displacement_contrastive):
    # Create a list to hold the data
    data = []

    # Populate the data for Cell & Time Aware
    for tau in taus:
        displacements_aware = displacement_aware.get(tau, [])
        for displacement in displacements_aware:
            data.append(
                {
                    "Time Shift (τ)": tau,
                    "Displacement": displacement,
                    "Sampling": "Cell & Time Aware (30 min interval)",
                }
            )

    # Populate the data for Classical Contrastive
    for tau in taus:
        displacements_contrastive = displacement_contrastive.get(tau, [])
        for displacement in displacements_contrastive:
            data.append(
                {
                    "Time Shift (τ)": tau,
                    "Displacement": displacement,
                    "Sampling": "Classical Contrastive (No Tracking)",
                }
            )

    # Convert to a DataFrame
    df = pd.DataFrame(data)
    return df


taus = list(displacement_per_tau_aware_cosine.keys())


# Prepare the violin plot data
df = prepare_violin_data(
    taus, displacement_per_tau_aware_cosine, displacement_per_tau_contrastive_cosine
)


# Create a violin plot using seaborn
plt.figure(figsize=(12, 8))
sns.violinplot(
    x="Time Shift (τ)",
    y="Displacement",
    hue="Sampling",
    data=df,
    palette="Set2",
    scale="width",
    bw=0.2,
    inner=None,
    split=True,
    cut=0,
)


# Add labels and title
plt.xlabel("Time Shift (τ)", fontsize=14)
plt.ylabel("Cosine Similarity", fontsize=14)
plt.title("Cosine Similarity Distribution on Features", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)  # Lighter grid lines for less distraction
plt.legend(title="Sampling", fontsize=12, title_fontsize=14)


# plt.ylim(0.5, 1.0)


# Save the violin plot as SVG and PDF
plt.savefig("1fixed_violin_plot_cosine_similarity.svg", format="svg")
# plt.savefig('violin_plot_cosine_similarity.pdf', format='pdf')


# Show the plot
plt.show()
# %% using umap violin plot

# %% Paths to datasets
features_path_30_min = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)
feature_path_no_track = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr"
)

# %% Read embedding datasets
embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)


# %% Compute UMAP on features
def compute_umap(dataset):
    features = dataset["features"]
    scaled_features = StandardScaler().fit_transform(features.values)
    umap = UMAP(n_components=2)  # Reduce to 2 dimensions
    embedding = umap.fit_transform(scaled_features)

    # Add UMAP coordinates using xarray functionality
    umap_features = features.assign_coords(
        UMAP1=("sample", embedding[:, 0]), UMAP2=("sample", embedding[:, 1])
    )
    return umap_features


# Apply UMAP to both datasets
umap_features_30_min = compute_umap(embedding_dataset_30_min)
umap_features_no_track = compute_umap(embedding_dataset_no_track)

# %%
print(umap_features_30_min)
# %% Visualize UMAP embeddings
# # Visualize UMAP embeddings for the 30 min interval
# plt.figure(figsize=(8, 6))
# plt.scatter(umap_features_30_min[:, 0], umap_features_30_min[:, 1], c=embedding_dataset_30_min["t"].values, cmap='viridis')
# plt.colorbar(label='Timepoints')
# plt.title('UMAP Projection of Features (30 min Interval)')
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')
# plt.show()

# # Visualize UMAP embeddings for the No Tracking dataset
# plt.figure(figsize=(8, 6))
# plt.scatter(umap_features_no_track[:, 0], umap_features_no_track[:, 1], c=embedding_dataset_no_track["t"].values, cmap='viridis')
# plt.colorbar(label='Timepoints')
# plt.title('UMAP Projection of Features (No Tracking)')
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')
# plt.show()
# %% Compute displacements using UMAP coordinates (using Cosine similarity)
max_tau = 10  # Maximum time shift (tau) to compute displacements

# Compute displacements for UMAP-processed Cell & Time Aware (30 min interval)
displacement_per_tau_aware_umap_cosine = compute_displacement(
    umap_features_30_min,
    max_tau,
    use_cosine=True,
    use_dissimilarity=False,
    use_umap=True,
)

# Compute displacements for UMAP-processed Classical Contrastive (No Tracking)
displacement_per_tau_contrastive_umap_cosine = compute_displacement(
    umap_features_no_track,
    max_tau,
    use_cosine=True,
    use_dissimilarity=False,
    use_umap=True,
)


# %% Prepare data for violin plot
def prepare_violin_data(taus, displacement_aware, displacement_contrastive):
    # Create a list to hold the data
    data = []

    # Populate the data for Cell & Time Aware
    for tau in taus:
        displacements_aware = displacement_aware.get(tau, [])
        for displacement in displacements_aware:
            data.append(
                {
                    "Time Shift (τ)": tau,
                    "Displacement": displacement,
                    "Sampling": "Cell & Time Aware (30 min interval)",
                }
            )

    # Populate the data for Classical Contrastive
    for tau in taus:
        displacements_contrastive = displacement_contrastive.get(tau, [])
        for displacement in displacements_contrastive:
            data.append(
                {
                    "Time Shift (τ)": tau,
                    "Displacement": displacement,
                    "Sampling": "Classical Contrastive (No Tracking)",
                }
            )

    # Convert to a DataFrame
    df = pd.DataFrame(data)
    return df


taus = list(displacement_per_tau_aware_umap_cosine.keys())

# Prepare the violin plot data
df = prepare_violin_data(
    taus,
    displacement_per_tau_aware_umap_cosine,
    displacement_per_tau_contrastive_umap_cosine,
)

# %% Create a violin plot using seaborn
plt.figure(figsize=(12, 8))
sns.violinplot(
    x="Time Shift (τ)",
    y="Displacement",
    hue="Sampling",
    data=df,
    palette="Set2",
    scale="width",
    bw=0.2,
    inner=None,
    split=True,
    cut=0,
)

# Add labels and title
plt.xlabel("Time Shift (τ)", fontsize=14)
plt.ylabel("Cosine Similarity", fontsize=14)
plt.title("Cosine Similarity Distribution using UMAP Features", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)  # Lighter grid lines for less distraction
plt.legend(title="Sampling", fontsize=12, title_fontsize=14)

# plt.ylim(0, 1)

# Save the violin plot as SVG and PDF
plt.savefig("fixed_plot_cosine_similarity.svg", format="svg")
# plt.savefig('violin_plot_cosine_similarity_umap.pdf', format='pdf')

# Show the plot
plt.show()


# %%
# %% Visualize Displacement Distributions (Example Code)
# Compare displacement distributions for τ = 1
# plt.figure(figsize=(10, 6))
# sns.histplot(displacement_per_tau_aware_umap_cosine[1], kde=True, label='UMAP - 30 min Interval', color='blue')
# sns.histplot(displacement_per_tau_contrastive_umap_cosine[1], kde=True, label='UMAP - No Tracking', color='green')
# plt.legend()
# plt.title('Comparison of Displacement Distributions for τ = 1 (UMAP)')
# plt.xlabel('Displacement')
# plt.show()

# # Compare displacement distributions for the full feature set (same τ = 1)
# plt.figure(figsize=(10, 6))
# sns.histplot(displacement_per_tau_aware_cosine[1], kde=True, label='Full Features - 30 min Interval', color='red')
# sns.histplot(displacement_per_tau_contrastive_cosine[1], kde=True, label='Full Features - No Tracking', color='orange')
# plt.legend()
# plt.title('Comparison of Displacement Distributions for τ = 1 (Full Features)')
# plt.xlabel('Displacement')
# plt.show()
# # %%
