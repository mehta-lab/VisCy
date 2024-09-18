""" Script to compute the correlation between PCA and UMAP features and computed features
* finds the computed features best representing the PCA and UMAP components
* outputs a heatmap of the correlation between PCA and UMAP features and computed features
"""

# %%
from pathlib import Path
import sys

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import (
    FeatureExtractor as FE,
)
from viscy.representation.evaluation import dataset_of_tracks

import matplotlib.pyplot as plt
import seaborn as sns

# %%
features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
)

# %%

source_channel = ["Phase3D", "RFP"]
z_range = (28, 43)
normalizations = None
# fov_name = "/B/4/5"
# track_id = 11

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

# %% umap analsis of the features

# load all unprojected features:
features = embedding_dataset["features"]
scaled_features = StandardScaler().fit_transform(features.values)

umap = UMAP()
embedding = umap.fit_transform(features.values)
features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)

# %% convert the xarray to dataframe structure and add columns for computed features
features_df = features.to_dataframe()
features_df = features_df.drop(columns=["features"])
df = features_df.drop_duplicates()
features = df.reset_index(drop=True)

features = features[features["fov_name"].str.startswith("/B/")]

features["Phase Symmetry Score"] = np.nan
features["Fluor Symmetry Score"] = np.nan
features["Sensor Area"] = np.nan
features["Masked Sensor Intensity"] = np.nan
features["Entropy Phase"] = np.nan
features["Entropy Fluor"] = np.nan
features["Contrast Phase"] = np.nan
features["Dissimilarity Phase"] = np.nan
features["Homogeneity Phase"] = np.nan
features["Contrast Fluor"] = np.nan
features["Dissimilarity Fluor"] = np.nan
features["Homogeneity Fluor"] = np.nan
features["Phase IQR"] = np.nan
features["Fluor Mean Intensity"] = np.nan
features["Phase Standard Deviation"] = np.nan
features["Fluor Standard Deviation"] = np.nan
features["Phase radial profile"] = np.nan
features["Fluor radial profile"] = np.nan

# %% compute the computed features and add them to the dataset

# fov_names_list = [
#     name for name in embedding_dataset["fov_name"].values if name.startswith("/B/4")
# ]
fov_names_list = features["fov_name"].unique()
unique_fov_names = sorted(list(set(fov_names_list)))

# features = pd.DataFrame()

for fov_name in unique_fov_names:

    # all_tracks_FOV = embedding_dataset.sel(fov_name=fov_name)

    # unique_track_ids = list(all_tracks_FOV["track_id"].values)
    unique_track_ids = features[features["fov_name"] == fov_name]["track_id"].unique()
    unique_track_ids = list(set(unique_track_ids))

    for track_id in unique_track_ids:
        # a_track_in_FOV = all_tracks_FOV.sel(track_id=track_id)
        # indices = np.arange(a_track_in_FOV.sizes["sample"])
        # features_track = a_track_in_FOV["features"]
        # time_stamp = features_track["t"][indices].astype(str)

        # feature_track_values = features_track.values

        # perform PCA analysis of features

        # pca = PCA(n_components=4)
        # if feature_track_values.shape[0] > 5:
        #     pca_features = pca.fit_transform(feature_track_values)
        # else:
        #     continue

        # features_track = (
        #     features_track.assign_coords(PCA1=("sample", pca_features[:, 0]))
        #     .assign_coords(PCA2=("sample", pca_features[:, 1]))
        #     .assign_coords(PCA3=("sample", pca_features[:, 2]))
        #     .assign_coords(PCA4=("sample", pca_features[:, 3]))
        #     .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4"], append=True)
        # )

        # umap = UMAP()
        # if feature_track_values.shape[0] > 5:
        #     umap_features = umap.fit_transform(features_track.values)
        # else:
        #     continue

        # features_track = (
        #     features_track.assign_coords(UMAP1=("sample", umap_features[:, 0]))
        #     .assign_coords(UMAP2=("sample", umap_features[:, 1]))
        #     .set_index(sample=["UMAP1", "UMAP2"], append=True)
        # )

        # load the image patches

        prediction_dataset = dataset_of_tracks(
            data_path,
            tracks_path,
            [fov_name],
            [track_id],
            source_channel=source_channel,
        )

        whole = np.stack([p["anchor"] for p in prediction_dataset])
        phase = whole[:, 0, 3]
        fluor = np.max(whole[:, 1], axis=1)
        # phase = np.stack([p["anchor"][0, 3].numpy() for p in predict_dataset])
        # fluor = np.stack([np.max(p["anchor"][1].numpy(), axis=0) for p in predict_dataset])

        # Compute Fourier descriptors for phase image
        # data = {
        #     "Phase Symmetry Score": [],
        #     "Fluor Symmetry Score": [],
        #     "Sensor Area": [],
        #     "Masked Sensor Intensity": [],
        #     "Entropy Phase": [],
        #     "Entropy Fluor": [],
        #     "Contrast Phase": [],
        #     "Dissimilarity Phase": [],
        #     "Homogeneity Phase": [],
        #     "Contrast Fluor": [],
        #     "Dissimilarity Fluor": [],
        #     "Homogeneity Fluor": [],
        #     "Phase IQR": [],
        #     "Fluor Mean Intensity": [],
        #     "Phase Standard Deviation": [],
        #     "Fluor Standard Deviation": [],
        #     "Phase radial profile": [],
        #     "Fluor radial profile": [],
        #     "Time": [],
        # }

        for t in range(phase.shape[0]):
            # Compute Fourier descriptors for phase image
            phase_descriptors = FE.compute_fourier_descriptors(phase[t])
            # Analyze symmetry of phase image
            phase_symmetry_score = FE.analyze_symmetry(phase_descriptors)

            # Compute Fourier descriptors for fluor image
            fluor_descriptors = FE.compute_fourier_descriptors(fluor[t])
            # Analyze symmetry of fluor image
            fluor_symmetry_score = FE.analyze_symmetry(fluor_descriptors)

            # Compute area of sensor
            masked_intensity, area = FE.compute_area(fluor[t])

            # Compute higher frequency features using spectral entropy
            entropy_phase = FE.compute_spectral_entropy(phase[t])
            entropy_fluor = FE.compute_spectral_entropy(fluor[t])

            # Compute texture analysis using GLCM
            contrast_phase, dissimilarity_phase, homogeneity_phase = (
                FE.compute_glcm_features(phase[t])
            )
            contrast_fluor, dissimilarity_fluor, homogeneity_fluor = (
                FE.compute_glcm_features(fluor[t])
            )

            # # Compute edge detection using Canny
            # edges_phase = FE.detect_edges(phase[t])
            # edges_fluor = FE.detect_edges(fluor[t])

            # Quantify the amount of edge feature in the phase image
            # edge_density_phase = np.sum(edges_phase) / (edges_phase.shape[0] * edges_phase.shape[1])

            # Quantify the amount of edge feature in the fluor image
            # edge_density_fluor = np.sum(edges_fluor) / (edges_fluor.shape[0] * edges_fluor.shape[1])

            # Compute interqualtile range of pixel intensities
            iqr = FE.compute_iqr(phase[t])

            # Compute mean pixel intensity
            fluor_mean_intensity = FE.compute_mean_intensity(fluor[t])

            # Compute standard deviation of pixel intensities
            phase_std_dev = FE.compute_std_dev(phase[t])
            fluor_std_dev = FE.compute_std_dev(fluor[t])

            # Compute radial intensity gradient
            phase_radial_profile = FE.compute_radial_intensity_gradient(phase[t])
            fluor_radial_profile = FE.compute_radial_intensity_gradient(fluor[t])

            # Append the computed features to the data dictionary
            # data["Phase Symmetry Score"].append(phase_symmetry_score)
            # data["Fluor Symmetry Score"].append(fluor_symmetry_score)
            # data["Sensor Area"].append(area)
            # data["Masked Sensor Intensity"].append(masked_intensity)
            # data["Entropy Phase"].append(entropy_phase)
            # data["Entropy Fluor"].append(entropy_fluor)
            # data["Contrast Phase"].append(contrast_phase)
            # data["Dissimilarity Phase"].append(dissimilarity_phase)
            # data["Homogeneity Phase"].append(homogeneity_phase)
            # data["Contrast Fluor"].append(contrast_fluor)
            # data["Dissimilarity Fluor"].append(dissimilarity_fluor)
            # data["Homogeneity Fluor"].append(homogeneity_fluor)
            # # data["Edge Density Phase"].append(edge_density_phase)
            # # data["Edge Density Fluor"].append(edge_density_fluor)
            # data["Phase IQR"].append(iqr)
            # data["Fluor Mean Intensity"].append(fluor_mean_intensity)
            # data["Phase Standard Deviation"].append(phase_std_dev)
            # data["Fluor Standard Deviation"].append(fluor_std_dev)
            # data["Phase radial profile"].append(phase_radial_profile)
            # data["Fluor radial profile"].append(fluor_radial_profile)
            # data["Time"].append(t)

            # update the features dataframe with the computed features

            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Fluor Symmetry Score",
            ] = fluor_symmetry_score
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Phase Symmetry Score",
            ] = phase_symmetry_score
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Sensor Area",
            ] = area
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Masked Sensor Intensity",
            ] = masked_intensity
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Entropy Phase",
            ] = entropy_phase
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Entropy Fluor",
            ] = entropy_fluor
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Contrast Phase",
            ] = contrast_phase
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Dissimilarity Phase",
            ] = dissimilarity_phase
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Homogeneity Phase",
            ] = homogeneity_phase
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Contrast Fluor",
            ] = contrast_fluor
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Dissimilarity Fluor",
            ] = dissimilarity_fluor
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Homogeneity Fluor",
            ] = homogeneity_fluor
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Phase IQR",
            ] = iqr
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Fluor Mean Intensity",
            ] = fluor_mean_intensity
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Phase Standard Deviation",
            ] = phase_std_dev
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Fluor Standard Deviation",
            ] = fluor_std_dev
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Phase radial profile",
            ] = phase_radial_profile
            features.loc[
                (features["fov_name"] == fov_name)
                & (features["track_id"] == track_id)
                & (features["t"] == t),
                "Fluor radial profile",
            ] = fluor_radial_profile

        # Create a dataframe to store the computed features
        # data_df = pd.DataFrame(data)

        # compute correlation between PCA features and computed features

        # Create a dataframe with PCA results
        # pca_results = pd.DataFrame(
        #     pca_features, columns=["PCA1", "PCA2", "PCA3", "PCA4"]
        # )
        # umap_results = pd.DataFrame(umap_features, columns=["UMAP1", "UMAP2"])
        # combined_df = pd.concat([data_df, pca_results, umap_results], axis=1)
        # combined_df["fov_name"] = fov_name
        # combined_df["track_id"] = track_id

        # features = pd.concat([features, combined_df], ignore_index=True)

# %%

# Save the features dataframe to a CSV file
features.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan.csv",
    index=False,
)

# sub_features = features[features["Time"] == 20]
feature_df_removed = features.drop(
    columns=["fov_name", "track_id", "t", "id", "parent_track_id", "parent_id"]
)

# Compute correlation between PCA features and computed features
correlation = feature_df_removed.corr(method="spearman")

# %% find the best correlated computed features with PCA features

# Find the best correlated computed features with PCA features
# best_correlated_features = correlation.loc["PCA1":"PCA4", :].idxmax()
# best_correlated_features

# %% display PCA correlation as a heatmap

# plt.figure(figsize=(20, 5))
# sns.heatmap(
#     correlation.drop(columns=["PCA1", "PCA2", "PCA3", "PCA4"]).loc["PCA1":"PCA4", :],
#     annot=True,
#     cmap="coolwarm",
#     fmt=".2f",
# )
# plt.title("Correlation between PCA features and computed features")
# plt.xlabel("Computed Features")
# plt.ylabel("PCA Features")
# plt.show()

# %% display UMAP correlation as a heatmap

plt.figure(figsize=(20, 5))
sns.heatmap(
    correlation.drop(columns=["UMAP1", "UMAP2"]).loc["UMAP1":"UMAP2", :],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Correlation between UMAP features and computed features")
plt.xlabel("Computed Features")
plt.ylabel("UMAP Features")
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan.svg"
)

# %%
