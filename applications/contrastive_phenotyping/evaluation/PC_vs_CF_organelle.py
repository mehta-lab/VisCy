""" Script to compute the correlation between PCA and UMAP features and computed features
* finds the computed features best representing the PCA and UMAP components
* outputs a heatmap of the correlation between PCA and UMAP features and computed features
"""

# %%
# Standard library imports
import os
import sys
from pathlib import Path

# Third party imports
import cv2
import matplotlib.pyplot as plt
import mahotas.features
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Local imports
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks
from viscy.representation.evaluation.feature import (
    FeatureExtractor as FE,
)

# %%
embeddings_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)

# %%

source_channel = ["Phase3D", "raw GFP EX488 EM525-45"]
seg_channel = ["nuclei_prediction_labels_labels"]
z_range = (30, 35)
normalizations = None
# fov_name = "/0/6/000000"
# track_id = 21

embedding_dataset = read_embedding_dataset(embeddings_path)
embedding_dataset

# load all unprojected features:
learned_features = embedding_dataset["features"]

# %% PCA analysis of the features

# Normalize features before PCA
features_normalized = (learned_features.values - learned_features.values.mean(axis=0)) / learned_features.values.std(axis=0)
pca = PCA(n_components=8)
pca_features = pca.fit_transform(features_normalized)
learned_features = (
    learned_features.assign_coords(PCA1=("sample", pca_features[:, 0]))
    .assign_coords(PCA2=("sample", pca_features[:, 1]))
    .assign_coords(PCA3=("sample", pca_features[:, 2]))
    .assign_coords(PCA4=("sample", pca_features[:, 3]))
    .assign_coords(PCA5=("sample", pca_features[:, 4]))
    .assign_coords(PCA6=("sample", pca_features[:, 5]))
    .assign_coords(PCA7=("sample", pca_features[:, 6]))
    .assign_coords(PCA8=("sample", pca_features[:, 7]))
    .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8"], append=True)
)


# %% convert the xarray to dataframe structure and add columns for computed features

# Convert the learned features to a DataFrame
features_df = learned_features.to_dataframe()

# Drop the 'features' column
features_df = features_df.drop(columns=["features"])

# Drop duplicate rows
features_df = features_df.drop_duplicates()

# Reset the index
features_df = features_df.reset_index(drop=True)

# Filter for specific FOV names
features_df = features_df[features_df["fov_name"].str.startswith("/C/2/000000")]

features_df["Phase Symmetry Score"] = np.nan
features_df["Fluor Symmetry Score"] = np.nan
features_df["Fluor Area"] = np.nan
features_df["Masked fluor Intensity"] = np.nan
features_df["Entropy Phase"] = np.nan
features_df["Contrast Phase"] = np.nan
features_df["Dissimilarity Phase"] = np.nan
features_df["Homogeneity Phase"] = np.nan
features_df["Contrast Fluor"] = np.nan
features_df["Dissimilarity Fluor"] = np.nan
features_df["Homogeneity Fluor"] = np.nan
features_df["Phase IQR"] = np.nan
features_df["Fluor Mean Intensity"] = np.nan
features_df["Phase Standard Deviation"] = np.nan
features_df["Fluor Standard Deviation"] = np.nan
features_df["Fluor weighted intensity gradient"] = np.nan
features_df["Fluor texture"] = np.nan
features_df["Phase texture"] = np.nan
features_df["Perimeter area ratio"] = np.nan
features_df["Nucleus eccentricity"] = np.nan
features_df["Instantaneous velocity"] = np.nan
features_df["Fluor localization"] = np.nan


# Display the first few rows of the filtered DataFrame
print(features_df.head())

# %% iterate over new features and compute them

# weighted intensity gradient

# %% compute the computed features and add them to the dataset

fov_names_list = learned_features["fov_name"].unique()
unique_fov_names = sorted(list(set(fov_names_list)))

iteration_count = 0
max_iterations = 100

for fov_name in unique_fov_names:
    csv_files = list((Path(str(tracks_path) + str(fov_name))).glob("*.csv"))
    tracks_df = pd.read_csv(str(csv_files[0]))

    unique_track_ids = learned_features[learned_features["fov_name"] == fov_name]["track_id"].unique()
    unique_track_ids = list(set(unique_track_ids))

    for track_id in unique_track_ids:
        if iteration_count >= max_iterations:
            break
        track_subdf = tracks_df[tracks_df["track_id"] == track_id]
            
        prediction_dataset = dataset_of_tracks(
            data_path,
            tracks_path,
            [fov_name],
            [track_id],
            z_range=z_range,
            source_channel=source_channel,
        )
        track_channel = dataset_of_tracks(
            tracks_path,
            tracks_path,
            [fov_name],
            [track_id],
            z_range=(0,1),
            source_channel=seg_channel,
        )

        whole = np.stack([p["anchor"] for p in prediction_dataset])
        seg_mask = np.stack([p["anchor"] for p in track_channel])
        phase = whole[:, 0, 2]
        fluor = np.max(whole[:, 1], axis=1)
        nucl_mask = seg_mask[:, 0, 0]

        for t in range(phase.shape[0]):

            # Compute Fourier descriptors for phase image
            phase_descriptors = FE.compute_fourier_descriptors(phase[t])
            # Analyze symmetry of phase image
            phase_symmetry_score = FE.analyze_symmetry(phase_descriptors)

            # Compute Fourier descriptors for fluor image
            fluor_descriptors = FE.compute_fourier_descriptors(fluor[t])
            # Analyze symmetry of fluor image
            fluor_symmetry_score = FE.analyze_symmetry(fluor_descriptors)

            # Compute area of fluor
            masked_intensity, area = FE.compute_area(fluor[t])

            # Compute higher frequency features using spectral entropy
            entropy_phase = FE.compute_spectral_entropy(phase[t])

            # Compute texture analysis using GLCM
            contrast_phase, dissimilarity_phase, homogeneity_phase = (
                FE.compute_glcm_features(phase[t])
            )
            contrast_fluor, dissimilarity_fluor, homogeneity_fluor = (
                FE.compute_glcm_features(fluor[t])
            )

            # Compute interqualtile range of pixel intensities
            iqr = FE.compute_iqr(phase[t])

            # Compute mean pixel intensity
            fluor_mean_intensity = FE.compute_mean_intensity(fluor[t])

            # Compute standard deviation of pixel intensities
            phase_std_dev = FE.compute_std_dev(phase[t])
            fluor_std_dev = FE.compute_std_dev(fluor[t])

            # Compute gradient for localization in organelle channel
            fluor_weighted_gradient = compute_weighted_intensity_gradient(fluor[t])

            # compute the texture features using haralick
            phase_texture = texture_features(phase[t])
            fluor_texture = texture_features(fluor[t])

            # compute the perimeter of the nuclear segmentations found inside the patch
            perimeter_area_ratio = compute_perimeter_area_ratio(nucl_mask[t])

            # compute the eccentricity of the nucleus
            seg_eccentricity = nucleus_eccentricity(nucl_mask[t])

            # compute instantaneous velocity of cell 
            inst_velocity = compute_instantaneous_velocity(track_subdf, t)

            # compute the localization of the fluor
            fluor_location = fluor_localization(fluor[t], nucl_mask[t])

            # Create dictionary mapping feature names to their computed values
            feature_values = {
                "Fluor Symmetry Score": fluor_symmetry_score,
                "Phase Symmetry Score": phase_symmetry_score,
                "Fluor Area": area,
                "Masked fluor Intensity": masked_intensity,
                "Entropy Phase": entropy_phase,
                "Contrast Phase": contrast_phase,
                "Dissimilarity Phase": dissimilarity_phase,
                "Homogeneity Phase": homogeneity_phase,
                "Contrast Fluor": contrast_fluor,
                "Dissimilarity Fluor": dissimilarity_fluor,
                "Homogeneity Fluor": homogeneity_fluor,
                "Phase IQR": iqr,
                "Fluor Mean Intensity": fluor_mean_intensity,
                "Phase Standard Deviation": phase_std_dev,
                "Fluor Standard Deviation": fluor_std_dev,
                "Fluor weighted intensity gradient": fluor_weighted_gradient,
                "Phase texture": phase_texture,
                "Fluor texture": fluor_texture,
                "Perimeter area ratio": perimeter_area_ratio,
                "Nucleus eccentricity": seg_eccentricity,
                "Instantaneous velocity": inst_velocity,
                "Fluor localization": fluor_location,
            }

            # Update features dataframe using the dictionary
            for feature_name, value in feature_values.items():
                learned_features.loc[
                    (learned_features["fov_name"] == fov_name)
                    & (learned_features["track_id"] == track_id)
                    & (learned_features["t"] == t),
                    feature_name
                ] = value

        iteration_count += 1
        print(f"Processed {iteration_count}")

# %%

# Save the features dataframe to a CSV file
learned_features.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan_organelle.csv",
    index=False,
)

# # read the features dataframe from the CSV file
# features = pd.read_csv(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan.csv"
# )
# remove the column "Perimeter"
# features = features.drop(columns=["Perimeter"])

# remove the rows with missing values
learned_features = learned_features.dropna()

# sub_features = features[features["Time"] == 20]
feature_df_removed = learned_features.drop(
    columns=["fov_name", "track_id", "t", "id", "parent_track_id", "parent_id", "PHATE1", "PHATE2", "UMAP1", "UMAP2"]
)

# Compute correlation between PCA features and computed features
correlation = feature_df_removed.corr(method="spearman")

# %% display PCA correlation as a heatmap

plt.figure(figsize=(20, 8))
sns.heatmap(
    correlation.drop(columns=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8"]).loc[
        "PCA1":"PCA8", :
    ],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={'size': 12}  # Increase annotation text size
)
plt.title("Correlation between PCA features and computed features", fontsize=12)
plt.xlabel("Computed Features", fontsize=12)
plt.ylabel("PCA Features", fontsize=12)
plt.xticks(fontsize=12)  # Increase x-axis tick labels
plt.yticks(fontsize=12)  # Increase y-axis tick labels
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_organelle.svg"
)


# %% plot PCA vs set of computed features

# set_features = [
#     "Phase Standard Deviation",
#     "Entropy Fluor",
#     "Fluor Standard Deviation",
#     "Dissimilarity Fluor",
#     "Fluor Area",
#     "Fluor Symmetry Score",
#     "Entropy Fluor",
#     "Masked fluor Intensity",
# ]

# plt.figure(figsize=(8, 10))
# sns.heatmap(
#     correlation.loc[set_features, "PCA1":"PCA8"],
#     annot=True,
#     cmap="coolwarm",
#     fmt=".2f",
#     vmin=-1,
#     vmax=1,
# )

# plt.savefig(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_setfeatures.svg"
# )
