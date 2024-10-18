""" Script to compute the correlation between PCA and UMAP features and computed features
* finds the computed features best representing the PCA and UMAP components
* outputs a heatmap of the correlation between PCA and UMAP features and computed features
"""

# %%
import os
import sys
from pathlib import Path

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks
from viscy.representation.evaluation.feature import (
    FeatureExtractor as FE,
)

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

# load all unprojected features:
features = embedding_dataset["features"]

# %% PCA analysis of the features

pca = PCA(n_components=5)
pca_features = pca.fit_transform(features.values)
features = (
    features.assign_coords(PCA1=("sample", pca_features[:, 0]))
    .assign_coords(PCA2=("sample", pca_features[:, 1]))
    .assign_coords(PCA3=("sample", pca_features[:, 2]))
    .assign_coords(PCA4=("sample", pca_features[:, 3]))
    .assign_coords(PCA5=("sample", pca_features[:, 4]))
    .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5"], append=True)
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

fov_names_list = features["fov_name"].unique()
unique_fov_names = sorted(list(set(fov_names_list)))


for fov_name in unique_fov_names:

    unique_track_ids = features[features["fov_name"] == fov_name]["track_id"].unique()
    unique_track_ids = list(set(unique_track_ids))

    for track_id in unique_track_ids:

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

# %%

# Save the features dataframe to a CSV file
features.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan.csv",
    index=False,
)

# # read the features dataframe from the CSV file
# features = pd.read_csv(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan.csv"
# )

# remove the rows with missing values
features = features.dropna()

# sub_features = features[features["Time"] == 20]
feature_df_removed = features.drop(
    columns=["fov_name", "track_id", "t", "id", "parent_track_id", "parent_id"]
)

# Compute correlation between PCA features and computed features
correlation = feature_df_removed.corr(method="spearman")

# %% display PCA correlation as a heatmap

plt.figure(figsize=(20, 5))
sns.heatmap(
    correlation.drop(columns=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5"]).loc[
        "PCA1":"PCA5", :
    ],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Correlation between PCA features and computed features")
plt.xlabel("Computed Features")
plt.ylabel("PCA Features")
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca.svg"
)


# %% plot PCA vs set of computed features

set_features = [
    "Fluor radial profile",
    "Homogeneity Phase",
    "Phase IQR",
    "Phase Standard Deviation",
    "Sensor Area",
    "Homogeneity Fluor",
    "Contrast Fluor",
    "Phase radial profile",
]

plt.figure(figsize=(8, 10))
sns.heatmap(
    correlation.loc[set_features, "PCA1":"PCA5"],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
)

plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_setfeatures.svg"
)

# %% find the cell patches with the highest and lowest value in each feature

def save_patches(fov_name, track_id):
    data_path = Path(
        "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
    )
    tracks_path = Path(
        "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
    )
    source_channel = ["Phase3D", "RFP"]
    prediction_dataset = dataset_of_tracks(
        data_path,
        tracks_path,
        [fov_name],
        [track_id],
        source_channel=source_channel,
    )
    whole = np.stack([p["anchor"] for p in prediction_dataset])
    phase = whole[:, 0]
    fluor = whole[:, 1]
    out_dir = "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/data/computed_features/"
    fov_name_out = fov_name.replace("/", "_")
    np.save(
        (os.path.join(out_dir, "phase" + fov_name_out + "_" + str(track_id) + ".npy")),
        phase,
    )
    np.save(
        (os.path.join(out_dir, "fluor" + fov_name_out + "_" + str(track_id) + ".npy")),
        fluor,
    )


# PCA1: Fluor radial profile
highest_fluor_radial_profile = features.loc[features["Fluor radial profile"].idxmax()]
print("Row with highest 'Fluor radial profile':")
# print(highest_fluor_radial_profile)
print(
    f"fov_name: {highest_fluor_radial_profile['fov_name']}, time: {highest_fluor_radial_profile['t']}"
)
save_patches(
    highest_fluor_radial_profile["fov_name"], highest_fluor_radial_profile["track_id"]
)

lowest_fluor_radial_profile = features.loc[features["Fluor radial profile"].idxmin()]
print("Row with lowest 'Fluor radial profile':")
# print(lowest_fluor_radial_profile)
print(
    f"fov_name: {lowest_fluor_radial_profile['fov_name']}, time: {lowest_fluor_radial_profile['t']}"
)
save_patches(
    lowest_fluor_radial_profile["fov_name"], lowest_fluor_radial_profile["track_id"]
)

# PCA2: Entropy phase
highest_entropy_phase = features.loc[features["Entropy Phase"].idxmax()]
print("Row with highest 'Entropy Phase':")
# print(highest_entropy_phase)
print(
    f"fov_name: {highest_entropy_phase['fov_name']}, time: {highest_entropy_phase['t']}"
)
save_patches(highest_entropy_phase["fov_name"], highest_entropy_phase["track_id"])

lowest_entropy_phase = features.loc[features["Entropy Phase"].idxmin()]
print("Row with lowest 'Entropy Phase':")
# print(lowest_entropy_phase)
print(
    f"fov_name: {lowest_entropy_phase['fov_name']}, time: {lowest_entropy_phase['t']}"
)
save_patches(lowest_entropy_phase["fov_name"], lowest_entropy_phase["track_id"])

# PCA3: Phase IQR
highest_phase_iqr = features.loc[features["Phase IQR"].idxmax()]
print("Row with highest 'Phase IQR':")
# print(highest_phase_iqr)
print(f"fov_name: {highest_phase_iqr['fov_name']}, time: {highest_phase_iqr['t']}")
save_patches(highest_phase_iqr["fov_name"], highest_phase_iqr["track_id"])

tenth_lowest_phase_iqr = features.nsmallest(10, "Phase IQR").iloc[9]
print("Row with tenth lowest 'Phase IQR':")
# print(tenth_lowest_phase_iqr)
print(
    f"fov_name: {tenth_lowest_phase_iqr['fov_name']}, time: {tenth_lowest_phase_iqr['t']}"
)
save_patches(tenth_lowest_phase_iqr["fov_name"], tenth_lowest_phase_iqr["track_id"])

# PCA4: Phase Standard Deviation
highest_phase_std_dev = features.loc[features["Phase Standard Deviation"].idxmax()]
print("Row with highest 'Phase Standard Deviation':")
# print(highest_phase_std_dev)
print(
    f"fov_name: {highest_phase_std_dev['fov_name']}, time: {highest_phase_std_dev['t']}"
)
save_patches(highest_phase_std_dev["fov_name"], highest_phase_std_dev["track_id"])

lowest_phase_std_dev = features.loc[features["Phase Standard Deviation"].idxmin()]
print("Row with lowest 'Phase Standard Deviation':")
# print(lowest_phase_std_dev)
print(
    f"fov_name: {lowest_phase_std_dev['fov_name']}, time: {lowest_phase_std_dev['t']}"
)
save_patches(lowest_phase_std_dev["fov_name"], lowest_phase_std_dev["track_id"])

# PCA5: Sensor area
highest_sensor_area = features.loc[features["Sensor Area"].idxmax()]
print("Row with highest 'Sensor Area':")
# print(highest_sensor_area)
print(f"fov_name: {highest_sensor_area['fov_name']}, time: {highest_sensor_area['t']}")
save_patches(highest_sensor_area["fov_name"], highest_sensor_area["track_id"])

tenth_lowest_sensor_area = features.nsmallest(10, "Sensor Area").iloc[9]
print("Row with tenth lowest 'Sensor Area':")
# print(tenth_lowest_sensor_area)
print(
    f"fov_name: {tenth_lowest_sensor_area['fov_name']}, time: {tenth_lowest_sensor_area['t']}"
)
save_patches(tenth_lowest_sensor_area["fov_name"], tenth_lowest_sensor_area["track_id"])
