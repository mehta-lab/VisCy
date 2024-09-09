# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import (
    FeatureExtractor as FE,
)
from viscy.representation.evaluation import dataset_of_tracks

# %%
features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/code_testing_soorya/output/June_140Patch_2chan/phaseRFP_140patch_99ckpt_Feb.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/2.1-register/registered.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr"
)

# %%

source_channel = ["Phase3D", "RFP"]
z_range = (28, 43)
normalizations = None
# fov_name = "/B/4/5"
# track_id = 11

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

fov_names_list = [
    name for name in embedding_dataset["fov_name"].values if name.startswith("/A/3/")
]
unique_fov_names = sorted(list(set(fov_names_list)))
correlation_sum = pd.DataFrame()
ii = 0
features = pd.DataFrame()
computed_pca = pd.DataFrame()


for fov_name in unique_fov_names:

    all_tracks_FOV = embedding_dataset.sel(fov_name=fov_name)

    unique_track_ids = list(all_tracks_FOV["track_id"].values)
    unique_track_ids = list(set(unique_track_ids))

    for track_id in unique_track_ids:
        a_track_in_FOV = all_tracks_FOV.sel(track_id=track_id)
        indices = np.arange(a_track_in_FOV.sizes["sample"])
        features_track = a_track_in_FOV["features"]
        time_stamp = features_track["t"][indices].astype(str)

        scaled_features_track = StandardScaler().fit_transform(features_track.values)

        # perform PCA analysis of features

        pca = PCA(n_components=5)
        if scaled_features_track.shape[0] > 5:
            pca_features = pca.fit_transform(scaled_features_track)
            ii += 1
        else:
            continue

        features_track = (
            features_track.assign_coords(PCA1=("sample", pca_features[:, 0]))
            .assign_coords(PCA2=("sample", pca_features[:, 1]))
            .assign_coords(PCA3=("sample", pca_features[:, 2]))
            .assign_coords(PCA4=("sample", pca_features[:, 3]))
            .assign_coords(PCA5=("sample", pca_features[:, 4]))
            .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5"], append=True)
        )

        # load the image patches

        prediction_dataset = dataset_of_tracks(
            data_path,
            tracks_path,
            [fov_name],
            [track_id],
            source_channel=source_channel,
        )

        whole = np.stack([p["anchor"] for p in predict_dataset])
        phase = whole[:, 0, 3]
        fluor = np.max(whole[:, 1], axis=1)
        # phase = np.stack([p["anchor"][0, 3].numpy() for p in predict_dataset])
        # fluor = np.stack([np.max(p["anchor"][1].numpy(), axis=0) for p in predict_dataset])

        # Compute Fourier descriptors for phase image
        data = {
            "Phase Symmetry Score": [],
            "Fluor Symmetry Score": [],
            "Sensor Area": [],
            "Masked Sensor Intensity": [],
            "Entropy Phase": [],
            "Entropy Fluor": [],
            "Contrast Phase": [],
            "Dissimilarity Phase": [],
            "Homogeneity Phase": [],
            "Contrast Fluor": [],
            "Dissimilarity Fluor": [],
            "Homogeneity Fluor": [],
            "Phase IQR": [],
            "Fluor Mean Intensity": [],
            "Phase Standard Deviation": [],
            "Fluor Standard Deviation": [],
        }

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

            # Append the computed features to the data dictionary
            data["Phase Symmetry Score"].append(phase_symmetry_score)
            data["Fluor Symmetry Score"].append(fluor_symmetry_score)
            data["Sensor Area"].append(area)
            data["Masked Sensor Intensity"].append(masked_intensity)
            data["Entropy Phase"].append(entropy_phase)
            data["Entropy Fluor"].append(entropy_fluor)
            data["Contrast Phase"].append(contrast_phase)
            data["Dissimilarity Phase"].append(dissimilarity_phase)
            data["Homogeneity Phase"].append(homogeneity_phase)
            data["Contrast Fluor"].append(contrast_fluor)
            data["Dissimilarity Fluor"].append(dissimilarity_fluor)
            data["Homogeneity Fluor"].append(homogeneity_fluor)
            # data["Edge Density Phase"].append(edge_density_phase)
            # data["Edge Density Fluor"].append(edge_density_fluor)
            data["Phase IQR"].append(iqr)
            data["Fluor Mean Intensity"].append(fluor_mean_intensity)
            data["Phase Standard Deviation"].append(phase_std_dev)
            data["Fluor Standard Deviation"].append(fluor_std_dev)

        # Create a dataframe to store the computed features
        features = pd.concat([features, pd.DataFrame(data)])

        # compute correlation between PCA features and computed features

        # Create a dataframe with PCA results
        pca_results = pd.DataFrame(
            pca_features, columns=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5"]
        )
        computed_pca = pd.concat([computed_pca, pca_results])

# %%

# Compute correlation between PCA features and computed features
correlation = pd.concat([computed_pca, features], axis=1).corr()
# correlation_sum = correlation_sum.add(correlation, fill_value=0)
# correlation_avg = correlation_sum / ii

# %% find the best correlated computed features with PCA features

# Find the best correlated computed features with PCA features
best_correlated_features = correlation.loc["PCA1":"PCA5", :].idxmax()
best_correlated_features

# %% display as a heatmap
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.show()

# %%
