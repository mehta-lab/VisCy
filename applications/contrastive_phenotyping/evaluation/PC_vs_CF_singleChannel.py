""" Script to compute the correlation between PCA and UMAP features and computed features
* finds the computed features best representing the PCA and UMAP components
* outputs a heatmap of the correlation between PCA and UMAP features and computed features
"""

# %%
import sys
from pathlib import Path

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks
from viscy.representation.evaluation.feature import (
    FeatureExtractor as FE,
)

# %%
features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval_phase/predictions/epoch_186/1chan_128patch_186ckpt_Febtest.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/9-lineage-cell-division/lineages_gt/track.zarr"
)

# %%

source_channel = ["Phase3D"]
z_range = (28, 43)
normalizations = None
# fov_name = "/B/4/5"
# track_id = 11

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

# load all unprojected features:
features = embedding_dataset["features"]

# %% PCA analysis of the features

pca = PCA(n_components=3)
embedding = pca.fit_transform(features.values)
features = (
    features.assign_coords(PCA1=("sample", embedding[:, 0]))
    .assign_coords(PCA2=("sample", embedding[:, 1]))
    .assign_coords(PCA3=("sample", embedding[:, 2]))
    .set_index(sample=["PCA1", "PCA2", "PCA3"], append=True)
)

# %% convert the xarray to dataframe structure and add columns for computed features
features_df = features.to_dataframe()
features_df = features_df.drop(columns=["features"])
df = features_df.drop_duplicates()
features = df.reset_index(drop=True)

features = features[features["fov_name"].str.startswith("/B/")]

features["Phase Symmetry Score"] = np.nan
features["Entropy Phase"] = np.nan
features["Contrast Phase"] = np.nan
features["Dissimilarity Phase"] = np.nan
features["Homogeneity Phase"] = np.nan
features["Phase IQR"] = np.nan
features["Phase Standard Deviation"] = np.nan
features["Phase radial profile"] = np.nan

# %% compute the computed features and add them to the dataset

fov_names_list = features["fov_name"].unique()
unique_fov_names = sorted(list(set(fov_names_list)))

for fov_name in unique_fov_names:

    unique_track_ids = features[features["fov_name"] == fov_name]["track_id"].unique()
    unique_track_ids = list(set(unique_track_ids))

    for track_id in unique_track_ids:

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

        for t in range(phase.shape[0]):
            # Compute Fourier descriptors for phase image
            phase_descriptors = FE.compute_fourier_descriptors(phase[t])
            # Analyze symmetry of phase image
            phase_symmetry_score = FE.analyze_symmetry(phase_descriptors)

            # Compute higher frequency features using spectral entropy
            entropy_phase = FE.compute_spectral_entropy(phase[t])

            # Compute texture analysis using GLCM
            contrast_phase, dissimilarity_phase, homogeneity_phase = (
                FE.compute_glcm_features(phase[t])
            )

            # Compute interqualtile range of pixel intensities
            iqr = FE.compute_iqr(phase[t])

            # Compute standard deviation of pixel intensities
            phase_std_dev = FE.compute_std_dev(phase[t])

            # Compute radial intensity gradient
            phase_radial_profile = FE.compute_radial_intensity_gradient(phase[t])

            # update the features dataframe with the computed features

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
                "Entropy Phase",
            ] = entropy_phase
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
                "Phase IQR",
            ] = iqr
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
                "Phase radial profile",
            ] = phase_radial_profile

# %%
# Save the features dataframe to a CSV file
features.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_oneChan.csv",
    index=False,
)

# read the csv file
# features = pd.read_csv(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_oneChan.csv"
# )

# remove the rows with missing values
features = features.dropna()

# sub_features = features[features["Time"] == 20]
feature_df_removed = features.drop(
    columns=["fov_name", "track_id", "t", "id", "parent_track_id", "parent_id"]
)

# Compute correlation between PCA features and computed features
correlation = feature_df_removed.corr(method="spearman")

# %% calculate the p-value and draw volcano plot to show the significance of the correlation

p_values = pd.DataFrame(index=correlation.index, columns=correlation.columns)

for i in correlation.index:
    for j in correlation.columns:
        if i != j:
            p_values.loc[i, j] = spearmanr(
                feature_df_removed[i], feature_df_removed[j]
            )[1]

p_values = p_values.astype(float)

# %% draw an interactive volcano plot showing -log10(p-value) vs fold change

# Flatten the correlation and p-values matrices and create a DataFrame
correlation_flat = correlation.values.flatten()
p_values_flat = p_values.values.flatten()
# Create a list of feature names for the flattened correlation and p-values
feature_names = [f"{i}_{j}" for i in correlation.index for j in correlation.columns]

data = pd.DataFrame(
    {
        "Correlation": correlation_flat,
        "-log10(p-value)": -np.log10(p_values_flat),
        "feature_names": feature_names,
    }
)

# Create an interactive scatter plot using Plotly
fig = px.scatter(
    data,
    x="Correlation",
    y="-log10(p-value)",
    title="Volcano plot showing significance of correlation",
    labels={"Correlation": "Correlation", "-log10(p-value)": "-log10(p-value)"},
    opacity=0.5,
    hover_data=["feature_names"],
)

fig.show()
# Save the interactive volcano plot as an HTML file
fig.write_html(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/volcano_plot_1chan.html"
)

# %%
