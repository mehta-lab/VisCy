from viscy.data.triplet import TripletDataModule
from viscy.light.embedding_writer import read_embedding_dataset

from pathlib import Path
import numpy as np
from skimage import io
from applications.contrastive_phenotyping.contrastive_cli.computed_features import FeatureExtractor as FE
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
fov_name = "/B/4/5"
track_id = 11

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset
all_tracks_FOV = embedding_dataset.sel(fov_name=fov_name)
a_track_in_FOV = all_tracks_FOV.sel(track_id=track_id)
# Why is sample dimension ~22000 long after the dataset is sliced by FOV and by track_id?
indices = np.arange(a_track_in_FOV.sizes["sample"])
features_track = a_track_in_FOV["features"]
time_stamp = features_track["t"][indices].astype(str)

scaled_features_track = StandardScaler().fit_transform(features_track.values)

# %% perform PCA analysis of features

pca = PCA(n_components=5)
pca_features = pca.fit_transform(scaled_features_track)

features_track = (
    features_track.assign_coords(PCA1=("sample", pca_features[:, 0]))
    .assign_coords(PCA2=("sample", pca_features[:, 1]))
    .assign_coords(PCA3=("sample", pca_features[:, 2]))
    .assign_coords(PCA4=("sample", pca_features[:, 3]))
    .assign_coords(PCA5=("sample", pca_features[:, 4]))
    .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5"], append=True)
)

# %% load the image patches

data_module = TripletDataModule(
    data_path=data_path,
    tracks_path=tracks_path,
    source_channel=source_channel,
    z_range=z_range,
    initial_yx_patch_size=(256, 256),
    final_yx_patch_size=(256, 256),
    batch_size=1,
    num_workers=16,
    normalizations=normalizations,
    predict_cells=True,
    include_fov_names=[fov_name],
    include_track_ids=[track_id],
)
# for train and val
data_module.setup("predict")
predict_dataset = data_module.predict_dataset

phase = np.stack([p["anchor"][0, 7].numpy() for p in predict_dataset])
fluor = np.stack([np.max(p["anchor"][1].numpy(), axis=0) for p in predict_dataset])

# %% first feature: symmetry of cell

# Compute Fourier descriptors for phase image
phase_descriptors = FE.compute_fourier_descriptors(phase)
# Analyze symmetry of phase image
phase_symmetry_score = FE.analyze_symmetry(phase_descriptors)

# Compute Fourier descriptors for fluor image
fluor_descriptors = FE.compute_fourier_descriptors(fluor)
# Analyze symmetry of fluor image
fluor_symmetry_score = FE.analyze_symmetry(fluor_descriptors)

# %% second feature: area of sensor

thresholded_image, area = FE.otsu_threshold_and_compute_area(fluor)

# %% third feature: higher fequency features using spectral entropy

entropy_phase = FE.compute_spectral_entropy(phase)
entropy_fluor = FE.compute_spectral_entropy(fluor)

# %% fourth feature: texture analysis using GLCM

contrast_phase, dissimilarity_phase, homogeneity_phase = FE.compute_glcm_features(phase)
contrast_fluor, dissimilarity_fluor, homogeneity_fluor = FE.compute_glcm_features(fluor)

# %% fifth feature: edge detection using Canny
edges_phase = FE.detect_edges(phase)
edges_fluor = FE.detect_edges(fluor)

# quantify the amount of edge feature in the phase image
edge_density_phase = np.sum(edges_phase) / (edges_phase.shape[0] * edges_phase.shape[1])

# quantify the amount of edge feature in the fluor image
edge_density_fluor = np.sum(edges_fluor) / (edges_fluor.shape[0] * edges_fluor.shape[1])

# %% sixth feature: interqualtile range of pixel intensities

iqr = FE.compute_iqr(phase)

# %% seventh feature: mean pixel intensity

mean_intensity = FE.compute_mean_intensity(fluor)

# %% eighth feature: standard deviation of pixel intensities

std_dev = FE.compute_std_dev(phase)

# Create a dataframe to store the computed features
data = {
    "Phase Symmetry Score": phase_symmetry_score,
    "Fluor Symmetry Score": fluor_symmetry_score,
    "Area": area,
    "Entropy Phase": entropy_phase,
    "Entropy Fluor": entropy_fluor,
    "Contrast Phase": contrast_phase,
    "Dissimilarity Phase": dissimilarity_phase,
    "Homogeneity Phase": homogeneity_phase,
    "Contrast Fluor": contrast_fluor,
    "Dissimilarity Fluor": dissimilarity_fluor,
    "Homogeneity Fluor": homogeneity_fluor,
    "Edge Density Phase": edge_density_phase,
    "Edge Density Fluor": edge_density_fluor,
    "IQR": iqr,
    "Mean Intensity": mean_intensity,
    "Standard Deviation": std_dev,
}

df = pd.DataFrame(data)
