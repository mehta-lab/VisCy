# %%
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import load_annotation

# %% Paths and parameters.


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
embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset


# %%
# Compute UMAP over all features
features = embedding_dataset["features"]
# or select a well:
# features = features[features["fov_name"].str.contains("B/4")]


scaled_features = StandardScaler().fit_transform(features.values)
umap = UMAP()
# Fit UMAP on all features
embedding = umap.fit_transform(scaled_features)


# %%
# Add UMAP coordinates to the dataset and plot w/ time


features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
features


sns.scatterplot(
    x=features["UMAP1"], y=features["UMAP2"], hue=features["t"], s=7, alpha=0.8
)


# Add the title to the plot
plt.title("Cell & Time Aware Sampling (30 min interval)")
plt.xlim(-10, 20)
plt.ylim(-10, 20)
# plt.savefig('umap_cell_time_aware_time.svg', format='svg')
plt.savefig("updated_cell_time_aware_time.png", format="png")
# Show the plot
plt.show()


# %%


any_features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_2chan_128patch_32projDim/2chan_128patch_56ckpt_FebTest.zarr"
)
embedding_dataset = read_embedding_dataset(any_features_path)
embedding_dataset


# %%
# Compute UMAP over all features
features = embedding_dataset["features"]
# or select a well:
# features = features[features["fov_name"].str.contains("B/4")]


scaled_features = StandardScaler().fit_transform(features.values)
umap = UMAP()
# Fit UMAP on all features
embedding = umap.fit_transform(scaled_features)


# %% Any time sampling plot


features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
features


sns.scatterplot(
    x=features["UMAP1"], y=features["UMAP2"], hue=features["t"], s=7, alpha=0.8
)


# Add the title to the plot
plt.title("Cell Aware Sampling")

plt.xlim(-10, 20)
plt.ylim(-10, 20)

plt.savefig("1_updated_cell_aware_time.png", format="png")
# plt.savefig('umap_cell_aware_time.pdf', format='pdf')
# Show the plot
plt.show()


# %%


contrastive_learning_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr"
)
embedding_dataset = read_embedding_dataset(contrastive_learning_path)
embedding_dataset


# %%
# Compute UMAP over all features
features = embedding_dataset["features"]
# or select a well:
# features = features[features["fov_name"].str.contains("B/4")]


scaled_features = StandardScaler().fit_transform(features.values)
umap = UMAP()
# Fit UMAP on all features
embedding = umap.fit_transform(scaled_features)


# %% Any time sampling plot


features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
features

sns.scatterplot(
    x=features["UMAP1"], y=features["UMAP2"], hue=features["t"], s=7, alpha=0.8
)

# Add the title to the plot
plt.title("Classical Contrastive Learning Sampling")
plt.xlim(-10, 20)
plt.ylim(-10, 20)
plt.savefig("updated_classical_time.png", format="png")
# plt.savefig('classical_time.pdf', format='pdf')

# Show the plot
plt.show()


# %% PCA


pca = PCA(n_components=4)
# scaled_features = StandardScaler().fit_transform(features.values)
# pca_features = pca.fit_transform(scaled_features)
pca_features = pca.fit_transform(features.values)


features = (
    features.assign_coords(PCA1=("sample", pca_features[:, 0]))
    .assign_coords(PCA2=("sample", pca_features[:, 1]))
    .assign_coords(PCA3=("sample", pca_features[:, 2]))
    .assign_coords(PCA4=("sample", pca_features[:, 3]))
    .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4"], append=True)
)


# %% plot PCA components w/ time


plt.figure(figsize=(10, 10))
sns.scatterplot(
    x=features["PCA1"], y=features["PCA2"], hue=features["t"], s=7, alpha=0.8
)


# %% OVERLAY INFECTION ANNOTATION
ann_root = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred"
)


infection = load_annotation(
    features,
    ann_root / "extracted_inf_state.csv",
    "infection_state",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)


# %%
sns.scatterplot(x=features["UMAP1"], y=features["UMAP2"], hue=infection, s=7, alpha=0.8)


# %% plot PCA components with infection hue
sns.scatterplot(x=features["PCA1"], y=features["PCA2"], hue=infection, s=7, alpha=0.8)


# %%
