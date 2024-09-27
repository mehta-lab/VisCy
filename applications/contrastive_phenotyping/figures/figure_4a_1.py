# %% Importing Necessary Libraries
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from viscy.representation.embedding_writer import read_embedding_dataset

# %% Defining Paths for February and June Datasets
feb_features_path = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/code_testing_soorya/output/June_140Patch_2chan/phaseRFP_140patch_99ckpt_Feb.zarr")
feb_data_path = Path("/hpc/projects/virtual_staining/2024_02_04_A549_DENV_ZIKV_timelapse/registered_chunked.zarr")
feb_tracks_path = Path("/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr")

# %% Function to Load and Process the Embedding Dataset
def compute_umap(embedding_dataset):
    features = embedding_dataset["features"]
    scaled_features = StandardScaler().fit_transform(features.values)
    umap = UMAP()
    embedding = umap.fit_transform(scaled_features)
    
    features = (
        features.assign_coords(UMAP1=("sample", embedding[:, 0]))
        .assign_coords(UMAP2=("sample", embedding[:, 1]))
        .set_index(sample=["UMAP1", "UMAP2"], append=True)
    )
    return features

# %% Function to Load Annotations
def load_annotation(da, path, name, categories: dict | None = None):
    annotation = pd.read_csv(path)
    annotation["fov_name"] = "/" + annotation["fov ID"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.loc[mi][name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected

# %% Function to Plot UMAP with Infection Annotations
def plot_umap_infection(features, infection, title):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features["UMAP1"], y=features["UMAP2"], hue=infection, s=7, alpha=0.8)
    plt.title(f"UMAP Plot - {title}")
    plt.show()

# %% Load and Process February Dataset
feb_embedding_dataset = read_embedding_dataset(feb_features_path)
feb_features = compute_umap(feb_embedding_dataset)

feb_ann_root = Path("/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track")
feb_infection = load_annotation(feb_features, feb_ann_root / "tracking_v1_infection.csv", "infection class", {0.0: "background", 1.0: "uninfected", 2.0: "infected"})

# %% Plot UMAP with Infection Status for February Dataset
plot_umap_infection(feb_features, feb_infection, "February Dataset")

# %%
print(feb_embedding_dataset)
print(feb_infection)
print(feb_features)
# %%


# %% Identify cells by infection type using fov_name
mock_cells = feb_features.sel(sample=feb_features['fov_name'].str.contains('/A/3') | feb_features['fov_name'].str.contains('/B/3'))
zika_cells = feb_features.sel(sample=feb_features['fov_name'].str.contains('/A/4'))
dengue_cells = feb_features.sel(sample=feb_features['fov_name'].str.contains('/B/4'))

# %% Plot UMAP with Infection Status
plt.figure(figsize=(10, 8))
sns.scatterplot(x=feb_features["UMAP1"], y=feb_features["UMAP2"], hue=feb_infection, s=7, alpha=0.8)

# Overlay with circled cells
plt.scatter(mock_cells["UMAP1"], mock_cells["UMAP2"], facecolors='none', edgecolors='blue', s=20, label='Mock Cells')
plt.scatter(zika_cells["UMAP1"], zika_cells["UMAP2"], facecolors='none', edgecolors='green', s=20, label='Zika MOI 5')
plt.scatter(dengue_cells["UMAP1"], dengue_cells["UMAP2"], facecolors='none', edgecolors='red', s=20, label='Dengue MOI 5')

# Add legend and show plot
plt.legend(loc='best')
plt.title("UMAP Plot - February Dataset with Mock, Zika, and Dengue Highlighted")
plt.show()

# %%
# %% Create a 1x3 grid of heatmaps
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# Mock Cells Heatmap
sns.histplot(x=mock_cells["UMAP1"], y=mock_cells["UMAP2"], bins=50, pmax=1, cmap="Blues", ax=axs[0])
axs[0].set_title('Mock Cells')
axs[0].set_xlim(feb_features["UMAP1"].min(), feb_features["UMAP1"].max())
axs[0].set_ylim(feb_features["UMAP2"].min(), feb_features["UMAP2"].max())

# Zika Cells Heatmap
sns.histplot(x=zika_cells["UMAP1"], y=zika_cells["UMAP2"], bins=50, pmax=1, cmap="Greens", ax=axs[1])
axs[1].set_title('Zika MOI 5')
axs[1].set_xlim(feb_features["UMAP1"].min(), feb_features["UMAP1"].max())
axs[1].set_ylim(feb_features["UMAP2"].min(), feb_features["UMAP2"].max())

# Dengue Cells Heatmap
sns.histplot(x=dengue_cells["UMAP1"], y=dengue_cells["UMAP2"], bins=50, pmax=1, cmap="Reds", ax=axs[2])
axs[2].set_title('Dengue MOI 5')
axs[2].set_xlim(feb_features["UMAP1"].min(), feb_features["UMAP1"].max())
axs[2].set_ylim(feb_features["UMAP2"].min(), feb_features["UMAP2"].max())

# Set labels and adjust layout
for ax in axs:
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %% Create a 2x3 grid of heatmaps (1 row for each heatmap, splitting infected and uninfected in the second row)
fig, axs = plt.subplots(2, 3, figsize=(24, 12), sharex=True, sharey=True)

# Mock Cells Heatmap
sns.histplot(x=mock_cells["UMAP1"], y=mock_cells["UMAP2"], bins=50, pmax=1, cmap="Blues", ax=axs[0, 0])
axs[0, 0].set_title('Mock Cells')
axs[0, 0].set_xlim(feb_features["UMAP1"].min(), feb_features["UMAP1"].max())
axs[0, 0].set_ylim(feb_features["UMAP2"].min(), feb_features["UMAP2"].max())

# Zika Cells Heatmap
sns.histplot(x=zika_cells["UMAP1"], y=zika_cells["UMAP2"], bins=50, pmax=1, cmap="Greens", ax=axs[0, 1])
axs[0, 1].set_title('Zika MOI 5')
axs[0, 1].set_xlim(feb_features["UMAP1"].min(), feb_features["UMAP1"].max())
axs[0, 1].set_ylim(feb_features["UMAP2"].min(), feb_features["UMAP2"].max())

# Dengue Cells Heatmap
sns.histplot(x=dengue_cells["UMAP1"], y=dengue_cells["UMAP2"], bins=50, pmax=1, cmap="Reds", ax=axs[0, 2])
axs[0, 2].set_title('Dengue MOI 5')
axs[0, 2].set_xlim(feb_features["UMAP1"].min(), feb_features["UMAP1"].max())
axs[0, 2].set_ylim(feb_features["UMAP2"].min(), feb_features["UMAP2"].max())

# Infected Cells Heatmap
sns.histplot(x=infected_cells["UMAP1"], y=infected_cells["UMAP2"], bins=50, pmax=1, cmap="Reds", ax=axs[1, 0])
axs[1, 0].set_title('Infected Cells')
axs[1, 0].set_xlim(feb_features["UMAP1"].min(), feb_features["UMAP1"].max())
axs[1, 0].set_ylim(feb_features["UMAP2"].min(), feb_features["UMAP2"].max())

# Uninfected Cells Heatmap
sns.histplot(x=uninfected_cells["UMAP1"], y=uninfected_cells["UMAP2"], bins=50, pmax=1, cmap="Greens", ax=axs[1, 1])
axs[1, 1].set_title('Uninfected Cells')
axs[1, 1].set_xlim(feb_features["UMAP1"].min(), feb_features["UMAP1"].max())
axs[1, 1].set_ylim(feb_features["UMAP2"].min(), feb_features["UMAP2"].max())

# Remove the last subplot (bottom right corner)
fig.delaxes(axs[1, 2])

# Set labels and adjust layout
for ax in axs.flat:
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

plt.tight_layout()
plt.show()



# %%
