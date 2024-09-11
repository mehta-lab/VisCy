# %% Importing Necessary Libraries
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from viscy.representation.embedding_writer import read_embedding_dataset
from sklearn.decomposition import PCA

# %% Defining Paths for February and June Datasets
feb_features_path = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/febtest_predict.zarr")
feb_data_path = Path("/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr")
feb_tracks_path = Path("/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr")

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
    print(annotation.columns)
    annotation["fov_name"] = "/" + annotation["fov name "]
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

feb_ann_root = Path("/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred")
feb_infection = load_annotation(feb_features, feb_ann_root / "extracted_inf_state.csv", "infection_state", {0.0: "background", 1.0: "uninfected", 2.0: "infected"})

# %% Plot UMAP with Infection Status for February Dataset
plot_umap_infection(feb_features, feb_infection, "February Dataset")

# %% Function to Load and Process the Embedding Dataset using PCA
def compute_pc(embedding_dataset):
    features = embedding_dataset["features"]
    
    # Standardize the features
    scaled_features = StandardScaler().fit_transform(features.values)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pc_embedding = pca.fit_transform(scaled_features)
    
    # Convert the PCA embedding into an xarray.DataArray, maintaining the same coordinates as the original dataset
    pc_data = xr.DataArray(
        data=pc_embedding,
        dims=["sample", "pc"],
        coords={
            "sample": features.coords["sample"],
            "fov_name": features.coords["fov_name"],
            "track_id": features.coords["track_id"],
            "t": features.coords["t"],
            "id": features.coords["id"],
            "PC1": ("sample", pc_embedding[:, 0]),
            "PC2": ("sample", pc_embedding[:, 1])
        }
    )
    
    return pc_data


# %% Function to Plot PCA with Infection Status
def plot_pca_infection(pca_data, infection, title):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_data["PCA1"], y=pca_data["PCA2"], hue=infection, s=7, alpha=0.8)
    plt.title(f"PCA Plot - {title}")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()

# %% Plot PCA for February Dataset
feb_pca_data = compute_pca(feb_embedding_dataset)
print(feb_pca_data)

#plot_pca_infection(feb_pca_data, feb_infection, "February Dataset")
# %%
print("PCA embedding shape: ", pc_embedding.shape)
print("PCA embedding head: ", pc_embedding[:5])
# %%

def plot_umap_histogram(umap_data, infection, title):
    plt.figure(figsize=(15, 5))
    states = infection.unique()  # Get unique infection states
    
    for i, state in enumerate(states):
        plt.subplot(1, len(states), i+1)
        
        # Align both `fov_name` and `id` by comparing their values
        condition = (umap_data.coords["fov_name"].values == infection.coords["fov_name"].values) & \
                    (umap_data.coords["id"].values == infection.coords["id"].values) & \
                    (infection == state)
        
        # Filter umap_data based on this condition
        subset = umap_data.where(condition, drop=True)
        
        # Create the hexbin plot for each state
        plt.hexbin(subset["UMAP1"].values, subset["UMAP2"].values, gridsize=50, cmap="inferno")
        plt.title(f"Infection = {state}")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
    
    # Set the main title
    plt.suptitle(f"{title} - UMAP Histogram")
    plt.show()




# %%
feb_umap_data = compute_umap(feb_embedding_dataset)

plot_umap_histogram(feb_umap_data, feb_infection, "February Dataset")

# %%
print(feb_umap_data)
print(feb_infection)
plot_umap_histogram(feb_umap_data, feb_infection, "February Dataset")

# %%
print(feb_umap_data.coords["fov_name"])

# %%
