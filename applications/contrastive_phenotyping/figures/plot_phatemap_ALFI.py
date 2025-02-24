# %% figure plotting phate maps from ALFI data at different time intervals and joint embeddings from time aware and classical models

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks

# %% load the embedding dataset
# Paths to datasets
feature_paths = {
    "7 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_7mins.zarr",
    "14 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_14mins.zarr",
    "28 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_28mins.zarr",
    "56 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_56mins.zarr",
    "91 min interval": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_91mins.zarr",
    "Classical": "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_classical.zarr",
}

# %% color using human annotation for cell cycle state

def load_annotation(da, path, name, exclude, categories: dict | None = None):

    PHATE1 = da["PHATE1"].values
    PHATE2 = da["PHATE2"].values

    annotation = pd.read_csv(path)
    # annotation_columns = annotation.columns.tolist()
    # print(annotation_columns)
    annotation["fov_name"] = "/" + annotation["fov ID"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.reindex(mi)[name]
    if categories:
        PHATE1 = PHATE1[selected != exclude]
        PHATE2 = PHATE2[selected != exclude]
        selected = selected[selected.isin(categories.keys())]
        selected = selected.astype("category").cat.rename_categories(categories)
    
    return selected, PHATE1, PHATE2

ann_root = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets"
)

# %% Create subplot figure with phatemaps from all time intervals
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot each PHATE map in its own subplot
num_plots = len(feature_paths)
for idx, (label, path) in enumerate(feature_paths.items()):
    embedding_dataset = read_embedding_dataset(path)
    division, PHATE1, PHATE2 = load_annotation(
        embedding_dataset,
        ann_root / "test_annotations.csv",
        "division",
        -1,
        {0: "interphase", 1: "mitosis"},
    )
    
    # Create scatter plot with annotations
    scatter = axes[idx].scatter(PHATE1, PHATE2, 
                              c=division.map({'interphase': 'blue', 'mitosis': 'red'}), 
                              alpha=0.6, s=1)
    axes[idx].set_title(label, fontsize=18)
    
    # Only show x-label on bottom subplots
    if idx >= num_plots/2:  # Assumes square grid layout
        axes[idx].set_xlabel('PHATE1', fontsize=18)
    else:
        axes[idx].set_xlabel('')
    
    # Only show y-label on leftmost subplots
    if idx in [0, 3]:  # Left edge of 2x3 grid
        axes[idx].set_ylabel('PHATE2', fontsize=18)
    else:
        axes[idx].set_ylabel('')
    
    # Increase tick label sizes
    axes[idx].tick_params(axis='both', which='major', labelsize=14)

# Add a common legend with larger font
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='blue', label='interphase', markersize=8),
                  plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='red', label='mitosis', markersize=8)]
fig.legend(handles=legend_elements, loc='center right', fontsize=18)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/ALFI/phatemap_timeinterval_ALFI.pdf", dpi=300)





# %% plot the joint embedding from time aware and classical models

train_data_timeaware = read_embedding_dataset(
    "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions_indistribution/ALFI_7mins.zarr"
)
features_train_timeaware = train_data_timeaware["features"].values

train_data_classical = read_embedding_dataset(
    "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions_indistribution/ALFI_classical.zarr"
)
features_train_classical = train_data_classical["features"].values

test_data_timeaware = read_embedding_dataset(
    "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_7mins.zarr"
)
features_test_timeaware = test_data_timeaware["features"].values

test_data_classical = read_embedding_dataset(
    "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_classical.zarr"
)
features_test_classical = test_data_classical["features"].values

# dataset paths
train_data_path = "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/float_phase_ome_zarr_output_valtrain.zarr"
train_track_path = "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/track_phase_ome_zarr_output_valtrain.zarr"
test_data_path = "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/float_phase_ome_zarr_output_test.zarr"
test_track_path = "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/track_phase_ome_zarr_output_test.zarr"


# %% load annotations for train and test data


def load_annotation(da, path, name, categories: dict | None = None):

    annotation = pd.read_csv(path)
    annotation["fov_name"] = "/" + annotation["fov ID"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.reindex(mi)[name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    
    return selected

ann_root = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets"
)

test_division = load_annotation(
    test_data_timeaware,
    ann_root / "test_annotations.csv",
    "division",
    {0: "HeLa+RPE: interphase", 1: "HeLa+RPE: mitosis"},
)
train_division = load_annotation(
    train_data_timeaware,
    ann_root / "train_annotations.csv",
    "division",
    {0: "U2OS: interphase", 1: "U2OS: mitosis"},
)

# concatenate the train and test data annotations after renaming the annotation to add cell type
division = pd.concat([train_division, test_division])

# %% joint the embedding from time aware and classical models and calculate phate coordinates

def compute_phate_from_feature(features):
    import phate

    # Compute PHATE embeddings
    phate_model = phate.PHATE(
        n_components=2, knn=5, decay=40,
    )
    phate_embedding = phate_model.fit_transform(features)
    return phate_embedding

# joint the train and test data embeddings
features_timeaware = np.concatenate([features_train_timeaware, features_test_timeaware], axis=0)
features_classical = np.concatenate([features_train_classical, features_test_classical], axis=0)

# compute phate coordinates
phate_embedding_timeaware = compute_phate_from_feature(features_timeaware)
phate_embedding_classical = compute_phate_from_feature(features_classical)

# Filter out excluded values after PHATE calculation
mask = division != -1
phate_embedding_timeaware = phate_embedding_timeaware[mask]
phate_embedding_classical = phate_embedding_classical[mask]
division = division[mask]


# %% # note image patches to show different phenotypes in the plot

fov_name_test_mitosis = "/0/0/0"
track_id_test_mitosis = [1]
t_test_mitosis = 24
cell_type_test_mitosis = "HeLa+RPE: mitosis"
fov_name_test_interphase = "/0/0/0"
track_id_test_interphase = [9]
t_test_interphase = 3
cell_type_test_interphase = "HeLa+RPE: interphase"

fov_name_train_mitosis = "/0/0/0"
track_id_train_mitosis = [34]
t_train_mitosis = 10
cell_type_train_mitosis = "U2OS: mitosis"
fov_name_train_interphase = "/0/0/0"
track_id_train_interphase = [36]
t_train_interphase = 87
cell_type_train_interphase = "U2OS: interphase"

z_range = (0, 1)
source_channel = ["DIC"]

def load_track_images(data_path, track_path, t, fov_name, track_id):
    """Load image tracks for a specific condition.
    
    Args:
        data_path (str): Path to the data zarr file
        track_path (str): Path to the track zarr file
        fov_name (str): FOV name
        track_id (list): List of track IDs
        t (int): Time point
    
    Returns:
        numpy.ndarray: Image data for the track
    """

    csv_files = list((Path(str(track_path) + str(fov_name))).glob("*.csv"))
    tracks_df = pd.read_csv(str(csv_files[0]))
    track_subdf = tracks_df[tracks_df["track_id"] == track_id]
    t_min = np.min(track_subdf["t"])
    dataset = dataset_of_tracks(
        data_path,
        track_path,
        [fov_name],
        [track_id],
        z_range=z_range,
        source_channel=source_channel,
        initial_yx_patch_size=(192,192),
        final_yx_patch_size=(192,192),
    )
    whole = np.stack([p["anchor"] for p in dataset])
    return whole[t-t_min, 0]

# Define conditions
conditions = [
    # (fov_name, track_id, t, data_path, track_path, name)
    (fov_name_test_mitosis, track_id_test_mitosis, t_test_mitosis, 
     test_data_path, test_track_path, cell_type_test_mitosis),
    (fov_name_test_interphase, track_id_test_interphase, t_test_interphase, 
     test_data_path, test_track_path, cell_type_test_interphase),
    (fov_name_train_mitosis, track_id_train_mitosis, t_train_mitosis, 
     train_data_path, train_track_path, cell_type_train_mitosis),
    (fov_name_train_interphase, track_id_train_interphase, t_train_interphase, 
     train_data_path, train_track_path, cell_type_train_interphase),
]

# Load all images
dic_images = {}
for fov_name, track_id, t, data_path, track_path, name in conditions:
    dic_images[name] = load_track_images(data_path, track_path, t, fov_name, track_id[0])

# get the dic data as a dataframe
dic_train_time = train_data_timeaware["t"].values
dic_train_fov_name = train_data_timeaware["fov_name"].values
dic_train_track_id = train_data_timeaware["track_id"].values

dic_train_data = pd.DataFrame({
    "time": dic_train_time,
    "fov_name": dic_train_fov_name,
    "track_id": dic_train_track_id,
})

dic_test_time = test_data_timeaware["t"].values
dic_test_fov_name = test_data_timeaware["fov_name"].values
dic_test_track_id = test_data_timeaware["track_id"].values

dic_test_data = pd.DataFrame({
    "time": dic_test_time,
    "fov_name": dic_test_fov_name,
    "track_id": dic_test_track_id,
})

# concatenate the train and test dataframes
dic_data = pd.concat([dic_train_data, dic_test_data])
bool_mask = mask.values # remove the index to get the boolean values only
dic_data = dic_data[bool_mask]

dic_data["PHATE1"] = phate_embedding_timeaware[:, 0]
dic_data["PHATE2"] = phate_embedding_timeaware[:, 1]
# add a column for specifying cell type
dic_data["cell_type"] = division.values

# %% plot the phate coordinates with annotations for timeaware model

# plot for timeaware model
# Create a color mapping dictionary
color_map = {
    'HeLa+RPE: mitosis': 'orange',
    'U2OS: mitosis': 'red',
    'HeLa+RPE: interphase': 'green',
    'U2OS: interphase': 'steelblue'
}

# Create figure with 1x3 layout (scatterplot on left, 2x2 image grid on right)
fig = plt.figure(figsize=(16, 8))  # Reduced overall width from 20 to 16
gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1], 
                        wspace=0.05,  # Reduced horizontal spacing between subplots
                        hspace=0.2)   # Reduced vertical spacing between subplots

# PHATE scatter plot (spans both rows on the left)
ax1 = fig.add_subplot(gs[:, 0])

# Create scatter plot with white edges
plt.scatter(phate_embedding_timeaware[:, 0], phate_embedding_timeaware[:, 1], 
           c=division.map(color_map), alpha=0.7, s=40,
           edgecolor='white', linewidth=0.5)  # Add white edges to points
for fov_name, track_id, t, data_path, track_path, name in conditions:
    highlight_data = dic_data[
        (dic_data["fov_name"] == fov_name) & 
        (dic_data["track_id"] == track_id[0]) & 
        (dic_data["time"] == t) &
        (dic_data["cell_type"] == name)
    ]
    if not highlight_data.empty:
        ax1.scatter(highlight_data["PHATE1"], highlight_data["PHATE2"], 
            c=highlight_data["cell_type"].map(color_map), alpha=0.7, s=500,
            edgecolor='black', linewidth=2, zorder=5)

# Update legend to match the new style
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=color, label=label, markersize=8,
                            markeredgecolor='white', markeredgewidth=0.5)  # Add white edges to legend
                  for label, color in color_map.items()]
ax1.legend(handles=legend_elements, fontsize=18)

ax1.set_xlabel('PHATE1', fontsize=18)
ax1.set_ylabel('PHATE2', fontsize=18)

# add image patches for the train and test data
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(dic_images[cell_type_test_mitosis][0], cmap='gray')
ax2.set_title(cell_type_test_mitosis, fontsize=18)
for spine in ax2.spines.values():
    spine.set_color(color_map[cell_type_test_mitosis])
    spine.set_linewidth(6)  # Increased from 3 to 6
ax2.axis('on')
ax2.set_xticks([])
ax2.set_yticks([])

ax3 = fig.add_subplot(gs[1, 1])
ax3.imshow(dic_images[cell_type_test_interphase][0], cmap='gray')
ax3.set_title(cell_type_test_interphase, fontsize=18)
for spine in ax3.spines.values():
    spine.set_color(color_map[cell_type_test_interphase])
    spine.set_linewidth(6)  # Increased from 3 to 6
ax3.axis('on')
ax3.set_xticks([])
ax3.set_yticks([])

ax4 = fig.add_subplot(gs[0, 2])
ax4.imshow(dic_images[cell_type_train_mitosis][0], cmap='gray')
ax4.set_title(cell_type_train_mitosis, fontsize=18)
for spine in ax4.spines.values():
    spine.set_color(color_map[cell_type_train_mitosis])
    spine.set_linewidth(6)  # Increased from 3 to 6
ax4.axis('on')
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig.add_subplot(gs[1, 2])
ax5.imshow(dic_images[cell_type_train_interphase][0], cmap='gray')
ax5.set_title(cell_type_train_interphase, fontsize=18)
for spine in ax5.spines.values():
    spine.set_color(color_map[cell_type_train_interphase])
    spine.set_linewidth(6)  # Increased from 3 to 6
ax5.axis('on')
ax5.set_xticks([])
ax5.set_yticks([])

plt.tight_layout()
plt.savefig("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/ALFI/phatemap_joint_ALFI_timeaware_wImages.pdf", dpi=300)

# %%
