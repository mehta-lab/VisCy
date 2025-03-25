# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset

#
features_data_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
num_PC_components = 8

embedding_dataset = read_embedding_dataset(features_data_path)
features = embedding_dataset["features"]
features_df = features["sample"].to_dataframe().reset_index(drop=True)

fov_patterns = ["/B/1/", "/C/2/"]
pattern_mask = features_df["fov_name"].apply(
    lambda x: any(pattern in x for pattern in fov_patterns)
)
features_df = features_df[pattern_mask]
filtered_features = features.values[pattern_mask]

scaled_features = StandardScaler().fit_transform(filtered_features)
pca = PCA(n_components=num_PC_components)
pca_coords = pca.fit_transform(scaled_features)

for i in range(num_PC_components):
    features_df[f"PCA{i+1}"] = pca_coords[:, i]

pca_explained_variance = [
    f"PC{i+1} ({var:.1f}%)" for i, var in enumerate(pca.explained_variance_ratio_ * 100)
]
# %%
timepoint_counts = features_df.groupby(["fov_name", "track_id"]).size()
valid_cells = timepoint_counts[timepoint_counts >= 30].reset_index()[
    ["fov_name", "track_id"]
]
filtered_features_df = features_df.merge(valid_cells, on=["fov_name", "track_id"])

selected_cells = []
for fov in filtered_features_df["fov_name"].unique():
    fov_cells = filtered_features_df[filtered_features_df["fov_name"] == fov][
        "track_id"
    ].unique()
    n = 5
    if len(fov_cells) >= n:
        selected_fov_cells = np.random.choice(fov_cells, n, replace=False)
        for cell in selected_fov_cells:
            selected_cells.append((fov, cell))
    else:
        for cell in fov_cells:
            selected_cells.append((fov, cell))

selected_cells_df = pd.DataFrame(selected_cells, columns=["fov_name", "track_id"])
filtered_features_df = filtered_features_df.merge(
    selected_cells_df, on=["fov_name", "track_id"]
)

cell_trajectories_df = (
    filtered_features_df.groupby(["fov_name", "track_id"])["PCA1"]
    .apply(list)
    .reset_index()
)
cell_trajectories = []
cell_labels = []
for _, row in cell_trajectories_df.iterrows():
    trajectory = np.array(row["PCA1"], dtype=float)
    cell_trajectories.append(trajectory)
    cell_labels.append(f"{row['fov_name']}-{row['track_id']}")

# %%
dtw_matrix = np.zeros((len(cell_trajectories), len(cell_trajectories)))
for i in range(len(cell_trajectories)):
    for j in range(i + 1, len(cell_trajectories)):
        dtw_matrix[i, j] = dtw.distance(cell_trajectories[i], cell_trajectories[j])
        dtw_matrix[j, i] = dtw_matrix[i, j]

linkage_matrix = linkage(dtw_matrix, method="ward")
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=cell_labels, leaf_rotation=90)
plt.xlabel("Cells")
plt.ylabel("DTW Distance")
plt.title("Hierarchical Clustering of Cells Based on DTW")
plt.tight_layout()
plt.show()

# %%
num_clusters = 3
cluster_labels = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
unique_clusters = np.unique(cluster_labels)

plt.figure(figsize=(8, 5))
for cluster in unique_clusters:
    cluster_cells = [
        cell_trajectories[i]
        for i in range(len(cluster_labels))
        if cluster_labels[i] == cluster
    ]

    max_length = max(len(traj) for traj in cluster_cells)
    aligned_trajectories = []
    for traj in cluster_cells:
        if len(traj) < max_length:
            padded_traj = np.pad(traj, (0, max_length - len(traj)), "edge")
            aligned_trajectories.append(padded_traj)
        else:
            aligned_trajectories.append(traj[:max_length])

    aligned_trajectories = np.array(aligned_trajectories)
    avg_trajectory = np.mean(aligned_trajectories, axis=0)
    timepoints = np.arange(len(avg_trajectory))
    plt.plot(timepoints, avg_trajectory, label=f"Cluster {cluster}")

plt.xlabel("Time")
plt.ylabel("PC1 Value")
plt.legend()
plt.title("Average PC1 Trajectories per DTW Cluster")
plt.show()

# %%
pc2_trajectories_df = (
    filtered_features_df.groupby(["fov_name", "track_id"])["PCA2"]
    .apply(list)
    .reset_index()
)

pc3_trajectories_df = (
    filtered_features_df.groupby(["fov_name", "track_id"])["PCA3"]
    .apply(list)
    .reset_index()
)

pc2_trajectories = []
for _, row in pc2_trajectories_df.iterrows():
    trajectory = np.array(row["PCA2"], dtype=float)
    pc2_trajectories.append(trajectory)

pc3_trajectories = []
for _, row in pc3_trajectories_df.iterrows():
    trajectory = np.array(row["PCA3"], dtype=float)
    pc3_trajectories.append(trajectory)

fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)


def plot_avg_trajectories(pc_trajectories, ax, pc_num):
    for cluster in unique_clusters:
        cluster_pc_trajectories = [
            pc_trajectories[i]
            for i in range(len(cluster_labels))
            if cluster_labels[i] == cluster
        ]

        max_length = max(len(traj) for traj in cluster_pc_trajectories)
        aligned_trajectories = []
        for traj in cluster_pc_trajectories:
            if len(traj) < max_length:
                padded_traj = np.pad(traj, (0, max_length - len(traj)), "edge")
                aligned_trajectories.append(padded_traj)
            else:
                aligned_trajectories.append(traj[:max_length])

        aligned_trajectories = np.array(aligned_trajectories)
        avg_trajectory = np.mean(aligned_trajectories, axis=0)
        timepoints = np.arange(len(avg_trajectory))
        ax.plot(timepoints, avg_trajectory, label=f"Cluster {cluster}")

    ax.set_ylabel(f"PC{pc_num} Value")
    ax.set_title(f"Average PC{pc_num} Trajectories per Cluster")
    ax.legend()


plot_avg_trajectories(cell_trajectories, axes[0], 1)
plot_avg_trajectories(pc2_trajectories, axes[1], 2)
plot_avg_trajectories(pc3_trajectories, axes[2], 3)

axes[2].set_xlabel("Time")
plt.tight_layout()
plt.show()

# %%
