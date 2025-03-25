# %%
import matplotlib.pyplot as plt
import numpy as np
from dtaidistance import dtw
from scipy.cluster.hierarchy import dendrogram, linkage

# DEMO test for the DTW distance using sinusoidals
np.random.seed(42)
timepoints = 50
cells = 5

# Create synthetic cell trajectories (e.g., PCA1 shape evolution)
cell_trajectories = [
    np.sin(np.linspace(0, 10, timepoints) + np.random.rand() * 2) for _ in range(cells)
]
# add extra transforms to signal
cell_trajectories[2] += np.sin(np.linspace(0, 5, timepoints))

# %%
# plot cell trajectories
plt.figure(figsize=(8, 5))
for i in range(cells):
    plt.plot(cell_trajectories[i], label=f"Cell {i+1}")
plt.legend()
plt.show()
# %%
# Compute DTW distance matrix
dtw_matrix = np.zeros((cells, cells))
for i in range(cells):
    for j in range(i + 1, cells):
        dtw_matrix[i, j] = dtw.distance(cell_trajectories[i], cell_trajectories[j])
        dtw_matrix[j, i] = dtw_matrix[i, j]

linkage_matrix = linkage(dtw_matrix, method="ward")

# Plot the dendrogram
plt.figure(figsize=(8, 5))
dendrogram(linkage_matrix, labels=[f"Cell {i+1}" for i in range(cells)])
plt.xlabel("Cells")
plt.ylabel("DTW Distance")
plt.title("Hierarchical Clustering of Cells Based on DTW")
plt.show()

# %%
# Align cells using DTW
reference_cell = 0  # Use first cell as reference
aligned_cell_trajectories = []

for i in range(cells):
    if i == reference_cell:
        aligned_cell_trajectories.append(cell_trajectories[i])
    else:
        # Find optimal warping path
        path = dtw.warping_path(cell_trajectories[reference_cell], cell_trajectories[i])

        # Create aligned trajectory using the warping path
        aligned_trajectory = np.zeros_like(cell_trajectories[reference_cell])
        for ref_idx, query_idx in path:
            aligned_trajectory[ref_idx] = cell_trajectories[i][query_idx]

        aligned_cell_trajectories.append(aligned_trajectory)

# %%
# plot aligned cell trajectories
plt.figure(figsize=(8, 5))
for i in range(cells):
    plt.plot(aligned_cell_trajectories[i], label=f"Cell {i+1}")
plt.legend()
plt.title("Aligned Cell Trajectories")
plt.show()
# %%
