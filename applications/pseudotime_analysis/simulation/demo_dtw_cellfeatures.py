# %%
import matplotlib.pyplot as plt
import numpy as np
from dtaidistance import dtw
from scipy.cluster.hierarchy import dendrogram, linkage

# testing if we can use DTW to align cell trajectories using a short reference pattern

np.random.seed(42)
timepoints = 50
cells = 8

# Create synthetic cell trajectories (e.g., PCA1 shape evolution)
cell_trajectories = [
    np.sin(np.linspace(0, 10, timepoints) + np.random.rand() * 2) for _ in range(cells)
]
# add extra transforms to signal
cell_trajectories[cells - 1] += np.sin(np.linspace(0, 5, timepoints)) * 2
cell_trajectories[cells - 2] += np.sin(np.linspace(0, 5, timepoints)) * 3

# %%
# plot cell trajectories
plt.figure(figsize=(8, 5))
for i in range(cells):
    plt.plot(cell_trajectories[i], label=f"Cell {i + 1}")
plt.legend()
plt.title("Original Cell Trajectories")
plt.show()
# %%
# Set reference cell for all subsequent analysis
reference_cell = 0  # Use first cell as reference

# Compute DTW distance matrix
dtw_matrix = np.zeros((cells, cells))
for i in range(cells):
    for j in range(i + 1, cells):
        dtw_matrix[i, j] = dtw.distance(cell_trajectories[i], cell_trajectories[j])
        dtw_matrix[j, i] = dtw_matrix[i, j]

# Print distance matrix for examination
print("DTW Distance Matrix:")
for i in range(cells):
    print(f"Cell {i + 1}: {dtw_matrix[reference_cell, i]:.2f}")

# Plot distance heatmap
plt.figure(figsize=(8, 6))
plt.imshow(dtw_matrix, cmap="viridis", origin="lower")
plt.colorbar(label="DTW Distance")
plt.title("DTW Distance Matrix Between Cells")
plt.xlabel("Cell Index")
plt.ylabel("Cell Index")
plt.tight_layout()
plt.show()

linkage_matrix = linkage(dtw_matrix, method="ward")

# Plot the dendrogram
plt.figure(figsize=(8, 5))
dendrogram(linkage_matrix, labels=[f"Cell {i + 1}" for i in range(cells)])
plt.xlabel("Cells")
plt.ylabel("DTW Distance")
plt.title("Hierarchical Clustering of Cells Based on DTW")
plt.show()

# %%
# Align cells using DTW with distance filtering
# Set a threshold for maximum allowed DTW distance
# Cells with distances above this threshold won't be aligned
# This can be set based on the distribution of distances or domain knowledge
distance_threshold = np.median(dtw_matrix[reference_cell, :]) * 1.5  # Example threshold

print(f"Using distance threshold: {distance_threshold:.2f}")
print("Distances from reference cell:")
for i in range(cells):
    distance = dtw_matrix[reference_cell, i]
    status = (
        "Included"
        if distance <= distance_threshold or i == reference_cell
        else "Excluded (too dissimilar)"
    )
    print(f"Cell {i + 1}: {distance:.2f} - {status}")

# Initialize aligned trajectories with the reference cell
aligned_cell_trajectories = [cell_trajectories[reference_cell].copy()]
alignment_status = [True]  # Reference cell is always included

for i in range(1, cells):
    distance = dtw_matrix[reference_cell, i]

    # Skip cells that are too dissimilar to the reference
    if distance > distance_threshold:
        aligned_cell_trajectories.append(
            np.full_like(cell_trajectories[reference_cell], np.nan)
        )
        alignment_status.append(False)
        continue

    # Find optimal warping path
    path = dtw.warping_path(cell_trajectories[reference_cell], cell_trajectories[i])

    # Create aligned trajectory by mapping query points to reference timeline
    aligned_trajectory = np.zeros_like(cell_trajectories[reference_cell])
    path_dict = {}

    # Group by reference indices
    for ref_idx, query_idx in path:
        if ref_idx not in path_dict:
            path_dict[ref_idx] = []
        path_dict[ref_idx].append(query_idx)

    # For each reference index, average the corresponding query values
    for ref_idx, query_indices in path_dict.items():
        query_values = [cell_trajectories[i][idx] for idx in query_indices]
        aligned_trajectory[ref_idx] = np.mean(query_values)

    aligned_cell_trajectories.append(aligned_trajectory)
    alignment_status.append(True)

# %%
# plot aligned cell trajectories (only included cells)
plt.figure(figsize=(10, 6))

# Plot reference cell first
plt.plot(aligned_cell_trajectories[0], "k-", linewidth=2.5, label="Reference (Cell 1)")

# Plot other cells that were successfully aligned
for i in range(1, cells):
    if alignment_status[i]:
        plt.plot(aligned_cell_trajectories[i], label=f"Cell {i + 1}")

plt.legend()
plt.title("Aligned Cell Trajectories (Filtered by DTW Distance)")
plt.show()

# %%
# Visualize warping paths for examples
plt.figure(figsize=(15, 10))

# First find cells to include based on distance threshold
included_cells = [i for i in range(1, cells) if alignment_status[i]]
excluded_cells = [i for i in range(1, cells) if not alignment_status[i]]

# Show included cells examples
for idx, target_cell in enumerate(included_cells[: min(2, len(included_cells))]):
    plt.subplot(2, 3, idx + 1)

    # Get warping path
    path = dtw.warping_path(
        cell_trajectories[reference_cell], cell_trajectories[target_cell]
    )

    # Plot both signals
    plt.plot(cell_trajectories[reference_cell], label="Reference", linewidth=2)
    plt.plot(
        cell_trajectories[target_cell], label=f"Cell {target_cell + 1}", linewidth=2
    )

    # Plot warping connections
    for ref_idx, query_idx in path:
        plt.plot(
            [ref_idx, query_idx],
            [
                cell_trajectories[reference_cell][ref_idx],
                cell_trajectories[target_cell][query_idx],
            ],
            "k-",
            alpha=0.1,
        )

    plt.title(
        f"Included - Cell {target_cell + 1} (Dist: {dtw_matrix[reference_cell, target_cell]:.2f})"
    )
    plt.legend()

# Show excluded cells examples
for idx, target_cell in enumerate(excluded_cells[: min(2, len(excluded_cells))]):
    plt.subplot(2, 3, 3 + idx)

    plt.plot(cell_trajectories[reference_cell], label="Reference", linewidth=2)
    plt.plot(
        cell_trajectories[target_cell], label=f"Cell {target_cell + 1}", linewidth=2
    )

    # Show distance value
    plt.title(
        f"Excluded - Cell {target_cell + 1} (Dist: {dtw_matrix[reference_cell, target_cell]:.2f})"
    )
    plt.legend()

# Compare original and aligned for an included cell

if included_cells:
    plt.subplot(2, 3, 5)
    target_cell = included_cells[0]
    plt.plot(cell_trajectories[reference_cell], label="Reference", linewidth=2)
    plt.plot(
        cell_trajectories[target_cell],
        label=f"Original Cell {target_cell + 1}",
        linewidth=2,
        linestyle="--",
        alpha=0.7,
    )
    plt.plot(
        aligned_cell_trajectories[target_cell],
        label=f"Aligned Cell {target_cell + 1}",
        linewidth=2,
    )
    plt.title("Alignment Example (Included)")
    plt.legend()

# Show distance distribution
plt.subplot(2, 3, 6)
distances = dtw_matrix[reference_cell, 1:]  # Skip distance to self
plt.hist(distances, bins=10)
plt.axvline(
    distance_threshold,
    color="r",
    linestyle="--",
    label=f"Threshold: {distance_threshold:.2f}",
)
plt.title("DTW Distance Distribution")
plt.xlabel("DTW Distance from Reference")
plt.ylabel("Count")
plt.legend()

plt.tight_layout()
plt.show()
# %%
