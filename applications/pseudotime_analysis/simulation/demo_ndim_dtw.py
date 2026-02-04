# %%
import matplotlib.pyplot as plt
import numpy as np
from dtaidistance.dtw_ndim import warping_path
from scipy.spatial.distance import cdist

# %%
# Simulation of embeddings with temporal warping
np.random.seed(42)  # For reproducibility

# Base parameters
num_cells = 8
num_timepoints = 30  # More timepoints for better visualization of warping
embedding_dim = 100

# Create a reference trajectory (Cell 1)
t_ref = np.linspace(0, 4 * np.pi, num_timepoints)  # 2 complete periods (0 to 4π)
base_pattern = np.zeros((num_timepoints, embedding_dim))

# Generate a structured pattern with clear sinusoidal shape
for dim in range(embedding_dim):
    # Use lower frequencies to ensure at least one full period
    freq = 0.2 + 0.3 * np.random.rand()  # Frequency between 0.2 and 0.5
    phase = np.random.rand() * np.pi  # Random phase
    amplitude = 0.7 + 0.6 * np.random.rand()  # Amplitude between 0.7 and 1.3

    # Create basic sine wave for this dimension
    base_pattern[:, dim] = amplitude * np.sin(freq * t_ref + phase)

# Cell embeddings with different temporal dynamics
cell_embeddings = np.zeros((num_cells, num_timepoints, embedding_dim))

# Cell 1 (reference) - standard progression
cell_embeddings[0] = base_pattern.copy()

# Cell 2 - similar to reference with small variations
cell_embeddings[1] = base_pattern + np.random.randn(num_timepoints, embedding_dim) * 0.2

# Cell 3 - starts slow, then accelerates (time warping)
# Map [0,1] -> [0,4π] with non-linear warping
t_warped = np.power(np.linspace(0, 1, num_timepoints), 1.7) * 4 * np.pi
for dim in range(embedding_dim):
    # Get the same frequencies and phases as the reference
    freq = 0.2 + 0.3 * np.random.rand()
    phase = np.random.rand() * np.pi
    amplitude = 0.7 + 0.6 * np.random.rand()

    # Apply the warping to the timepoints
    cell_embeddings[2, :, dim] = amplitude * np.sin(freq * t_warped + phase)
cell_embeddings[2] += np.random.randn(num_timepoints, embedding_dim) * 0.15

# Cell 4 - starts fast, then slows down (time warping)
t_warped = np.power(np.linspace(0, 1, num_timepoints), 0.6) * 4 * np.pi
for dim in range(embedding_dim):
    freq = 0.2 + 0.3 * np.random.rand()
    phase = np.random.rand() * np.pi
    amplitude = 0.7 + 0.6 * np.random.rand()
    cell_embeddings[3, :, dim] = amplitude * np.sin(freq * t_warped + phase)
cell_embeddings[3] += np.random.randn(num_timepoints, embedding_dim) * 0.15

# Cell 5 - missing middle section (temporal gap)
t_warped = np.concatenate(
    [
        np.linspace(0, 1.5 * np.pi, num_timepoints // 2),  # First 1.5 periods
        np.linspace(
            2.5 * np.pi, 4 * np.pi, num_timepoints // 2
        ),  # Last 1.5 periods (sshkip middle)
    ]
)
for dim in range(embedding_dim):
    freq = 0.2 + 0.3 * np.random.rand()
    phase = np.random.rand() * np.pi
    amplitude = 0.7 + 0.6 * np.random.rand()
    cell_embeddings[4, :, dim] = amplitude * np.sin(freq * t_warped + phase)
cell_embeddings[4] += np.random.randn(num_timepoints, embedding_dim) * 0.15

# Cell 6 - phase shifted (out of sync with reference)
cell_embeddings[5] = np.roll(
    base_pattern, shift=num_timepoints // 4, axis=0
)  # 1/4 cycle shift
cell_embeddings[5] += np.random.randn(num_timepoints, embedding_dim) * 0.2

# Cell 7 - Double frequency (faster oscillations)
for dim in range(embedding_dim):
    freq = (0.2 + 0.3 * np.random.rand()) * 2  # Double frequency
    phase = np.random.rand() * np.pi
    amplitude = 0.7 + 0.6 * np.random.rand()
    cell_embeddings[6, :, dim] = amplitude * np.sin(freq * t_ref + phase)
cell_embeddings[6] += np.random.randn(num_timepoints, embedding_dim) * 0.15

# Cell 8 - Very different pattern with trend
cell_embeddings[7] = np.random.randn(num_timepoints, embedding_dim) * 1.5
trend = np.linspace(0, 3, num_timepoints).reshape(-1, 1)
cell_embeddings[7] += trend * np.random.randn(1, embedding_dim)

# %%
# Visualize the first two dimensions of each cell's embeddings to see the temporal patterns
plt.figure(figsize=(18, 10))

# Create subplots for 4 dimensions
for dim in range(4):
    plt.subplot(2, 2, dim + 1)

    for i in range(num_cells):
        plt.plot(
            range(num_timepoints),
            cell_embeddings[i, :, dim],
            label=f"Cell {i + 1}",
            linewidth=2,
        )

    plt.title(f"Dimension {dim + 1} over time")
    plt.xlabel("Timepoint")
    plt.ylabel(f"Value (Dim {dim + 1})")
    plt.grid(alpha=0.3)

    if dim == 0:
        plt.legend(loc="upper right")

plt.tight_layout()
plt.show()


# %%
# Helper function to compute DTW warping matrix
def compute_dtw_matrix(s1, s2):
    """
    Compute the DTW warping matrix and best path manually.

    Args:
        s1: First sequence (reference)
        s2: Second sequence (query)

    Returns:
        warping_matrix: The accumulated cost matrix
        best_path: The optimal warping path
    """
    # Compute pairwise distances between all timepoints
    distance_matrix = cdist(s1, s2)

    n, m = distance_matrix.shape

    # Initialize the accumulated cost matrix
    warping_matrix = np.full((n, m), np.inf)
    warping_matrix[0, 0] = distance_matrix[0, 0]

    # Fill the first column and row
    for i in range(1, n):
        warping_matrix[i, 0] = warping_matrix[i - 1, 0] + distance_matrix[i, 0]
    for j in range(1, m):
        warping_matrix[0, j] = warping_matrix[0, j - 1] + distance_matrix[0, j]

    # Fill the rest of the matrix
    for i in range(1, n):
        for j in range(1, m):
            warping_matrix[i, j] = distance_matrix[i, j] + min(
                warping_matrix[i - 1, j],  # insertion
                warping_matrix[i, j - 1],  # deletion
                warping_matrix[i - 1, j - 1],  # match
            )

    # Backtrack to find the optimal path
    i, j = n - 1, m - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_cost = min(
                warping_matrix[i - 1, j],
                warping_matrix[i, j - 1],
                warping_matrix[i - 1, j - 1],
            )

            if min_cost == warping_matrix[i - 1, j - 1]:
                i, j = i - 1, j - 1
            elif min_cost == warping_matrix[i - 1, j]:
                i -= 1
            else:
                j -= 1

        path.append((i, j))

    path.reverse()

    return warping_matrix, path


# %%
# Compute DTW distances and warping paths between cell 1 and all other cells
reference_cell = 0  # Cell 1 (0-indexed)
dtw_results = []

for i in range(num_cells):
    if i != reference_cell:
        # Get distance and path from dtaidistance
        path, dist = warping_path(
            cell_embeddings[reference_cell],
            cell_embeddings[i],
            include_distance=True,
        )

        # Compute our own warping matrix for visualization
        warping_matrix, _ = compute_dtw_matrix(
            cell_embeddings[reference_cell], cell_embeddings[i]
        )

        dtw_results.append(
            (i + 1, dist, path, warping_matrix)
        )  # Store cell number, distance, path, matrix
        print(f"DTW distance between Cell 1 and Cell {i + 1} dist: {dist:.4f}")

# %%
# Visualize the DTW distances
cell_ids = [result[0] for result in dtw_results]
distances = [result[1] for result in dtw_results]

plt.figure(figsize=(10, 6))
plt.bar(cell_ids, distances)
plt.xlabel("Cell ID")
plt.ylabel("DTW Distance from Cell 1")
plt.title("DTW Distances from Cell 1 to Other Cells")
plt.xticks(cell_ids)
plt.tight_layout()
plt.show()

# %%
# Create a grid of all warping matrices in a 4x2 layout
fig, axes = plt.subplots(4, 2, figsize=(16, 24))
axes = axes.flatten()

# Common colorbar limits for better comparison
all_matrices = [result[3] for result in dtw_results]
vmin = min(matrix.min() for matrix in all_matrices)
vmax = max(matrix.max() for matrix in all_matrices)

# Add diagonal reference line for comparison
diagonal = np.linspace(0, num_timepoints - 1, 100)

for i, result in enumerate(dtw_results):
    cell_id, dist, path, warping_matrix = result
    ax = axes[i]

    # Plot the warping matrix
    im = ax.imshow(
        warping_matrix,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    # Plot diagonal reference line
    ax.plot(diagonal, diagonal, "w--", alpha=0.7, linewidth=1, label="Diagonal")

    # Extract and plot the best path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_y, path_x, "r-", linewidth=2, label="Best path")

    # Add some arrows to show direction
    step = max(1, len(path) // 5)  # Show 5 arrows along the path
    for j in range(0, len(path) - 1, step):
        ax.annotate(
            "",
            xy=(path_y[j + 1], path_x[j + 1]),
            xytext=(path_y[j], path_x[j]),
            arrowprops=dict(arrowstyle="->", color="orange", lw=1.5),
        )

    # Add title and axes labels
    cell_desc = ""
    if cell_id == 2:
        cell_desc = " (Small variations)"
    elif cell_id == 3:
        cell_desc = " (Slow→Fast)"
    elif cell_id == 4:
        cell_desc = " (Fast→Slow)"
    elif cell_id == 5:
        cell_desc = " (Missing middle)"
    elif cell_id == 6:
        cell_desc = " (Phase shift)"
    elif cell_id == 7:
        cell_desc = " (Double frequency)"
    elif cell_id == 8:
        cell_desc = " (Different pattern)"

    ax.set_title(f"Cell 1 vs Cell {cell_id}{cell_desc} (Dist: {dist:.2f})")
    ax.set_xlabel("Cell {} Timepoints".format(cell_id))
    ax.set_ylabel("Cell 1 Timepoints")

    # Add legend
    ax.legend(loc="lower right", fontsize=8)

# Add a colorbar at the bottom spanning all subplots
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Accumulated Cost")

plt.suptitle("DTW Warping Matrices: Cell 1 vs All Other Cells", fontsize=16)
plt.tight_layout(rect=[0, 0.07, 1, 0.98])
plt.show()


# %%
# Compute average embedding by aligning all cells to the reference cell
def align_to_reference(
    reference: np.ndarray, query: np.ndarray, path: list[tuple[int, int]]
) -> np.ndarray:
    """
    Align a query embedding to the reference timepoints based on the DTW path.

    Args:
        reference: Reference embedding (n_timepoints x n_dims)
        query: Query embedding to align (m_timepoints x n_dims)
        path: DTW path as list of (ref_idx, query_idx) tuples

    Returns:
        aligned_query: Query embeddings aligned to reference timepoints
    """
    n_ref, n_dims = reference.shape
    aligned_query = np.zeros_like(reference)

    # Count how many query timepoints map to each reference timepoint
    counts = np.zeros(n_ref)

    # Sum query embeddings for each reference timepoint based on the path
    for ref_idx, query_idx in path:
        aligned_query[ref_idx] += query[query_idx]
        counts[ref_idx] += 1

    # Average when multiple query timepoints map to the same reference timepoint
    for i in range(n_ref):
        if counts[i] > 0:
            aligned_query[i] /= counts[i]
        else:
            # If no query timepoints map to this reference, use nearest neighbors
            nearest_idx = min(range(len(path)), key=lambda j: abs(path[j][0] - i))
            aligned_query[i] = query[path[nearest_idx][1]]

    return aligned_query


# Identify the top 5 most similar cells
# Sort cells by distance
sorted_results = sorted(dtw_results, key=lambda x: x[1])  # Sort by distance (x[1])

# Select top 5 closest cells
top_5_cells = sorted_results[:5]
print("Top 5 cells (by DTW distance to reference):")
for cell_id, dist, _, _ in top_5_cells:
    print(f"Cell {cell_id}: Distance = {dist:.4f}")

# First, align all the top 5 cells to the reference timepoints
aligned_cells = []
all_cell_ids = [reference_cell + 1] + [cell_id for cell_id, _, _, _ in top_5_cells]

# Include reference cell as-is (it's already aligned)
aligned_cells.append(cell_embeddings[reference_cell])

# Align each of the top 5 cells
for cell_id, _, path, _ in top_5_cells:
    cell_idx = cell_id - 1  # Convert to 0-indexed

    # Get aligned version of this cell
    aligned_cell = align_to_reference(
        cell_embeddings[reference_cell], cell_embeddings[cell_idx], path
    )

    # Store the aligned cell
    aligned_cells.append(aligned_cell)

# %%
# Visualize the aligned cells before averaging
plt.figure(figsize=(18, 10))

# Create subplots for the first 4 dimensions
for dim in range(4):
    plt.subplot(2, 2, dim + 1)

    # Plot each aligned cell
    for i, (cell_id, aligned_cell) in enumerate(zip(all_cell_ids, aligned_cells)):
        if i == 0:
            # Reference cell
            plt.plot(
                range(num_timepoints),
                aligned_cell[:, dim],
                "b-",
                linewidth=2,
                label="Cell 1 (Reference)",
            )
        else:
            # Other aligned cells
            plt.plot(
                range(num_timepoints),
                aligned_cell[:, dim],
                "g-",
                alpha=0.5,
                linewidth=1,
                label=f"Cell {cell_id} (Aligned)" if i == 1 else None,
            )

    plt.title(f"Dimension {dim + 1}: Aligned Cells")
    plt.xlabel("Reference Timepoint")
    plt.ylabel(f"Value (Dim {dim + 1})")
    plt.grid(alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.suptitle("Cells Aligned to Reference before Averaging", fontsize=16, y=1.02)
plt.show()

# %%
# Now compute the average of the aligned cells
average_embedding = np.zeros_like(cell_embeddings[reference_cell])

# Add all aligned cells
for aligned_cell in aligned_cells:
    average_embedding += aligned_cell

# Divide by number of cells
average_embedding /= len(aligned_cells)

# %%
# Visualize the original embeddings and the average embedding
plt.figure(figsize=(18, 8))

# Create subplots for the first 4 dimensions
for dim in range(4):
    plt.subplot(2, 2, dim + 1)

    # Plot all original cells (transparent)
    for i in range(num_cells):
        # Determine if this cell is in the top 5
        is_top5 = False
        for cell_id, _, _, _ in top_5_cells:
            if i + 1 == cell_id:  # Convert 1-indexed to 0-indexed
                is_top5 = True
                break

        # Style based on cell type
        alpha = 0.3
        color = "gray"
        label = None

        if i == reference_cell:
            alpha = 0.7
            color = "blue"
            label = "Cell 1 (Reference)"
        elif is_top5:
            alpha = 0.5
            color = "green"
            if dim == 0 and i == top_5_cells[0][0] - 1:  # Only label once
                label = "Top 5 Cells"

        plt.plot(
            range(num_timepoints),
            cell_embeddings[i, :, dim],
            alpha=alpha,
            color=color,
            linewidth=1,
            label=label,
        )

    # Plot the average embedding
    plt.plot(
        range(num_timepoints),
        average_embedding[:, dim],
        "r-",
        linewidth=2,
        label="Average Embedding",
    )

    plt.title(f"Dimension {dim + 1}: Original vs Average")
    plt.xlabel("Timepoint")
    plt.ylabel(f"Value (Dim {dim + 1})")
    plt.grid(alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.suptitle(
    "Average Embedding from Top 5 Similar Cells (via DTW)", fontsize=16, y=1.02
)
plt.show()

# %%
# Evaluate the average embedding as a reference
# Compute DTW distances from average to all cells
average_dtw_results = []

for i in range(num_cells):
    # Get distance and path from the average to each cell
    path, dist = warping_path(
        average_embedding,
        cell_embeddings[i],
        include_distance=True,
    )

    # Compute warping matrix for visualization
    warping_matrix, _ = compute_dtw_matrix(average_embedding, cell_embeddings[i])

    average_dtw_results.append((i + 1, dist, path, warping_matrix))
    print(f"DTW distance between Average and Cell {i + 1} dist: {dist:.4f}")

# %%
# Compare distances: Cell 1 as reference vs Average as reference
# Combine the DTW distances for comparison
comparison_data = []

# Add Cell 1 reference distances
for i in range(num_cells):
    if i == reference_cell:
        # Distance to self is 0
        comparison_data.append(
            {
                "Cell ID": i + 1,
                "To Cell 1": 0.0,
                "To Average": average_dtw_results[i][1],
            }
        )
    else:
        # Find the matching result from dtw_results
        for cell_id, dist, _, _ in dtw_results:
            if cell_id == i + 1:
                comparison_data.append(
                    {
                        "Cell ID": i + 1,
                        "To Cell 1": dist,
                        "To Average": average_dtw_results[i][1],
                    }
                )
                break

# Prepare bar chart data
cell_ids = [d["Cell ID"] for d in comparison_data]
to_cell1 = [d["To Cell 1"] for d in comparison_data]
to_average = [d["To Average"] for d in comparison_data]

# Compute some statistics
total_to_cell1 = sum(to_cell1)
total_to_average = sum(to_average)
avg_to_cell1 = total_to_cell1 / len(cell_ids)
avg_to_average = total_to_average / len(cell_ids)

# Create a comparison bar chart
plt.figure(figsize=(12, 6))
x = np.arange(len(cell_ids))
width = 0.35

plt.bar(x - width / 2, to_cell1, width, label="Distance to Cell 1")
plt.bar(x + width / 2, to_average, width, label="Distance to Average")

plt.xlabel("Cell ID")
plt.ylabel("DTW Distance")
plt.title("Comparison: Cell 1 vs Average as Reference")
plt.xticks(x, cell_ids)
plt.legend()

# Add summary as text annotation
plt.figtext(
    0.5,
    0.01,
    f"Total distance - Cell 1: {total_to_cell1:.2f}, Average: {total_to_average:.2f}\n"
    f"Mean distance - Cell 1: {avg_to_cell1:.2f}, Average: {avg_to_average:.2f}",
    ha="center",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8),
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# %%
# Visualize selected warping matrices using the average as reference
# Select a few representative cells (one similar, one different)
similar_cell_idx = 1  # Cell 2
different_cell_idx = 7  # Cell 8

plt.figure(figsize=(16, 7))

# Plot warping matrix for similar cell
plt.subplot(1, 2, 1)
warping_matrix = average_dtw_results[similar_cell_idx - 1][3]
path = average_dtw_results[similar_cell_idx - 1][2]
dist = average_dtw_results[similar_cell_idx - 1][1]

plt.imshow(warping_matrix, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="Accumulated Cost")

# Plot diagonal and path
diagonal = np.linspace(0, num_timepoints - 1, 100)
plt.plot(diagonal, diagonal, "w--", alpha=0.7, linewidth=1, label="Diagonal")

# Extract and plot the best path
path_x = [p[0] for p in path]
path_y = [p[1] for p in path]
plt.plot(path_y, path_x, "r-", linewidth=2, label="Best path")

plt.title(f"Average vs Cell {similar_cell_idx} (Similar, Dist: {dist:.2f})")
plt.xlabel(f"Cell {similar_cell_idx} Timepoints")
plt.ylabel("Average Timepoints")
plt.legend(loc="lower right", fontsize=8)

# Plot warping matrix for different cell
plt.subplot(1, 2, 2)
warping_matrix = average_dtw_results[different_cell_idx - 1][3]
path = average_dtw_results[different_cell_idx - 1][2]
dist = average_dtw_results[different_cell_idx - 1][1]

plt.imshow(warping_matrix, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="Accumulated Cost")

# Plot diagonal and path
plt.plot(diagonal, diagonal, "w--", alpha=0.7, linewidth=1, label="Diagonal")

# Extract and plot the best path
path_x = [p[0] for p in path]
path_y = [p[1] for p in path]
plt.plot(path_y, path_x, "r-", linewidth=2, label="Best path")

plt.title(f"Average vs Cell {different_cell_idx} (Different, Dist: {dist:.2f})")
plt.xlabel(f"Cell {different_cell_idx} Timepoints")
plt.ylabel("Average Timepoints")
plt.legend(loc="lower right", fontsize=8)

plt.suptitle("DTW Warping Matrices: Average as Reference", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
