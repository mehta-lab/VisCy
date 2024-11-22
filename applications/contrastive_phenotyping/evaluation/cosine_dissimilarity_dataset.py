# %%
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.clustering import (
    compare_time_offset,
    cross_dissimilarity,
    rank_nearest_neighbors,
    select_block,
)
import numpy as np

plt.style.use("../evaluation/figure.mplstyle")


# %%
prediction_path = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_7mins.zarr"
)

embeddings = read_embedding_dataset(prediction_path)
features = embeddings["features"]

# %%
scaled_features = StandardScaler().fit_transform(features.values)

# %%
cross_dist = cross_dissimilarity(scaled_features, metric="cosine")
rank_fractions = rank_nearest_neighbors(cross_dist, normalize=True)

plt.imshow(cross_dist)

# %%
""" 
Computing the smoothness and dynamic range
- Get the off diagonal per block and compute the mode
- The blocks are not square, so we need to get the off diagonal elements
- Get the 1 and 99 percentile of the off diagonal per block
"""
features_df = features["sample"].to_dataframe().reset_index(drop=True)

piece_wise_dissimilarity_per_track = []
piece_wise_rank_difference_per_track = []
for name, subdata in features_df.groupby(["fov_name", "track_id"]):
    if len(subdata) > 1:
        single_track_dissimilarity = select_block(cross_dist, subdata.index.values)
        single_track_rank_fraction = select_block(rank_fractions, subdata.index.values)
        piece_wise_dissimilarity = compare_time_offset(
            single_track_dissimilarity, time_offset=1
        )
        piece_wise_rank_difference = compare_time_offset(
            single_track_rank_fraction, time_offset=1
        )
        piece_wise_dissimilarity_per_track.append(piece_wise_dissimilarity)
        piece_wise_rank_difference_per_track.append(piece_wise_rank_difference)

# Get the median/mode of the off diagonal elements
median_piece_wise_dissimilarity = [
    np.median(track) for track in piece_wise_dissimilarity_per_track
]
p99_piece_wise_dissimilarity = [
    np.percentile(track, 99) for track in piece_wise_dissimilarity_per_track
]
p1_percentile_piece_wise_dissimilarity = [
    np.percentile(track, 1) for track in piece_wise_dissimilarity_per_track
]

# Plot the histogram of the median dissimilarity
plt.figure()
plt.title("Adjacent Frame Median Dissimilarity per Track")
sns.histplot(median_piece_wise_dissimilarity, bins=30, kde=True)
plt.xlabel("Cosine Dissimilarity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot the histogram of the 1 percentile dissimilarity
plt.figure()
plt.title("Adjacent Frame 1 Percentile Dissimilarity per Track")
sns.histplot(p1_percentile_piece_wise_dissimilarity, bins=30, kde=True)
plt.xlabel("Cosine Dissimilarity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot the histogram of the 99 percentile dissimilarity
plt.figure()
plt.title("Adjacent Frame 99 Percentile Dissimilarity per Track")
sns.histplot(p99_piece_wise_dissimilarity, bins=30, kde=True)
plt.xlabel("Cosine Dissimilarity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# %%
# Random sampling values in the values in the dissimilarity matrix
# Random Sampling
n_samples = 2000
sampled_values = []
for _ in range(n_samples):
    i, j = np.random.randint(0, len(cross_dist), size=2)
    sampled_values.append(cross_dist[i, j])


# Plot the histogram of the sampled values
plt.figure()
plt.title("Random Sampling of Dissimilarity Values")
sns.histplot(sampled_values, bins=30, kde=True, stat="density")
plt.xlabel("Cosine Dissimilarity")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# %%
# Plot the median and the random sampling in one plot each with different colors
plt.figure()
sns.histplot(
    median_piece_wise_dissimilarity,
    bins=30,
    kde=True,
    color="cyan",
    alpha=0.5,
    stat="density",
)
sns.histplot(sampled_values, bins=30, kde=True, color="red", alpha=0.5, stat="density")
plt.xlabel("Cosine Dissimilarity")
plt.ylabel("Density")
plt.tight_layout()
plt.legend(["Adjacent Frame", "Random Sample"])
plt.show()

# %%
