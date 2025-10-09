# %%
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.clustering import (
    compare_time_offset,
    pairwise_distance_matrix,
    rank_nearest_neighbors,
    select_block,
)

# %%
prediction_path = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_7mins.zarr"
)

embeddings = read_embedding_dataset(prediction_path)
features = embeddings["features"]

# %%
scaled_features = StandardScaler().fit_transform(features.values)

# %%
cross_dist = pairwise_distance_matrix(scaled_features, metric="cosine")
rank_fractions = rank_nearest_neighbors(cross_dist, normalize=True)

# %%
# select a single track in a single fov
fov = "/0/0/0"
fov_idx = (features["fov_name"] == fov).values

track_id = 1
track_idx = (features["track_id"] == track_id).values

fov_and_track_idx = fov_idx & track_idx

single_track_dissimilarity = select_block(cross_dist, fov_and_track_idx)
single_track_rank_fraction = select_block(rank_fractions, fov_and_track_idx)

piece_wise_dissimilarity = compare_time_offset(
    single_track_dissimilarity, time_offset=1
)
piece_wise_rank_difference = compare_time_offset(
    single_track_rank_fraction, time_offset=1
)

# %%
f = plt.figure(figsize=(8, 12))
f.suptitle(f"Track {track_id} in FOV {fov}")
subfigs = f.subfigures(2, 1, height_ratios=[1, 2])

umap = subfigs[0].subplots(1, 1)
single_cell_features = features.sel(fov_name=fov, track_id=track_id).sortby("t")
sns.lineplot(
    x=single_cell_features["UMAP1"],
    y=single_cell_features["UMAP2"],
    ax=umap,
    color="k",
    alpha=0.5,
)
sns.scatterplot(
    x=features["UMAP1"], y=features["UMAP2"], ax=umap, color="k", s=100, alpha=0.01
)
sns.scatterplot(
    x=single_cell_features["UMAP1"],
    y=single_cell_features["UMAP2"],
    hue=single_cell_features["t"],
    ax=umap,
    palette="RdYlGn",
)

f1 = subfigs[1]
ax = f1.subplots(2, 2)

sns.heatmap(single_track_dissimilarity, ax=ax[0, 0], square=True)
ax[0, 0].set_title("Cosine dissimilarity")
ax[0, 0].set_xlabel("Frame")
ax[0, 0].set_ylabel("Frame")

sns.heatmap(single_track_rank_fraction, ax=ax[0, 1], square=True)
ax[0, 1].set_title("Column-wise normalized neighborhood distance")
ax[0, 1].set_xlabel("Frame")
ax[0, 1].set_ylabel("Frame")

sns.lineplot(piece_wise_dissimilarity, ax=ax[1, 0])
ax[1, 0].set_title("$1 - \cos{(t_i, t_{i+1})}$")
ax[1, 0].set_xlabel("Frame")
ax[1, 0].set_ylabel("Cosine dissimilarity")

sns.lineplot(piece_wise_rank_difference, ax=ax[1, 1])
ax[1, 1].set_title("Nearest neighbor fraction difference")
ax[1, 1].set_xlabel("Frame")
ax[1, 1].set_ylabel("Rank fraction")

# %%
