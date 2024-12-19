# %%
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.clustering import (
    compare_time_offset,
    cross_dissimilarity,
    rank_nearest_neighbors,
    select_block,
)
import numpy as np
from tqdm import tqdm
import pandas as pd


plt.style.use("../evaluation/figure.mplstyle")

# plotting
VERBOSE = False


def compute_piece_wise_dissimilarity(
    features_df: pd.DataFrame, cross_dist: NDArray, rank_fractions: NDArray
):
    """
    Computing the smoothness and dynamic range
    - Get the off diagonal per block and compute the mode
    - The blocks are not square, so we need to get the off diagonal elements
    - Get the 1 and 99 percentile of the off diagonal per block
    """
    piece_wise_dissimilarity_per_track = []
    piece_wise_rank_difference_per_track = []
    for name, subdata in features_df.groupby(["fov_name", "track_id"]):
        if len(subdata) > 1:
            indices = subdata.index.values
            single_track_dissimilarity = select_block(cross_dist, indices)
            single_track_rank_fraction = select_block(rank_fractions, indices)
            piece_wise_dissimilarity = compare_time_offset(
                single_track_dissimilarity, time_offset=1
            )
            piece_wise_rank_difference = compare_time_offset(
                single_track_rank_fraction, time_offset=1
            )
            piece_wise_dissimilarity_per_track.append(piece_wise_dissimilarity)
            piece_wise_rank_difference_per_track.append(piece_wise_rank_difference)
    return piece_wise_dissimilarity_per_track, piece_wise_rank_difference_per_track


def plot_histogram(
    data, title, xlabel, ylabel, color="blue", alpha=0.5, stat="frequency"
):
    plt.figure()
    plt.title(title)
    sns.histplot(data, bins=30, kde=True, color=color, alpha=alpha, stat=stat)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# %%
PATH_TO_GDRIVE_FIGUE = "/home/eduardo.hirata/mydata/gdrive/publications/learning_impacts_of_infection/fig_manuscript/rev2_ICLR_fig/"

prediction_path_1 = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_98ckpt_rev6_2.zarr"
)
prediction_path_2 = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev5_sensorPhase_infection/2chan_160patch_97ckpt_rev5_2.zarr"
)

for prediction_path, loss_name in tqdm(
    [(prediction_path_1, "ntxent"), (prediction_path_2, "triplet")]
):

    # Read the dataset
    embeddings = read_embedding_dataset(prediction_path)
    features = embeddings["features"]

    scaled_features = StandardScaler().fit_transform(features.values)
    # Compute the cosine dissimilarity
    cross_dist = cross_dissimilarity(scaled_features, metric="cosine")
    rank_fractions = rank_nearest_neighbors(cross_dist, normalize=True)

    plt.figure()
    plt.imshow(cross_dist)
    plt.show()

    # Compute piece-wise dissimilarity and rank difference
    features_df = features["sample"].to_dataframe().reset_index(drop=True)
    piece_wise_dissimilarity_per_track, piece_wise_rank_difference_per_track = (
        compute_piece_wise_dissimilarity(features_df, cross_dist, rank_fractions)
    )

    all_dissimilarity = np.concatenate(piece_wise_dissimilarity_per_track)

    # # Get the median/mode of the off diagonal elements
    # median_piece_wise_dissimilarity = np.array(
    #     [np.median(track) for track in piece_wise_dissimilarity_per_track]
    # )
    p99_piece_wise_dissimilarity = np.array(
        [np.percentile(track, 99) for track in piece_wise_dissimilarity_per_track]
    )
    p1_percentile_piece_wise_dissimilarity = np.array(
        [np.percentile(track, 1) for track in piece_wise_dissimilarity_per_track]
    )

    # Random sampling values in the dissimilarity matrix
    n_samples = 3000
    random_indices = np.random.randint(0, len(cross_dist), size=(n_samples, 2))
    sampled_values = cross_dist[random_indices[:, 0], random_indices[:, 1]]

    print(f"Dissimilarity Statistics for {prediction_path.stem}")
    print(f"Mean: {np.mean(all_dissimilarity)}")
    print(f"Std: {np.std(all_dissimilarity)}")
    print(f"Median: {np.median(all_dissimilarity)}")

    print(f"Distance Statistics for random sampling")
    print(f"Mean: {np.mean(sampled_values)}")
    print(f"Std: {np.std(sampled_values)}")
    print(f"Median: {np.median(sampled_values)}")

    if VERBOSE:
        # Plot histograms
        # plot_histogram(
        #     median_piece_wise_dissimilarity,
        #     "Adjacent Frame Median Dissimilarity per Track",
        #     "Cosine Dissimilarity",
        #     "Frequency",
        # )
        # plot_histogram(
        #     p1_percentile_piece_wise_dissimilarity,
        #     "Adjacent Frame 1 Percentile Dissimilarity per Track",
        #     "Cosine Dissimilarity",
        #     "Frequency",
        # )
        # plot_histogram(
        #     p99_piece_wise_dissimilarity,
        #     "Adjacent Frame 99 Percentile Dissimilarity per Track",
        #     "Cosine Dissimilarity",
        #     "Frequency",
        # )

        plot_histogram(
            piece_wise_dissimilarity_per_track,
            "Adjacent Frame Dissimilarity per Track",
            "Cosine Dissimilarity",
            "Frequency",
        )
        # Plot the histogram of the sampled values
        plot_histogram(
            sampled_values,
            "Random Sampling of Dissimilarity Values",
            "Cosine Dissimilarity",
            "Density",
            color="red",
            alpha=0.5,
            stat="density",
        )

    # Plot the median and the random sampling in one plot each with different colors
    fig = plt.figure()
    sns.histplot(
        all_dissimilarity,
        bins=30,
        kde=True,
        color="cyan",
        alpha=0.5,
        stat="density",
    )
    sns.histplot(
        sampled_values, bins=30, kde=True, color="red", alpha=0.5, stat="density"
    )
    plt.xlabel("Cosine Dissimilarity")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.legend(["Adjacent Frame", "Random Sample"])
    plt.show()
    fig.savefig(
        f"{PATH_TO_GDRIVE_FIGUE}/cosine_dissimilarity_smoothness_{prediction_path.stem}_{loss_name}.pdf",
        dpi=600,
    )

# %%
