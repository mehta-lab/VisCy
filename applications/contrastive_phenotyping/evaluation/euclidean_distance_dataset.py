# %%
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.clustering import (
    compare_time_offset,
    pairwise_distance_matrix,
    rank_nearest_neighbors,
    select_block,
)
import numpy as np
from tqdm import tqdm
import pandas as pd

from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar


plt.style.use("../evaluation/figure.mplstyle")


def compute_piece_wise_distance(
    features_df: pd.DataFrame, cross_dist: NDArray, rank_fractions: NDArray
):
    """
    Computing the smoothness and dynamic range
    - Get the off diagonal per block and compute the mode
    - The blocks are not square, so we need to get the off diagonal elements
    - Get the 1 and 99 percentile of the off diagonal per block
    """
    piece_wise_distance_per_track = []
    piece_wise_rank_difference_per_track = []
    for name, subdata in features_df.groupby(["fov_name", "track_id"]):
        if len(subdata) > 1:
            indices = subdata.index.values
            single_track_distance = select_block(cross_dist, indices)
            single_track_rank_fraction = select_block(rank_fractions, indices)
            piece_wise_distance = compare_time_offset(
                single_track_distance, time_offset=1
            )
            piece_wise_rank_difference = compare_time_offset(
                single_track_rank_fraction, time_offset=1
            )
            piece_wise_distance_per_track.append(piece_wise_distance)
            piece_wise_rank_difference_per_track.append(piece_wise_rank_difference)
    return piece_wise_distance_per_track, piece_wise_rank_difference_per_track


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


def find_distribution_peak(data: np.ndarray) -> float:
    """
    Find the peak (mode) of a distribution using kernel density estimation.

    Args:
        data: Array of values to find the peak for

    Returns:
        float: The x-value where the peak occurs
    """
    kde = gaussian_kde(data)
    # Find the peak (maximum) of the KDE
    result = minimize_scalar(
        lambda x: -kde(x), bounds=(np.min(data), np.max(data)), method="bounded"
    )
    return result.x


def analyze_embedding_smoothness(
    prediction_path: Path,
    verbose: bool = False,
    output_path: Optional[str] = None,
    loss_name: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """
    Analyze the smoothness and dynamic range of embeddings using Euclidean distance.

    Args:
        prediction_path: Path to the embedding dataset
        verbose: If True, generates additional plots
        output_path: Path to save the final plot (optional)
        loss_name: Name of the loss function used (optional)
        overwrite: If True, overwrites existing files. If False, raises error if file exists (default: False)

    Returns:
        dict: Dictionary containing metrics including:
            - distance_mean: Mean of adjacent frame distance
            - distance_std: Standard deviation of adjacent frame distance
            - distance_median: Median of adjacent frame distance
            - distance_peak: Peak of adjacent frame distribution
            - distance_p99: 99th percentile of adjacent frame distance
            - distance_p1: 1st percentile of adjacent frame distance
            - distance_distribution: Full distribution of adjacent frame distances
            - random_mean: Mean of random sampling distance
            - random_std: Standard deviation of random sampling distance
            - random_median: Median of random sampling distance
            - random_peak: Peak of random sampling distribution
            - random_distribution: Full distribution of random sampling distances
            - dynamic_range: Difference between random and adjacent peaks
    """
    # Read the dataset
    embeddings = read_embedding_dataset(prediction_path)
    features = embeddings["features"]

    # Compute the Euclidean distance
    cross_dist = pairwise_distance_matrix(features, metric="euclidean")
    rank_fractions = rank_nearest_neighbors(cross_dist, normalize=True)

    # Compute piece-wise distance and rank difference
    features_df = features["sample"].to_dataframe().reset_index(drop=True)
    piece_wise_distance_per_track, piece_wise_rank_difference_per_track = (
        compute_piece_wise_distance(features_df, cross_dist, rank_fractions)
    )

    all_distance = np.concatenate(piece_wise_distance_per_track)

    p99_piece_wise_distance = np.array(
        [np.percentile(track, 99) for track in piece_wise_distance_per_track]
    )
    p1_percentile_piece_wise_distance = np.array(
        [np.percentile(track, 1) for track in piece_wise_distance_per_track]
    )

    # Random sampling values in the distance matrix with same size as adjacent frame measurements
    n_samples = len(all_distance)
    random_indices = np.random.randint(0, len(cross_dist), size=(n_samples, 2))
    sampled_values = cross_dist[random_indices[:, 0], random_indices[:, 1]]

    # Compute the peaks of both distributions using KDE
    adjacent_peak = float(find_distribution_peak(all_distance))
    random_peak = float(find_distribution_peak(sampled_values))
    dynamic_range = float(random_peak - adjacent_peak)

    metrics = {
        "distance_mean": float(np.mean(all_distance)),
        "distance_std": float(np.std(all_distance)),
        "distance_median": float(np.median(all_distance)),
        "distance_peak": adjacent_peak,
        "distance_p99": p99_piece_wise_distance,
        "distance_p1": p1_percentile_piece_wise_distance,
        "distance_distribution": all_distance,
        "random_mean": float(np.mean(sampled_values)),
        "random_std": float(np.std(sampled_values)),
        "random_median": float(np.median(sampled_values)),
        "random_peak": random_peak,
        "random_distribution": sampled_values,
        "dynamic_range": dynamic_range,
    }

    if verbose:
        # Plot the comparison histogram and save if output_path is provided
        fig = plt.figure()
        sns.histplot(
            metrics["distance_distribution"],
            bins=30,
            kde=True,
            color="cyan",
            alpha=0.5,
            stat="density",
        )
        sns.histplot(
            metrics["random_distribution"],
            bins=30,
            kde=True,
            color="red",
            alpha=0.5,
            stat="density",
        )
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Density")
        # Add vertical lines for the peaks
        plt.axvline(x=metrics["distance_peak"], color="cyan", linestyle="--", alpha=0.8)
        plt.axvline(x=metrics["random_peak"], color="red", linestyle="--", alpha=0.8)
        plt.tight_layout()
        plt.legend(["Adjacent Frame", "Random Sample", "Adjacent Peak", "Random Peak"])

        if output_path and loss_name:
            output_file = Path(
                f"{output_path}/euclidean_distance_smoothness_{prediction_path.stem}_{loss_name}.pdf"
            )
            if output_file.exists() and not overwrite:
                raise FileExistsError(
                    f"File {output_file} already exists and overwrite=False"
                )
            fig.savefig(
                output_file,
                dpi=600,
            )
        plt.show()

    return metrics


# Example usage:
if __name__ == "__main__":
    # plotting
    VERBOSE = True
    PATH_TO_GDRIVE_FIGUE = "./"

    # Define models as a dictionary with meaningful keys
    prediction_paths = {
        "ntxent_sensor_phase": Path(
            "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_98ckpt_rev6_2.zarr"
        ),
        "triplet_sensor_phase": Path(
            "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev5_sensorPhase_infection/2chan_160patch_97ckpt_rev5_2.zarr"
        ),
    }

    # output_folder to save the distributions as .csv
    output_folder = Path(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/euclidean_distance_distributions"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    # Evaluate each model
    for model_name, prediction_path in tqdm(
        prediction_paths.items(), desc="Evaluating models"
    ):
        print(f"\nAnalyzing model: {prediction_path.stem} (Loss: {model_name})")
        print("-" * 80)

        metrics = analyze_embedding_smoothness(
            prediction_path,
            verbose=VERBOSE,
            output_path=PATH_TO_GDRIVE_FIGUE,
            loss_name=model_name,
            overwrite=True,
        )

        # Save distributions to CSV
        distributions_df = pd.DataFrame(
            {
                "adjacent_frame": pd.Series(metrics["distance_distribution"]),
                "random_sampling": pd.Series(metrics["random_distribution"]),
            }
        )
        csv_path = (
            output_folder / f"{prediction_path.stem}_{model_name}_distributions.csv"
        )
        distributions_df.to_csv(csv_path, index=False)

        # Print statistics (existing code)
        print("\nAdjacent Frame Distance Statistics:")
        print(f"{'Mean:':<15} {metrics['distance_mean']:.3f}")
        print(f"{'Std:':<15} {metrics['distance_std']:.3f}")
        print(f"{'Median:':<15} {metrics['distance_median']:.3f}")
        print(f"{'Peak:':<15} {metrics['distance_peak']:.3f}")
        print(f"{'P1:':<15} {np.mean(metrics['distance_p1']):.3f}")
        print(f"{'P99:':<15} {np.mean(metrics['distance_p99']):.3f}")

        # Print random sampling statistics
        print("\nRandom Sampling Statistics:")
        print(f"{'Mean:':<15} {metrics['random_mean']:.3f}")
        print(f"{'Std:':<15} {metrics['random_std']:.3f}")
        print(f"{'Median:':<15} {metrics['random_median']:.3f}")
        print(f"{'Peak:':<15} {metrics['random_peak']:.3f}")

        # Print dynamic range
        print("\nComparison Metrics:")
        print(f"{'Dynamic Range:':<15} {metrics['dynamic_range']:.3f}")

        # Print distribution sizes
        print("\nDistribution Sizes:")
        print(
            f"{'Adjacent Frame:':<15} {len(metrics['distance_distribution']):,d} samples"
        )
        print(f"{'Random:':<15} {len(metrics['random_distribution']):,d} samples")

# %%
