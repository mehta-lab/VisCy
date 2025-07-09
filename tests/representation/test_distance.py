# %%
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from viscy.representation.evaluation.clustering import (
    compare_time_offset,
    pairwise_distance_matrix,
)


def generate_directional_embeddings_corrected(
    n_timepoints: int = 100,
    embedding_dim: int = 3,
    n_tracks: int = 5,
    movement_type: Literal[
        "smooth", "mild_chaos", "moderate_chaos", "high_chaos"
    ] = "smooth",
    target_direction: np.ndarray = None,
    noise_std: float = 0.05,
    seed: int = 42,
    normalize_method: Literal["zscore", "l2"] | None = "zscore",
) -> xr.Dataset:
    """
    Generate embeddings with multiple chaos levels.

    Parameters
    ----------
    movement_type : str
        - "smooth": Consistent direction and step size
        - "mild_chaos": Slight randomness, similar to smooth
        - "moderate_chaos": Moderate randomness and variability
        - "high_chaos": High randomness and large jumps
    """
    np.random.seed(seed)

    # Default target direction (toward positive x-axis)
    if target_direction is None:
        target_direction = np.zeros(embedding_dim)
        target_direction[0] = 2.0

    # Normalize target direction
    target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-8)

    # Define chaos parameters for each movement type
    chaos_params = {
        "smooth": {
            "random_prob": 0.0,
            "noise_scale": 0.15,
            "jump_prob": 0.0,
            "base_step": 0.12,
            "step_std": 0.15,
        },
        "mild_chaos": {
            "random_prob": 0.1,
            "noise_scale": 0.2,
            "jump_prob": 0.03,
            "exp_scales": [0.15, 0.25],
            "jump_range": (1.5, 2.5),
        },
        "moderate_chaos": {
            "random_prob": 0.25,
            "noise_scale": 0.3,
            "jump_prob": 0.08,
            "exp_scales": [0.12, 0.3, 0.6],
            "jump_range": (2.0, 4.0),
        },
        "high_chaos": {
            "random_prob": 0.4,
            "noise_scale": 0.4,
            "jump_prob": 0.15,
            "exp_scales": [0.1, 0.3, 0.8],
            "jump_range": (3.0, 8.0),
        },
    }

    params = chaos_params[movement_type]

    all_embeddings = []
    all_indices = []
    fov_name = "000000"

    for track_id in range(n_tracks):
        timepoints = np.arange(n_timepoints)
        embeddings = np.zeros((n_timepoints, embedding_dim))
        embeddings[0] = np.random.randn(embedding_dim) * 0.5

        for t in range(1, n_timepoints):
            if movement_type == "smooth":
                # Smooth movement (original logic)
                random_component = (
                    np.random.randn(embedding_dim) * params["noise_scale"]
                )
                direction = target_direction + random_component
                direction = direction / (np.linalg.norm(direction) + 1e-8)

                step_size = params["base_step"] * (
                    1 + np.random.normal(0, params["step_std"])
                )
                step_size = max(0.05, step_size)

            else:
                # Chaotic movement with varying levels
                # Direction logic
                if np.random.random() < params["random_prob"]:
                    direction = np.random.randn(embedding_dim)
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                else:
                    random_component = (
                        np.random.randn(embedding_dim) * params["noise_scale"]
                    )
                    direction = target_direction + random_component
                    direction = direction / (np.linalg.norm(direction) + 1e-8)

                # Step size distribution
                exp_scales = params["exp_scales"]
                if len(exp_scales) == 2:  # mild_chaos
                    if np.random.random() < 0.5:
                        step_size = np.random.exponential(exp_scales[0])
                    else:
                        step_size = np.random.exponential(exp_scales[1])
                else:  # moderate_chaos, high_chaos
                    rand_val = np.random.random()
                    if rand_val < 0.2:
                        step_size = np.random.exponential(exp_scales[0])
                    elif rand_val < 0.5:
                        step_size = np.random.exponential(exp_scales[1])
                    else:
                        step_size = np.random.exponential(exp_scales[2])

                # Large jumps
                if np.random.random() < params["jump_prob"]:
                    step_size *= np.random.uniform(*params["jump_range"])

            # Take step
            step = step_size * direction
            embeddings[t] = embeddings[t - 1] + step
            embeddings[t] += np.random.normal(0, noise_std, embedding_dim)

        # Optional normalization
        if normalize_method == "zscore":
            embeddings = (embeddings - np.mean(embeddings, axis=0)) / (
                np.std(embeddings, axis=0) + 1e-8
            )
        if normalize_method == "l2":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        all_embeddings.append(embeddings)

        # Create indices
        for t in range(n_timepoints):
            all_indices.append(
                {
                    "fov_name": fov_name,
                    "track_id": track_id,
                    "t": timepoints[t],
                    "id": len(all_indices),
                }
            )

    # Combine all tracks
    all_embeddings = np.vstack(all_embeddings)
    ultrack_indices = pd.DataFrame(all_indices)
    index = pd.MultiIndex.from_frame(ultrack_indices)

    dataset_dict = {"features": (("sample", "features"), all_embeddings)}
    dataset = xr.Dataset(dataset_dict, coords={"sample": index}).reset_index("sample")

    return dataset


def analyze_step_sizes_before_and_after_normalization(
    n_tracks: int = 5,
    n_timepoints: int = 100,
    embedding_dim: int = 3,
    target_direction: np.ndarray = None,
    seed: int = 42,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Compare step size distributions before and after normalization.

    This demonstrates how normalization affects the step size magnitudes.
    """
    # Generate datasets with and without normalization
    unnormalized_smooth = generate_directional_embeddings_corrected(
        n_tracks=n_tracks,
        n_timepoints=n_timepoints,
        embedding_dim=embedding_dim,
        movement_type="smooth",
        target_direction=target_direction,
        normalize_method=None,  # Key difference
        seed=seed,
    )

    unnormalized_chaotic = generate_directional_embeddings_corrected(
        n_tracks=n_tracks,
        n_timepoints=n_timepoints,
        embedding_dim=embedding_dim,
        movement_type="mild_chaos",
        target_direction=target_direction,
        normalize_method=None,  # Key difference
        seed=seed,
    )

    normalized_smooth = generate_directional_embeddings_corrected(
        n_tracks=n_tracks,
        n_timepoints=n_timepoints,
        embedding_dim=embedding_dim,
        movement_type="smooth",
        target_direction=target_direction,
        normalize_method=None,
        seed=seed,
    )

    normalized_chaotic = generate_directional_embeddings_corrected(
        n_tracks=n_tracks,
        n_timepoints=n_timepoints,
        embedding_dim=embedding_dim,
        movement_type="mild_chaos",
        target_direction=target_direction,
        normalize_method=None,
        seed=seed,
    )

    # Extract step sizes using the debug function logic
    def extract_step_sizes_simple(dataset):
        all_step_sizes = []
        unique_track_ids = np.unique(dataset["track_id"].values)

        for track_id in unique_track_ids:
            track_mask = dataset["track_id"] == track_id
            track_times = dataset["t"].values[track_mask]
            track_embeddings = dataset["features"].values[track_mask]

            time_order = np.argsort(track_times)
            sorted_embeddings = track_embeddings[time_order]

            if len(sorted_embeddings) > 1:
                steps = np.diff(sorted_embeddings, axis=0)
                step_sizes = np.linalg.norm(steps, axis=1)
                all_step_sizes.extend(step_sizes)

        return np.array(all_step_sizes)

    # Extract step sizes
    smooth_unnorm_steps = extract_step_sizes_simple(unnormalized_smooth)
    chaotic_unnorm_steps = extract_step_sizes_simple(unnormalized_chaotic)
    smooth_norm_steps = extract_step_sizes_simple(normalized_smooth)
    chaotic_norm_steps = extract_step_sizes_simple(normalized_chaotic)

    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Before normalization
    ax1.hist(
        smooth_unnorm_steps,
        bins=50,
        alpha=0.7,
        color="#2ca02c",
        label=f"Smooth (μ={np.mean(smooth_unnorm_steps):.3f}, σ={np.std(smooth_unnorm_steps):.3f})",
    )
    ax1.hist(
        chaotic_unnorm_steps,
        bins=50,
        alpha=0.7,
        color="#d62728",
        label=f"Chaotic (μ={np.mean(chaotic_unnorm_steps):.3f}, σ={np.std(chaotic_unnorm_steps):.3f})",
    )
    ax1.set_title("Before Normalization")
    ax1.set_xlabel("Step Size")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # After normalization
    ax2.hist(
        smooth_norm_steps,
        bins=50,
        alpha=0.7,
        color="#2ca02c",
        label=f"Smooth (μ={np.mean(smooth_norm_steps):.3f}, σ={np.std(smooth_norm_steps):.3f})",
    )
    ax2.hist(
        chaotic_norm_steps,
        bins=50,
        alpha=0.7,
        color="#d62728",
        label=f"Chaotic (μ={np.mean(chaotic_norm_steps):.3f}, σ={np.std(chaotic_norm_steps):.3f})",
    )
    ax2.set_title("After Normalization")
    ax2.set_xlabel("Step Size")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    # Log-scale comparison (before normalization)
    ax3.hist(smooth_unnorm_steps, bins=50, alpha=0.7, color="#2ca02c", label="Smooth")
    ax3.hist(chaotic_unnorm_steps, bins=50, alpha=0.7, color="#d62728", label="Chaotic")
    ax3.set_yscale("log")
    ax3.set_title("Before Normalization (Log Scale)")
    ax3.set_xlabel("Step Size")
    ax3.set_ylabel("Frequency (log)")
    ax3.legend()

    # Coefficient of variation comparison
    cv_smooth_unnorm = np.std(smooth_unnorm_steps) / np.mean(smooth_unnorm_steps)
    cv_chaotic_unnorm = np.std(chaotic_unnorm_steps) / np.mean(chaotic_unnorm_steps)
    cv_smooth_norm = np.std(smooth_norm_steps) / np.mean(smooth_norm_steps)
    cv_chaotic_norm = np.std(chaotic_norm_steps) / np.mean(chaotic_norm_steps)

    categories = [
        "Smooth\n(Unnorm)",
        "Chaotic\n(Unnorm)",
        "Smooth\n(Norm)",
        "Chaotic\n(Norm)",
    ]
    cv_values = [cv_smooth_unnorm, cv_chaotic_unnorm, cv_smooth_norm, cv_chaotic_norm]
    colors = ["#2ca02c", "#d62728", "#2ca02c", "#d62728"]
    alphas = [1.0, 1.0, 0.5, 0.5]

    # Create individual bars with their own alpha values
    bars = []
    for i, (cat, val, color, alpha) in enumerate(
        zip(categories, cv_values, colors, alphas)
    ):
        bar = ax4.bar(cat, val, color=color, alpha=alpha)
        bars.extend(bar)

    ax4.set_ylabel("Coefficient of Variation (σ/μ)")
    ax4.set_title("Step Size Variability Comparison")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)


def compute_msd_pairwise_optimized(
    embedding_dataset: xr.Dataset,
    distance_metric: Literal["euclidean", "cosine"] = "euclidean",
) -> dict[int, list[float]]:
    """
    Compute Mean Squared Displacement using pairwise distance matrix.

    Uses compare_time_offset for efficient diagonal extraction.

    Parameters
    ----------
    embedding_dataset : xr.Dataset
        Dataset containing embeddings and metadata
    distance_metric : Literal["euclidean", "cosine"]
        Distance metric to use

    Returns
    -------
    dict[int, list[float]]
        Dictionary mapping time lag τ to list of squared displacements
    """
    from collections import defaultdict

    unique_tracks_df = (
        embedding_dataset[["fov_name", "track_id"]].to_dataframe().drop_duplicates()
    )

    displacement_per_tau = defaultdict(list)

    for fov_name, track_id in zip(
        unique_tracks_df["fov_name"], unique_tracks_df["track_id"]
    ):
        # Filter data for this track
        track_data = embedding_dataset.where(
            (embedding_dataset["fov_name"] == fov_name)
            & (embedding_dataset["track_id"] == track_id),
            drop=True,
        )

        # Sort by time
        time_order = np.argsort(track_data["t"].values)
        times = track_data["t"].values[time_order]
        track_embeddings = track_data["features"].values[time_order]

        # Compute pairwise distance matrix
        if distance_metric == "euclidean":
            distance_matrix = pairwise_distance_matrix(
                track_embeddings, metric="euclidean"
            )
            distance_matrix = distance_matrix**2  # Square for MSD
        elif distance_metric == "cosine":
            distance_matrix = pairwise_distance_matrix(
                track_embeddings, metric="cosine"
            )
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # Extract displacements using diagonal offsets
        n_timepoints = len(times)
        for time_offset in range(1, n_timepoints):
            diagonal_displacements = compare_time_offset(distance_matrix, time_offset)

            for i, displacement in enumerate(diagonal_displacements):
                tau = int(times[i + time_offset] - times[i])
                displacement_per_tau[tau].append(displacement)

    return dict(displacement_per_tau)


def normalize_msd_by_embedding_variance(
    msd_data_dict: dict[str, dict[int, list[float]]],
    datasets: dict[str, xr.Dataset],
) -> dict[str, dict[int, list[float]]]:
    """
    Normalize MSD values by the embedding variance for each movement type.

    This enables fair comparison between different embedding models or movement types
    by removing scale differences.

    Parameters
    ----------
    msd_data_dict : dict[str, dict[int, list[float]]]
        Dictionary mapping movement type to MSD data
    datasets : dict[str, xr.Dataset]
        Dictionary mapping movement type to dataset (for computing variance)

    Returns
    -------
    dict[str, dict[int, list[float]]]
        Normalized MSD data with same structure as input
    """
    normalized_msd_data = {}

    for movement_type, msd_data in msd_data_dict.items():
        # Calculate embedding variance for this movement type
        embeddings = datasets[movement_type]["features"].values
        embedding_variance = np.var(embeddings)

        print(f"{movement_type}: embedding_variance = {embedding_variance:.4f}")

        # Normalize all MSD values by this variance
        normalized_msd_data[movement_type] = {}
        for tau, displacements in msd_data.items():
            normalized_msd_data[movement_type][tau] = [
                disp / embedding_variance for disp in displacements
            ]

    return normalized_msd_data


def normalize_step_sizes_by_embedding_variance(
    datasets: dict[str, xr.Dataset],
) -> dict[str, dict[str, float]]:
    """
    Normalize step size statistics by embedding variance for fair comparison.

    Parameters
    ----------
    datasets : dict[str, xr.Dataset]
        Dictionary mapping movement type to dataset

    Returns
    -------
    dict[str, dict[str, float]]
        Dictionary with normalized step size statistics
    """
    step_stats = {}

    print("\n=== Step Size Statistics (Normalized by Embedding Variance) ===")
    print("-" * 70)

    for movement_type, dataset in datasets.items():
        # Calculate embedding variance for normalization
        embeddings = dataset["features"].values
        embedding_variance = np.var(embeddings)

        # Extract step sizes
        all_step_sizes = []
        unique_track_ids = np.unique(dataset["track_id"].values)

        for track_id in unique_track_ids:
            track_mask = dataset["track_id"] == track_id
            track_embeddings = dataset["features"].values[track_mask]
            track_times = dataset["t"].values[track_mask]

            # Sort by time and remove duplicates
            time_order = np.argsort(track_times)
            sorted_embeddings = track_embeddings[time_order]
            sorted_times = track_times[time_order]
            unique_times, unique_indices = np.unique(sorted_times, return_index=True)
            final_embeddings = sorted_embeddings[unique_indices]

            if len(final_embeddings) > 1:
                steps = np.diff(final_embeddings, axis=0)
                step_sizes = np.linalg.norm(steps, axis=1)
                all_step_sizes.extend(step_sizes)

        step_sizes = np.array(all_step_sizes)

        # Calculate raw statistics
        raw_mean = np.mean(step_sizes)
        raw_std = np.std(step_sizes)
        raw_cv = raw_std / raw_mean

        # Calculate normalized statistics
        norm_mean = raw_mean / np.sqrt(embedding_variance)
        norm_std = raw_std / np.sqrt(embedding_variance)
        norm_cv = norm_std / norm_mean  # CV remains the same after scaling

        step_stats[movement_type] = {
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "raw_cv": raw_cv,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "norm_cv": norm_cv,
            "embedding_variance": embedding_variance,
            "n_steps": len(step_sizes),
        }

        print(
            f"{movement_type:15} | Raw: μ={raw_mean:.4f}, σ={raw_std:.4f}, CV={raw_cv:.4f}"
        )
        print(f"{'':15} | Norm: μ={norm_mean:.4f}, σ={norm_std:.4f}, CV={norm_cv:.4f}")
        print(f"{'':15} | Var={embedding_variance:.4f}, N={len(step_sizes)}")
        print("-" * 70)

    return step_stats


def plot_msd_comparison(
    msd_data_dict: dict[str, dict[int, list[float]]],
    title: str = "MSD: Smooth vs Chaotic Diffusion (Same Direction)",
    log_scale: bool = True,
    show_power_law_fits: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot MSD curves comparing smooth and chaotic diffusion.

    Parameters
    ----------
    msd_data_dict : dict[str, dict[int, list[float]]]
        Dictionary mapping movement type to MSD data
    title : str
        Plot title
    log_scale : bool
        Whether to use log-log scale
    show_power_law_fits : bool
        Whether to show power law fits

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"smooth": "#2ca02c", "chaotic": "#d62728"}

    for movement_type, msd_data in msd_data_dict.items():
        time_lags = sorted(msd_data.keys())
        msd_means = []
        msd_stds = []

        for tau in time_lags:
            displacements = np.array(msd_data[tau])
            msd_means.append(np.mean(displacements))
            msd_stds.append(np.std(displacements) / np.sqrt(len(displacements)))

        time_lags = np.array(time_lags)
        msd_means = np.array(msd_means)
        msd_stds = np.array(msd_stds)

        # Plot with error bars
        color = colors.get(movement_type, "#1f77b4")
        ax.errorbar(
            time_lags,
            msd_means,
            yerr=msd_stds,
            marker="o",
            label=f"{movement_type.title()} Diffusion",
            color=color,
            capsize=3,
            capthick=1,
            linewidth=2,
        )

        # Fit power law if requested
        if show_power_law_fits and len(time_lags) > 3:
            valid_mask = (time_lags > 0) & (msd_means > 0)
            if np.sum(valid_mask) > 3:
                log_tau = np.log(time_lags[valid_mask])
                log_msd = np.log(msd_means[valid_mask])

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_tau, log_msd
                )

                # Plot fit line
                tau_fit = np.linspace(
                    time_lags[valid_mask][0], time_lags[valid_mask][-1], 50
                )
                msd_fit = np.exp(intercept) * tau_fit**slope

                ax.plot(
                    tau_fit,
                    msd_fit,
                    "--",
                    color=color,
                    alpha=0.7,
                    label=f"{movement_type}: α={slope:.2f} (R²={r_value**2:.3f})",
                )

    ax.set_xlabel("Time Lag (τ)", fontsize=12)
    ax.set_ylabel("Mean Squared Displacement", fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_trajectory_comparison_3d(
    smooth_dataset: xr.Dataset,
    chaotic_dataset: xr.Dataset,
    target_direction: np.ndarray = None,
    title: str = "3D Trajectory Comparison: Smooth vs Chaotic",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot 3D trajectories comparing smooth and chaotic diffusion side by side.

    Parameters
    ----------
    smooth_dataset : xr.Dataset
        Dataset with smooth diffusion trajectories
    chaotic_dataset : xr.Dataset
        Dataset with chaotic diffusion trajectories
    target_direction : np.ndarray
        Target direction vector
    title : str
        Plot title

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and axes objects
    """
    fig = plt.figure(figsize=(16, 7))

    # Default target direction
    if target_direction is None:
        target_direction = np.array([2.0, 0.0, 0.0])

    # Smooth diffusion plot
    ax1 = fig.add_subplot(121, projection="3d")
    plot_single_trajectory_3d(smooth_dataset, ax1, "Smooth Diffusion", target_direction)

    # Chaotic diffusion plot
    ax2 = fig.add_subplot(122, projection="3d")
    plot_single_trajectory_3d(
        chaotic_dataset, ax2, "Chaotic Diffusion", target_direction
    )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_single_trajectory_3d(
    dataset: xr.Dataset,
    ax: plt.Axes,
    subtitle: str,
    target_direction: np.ndarray,
):
    """
    Plot trajectories for a single dataset in 3D.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing trajectories
    ax : plt.Axes
        3D axes object
    subtitle : str
        Subtitle for the plot
    target_direction : np.ndarray
        Target direction vector
    """
    n_tracks = len(np.unique(dataset["track_id"].values))
    colors = plt.cm.tab10(np.linspace(0, 1, n_tracks))

    unique_tracks_df = (
        dataset[["fov_name", "track_id"]].to_dataframe().drop_duplicates()
    )

    for i, (fov_name, track_id) in enumerate(
        zip(unique_tracks_df["fov_name"], unique_tracks_df["track_id"])
    ):
        track_data = dataset.where(
            (dataset["fov_name"] == fov_name) & (dataset["track_id"] == track_id),
            drop=True,
        )

        # Sort by time
        time_order = np.argsort(track_data["t"].values)
        embeddings = track_data["features"].values[time_order]

        x, y, z = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
        color = colors[int(track_id) % len(colors)]

        # Plot trajectory
        ax.plot(x, y, z, "-", color=color, alpha=0.7, linewidth=2)

        # Start and end points
        ax.scatter(
            x[0],
            y[0],
            z[0],
            color=color,
            s=100,
            marker="o",
            edgecolors="black",
            linewidth=1,
        )
        ax.scatter(
            x[-1],
            y[-1],
            z[-1],
            color=color,
            s=150,
            marker="*",
            edgecolors="black",
            linewidth=1,
        )

    # Show target direction arrow
    origin = np.array([0, 0, 0])
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        target_direction[0],
        target_direction[1],
        target_direction[2],
        color="red",
        arrow_length_ratio=0.1,
        linewidth=3,
        label="Target Direction",
    )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.set_title(subtitle)
    ax.legend()


def analyze_step_size_distributions_debug(
    smooth_dataset: xr.Dataset,
    chaotic_dataset: xr.Dataset,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Analyze and plot step size distributions with debugging information.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def extract_step_sizes_simple(dataset, dataset_name):
        """Extract step sizes with simple coordinate access."""
        all_step_sizes = []

        # Get unique track IDs
        unique_track_ids = np.unique(dataset["track_id"].values)

        print(f"\n{dataset_name} Dataset:")
        print(f"Total samples: {len(dataset['track_id'])}")
        print(f"Unique track IDs: {unique_track_ids}")

        for track_id in unique_track_ids:
            # Get all data for this track
            track_mask = dataset["track_id"] == track_id
            track_times = dataset["t"].values[track_mask]
            track_embeddings = dataset["features"].values[track_mask]

            # Sort by time
            time_order = np.argsort(track_times)
            sorted_embeddings = track_embeddings[time_order]
            sorted_times = track_times[time_order]

            # Remove duplicates in time (this might be the issue)
            unique_times, unique_indices = np.unique(sorted_times, return_index=True)
            final_embeddings = sorted_embeddings[unique_indices]

            print(
                f"Track {track_id}: {len(sorted_times)} total, {len(unique_times)} unique timepoints"
            )

            # Calculate step sizes
            if len(final_embeddings) > 1:
                steps = np.diff(final_embeddings, axis=0)
                step_sizes = np.linalg.norm(steps, axis=1)
                all_step_sizes.extend(step_sizes)
                print(f"Track {track_id}: {len(step_sizes)} steps")

        print(f"Total steps in {dataset_name}: {len(all_step_sizes)}")
        return np.array(all_step_sizes)

    # Extract step sizes with debug info
    smooth_steps = extract_step_sizes_simple(smooth_dataset, "Smooth")
    chaotic_steps = extract_step_sizes_simple(chaotic_dataset, "Chaotic")

    # Plot histograms
    ax1.hist(
        smooth_steps,
        bins=50,
        alpha=0.7,
        color="#2ca02c",
        label=f"Smooth (n={len(smooth_steps)}, μ={np.mean(smooth_steps):.3f}, σ={np.std(smooth_steps):.3f})",
    )
    ax1.hist(
        chaotic_steps,
        bins=50,
        alpha=0.7,
        color="#d62728",
        label=f"Chaotic (n={len(chaotic_steps)}, μ={np.mean(chaotic_steps):.3f}, σ={np.std(chaotic_steps):.3f})",
    )
    ax1.set_xlabel("Step Size")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Step Size Distribution")
    ax1.legend()

    # Plot coefficient of variation
    cv_smooth = np.std(smooth_steps) / np.mean(smooth_steps)
    cv_chaotic = np.std(chaotic_steps) / np.mean(chaotic_steps)

    ax2.bar(
        ["Smooth", "Chaotic"],
        [cv_smooth, cv_chaotic],
        color=["#2ca02c", "#d62728"],
        alpha=0.7,
    )
    ax2.set_ylabel("Coefficient of Variation (σ/μ)")
    ax2.set_title("Step Size Variability")

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_trajectory_comparison_3d_multi(
    datasets: dict[str, xr.Dataset],
    target_direction: np.ndarray = None,
    title: str = "3D Trajectory Comparison: Multiple Movement Types",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot 3D trajectories for multiple movement types.

    Parameters
    ----------
    datasets : dict[str, xr.Dataset]
        Dictionary mapping movement type name to dataset
    target_direction : np.ndarray
        Target direction vector
    title : str
        Plot title
    """
    n_types = len(datasets)
    cols = 2
    rows = (n_types + 1) // 2

    fig = plt.figure(figsize=(12, 6 * rows))

    # Default target direction
    if target_direction is None:
        target_direction = np.array([2.0, 0.0, 0.0])

    axes = []
    for i, (movement_type, dataset) in enumerate(datasets.items()):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        plot_single_trajectory_3d(
            dataset,
            ax,
            f"{movement_type.replace('_', ' ').title()} Movement",
            target_direction,
        )
        axes.append(ax)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig, axes


def plot_msd_comparison_multi(
    msd_data_dict: dict[str, dict[int, list[float]]],
    title: str = "MSD: Multiple Movement Types Comparison",
    log_scale: bool = True,
    show_power_law_fits: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot MSD curves for multiple movement types.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette for different movement types
    colors = {
        "smooth": "#2ca02c",
        "mild_chaos": "#ff7f0e",
        "moderate_chaos": "#d62728",
        "high_chaos": "#9467bd",
    }

    for movement_type, msd_data in msd_data_dict.items():
        time_lags = sorted(msd_data.keys())
        msd_means = []
        msd_stds = []

        for tau in time_lags:
            displacements = np.array(msd_data[tau])
            msd_means.append(np.mean(displacements))
            msd_stds.append(np.std(displacements) / np.sqrt(len(displacements)))

        time_lags = np.array(time_lags)
        msd_means = np.array(msd_means)
        msd_stds = np.array(msd_stds)

        # Plot with error bars
        color = colors.get(movement_type, "#1f77b4")
        ax.errorbar(
            time_lags,
            msd_means,
            yerr=msd_stds,
            marker="o",
            label=f"{movement_type.replace('_', ' ').title()}",
            color=color,
            capsize=3,
            capthick=1,
            linewidth=2,
            markersize=6,
        )

        # Fit power law if requested
        if show_power_law_fits and len(time_lags) > 3:
            valid_mask = (time_lags > 0) & (msd_means > 0)
            if np.sum(valid_mask) > 3:
                log_tau = np.log(time_lags[valid_mask])
                log_msd = np.log(msd_means[valid_mask])

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_tau, log_msd
                )

                # Plot fit line
                tau_fit = np.linspace(
                    time_lags[valid_mask][0], time_lags[valid_mask][-1], 50
                )
                msd_fit = np.exp(intercept) * tau_fit**slope

                ax.plot(
                    tau_fit,
                    msd_fit,
                    "--",
                    color=color,
                    alpha=0.7,
                    label=f"{movement_type}: α={slope:.2f} (R²={r_value**2:.3f})",
                )

    ax.set_xlabel("Time Lag (τ)", fontsize=12)
    ax.set_ylabel("Mean Squared Displacement", fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig, ax


def analyze_step_size_distributions_multi(
    datasets: dict[str, xr.Dataset],
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Analyze step size distributions for multiple movement types.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = {
        "smooth": "#2ca02c",
        "mild_chaos": "#ff7f0e",
        "moderate_chaos": "#d62728",
        "high_chaos": "#9467bd",
    }

    def extract_step_sizes_simple(dataset):
        """Extract step sizes with simple coordinate access."""
        all_step_sizes = []
        unique_track_ids = np.unique(dataset["track_id"].values)

        for track_id in unique_track_ids:
            track_mask = dataset["track_id"] == track_id
            track_times = dataset["t"].values[track_mask]
            track_embeddings = dataset["features"].values[track_mask]

            time_order = np.argsort(track_times)
            sorted_embeddings = track_embeddings[time_order]
            sorted_times = track_times[time_order]

            # Remove duplicates in time
            unique_times, unique_indices = np.unique(sorted_times, return_index=True)
            final_embeddings = sorted_embeddings[unique_indices]

            if len(final_embeddings) > 1:
                steps = np.diff(final_embeddings, axis=0)
                step_sizes = np.linalg.norm(steps, axis=1)
                all_step_sizes.extend(step_sizes)

        return np.array(all_step_sizes)

    # Extract step sizes for all datasets
    all_step_data = {}
    cv_values = []
    labels = []

    for movement_type, dataset in datasets.items():
        steps = extract_step_sizes_simple(dataset)
        all_step_data[movement_type] = steps

        # Calculate coefficient of variation
        cv = np.std(steps) / np.mean(steps)
        cv_values.append(cv)
        labels.append(movement_type.replace("_", " ").title())

    # Plot histograms
    for movement_type, steps in all_step_data.items():
        color = colors.get(movement_type, "#1f77b4")
        ax1.hist(
            steps,
            bins=50,
            alpha=0.7,
            color=color,
            label=f"{movement_type.replace('_', ' ').title()} (n={len(steps)}, μ={np.mean(steps):.3f}, σ={np.std(steps):.3f})",
        )

    ax1.set_xlabel("Step Size")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Step Size Distributions")
    ax1.legend()

    # Plot coefficient of variation
    bar_colors = [
        colors.get(movement_type, "#1f77b4") for movement_type in datasets.keys()
    ]
    bars = ax2.bar(labels, cv_values, color=bar_colors, alpha=0.7)
    ax2.set_ylabel("Coefficient of Variation (σ/μ)")
    ax2.set_title("Step Size Variability")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig, (ax1, ax2)


# %%
if __name__ == "__main__":
    # Note: direction of the embedding to simulate movement/infection.
    target_direction = np.array([10.0, 0, 0.0])

    movement_types = ["smooth", "mild_chaos", "moderate_chaos", "high_chaos"]

    datasets = {}
    print("=== Generating Datasets ===")
    for movement_type in movement_types:
        print(f"Generating {movement_type} dataset...")
        datasets[movement_type] = generate_directional_embeddings_corrected(
            n_tracks=5,
            n_timepoints=100,
            movement_type=movement_type,
            target_direction=target_direction,
            normalize_method=None,
            seed=42,
        )

    print("=== Computing MSD for All Movement Types ===")
    msd_data_dict = {}
    for movement_type, dataset in datasets.items():
        print(f"Computing MSD for {movement_type}...")
        msd_data_dict[movement_type] = compute_msd_pairwise_optimized(dataset)

    print("\n=== Normalizing MSD by Embedding Variance ===")
    normalized_msd_data_dict = normalize_msd_by_embedding_variance(
        msd_data_dict, datasets
    )

    print("=== MSD vs Time Plot (Raw) ===")
    fig_msd_raw, ax_msd_raw = plot_msd_comparison_multi(
        msd_data_dict, title="MSD: Raw Values (All Movement Types)"
    )
    plt.show()

    print("=== MSD vs Time Plot (Normalized by Embedding Variance) ===")
    fig_msd_norm, ax_msd_norm = plot_msd_comparison_multi(
        normalized_msd_data_dict,
        title="MSD: Normalized by Embedding Variance (All Movement Types)",
    )
    plt.show()

    print("=== 3D Trajectory Comparison (All Types) ===")
    fig_3d, axes_3d = plot_trajectory_comparison_3d_multi(datasets, target_direction)
    plt.show()

    print("=== Step Size Distribution Analysis (All Types) ===")
    fig_step, (ax_step1, ax_step2) = analyze_step_size_distributions_multi(datasets)
    plt.show()

    print("=== Step Size Normalization Analysis ===")
    step_stats = normalize_step_sizes_by_embedding_variance(datasets)

    print("=== Summary Statistics ===")
    for movement_type, dataset in datasets.items():
        print(f"\n{movement_type.replace('_', ' ').title()} Movement:")
        print(f"  Dataset shape: {dataset.dims}")
        print(f"  Total samples: {len(dataset.sample)}")

        # Calculate mean step size and CV
        def get_step_stats(dataset):
            all_step_sizes = []
            unique_track_ids = np.unique(dataset["track_id"].values)
            for track_id in unique_track_ids:
                track_mask = dataset["track_id"] == track_id
                track_embeddings = dataset["features"].values[track_mask]
                if len(track_embeddings) > 1:
                    steps = np.diff(track_embeddings, axis=0)
                    step_sizes = np.linalg.norm(steps, axis=1)
                    all_step_sizes.extend(step_sizes)
            return np.array(all_step_sizes)

        steps = get_step_stats(dataset)
        mean_step = np.mean(steps)
        std_step = np.std(steps)
        cv = std_step / mean_step

        print(f"  Mean step size: {mean_step:.4f}")
        print(f"  Step size std: {std_step:.4f}")
        print(f"  Coefficient of variation: {cv:.4f}")


# %%
