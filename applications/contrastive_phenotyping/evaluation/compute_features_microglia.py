# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dimensionality_reduction import compute_pca
from viscy.representation.evaluation.feature import CellFeatures, DynamicFeatures
from viscy.representation.evaluation.utils import preload_images
from viscy.transforms import NormalizeSampled

# %%


def plot_track_velocities(
    track_dynamics: pd.DataFrame,
    fov: str,
    max_tracks: int = 10,
    figsize: tuple = (12, 6),
):
    """Plot velocities over time for tracks in a specific FOV.

    Args:
        track_dynamics: DataFrame containing track features including instantaneous_velocity
        fov: Name of the FOV to plot
        max_tracks: Maximum number of tracks to plot (default: 10)
        figsize: Figure size as (width, height) tuple (default: (12, 6))
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get data for this FOV
    fov_data = track_dynamics[track_dynamics["fov_name"] == fov]

    if len(fov_data) == 0:
        logger.warning(f"No data found for FOV: {fov}")
        return fig

    # Get unique track IDs for this FOV and limit to max_tracks
    track_ids = fov_data["track_id"].unique()[:max_tracks]

    # Plot each track
    for track_id in track_ids:
        track_data = fov_data[fov_data["track_id"] == track_id]
        velocities = track_data["instantaneous_velocity"].iloc[
            0
        ]  # Get the velocity array
        times = np.arange(len(velocities))  # Create time points
        ax.plot(times, velocities, label=f"Track {track_id}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Cell Velocities Over Time - FOV: {fov}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_track_dynamics(
    track_dynamics: pd.DataFrame, fov: str = None, figsize: tuple = (15, 10)
):
    """Plot various dynamic measurements.

    Args:
        track_dynamics: DataFrame containing track dynamics features
        fov: Optional FOV to filter by
        figsize: Figure size as (width, height) tuple
    """
    if fov is not None:
        plot_data = track_dynamics[track_dynamics["fov_name"] == fov].copy()
    else:
        plot_data = track_dynamics.copy()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Track Dynamics Analysis")

    # Plot 1: Mean Velocity vs Net Displacement
    axes[0, 0].scatter(plot_data["mean_velocity"], plot_data["net_displacement"])
    axes[0, 0].set_xlabel("Mean Velocity")
    axes[0, 0].set_ylabel("Net Displacement")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Angular Velocity vs Directional Persistence
    axes[0, 1].scatter(
        plot_data["mean_angular_velocity"], plot_data["directional_persistence"]
    )
    axes[0, 1].set_xlabel("Mean Angular Velocity")
    axes[0, 1].set_ylabel("Directional Persistence")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Total Distance vs Net Displacement
    axes[1, 0].scatter(plot_data["total_distance"], plot_data["net_displacement"])
    axes[1, 0].set_xlabel("Total Distance")
    axes[1, 0].set_ylabel("Net Displacement")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Directional Persistence Distribution
    axes[1, 1].hist(plot_data["directional_persistence"], bins=20)
    axes[1, 1].set_xlabel("Directional Persistence")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# %%
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# %%
data_path = "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/test_dataset/test_fovs_20191107_1209_1_GW23_blank_bg_stabilized.zarr"
# features_path = "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/time_interval_1_microglia_test_fovs.zarr"
features_path = Path(
    "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/time_interval_1_microglia_test_fovs_updatedEmbedWritter.zarr"
)
tracks_path = "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/test_dataset/test_fovs_20191107_1209_1_GW23_blank_bg_stabilized_tracks.zarr"

# %%
embedding_dataset = read_embedding_dataset(features_path)
features = embedding_dataset["features"]
features_df = features["sample"].to_dataframe().reset_index(drop=True)

# %%
# NOTE: processing all
fov_tracks = {"/B/2/0": "all"}
channels_to_display = ["Phase3D"]
z_range = [0, 1]
yx_patch_size = (128, 128)
num_loading_workers = 16

# Use preload_images to load all images
image_cache = preload_images(
    data_path=data_path,
    tracks_path=tracks_path,
    features_path=features_path,
    fov_tracks=fov_tracks,
    channels_to_display=channels_to_display,
    z_range=z_range,
    yx_patch_size=yx_patch_size,
    num_loading_workers=num_loading_workers,
    normalizations=NormalizeSampled(
        keys=["Phase3D"],
        level="fov_statistics",
        subtrahend="mean",
        divisor="std",
    ),
)

# %%
# image_cache has the track in fov, track_id and t as the key and the channels as the value
for img in image_cache.values():
    for key, value in img.items():
        print(key, value)

    break
# %%
DF = DynamicFeatures(features_df)

# Compute features for all tracks and combine into a single DataFrame
track_dynamics = []
for track_id in features_df["track_id"].unique():
    features = DF.compute_all_features(track_id)
    features["track_id"] = track_id
    features["fov_name"] = features_df[features_df["track_id"] == track_id][
        "fov_name"
    ].iloc[0]
    track_dynamics.append(features)

track_dynamics = pd.concat(track_dynamics, ignore_index=True)

# Plot track dynamics
plot_track_dynamics(track_dynamics, fov="/B/2/0")
plt.show()

# Print summary statistics
print("\nDynamics Summary:")
print(track_dynamics.describe())

# %%
PCA_features, PCA_projection, pca_df = compute_pca(embedding_dataset)

# %%
# Plot PC1 vs mean velocity instead of instantaneous velocity
plt.figure()
plt.scatter(pca_df["PCA1"], track_dynamics["mean_velocity"])
plt.xlabel("PCA1")
plt.ylabel("Mean Velocity")
plt.title("PCA1 vs Mean Velocity")
plt.show()

# %%
# Or for a specific FOV
dynamics_df = DF.compute_track_dynamics()
plot_track_dynamics(dynamics_df, fov="/B/2/0")
plt.show()

# Print summary statistics
print("\nDynamics Summary:")
print(dynamics_df.describe())
# %%
