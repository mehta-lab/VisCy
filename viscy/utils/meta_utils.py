import os
import sys
from pathlib import Path

import iohub.ngff as ngff
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import viscy.utils.mp_utils as mp_utils
from viscy.utils.cli_utils import show_progress_bar


def write_meta_field(position: ngff.Position, metadata, field_name, subfield_name):
    """Write metadata to position's plate-level or FOV level .zattrs metadata.

    Write metadata to position's plate-level or FOV level .zattrs metadata by either
    creating a new field (field_name) according to metadata, or updating the metadata
    to an existing field if found, or concatenating the metadata from different channels.

    Assumes that the zarr store group given follows the OME-NGFF HCS
    format as specified here: https://ngff.openmicroscopy.org/latest/#hcs-layout

    Warning: Dangerous. Writing metadata fields above the image-level of
    an HCS hierarchy can break HCS compatibility.

    Parameters
    ----------
    position : ngff.Position
        NGFF position node object.
    metadata : dict
        Metadata dictionary to write to JSON .zattrs.
    field_name : str
        Name of the main metadata field.
    subfield_name : str
        Name of subfield inside the main field (values for different channels).
    """
    if field_name in position.zattrs:
        if subfield_name in position.zattrs[field_name]:
            position.zattrs[field_name][subfield_name].update(metadata)
        else:
            D1 = position.zattrs[field_name]
            field_metadata = {
                subfield_name: metadata,
            }
            # position.zattrs[field_name][subfield_name] = metadata
            position.zattrs[field_name] = {**D1, **field_metadata}
    else:
        field_metadata = {
            subfield_name: metadata,
        }
        position.zattrs[field_name] = field_metadata


def generate_normalization_metadata(
    zarr_dir: str | Path,
    num_workers: int = 4,
    channel_ids: list[int] | int = -1,
    grid_spacing: int = 32,
):
    """Generate pixel intensity metadata for on-the-fly normalization.

    Generate pixel intensity metadata to be later used in on-the-fly normalization
    during training and inference. Sampling is used for efficient estimation of median
    and interquartile range for intensity values on both a dataset and field-of-view
    level.

    Normalization values are recorded in the image-level metadata in the corresponding
    position of each zarr_dir store. Format of metadata is as follows:
    {
        channel_idx : {
            dataset_statistics: dataset level normalization values (positive float),
            fov_statistics: field-of-view level normalization values (positive float)
        }
    }

    Warning: Dangerous. Writing metadata fields above the image-level of
    an HCS hierarchy can break HCS compatibility.

    Parameters
    ----------
    zarr_dir : str
        Path to zarr store directory containing dataset.
    num_workers : int, optional
        Number of CPU workers for multiprocessing, by default 4.
    channel_ids : list[int] | int, optional
        Indices of channels to process in dataset arrays, by default -1 (all channels).
    grid_spacing : int, optional
        Distance between points in sampling grid, by default 32.
    """
    plate = ngff.open_ome_zarr(zarr_dir, mode="r+")
    position_map = list(plate.positions())

    # Prepare parameters for multiprocessing
    zarr_dir_path = os.path.dirname(os.path.dirname(zarr_dir))

    # Get channels to process
    if channel_ids == -1:
        # Get channel IDs from first position
        first_position = position_map[0][1]
        first_images = list(first_position.images())
        first_image = first_images[0][1]
        # shape is (t, c, z, y, x)
        channel_ids = list(range(first_image.data.shape[1]))

    if isinstance(channel_ids, int):
        channel_ids = [channel_ids]

    # Prepare parameters for each position and channel
    params_list = []
    for position_idx, (position_key, position) in enumerate(position_map):
        for channel_id in channel_ids:
            params = {
                "zarr_dir": zarr_dir,
                "position_key": position_key,
                "channel_id": channel_id,
                "grid_spacing": grid_spacing,
            }
            params_list.append(params)

    # Use multiprocessing to compute normalization statistics
    progress_bar = show_progress_bar()
    if num_workers > 1:
        with mp_utils.get_context("spawn").Pool(num_workers) as pool:
            results = pool.map(mp_utils.normalize_meta_worker, params_list)
            progress_bar.update(len(params_list))
    else:
        results = []
        for params in params_list:
            result = mp_utils.normalize_meta_worker(params)
            results.append(result)
            progress_bar.update(1)

    progress_bar.close()

    # Aggregate results and write to metadata
    all_dataset_stats = {}
    for result in results:
        if result is not None:
            position_key, channel_id, dataset_stats, fov_stats = result

            if channel_id not in all_dataset_stats:
                all_dataset_stats[channel_id] = []
            all_dataset_stats[channel_id].append(dataset_stats)

    # Calculate dataset-level statistics
    final_dataset_stats = {}
    for channel_id, stats_list in all_dataset_stats.items():
        if stats_list:
            # Aggregate median and IQR across all positions
            medians = [stats["median"] for stats in stats_list if "median" in stats]
            iqrs = [stats["iqr"] for stats in stats_list if "iqr" in stats]

            if medians and iqrs:
                final_dataset_stats[channel_id] = {
                    "median": np.median(medians),
                    "iqr": np.median(iqrs),
                }

    # Write metadata to each position
    for result in results:
        if result is not None:
            position_key, channel_id, dataset_stats, fov_stats = result

            # Get position object
            position = dict(plate.positions())[position_key]

            # Prepare metadata
            metadata = {
                "dataset_statistics": final_dataset_stats.get(channel_id, {}),
                "fov_statistics": fov_stats,
            }

            # Write metadata
            write_meta_field(
                position=position,
                metadata=metadata,
                field_name="normalization",
                subfield_name=str(channel_id),
            )

    print(f"Generated normalization metadata for {len(channel_ids)} channels")
    print(f"Dataset-level statistics: {final_dataset_stats}")


def compute_normalization_stats(
    image_data: NDArray, grid_spacing: int = 32
) -> dict[str, float]:
    """Compute normalization statistics from image data using grid sampling.

    Parameters
    ----------
    image_data : np.ndarray
        3D or 4D image array of shape (z, y, x) or (t, z, y, x).
    grid_spacing : int, optional
        Spacing between grid points for sampling, by default 32.

    Returns
    -------
    dict[str, float]
        Dictionary with median and IQR statistics for normalization.
    """
    # Handle different input shapes
    if image_data.ndim == 4:
        # Assume (t, z, y, x) and take first timepoint
        image_data = image_data[0]

    if image_data.ndim == 3:
        # Assume (z, y, x) and use middle z-slice if available
        if image_data.shape[0] > 1:
            z_mid = image_data.shape[0] // 2
            image_data = image_data[z_mid]
        else:
            image_data = image_data[0]

    # Now image_data should be 2D (y, x)
    if image_data.ndim != 2:
        raise ValueError(f"Expected 2D image after processing, got {image_data.ndim}D")

    # Create sampling grid
    y_indices = np.arange(0, image_data.shape[0], grid_spacing)
    x_indices = np.arange(0, image_data.shape[1], grid_spacing)

    # Sample values at grid points
    sampled_values = image_data[np.ix_(y_indices, x_indices)].flatten()

    # Remove any NaN or infinite values
    sampled_values = sampled_values[np.isfinite(sampled_values)]

    if len(sampled_values) == 0:
        return {"median": 0.0, "iqr": 1.0}

    # Compute statistics
    median = np.median(sampled_values)
    q25 = np.percentile(sampled_values, 25)
    q75 = np.percentile(sampled_values, 75)
    iqr = q75 - q25

    # Avoid zero IQR
    if iqr == 0:
        iqr = 1.0

    return {"median": float(median), "iqr": float(iqr)}
