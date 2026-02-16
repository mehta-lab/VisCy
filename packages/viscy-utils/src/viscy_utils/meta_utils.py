"""Normalization metadata generation for OME-Zarr datasets."""

import iohub.ngff as ngff
import numpy as np
import tensorstore
from tqdm import tqdm

from viscy_utils.mp_utils import get_val_stats


def write_meta_field(position, metadata, field_name, subfield_name):
    """Write metadata to position's .zattrs.

    Parameters
    ----------
    position : ngff.Position
        NGFF position node object.
    metadata : dict
        Metadata dictionary to write.
    field_name : str
        Name of the top-level field.
    subfield_name : str
        Name of the subfield (e.g. channel name).
    """
    if field_name in position.zattrs:
        if subfield_name in position.zattrs[field_name]:
            updated_subfield = {
                **position.zattrs[field_name][subfield_name],
                **metadata,
            }
            position.zattrs[field_name] = {
                **position.zattrs[field_name],
                subfield_name: updated_subfield,
            }
        else:
            D1 = position.zattrs[field_name]
            field_metadata = {
                subfield_name: metadata,
            }
            position.zattrs[field_name] = {**D1, **field_metadata}
    else:
        field_metadata = {
            subfield_name: metadata,
        }
        position.zattrs[field_name] = field_metadata


def _grid_sample(position, grid_spacing, channel_index, num_workers):
    """Sample a position using grid sampling across all timepoints."""
    return (
        position["0"]
        .tensorstore(
            context=tensorstore.Context(
                {"data_copy_concurrency": {"limit": num_workers}}
            )
        )[:, channel_index, :, ::grid_spacing, ::grid_spacing]
        .read()
        .result()
    )


def _grid_sample_timepoint(
    position, grid_spacing, channel_index, timepoint_index, num_workers
):
    """Sample a specific timepoint from a position using grid sampling.

    Parameters
    ----------
    position : ngff.Position
        NGFF position node object.
    grid_spacing : int
        Distance between points in sampling grid.
    channel_index : int
        Index of channel to sample.
    timepoint_index : int
        Index of timepoint to sample.
    num_workers : int
        Number of cpu workers.

    Returns
    -------
    np.ndarray
        Sampled values for the specified timepoint.
    """
    return (
        position["0"]
        .tensorstore(
            context=tensorstore.Context(
                {"data_copy_concurrency": {"limit": num_workers}}
            )
        )[timepoint_index, channel_index, :, ::grid_spacing, ::grid_spacing]
        .read()
        .result()
    )


def generate_normalization_metadata(
    zarr_dir, num_workers=4, channel_ids=-1, grid_spacing=32
):
    """Generate pixel intensity metadata for normalization.

    Normalization values are recorded in the image-level metadata in the
    corresponding position of each zarr_dir store.

    Parameters
    ----------
    zarr_dir : str or Path
        Path to zarr store directory containing dataset.
    num_workers : int, optional
        Number of cpu workers, by default 4.
    channel_ids : list or int, optional
        Indices of channels to process, by default -1 (all).
    grid_spacing : int, optional
        Distance between points in sampling grid, by default 32.
    """
    plate = ngff.open_ome_zarr(zarr_dir, mode="r+")
    position_map = list(plate.positions())

    if channel_ids == -1:
        channel_ids = range(len(plate.channel_names))
    elif isinstance(channel_ids, int):
        channel_ids = [channel_ids]

    _, first_position = position_map[0]
    num_timepoints = first_position["0"].shape[0]
    print(f"Detected {num_timepoints} timepoints in dataset")

    for i, channel_index in enumerate(channel_ids):
        print(f"Sampling channel index {channel_index} ({i + 1}/{len(channel_ids)})")

        channel_name = plate.channel_names[channel_index]
        dataset_sample_values = []
        position_and_statistics = []

        for _, pos in tqdm(position_map, desc="Positions"):
            samples = _grid_sample(pos, grid_spacing, channel_index, num_workers)
            dataset_sample_values.append(samples)
            fov_level_statistics = {"fov_statistics": get_val_stats(samples)}
            position_and_statistics.append((pos, fov_level_statistics))

        dataset_statistics = {
            "dataset_statistics": get_val_stats(np.stack(dataset_sample_values)),
        }

        print(f"Computing per-timepoint statistics for channel {channel_name}")
        timepoint_statistics = {}
        for t in tqdm(range(num_timepoints), desc="Timepoints"):
            timepoint_samples = []
            for _, pos in position_map:
                t_samples = _grid_sample_timepoint(
                    pos, grid_spacing, channel_index, t, num_workers
                )
                timepoint_samples.append(t_samples)
            timepoint_statistics[str(t)] = get_val_stats(np.stack(timepoint_samples))

        write_meta_field(
            position=plate,
            metadata=dataset_statistics
            | {"timepoint_statistics": timepoint_statistics},
            field_name="normalization",
            subfield_name=channel_name,
        )

        for pos, position_statistics in position_and_statistics:
            write_meta_field(
                position=pos,
                metadata=dataset_statistics
                | position_statistics
                | {"timepoint_statistics": timepoint_statistics},
                field_name="normalization",
                subfield_name=channel_name,
            )

    plate.close()
