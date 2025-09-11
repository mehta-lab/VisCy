"""Auxiliary utility functions."""

from pathlib import Path

import iohub.ngff as ngff
import yaml


def _assert_unique_subset(subset, superset, name):
    """Check that unique elements of subset are a subset of superset.

    Helper function to allow for clean code: Throws error if unique elements
    of subset are not a subset of unique elements of superset.

    Parameters
    ----------
    subset : list or int
        Subset to validate. If -1, returns all unique elements of superset.
    superset : list
        Superset to validate against.
    name : str
        Name of the parameter being validated (for error messages).

    Returns
    -------
    set
        Unique elements of subset if given a list. If subset is -1,
        returns all unique elements of superset.

    Raises
    ------
    AssertionError
        If subset is not a subset of superset.
    """
    if subset == -1:
        subset = superset
    if not (isinstance(subset, list) or isinstance(subset, tuple)):
        subset = list(subset)
    unique_subset = set(subset)
    unique_superset = set(superset)
    assert unique_subset.issubset(unique_superset), (
        f"{name} in requested {name}: {unique_subset}"
        f" not in available {name}: {unique_superset}"
    )
    return unique_subset


def validate_metadata_indices(
    zarr_dir: str | Path,
    time_ids=[],
    channel_ids=[],
    slice_ids=[],
    pos_ids=[],
):
    """Check availability of indices for timepoints, channels, positions and slices.

    Returns only the available indices from the specified indices.
    If input ids are None, the indices for that parameter will not be
    evaluated. If input ids are -1, all indices for that parameter will
    be returned.

    Assumes uniform structure, as such structure is required for HCS compatibility.

    Parameters
    ----------
    zarr_dir : str | Path
        HCS-compatible zarr directory to validate indices against.
    time_ids : list, optional
        Check availability of these timepoints in image metadata, by default [].
    channel_ids : list, optional
        Check availability of these channels in image metadata, by default [].
    slice_ids : list, optional
        Check availability of z slices in image metadata, by default [].
    pos_ids : list, optional
        Check availability of positions in zarr_dir, by default [].

    Returns
    -------
    dict
        Dictionary with keys 'time_ids', 'channel_ids', 'slice_ids', 'pos_ids'
        containing all indices found given input.

    Raises
    ------
    AssertionError
        If not all channels, timepoints, positions or slices are present.
    """
    plate = ngff.open_ome_zarr(zarr_dir, layout="hcs", mode="r")
    position_path, position = next(plate.positions())

    # read available channel indices from zarr store
    available_time_ids = range(position.data.shape[0])
    if isinstance(channel_ids, int):
        available_channel_ids = range(len(plate.channel_names))
    elif isinstance(channel_ids[0], int):
        available_channel_ids = range(len(plate.channel_names))
    else:
        available_channel_ids = len(plate.channel_names)
    available_slice_ids = range(position.data.shape[-3])
    available_pos_ids = [x[0] for x in list(plate.positions())]

    # enforce that requested indices are subsets of available indices
    time_ids = _assert_unique_subset(time_ids, available_time_ids, "slices")
    channel_ids = _assert_unique_subset(channel_ids, available_channel_ids, "channels")
    slice_ids = _assert_unique_subset(slice_ids, available_slice_ids, "slices")
    pos_ids = _assert_unique_subset(pos_ids, available_pos_ids, "positions")

    indices_metadata = {
        "time_ids": list(time_ids),
        "channel_ids": list(channel_ids),
        "slice_ids": list(slice_ids),
        "pos_ids": list(pos_ids),
    }
    plate.close()
    return indices_metadata


def read_config(config_fname: str | Path):
    """Read the config file in yml format.

    Parameters
    ----------
    config_fname : str | Path
        Filename of config yaml with its full path.

    Returns
    -------
    dict
        Configuration parameters.
    """
    with open(config_fname) as f:
        config = yaml.safe_load(f)

    return config
