"""Auxiliary utility functions"""

import iohub.ngff as ngff
import yaml


def _assert_unique_subset(subset, superset, name):
    """
    Helper function to allow for clean code:
        Throws error if unique elements of subset are not a subset of
        unique elements of superset.

    Returns unique elements of subset if given a list. If subset is -1,
    returns all unique elements of superset
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
    zarr_dir,
    time_ids=[],
    channel_ids=[],
    slice_ids=[],
    pos_ids=[],
):
    """
    Check the availability of indices provided timepoints, channels, positions
    and slices for all data, and returns only the available of the specified
    indices.

    If input ids are None, the indices for that parameter will not be
    evaluated. If input ids are -1, all indices for that parameter will
    be returned.

    Assumes uniform structure, as such structure is required for HCS compatibility

    :param str zarr_dir: HCS-compatible zarr directory to validate indices against
    :param list time_ids: check availability of these timepoints in image
                                metadata
    :param list channel_ids: check availability of these channels in image
                                    metadata
    :param list pos_ids: Check availability of positions in zarr_dir
    :param list slice_ids: Check availability of z slices in image metadata

    :return dict indices_metadata: All indices found given input
    :raise AssertionError: If not all channels, timepoints, positions
        or slices are present
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


def read_config(config_fname):
    """Read the config file in yml format

    :param str config_fname: fname of config yaml with its full path
    :return: dict config: Configuration parameters
    """

    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    return config
