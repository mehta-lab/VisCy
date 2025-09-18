import os
import sys

import iohub.ngff as ngff
import numpy as np
import pandas as pd
import tensorstore
from tqdm import tqdm

from viscy.utils.mp_utils import get_val_stats


def write_meta_field(
    position: ngff.Position, metadata: dict, field_name: str, subfield_name: str
):
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


def _grid_sample(
    position: ngff.Position, grid_spacing: int, channel_index: int, num_workers: int
):
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


def generate_normalization_metadata(
    zarr_dir: str, num_workers: int = 4, channel_ids: int = -1, grid_spacing: int = 32
):
    """Generate pixel intensity metadata for on-the-fly normalization.

    Generate pixel intensity metadata to be later used in on-the-fly normalization
    during training and inference. Sampling is used for efficient estimation of median
    and interquartile range for intensity values on both a dataset and field-of-view
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

    if channel_ids == -1:
        channel_ids = range(len(plate.channel_names))
    elif isinstance(channel_ids, int):
        channel_ids = [channel_ids]

    # get arguments for multiprocessed grid sampling
    mp_grid_sampler_args = []
    for _, position in position_map:
        mp_grid_sampler_args.append([position, grid_spacing])

    # sample values and use them to get normalization statistics
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
        write_meta_field(
            position=plate,
            metadata=dataset_statistics,
            field_name="normalization",
            subfield_name=channel_name,
        )

        for pos, position_statistics in position_and_statistics:
            write_meta_field(
                position=pos,
                metadata=dataset_statistics | position_statistics,
                field_name="normalization",
                subfield_name=channel_name,
            )

    plate.close()


def compute_zscore_params(
    frames_meta, ints_meta, input_dir, normalize_im, min_fraction=0.99
):
    """Compute normalization statistics from image data using grid sampling.

    Compute zscore median and interquartile range.

    Parameters
    ----------
    frames_meta : pd.DataFrame
        Dataframe containing all metadata.
    ints_meta : pd.DataFrame
        Metadata containing intensity statistics each z-slice and foreground fraction for masks.
    input_dir : str
        Directory containing images.
    normalize_im : None or str
        Normalization scheme for input images.
    min_fraction : float
        Minimum foreground fraction (in case of masks) for computing intensity statistics.
        for computing intensity statistics.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing:
        - pd.DataFrame frames_meta: Dataframe containing all metadata
        - pd.DataFrame ints_meta: Metadata containing intensity statistics of each z-slice
    """
    assert normalize_im in [
        None,
        "slice",
        "volume",
        "dataset",
    ], 'normalize_im must be None or "slice" or "volume" or "dataset"'

    if normalize_im is None:
        # No normalization
        frames_meta["zscore_median"] = 0
        frames_meta["zscore_iqr"] = 1
        return frames_meta
    elif normalize_im == "dataset":
        agg_cols = ["time_idx", "channel_idx", "dir_name"]
    elif normalize_im == "volume":
        agg_cols = ["time_idx", "channel_idx", "dir_name", "pos_idx"]
    else:
        agg_cols = ["time_idx", "channel_idx", "dir_name", "pos_idx", "slice_idx"]
    # median and inter-quartile range are more robust than mean and std
    ints_meta_sub = ints_meta[ints_meta["fg_frac"] >= min_fraction]
    ints_agg_median = ints_meta_sub[agg_cols + ["intensity"]].groupby(agg_cols).median()
    ints_agg_hq = (
        ints_meta_sub[agg_cols + ["intensity"]].groupby(agg_cols).quantile(0.75)
    )
    ints_agg_lq = (
        ints_meta_sub[agg_cols + ["intensity"]].groupby(agg_cols).quantile(0.25)
    )
    ints_agg = ints_agg_median
    ints_agg.columns = ["zscore_median"]
    ints_agg["zscore_iqr"] = ints_agg_hq["intensity"] - ints_agg_lq["intensity"]
    ints_agg.reset_index(inplace=True)

    cols_to_merge = frames_meta.columns[
        [col not in ["zscore_median", "zscore_iqr"] for col in frames_meta.columns]
    ]
    frames_meta = pd.merge(
        frames_meta[cols_to_merge],
        ints_agg,
        how="left",
        on=agg_cols,
    )
    if frames_meta["zscore_median"].isnull().values.any():
        raise ValueError(
            "Found NaN in normalization parameters. \
        min_fraction might be too low or images might be corrupted."
        )
    frames_meta_filename = os.path.join(input_dir, "frames_meta.csv")
    frames_meta.to_csv(frames_meta_filename, sep=",")

    cols_to_merge = ints_meta.columns[
        [col not in ["zscore_median", "zscore_iqr"] for col in ints_meta.columns]
    ]
    ints_meta = pd.merge(
        ints_meta[cols_to_merge],
        ints_agg,
        how="left",
        on=agg_cols,
    )
    ints_meta["intensity_norm"] = (
        ints_meta["intensity"] - ints_meta["zscore_median"]
    ) / (ints_meta["zscore_iqr"] + sys.float_info.epsilon)

    return frames_meta, ints_meta
