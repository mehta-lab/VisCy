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
        .tensorstore(context=tensorstore.Context({"data_copy_concurrency": {"limit": num_workers}}))[
            :, channel_index, :, ::grid_spacing, ::grid_spacing
        ]
        .read()
        .result()
    )


def generate_normalization_metadata(
    zarr_dir, num_workers=4, channel_ids=-1, grid_spacing=32, compute_otsu=False, otsu_grid_spacing=8
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
    compute_otsu : bool, optional
        Whether to compute Otsu thresholds for foreground estimation,
        by default False. Required for Spotlight loss.
    otsu_grid_spacing : int, optional
        Grid spacing for Otsu sampling, by default 8. Denser than the
        default ``grid_spacing=32`` to capture inter-cell gaps. A median
        filter is applied before thresholding to smooth noise.
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

    if compute_otsu:
        from scipy.ndimage import median_filter
        from skimage.filters import threshold_otsu

    for i, channel_index in enumerate(channel_ids):
        print(f"Sampling channel index {channel_index} ({i + 1}/{len(channel_ids)})")

        channel_name = plate.channel_names[channel_index]
        dataset_sample_values = []
        position_and_statistics = []

        for _, pos in tqdm(position_map, desc="Positions"):
            samples = _grid_sample(pos, grid_spacing, channel_index, num_workers)
            dataset_sample_values.append(samples)
            fov_stats = get_val_stats(samples)
            if compute_otsu:
                otsu_samples = _grid_sample(pos, otsu_grid_spacing, channel_index, num_workers)
                smoothed = median_filter(otsu_samples, size=(1, 1, 3, 3))
                fov_stats["otsu_threshold"] = float(threshold_otsu(smoothed.ravel()))
            fov_statistics = {"fov_statistics": fov_stats}
            fov_timepoint_statistics = {}
            for t in range(num_timepoints):
                fov_timepoint_statistics[str(t)] = get_val_stats(samples[t])
            fov_statistics["timepoint_statistics"] = fov_timepoint_statistics
            position_and_statistics.append((pos, fov_statistics))

        dataset_statistics = {
            "dataset_statistics": get_val_stats(np.stack(dataset_sample_values)),
        }

        print(f"Computing per-timepoint statistics for channel {channel_name}")
        dataset_timepoint_statistics = {}
        for t in tqdm(range(num_timepoints), desc="Timepoints"):
            all_fov_samples_at_t = np.stack([samples[t] for samples in dataset_sample_values])
            dataset_timepoint_statistics[str(t)] = get_val_stats(all_fov_samples_at_t)

        write_meta_field(
            position=plate,
            metadata=dataset_statistics | {"timepoint_statistics": dataset_timepoint_statistics},
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


def generate_fg_masks(
    zarr_dir,
    channel_names,
    fg_mask_key="fg_mask",
    num_workers=4,
):
    """Precompute binary foreground masks from Otsu thresholds.

    For each FOV and specified channel, loads the full-resolution image one
    timepoint at a time, smooths with the same median filter used for Otsu
    thresholding, and writes the binary mask as a zarr array alongside the
    image data.

    Requires ``generate_normalization_metadata`` with ``compute_otsu=True``
    to have been run first (Otsu thresholds must be stored in zattrs).

    Parameters
    ----------
    zarr_dir : str or Path
        Path to the HCS OME-Zarr dataset.
    channel_names : list[str]
        Channel names to compute masks for (typically the target channels).
    fg_mask_key : str, optional
        Zarr array key for the mask, by default ``"fg_mask"``.
    num_workers : int, optional
        Number of CPU workers for reading, by default 4.
    """
    from scipy.ndimage import median_filter

    with ngff.open_ome_zarr(zarr_dir, mode="r+") as plate:
        all_channel_names = plate.channel_names
        channel_indices = [all_channel_names.index(name) for name in channel_names]

        for pos_name, pos in tqdm(plate.positions(), desc="Generating FG masks"):
            if fg_mask_key in pos:
                raise FileExistsError(
                    f"Mask array '{fg_mask_key}' already exists at {pos_name}. Delete it first to regenerate."
                )

            img_arr = pos["0"]
            t_total, c_total = img_arr.shape[0], img_arr.shape[1]
            zyx_shape = img_arr.shape[2:]

            # Build full mask array: (T, C_all, Z, Y, X), zeros for non-target channels
            mask_all = np.zeros((t_total, c_total, *zyx_shape), dtype=np.uint8)

            for ch_name, ch_idx in zip(channel_names, channel_indices):
                norm = pos.zattrs["normalization"][ch_name]["fov_statistics"]
                otsu_threshold = norm["otsu_threshold"]

                for t in range(t_total):
                    # Read one timepoint, one channel: (Z, Y, X)
                    data = img_arr[t, ch_idx].astype(np.float32)
                    smoothed = median_filter(data, size=(1, 3, 3))
                    mask_all[t, ch_idx] = (smoothed >= otsu_threshold).astype(np.uint8)

            pos.create_image(
                fg_mask_key,
                mask_all,
                chunks=(1, 1, *zyx_shape),
            )
