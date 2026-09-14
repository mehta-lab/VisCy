"""Normalization metadata generation for OME-Zarr datasets."""

import logging

import iohub.ngff as ngff
import numpy as np
from iohub.core.config import TensorStoreConfig
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu
from tqdm import tqdm

from viscy_utils.mp_utils import get_val_stats

try:
    # cubic is how the rest of the codebase reaches the GPU (see
    # dynacell.evaluation.metrics): its proxies dispatch on the INPUT ARRAY's
    # device, so the same call runs cupyx on a cupy array and scipy on a numpy
    # one. Optional here because viscy-utils does not depend on cubic; it ships
    # in dynacell's `eval` and `preprocess` extras.
    import torch
    from cubic.cuda import ascupy, asnumpy
    from cubic.scipy import ndimage as _cubic_ndimage
except ImportError:  # pragma: no cover - exercised by the CPU-only environments
    ascupy = None

_logger = logging.getLogger(__name__)
_BACKEND: str | None = None


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


def _grid_sample(position, grid_spacing, channel_index):
    """Sample a position using grid sampling across all timepoints.

    The underlying plate must be opened with ``implementation="tensorstore"``
    (see :func:`generate_normalization_metadata`) so ``.native`` returns a
    ``tensorstore.TensorStore`` handle with the configured
    ``data_copy_concurrency``.
    """
    return position["0"].native[:, channel_index, :, ::grid_spacing, ::grid_spacing].read().result()


def smooth_median(array, size):
    """Median-filter ``array``, on the GPU when cubic and a device are available.

    The mask pass in :func:`generate_fg_masks` spends essentially all of its
    time here. Measured on a (44, 624, 924) float32 volume with the (1, 3, 3)
    footprint cell.zarr actually uses, on an A40:

        scipy.ndimage.median_filter                         5.44 s
        cubic.scipy.ndimage.median_filter via ascupy        0.03 s

    Routing through ``cubic.scipy.ndimage`` rather than calling cupyx directly
    is the codebase convention (``dynacell.evaluation.metrics`` does the same):
    the proxy dispatches on the input array's device, so uploading with
    ``ascupy`` selects the GPU implementation and passing plain numpy selects
    SciPy, with one call site either way.

    A median filter selects an existing element rather than computing a new
    one, so there is no floating-point reassociation and the backends agree
    exactly -- pinned by a test on both footprints used here.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.
    size : tuple of int
        Filter footprint, one entry per axis of ``array``.

    Returns
    -------
    numpy.ndarray
        Filtered array, on the host, with the input's dtype and shape.
    """
    global _BACKEND
    if ascupy is None:
        if _BACKEND is None:
            _BACKEND = "scipy"
            _logger.info("median filter: cubic not installed, using SciPy on the CPU")
        return median_filter(array, size=size)
    # Gate the upload on an actual device, as metrics.py does: cubic imports
    # cleanly without cupy/cucim and ascupy would raise "GPU requested but not
    # available". Falling through to numpy keeps cubic's own CPU path.
    if not torch.cuda.is_available():
        if _BACKEND is None:
            _BACKEND = "cubic-cpu"
            _logger.info("median filter: no CUDA device, using cubic's CPU path")
        return asnumpy(_cubic_ndimage.median_filter(array, size=size))
    if _BACKEND is None:
        _BACKEND = "cubic-gpu"
        _logger.info("median filter: using cubic on the GPU")
    return asnumpy(_cubic_ndimage.median_filter(ascupy(array), size=size))


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
    with ngff.open_ome_zarr(
        zarr_dir,
        mode="r+",
        implementation="tensorstore",
        implementation_config=TensorStoreConfig(data_copy_concurrency=num_workers),
    ) as plate:
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
                samples = _grid_sample(pos, grid_spacing, channel_index)
                dataset_sample_values.append(samples)
                fov_stats = get_val_stats(samples)
                if compute_otsu:
                    otsu_samples = _grid_sample(pos, otsu_grid_spacing, channel_index)
                    smoothed = smooth_median(otsu_samples, size=(1, 1, 3, 3))
                    flat = smoothed.ravel()
                    # Otsu's method is undefined for constant-valued inputs.
                    # Use the constant value itself so generate_fg_masks marks
                    # nothing as foreground (no meaningful structure to supervise).
                    if flat.min() == flat.max():
                        fov_stats["otsu_threshold"] = float(flat.min())
                    else:
                        fov_stats["otsu_threshold"] = float(threshold_otsu(flat))
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


def generate_fg_masks(
    zarr_dir,
    channel_names,
    fg_mask_key="fg_mask",
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
    """
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

            # Inherit source image's spatial chunking for aligned I/O;
            # cap at 512 per axis in case source data is unchunked.
            src_chunks = img_arr.chunks
            mask_chunks = (
                1,
                1,
                min(src_chunks[2], zyx_shape[0]),
                min(src_chunks[3], 512),
                min(src_chunks[4], 512),
            )
            mask_arr = pos.create_zeros(
                fg_mask_key,
                shape=(t_total, c_total, *zyx_shape),
                dtype=np.uint8,
                chunks=mask_chunks,
            )

            # Fill non-target channels with 1s (full supervision default)
            non_target = sorted(set(range(c_total)) - set(channel_indices))
            for c in non_target:
                mask_arr[:, c] = 1

            # Compute and write target channel masks per timepoint
            for ch_name, ch_idx in zip(channel_names, channel_indices):
                norm = pos.zattrs["normalization"][ch_name]["fov_statistics"]
                otsu_threshold = norm["otsu_threshold"]

                for t in range(t_total):
                    data = img_arr[t, ch_idx].astype(np.float32)
                    smoothed = smooth_median(data, size=(1, 3, 3))
                    mask_arr[t, ch_idx] = (smoothed >= otsu_threshold).astype(np.uint8)
