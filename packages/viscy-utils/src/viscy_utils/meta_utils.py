"""Normalization metadata generation for OME-Zarr datasets."""

import logging

import iohub.ngff as ngff
import numpy as np
from iohub.core.config import TensorStoreConfig
from scipy.ndimage import median_filter
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian, threshold_otsu
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
    from cubic.skimage import exposure as _cubic_exposure
    from cubic.skimage import filters as _cubic_filters
except ImportError:  # pragma: no cover - exercised by the CPU-only environments
    ascupy = None

_logger = logging.getLogger(__name__)
_BACKEND: dict[str, str] = {}


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


def _select_backend(op: str) -> str:
    """Pick the array backend for ``op``, logging the choice once per operation."""
    if ascupy is None:
        chosen = "scipy"
    # Gate the upload on an actual device, as metrics.py does: cubic imports
    # cleanly without cupy/cucim and ascupy would raise "GPU requested but not
    # available". Falling through to numpy keeps cubic's own CPU path.
    elif not torch.cuda.is_available():
        chosen = "cubic-cpu"
    else:
        chosen = "cubic-gpu"
    if _BACKEND.get(op) != chosen:
        _BACKEND[op] = chosen
        _logger.info("%s: using the %s backend", op, chosen)
    return chosen


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
    backend = _select_backend("median filter")
    if backend == "scipy":
        return median_filter(array, size=size)
    if backend == "cubic-cpu":
        return asnumpy(_cubic_ndimage.median_filter(array, size=size))
    return asnumpy(_cubic_ndimage.median_filter(ascupy(array), size=size))


def smooth_gaussian(array, sigma):
    """Gaussian-blur ``array``, on the GPU when cubic and a device are available.

    Used only to pick the Otsu threshold, never to build the mask: blurring
    this hard makes the intensity histogram separable but smears object
    boundaries, so the threshold it yields is applied to the median-filtered
    image instead. Measured against binarized cpdino segmentations on the 100
    ``dataset_v4/test_cropped`` positions (nucleus, mean over positions):

        threshold from median 5^3 alone, applied to median 5^3      IoU 0.7292
        threshold from median 5^3 + this blur, applied to median    IoU 0.7647
        same threshold, applied to the blurred image instead        IoU 0.7288

    Unlike :func:`smooth_median`, a Gaussian computes new values by weighted
    summation, so GPU and CPU results are close but not bit-identical; the
    tests assert a tolerance rather than equality.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.
    sigma : float or tuple of float
        Standard deviation, scalar for all axes or one entry per axis.

    Returns
    -------
    numpy.ndarray
        Blurred array, on the host, with the input's shape.
    """
    backend = _select_backend("gaussian filter")
    if backend == "scipy":
        return gaussian(array, sigma=sigma, preserve_range=True)
    if backend == "cubic-cpu":
        return asnumpy(_cubic_filters.gaussian(array, sigma=sigma, preserve_range=True))
    return asnumpy(_cubic_filters.gaussian(ascupy(array), sigma=sigma, preserve_range=True))


_CLAHE_TILE = (50, 120, 120)
_CLAHE_CLIP_LIMIT = 0.015
_CLAHE_PERCENTILE = 0.1


def equalize_clahe(array, tile_shape=_CLAHE_TILE, clip_limit=_CLAHE_CLIP_LIMIT, nbins=256):
    """Contrast-limited adaptive histogram equalization, on the GPU when available.

    ``equalize_adapthist`` takes a tile SIZE, but the number of tiles is what has
    to adapt: the established config computes ``shape // (50, 120, 120)`` so tiles
    stay roughly constant in pixels across differently-shaped stacks. That floors
    to zero in Z for a 44-plane stack, so it is clamped to at least one tile.

    The input is percentile-clipped to ``[0, 1]`` first because
    ``equalize_adapthist`` rejects float input outside ``[-1, 1]``. Clipping at
    ``_CLAHE_PERCENTILE`` rather than min-max keeps one bright speck from
    compressing the rest of the volume toward zero.

    Parameters
    ----------
    array : numpy.ndarray
        Input ``(Z, Y, X)`` volume.
    tile_shape : tuple of int
        Target tile size in voxels; the tile COUNT is derived from it.
    clip_limit : float
        Contrast-limiting threshold.
    nbins : int
        Histogram bins.

    Returns
    -------
    numpy.ndarray
        Equalized volume in ``[0, 1]`` -- a different intensity space from the
        input, so a threshold taken here is not comparable to a raw one.
    """
    lo, hi = np.percentile(array, (_CLAHE_PERCENTILE, 100 - _CLAHE_PERCENTILE))
    scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    n_tiles = tuple(max(int(dim // tile), 1) for dim, tile in zip(array.shape, tile_shape))
    kernel = tuple(int(dim // n) for dim, n in zip(array.shape, n_tiles))
    backend = _select_backend("clahe")
    if backend == "scipy":
        return equalize_adapthist(scaled, kernel_size=kernel, clip_limit=clip_limit, nbins=nbins)
    device = scaled if backend == "cubic-cpu" else ascupy(scaled)
    return asnumpy(_cubic_exposure.equalize_adapthist(device, kernel_size=kernel, clip_limit=clip_limit, nbins=nbins))


def foreground_mask_from_volume(volume_zyx):
    """Binary foreground mask for one ``(Z, Y, X)`` volume.

    The binarization path of the original Spotlight runs
    (``scripts/blastoid/preprocess.py::read_preprocess`` in the fnet_astr_vpa
    repo): CLAHE, then ``otsu_binarize(filter_size=5)`` -- a 5x5x5 median cube and
    ``threshold_otsu``, with no Gaussian, since CLAHE already separates the
    histogram. Foreground extraction is a preprocessing choice and not part of
    the Spotlight loss, so this is free to differ from :func:`otsu_threshold_from_volume`,
    which still supplies the raw-space ``otsu_threshold`` that ``NormalizeSampled``
    subtracts.

    Measured against binarized cpdino segmentations, nucleus channel, on each
    sample's own ``nucleus_area`` focus plane (mean IoU / recall):

                                          iPSC             A549
        median 5^3 + gaussian, raw space  0.7637 / 0.7851  0.7713 / 0.7916
        this recipe                       0.8262 / 0.8539  0.8898 / 0.9654

    A Gaussian on top of CLAHE was measured and rejected: it helps iPSC (0.8644)
    but hurts A549 (0.8807), while no-blur improves both. The repo's cleanup step
    (``min_obj_size=100``, ``max_hole_size=25``) was measured as a no-op here
    (+0.0003 and +0.0000) and is omitted.

    Parameters
    ----------
    volume_zyx : numpy.ndarray
        Full-resolution single-channel volume.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the input's shape.
    """
    equalized = smooth_median(equalize_clahe(volume_zyx), size=_OTSU_MEDIAN_SIZE)
    flat = equalized.ravel()
    if flat.min() == flat.max():
        return np.zeros(volume_zyx.shape, dtype=bool)
    return equalized >= float(threshold_otsu(flat, nbins=256))


_OTSU_MEDIAN_SIZE = 5
_OTSU_BLUR_SIGMA = 5.0


def otsu_threshold_from_volume(volume_tzyx):
    """Compute one FOV's Otsu threshold from a full-resolution ``(T, Z, Y, X)`` volume.

    This is the recipe of the original Spotlight implementation
    (``norm_threshold`` in the fnet_astr_vpa notebooks, which resolves to
    ``cubic``'s ``downscale_and_filter(downscale_factor=1, filter_size=5)`` then
    ``get_threshold_otsu``): a 5x5x5 median cube at full resolution, a sigma=5
    Gaussian, then Otsu. Both filters span Z. On the strongly anisotropic iPSC
    stacks that matters a lot -- restricting them to the plane scores worse
    than the decimated recipe they replace -- so the footprints are kept
    isotropic in voxels rather than matched to physical spacing.

    Measured against binarized cpdino segmentations, nucleus channel, scored on
    each sample's own ``nucleus_area`` focus plane (mean IoU / recall), on two
    datasets with very different voxel geometry -- iPSC ``test_cropped`` (100
    positions, z=0.290 / xy=0.108) and A549 ``dual_nucl_memb_mock`` (36
    position-timepoint samples, z=0.174 / xy=0.1494):

                                                   iPSC             A549
        decimate ::8 + median (1, 3, 3), previous   0.7086 / 0.7309  0.7406 / 0.7558
        this recipe, applied to the median volume   0.7647 / 0.7862  0.7713 / 0.7916
        both footprints restricted to XY            0.6967 / 0.7135  --

    Otsu erodes on both: precision stays at 0.94-0.97 while recall sits near
    0.76, so the masks are shrunken versions of the truth rather than
    mislocated ones, and this recipe buys back recall at near-flat precision.

    ``_OTSU_BLUR_SIGMA`` is the published value, deliberately NOT tuned. A
    larger blur scores substantially better on both datasets (iPSC 0.8713 at
    sigma=30 and still climbing, A549 0.8649 peaking at sigma=20), but the
    optimum is dataset-specific and does not follow from voxel spacing -- a
    spacing-derived sigma was measured and refuted. Tuning it would also mean
    fitting a training-time parameter on the split the models are scored on,
    which biases the Spotlight arms specifically, since the baselines never
    read ``fg_mask``. Tune it on a held-out non-eval split or not at all.

    The threshold is pooled over every timepoint and Z plane, so a position
    gets one scalar per channel.

    Parameters
    ----------
    volume_tzyx : numpy.ndarray
        Full-resolution ``(T, Z, Y, X)`` volume for a single channel.

    Returns
    -------
    float
        The Otsu threshold, or the constant value itself for a constant input,
        for which Otsu is undefined.
    """
    median = smooth_median(volume_tzyx, size=(1,) + (_OTSU_MEDIAN_SIZE,) * 3)
    flat = smooth_gaussian(median, sigma=(0,) + (_OTSU_BLUR_SIGMA,) * 3).ravel()
    if flat.min() == flat.max():
        return float(flat.min())
    return float(threshold_otsu(flat))


def generate_normalization_metadata(zarr_dir, num_workers=4, channel_ids=-1, grid_spacing=32, compute_otsu=False):
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
                    # Full resolution: the sigma=5 blur below is defined in source
                    # pixels, so thresholding a decimated grid would scale it by the
                    # stride and erase the effect.
                    fov_stats["otsu_threshold"] = otsu_threshold_from_volume(_grid_sample(pos, 1, channel_index))
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
    timepoint at a time and binarizes it with
    :func:`foreground_mask_from_volume`, writing the result as a zarr array
    alongside the image data.

    Does NOT read ``otsu_threshold``. That value is computed in raw intensity
    units for ``NormalizeSampled`` to subtract, whereas this pass thresholds in
    CLAHE's equalized ``[0, 1]`` space, so the two are not interchangeable and
    the mask is deliberately not derivable from the stored scalar. Running
    ``generate_normalization_metadata`` with ``compute_otsu=True`` is still
    required for Spotlight, but for normalization rather than for masking.

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
            for ch_idx in channel_indices:
                for t in range(t_total):
                    data = img_arr[t, ch_idx].astype(np.float32)
                    mask_arr[t, ch_idx] = foreground_mask_from_volume(data).astype(np.uint8)
