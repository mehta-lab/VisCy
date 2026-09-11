"""Assemble A549 mantis plates into per-organelle × per-condition pools.

Reads authoring platemap + splits YAMLs plus each plate's dynacell zarr.
For each requested (target, condition) pair, walks every authoring plate
that hosts the target, picks positions whose well's ``condition`` matches,
and writes one pooled HCS store at
``<output_root>/<split>/<TARGET>_<CONDITION>.<ext>`` with sequential
``0/0/fov<NNNN>`` position naming. Plate provenance is preserved in
per-position zattrs (``plate_id``, ``source_position``, …) and in a
sidecar JSON next to the store.

For each requested target on the train side, also emits
``<TARGET>_all.zarr`` (a zarr — not ozx — directory store) covering all
conditions together. Used as the training corpus when downstream models
expect a single data_path; per-condition ozx stores remain the canonical
eval inputs.

Driven by the packaged ``_configs/preprocess/dynacell/a549_assemble_pool.yaml``
loaded with :func:`dynacell.preprocess.load_preprocess_config` and passed to
:func:`assemble_pool`; there is no ``dynacell`` CLI subcommand for it.
"""

import contextlib
import hashlib
import json
import math
import shutil
from importlib.resources import files
from pathlib import Path

import numpy as np
from cubic.cuda import CUDAManager
from cubic.cuda import ascupy as _ascupy
from cubic.cuda import asnumpy as _asnumpy
from cubic.skimage import transform as _ski_transform
from iohub.core.ozx import pack_ozx
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from omegaconf import DictConfig
from tqdm import tqdm

from dynacell.preprocess.a549_mantis.authoring import (
    Platemap,
    load_platemap,
    load_splits,
)
from dynacell.preprocess.a549_mantis.channels import (
    DECONVOLVED_SUFFIX,
    ChannelSelection,
    resolve_channels,
    resolve_target_genes,
)
from dynacell.preprocess.a549_mantis.grid import GridFrame, build_grid
from viscy_utils.meta_utils import write_meta_field as _write_meta_field
from viscy_utils.mp_utils import get_val_stats

GENE_TO_FILENAME: dict[str, str] = {
    "sec61b": "SEC61B",
    "tomm20": "TOMM20",
    "h2b": "H2B",
    "caax": "CAAX",
    # Combined nucleus+membrane pool (both v2 fluor genes in one store).
    "dual_nucl_memb": "dual_nucl_memb",
}

VALID_CONDITIONS: tuple[str, ...] = ("mock", "ZIKV", "DENV")

_SPLIT_NAMES: tuple[str, ...] = ("train", "test")
_VALID_FORMATS: tuple[str, ...] = ("zarr", "ozx")

# Defaults applied when the corresponding key is absent from cfg. Both
# splits follow the v2-downsampled-to-v1-pixel-size plan (640x960 @
# 0.1494 µm/px): the test side is uncropped from the legacy 512x512 to
# the full 640x960 window and resampled to the uniform 0.1494 pitch.
# Opt out by setting the explicit value to null.
_DEFAULT_CONDITION_FORMAT: str = "zarr"
_DEFAULT_POOL_ALL_FORMAT: str = "zarr"
_DEFAULT_T_CAP: int | None = 10
_DEFAULT_TRAIN_CENTER_CROP_YX: tuple[int, int] = (640, 960)
_DEFAULT_TEST_CENTER_CROP_YX: tuple[int, int] = (640, 960)
_DEFAULT_TRAIN_TARGET_YX_PIXEL_SIZE_UM: float | None = 0.1494
_DEFAULT_TEST_TARGET_YX_PIXEL_SIZE_UM: float | None = 0.1494
_DEFAULT_EMIT_TRAIN_POOL_ALL: bool = True
# Publication artifact: colocated raw-only .ozx (drops *_deconvolved
# channels) next to each internal directory-zarr store. Off by default;
# the campaign turns it on so per-condition test stores publish raw-only.
_DEFAULT_EMIT_RAW_ONLY_OZX: bool = False

# Chunking: VisCy's reader reads full Y×X and slices Z by z_window_size,
# so we keep YX as one chunk and split Z into 16-slice slabs (48 = 3×16,
# covers UNetViT3D z=13 in 1 chunk, FNet z=32 in 2, CellDiff predict
# z=40 in 3). Sharding packs the whole FOV into one file, dropping
# filesystem-metadata cost from ~90 chunks/FOV to 1 shard/FOV.
_Z_CHUNK: int = 16
_OME_NGFF_VERSION: str = "0.5"  # zarr v3, required for sharding


def _pool_filename(target: str, condition: str | None) -> str:
    """``SEC61B_mock``, ``SEC61B_ZIKV``, …, or ``SEC61B_all`` for condition=None."""
    suffix = condition if condition is not None else "all"
    return f"{GENE_TO_FILENAME[target]}_{suffix}"


def _pool_output_path(
    output_root: Path,
    split: str,
    target: str,
    condition: str | None,
    output_format: str,
) -> Path:
    """Compose the canonical pool store path."""
    if output_format not in _VALID_FORMATS:
        raise ValueError(f"output_format={output_format!r} not in {_VALID_FORMATS}")
    suffix = "ozx" if output_format == "ozx" else "zarr"
    return output_root / split / f"{_pool_filename(target, condition)}.{suffix}"


def _provenance_sidecar_path(store_path: Path) -> Path:
    """JSON sidecar colocated with the store: ``<stem>.provenance.json``."""
    return store_path.parent / f"{store_path.stem}.provenance.json"


def _plate_zarr_path(plate_zarr_root: Path, platemap: Platemap) -> Path:
    return plate_zarr_root / platemap.experiment / "dynacell" / platemap.run_dir / platemap.zarr_filename


def _parse_position(pos_name: str) -> tuple[str, str]:
    """Split a HCS position name (``row/col/fov``) into (well_id, fov)."""
    parts = pos_name.split("/")
    if len(parts) != 3:
        raise ValueError(f"Position name {pos_name!r} must have form 'row/col/fov'")
    return f"{parts[0]}/{parts[1]}", parts[2]


def _check_output_safety(output_root: Path, plate_zarr_root: Path) -> None:
    """Refuse to write into read-only source directories.

    Parameters
    ----------
    output_root : Path
        Target directory for assembled zarrs.
    plate_zarr_root : Path
        Root of per-plate source zarrs (e.g. ``intracellular_dashboard``).

    Raises
    ------
    ValueError
        If ``output_root`` is ``plate_zarr_root`` or any descendant of it.
    """
    out_resolved = output_root.resolve()
    src_resolved = plate_zarr_root.resolve()
    if out_resolved == src_resolved or src_resolved in out_resolved.parents:
        raise ValueError(
            f"output_root {out_resolved} is inside plate_zarr_root "
            f"{src_resolved}; refusing to write into read-only source tree"
        )


def _provenance_attrs(
    platemap: Platemap,
    well_id: str,
    fov: str,
    grid_frames: list[GridFrame],
    stride_h: float,
) -> dict:
    """Build the per-position provenance zattrs for one assembled FOV.

    ``grid_stride_h`` records the *requested* grid stride (``cfg.grid.stride_h``).
    The realized spacing is irregular wherever :func:`build_grid` tail-snaps the
    last tick onto the final native frame, so ``hpi_values`` — not this scalar —
    is the per-frame source of truth. Three of the eight authoring plates have a
    short final interval for exactly that reason.
    """
    well = platemap.wells[well_id]
    return {
        "plate_id": platemap.experiment,
        "well_id": well_id,
        "fov": fov,
        "source_position": f"{well_id}/{fov}",
        "condition": well.condition,
        "hpi_start": platemap.hpi_start,
        "grid_stride_h": float(stride_h),
        "native_delta_t_min": platemap.native_delta_t_min,
        "hpi_values": [float(f.hpi) for f in grid_frames],
        "tick_hpi_values": [float(f.tick_hpi) for f in grid_frames],
        "native_frame_indices": [int(f.native_idx) for f in grid_frames],
    }


def _default_authoring_root() -> Path:
    """Resolve the authoring-root from packaged resources."""
    root = files("dynacell") / "_configs" / "datasets" / "a549-mantis" / "authoring"
    return Path(str(root))


def _read_source_spacing(source_zarr_path: Path) -> tuple[float, float, float]:
    """Return (z, y, x) µm/px spacing from a source plate's NGFF metadata."""
    with open_ome_zarr(source_zarr_path, mode="r", layout="hcs") as plate:
        _, position = next(iter(plate.positions()))
        scale = position.scale
        spacing = tuple(scale[position.get_axis_index(axis)] for axis in "zyx")
    for axis, value in zip("zyx", spacing):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"source zarr {source_zarr_path}: invalid {axis} spacing {value!r}")
    return float(spacing[0]), float(spacing[1]), float(spacing[2])


def _apply_center_crop_yx(block: np.ndarray, center_crop_yx: tuple[int, int] | None) -> np.ndarray:
    """Center-crop the YX plane of a (T, C, Z, Y, X) block."""
    if center_crop_yx is None:
        return block
    crop_y, crop_x = int(center_crop_yx[0]), int(center_crop_yx[1])
    src_y, src_x = block.shape[-2], block.shape[-1]
    if crop_y > src_y or crop_x > src_x:
        raise ValueError(f"center_crop_yx=({crop_y},{crop_x}) exceeds source YX ({src_y},{src_x})")
    y0 = (src_y - crop_y) // 2
    x0 = (src_x - crop_x) // 2
    return block[..., y0 : y0 + crop_y, x0 : x0 + crop_x]


def _resample_yx_to_pixel_size(
    block: np.ndarray,
    *,
    source_y_um: float,
    source_x_um: float,
    target_yx_um: float,
) -> np.ndarray:
    """Resample the (Y, X) plane of a (T, C, Z, Y, X) block to a target pixel size.

    Downsample only — refuses upsampling at assembly time. Uses
    ``cubic.skimage.transform.resize`` with anti-aliased bilinear
    interpolation.

    GPU dispatch is by INPUT ARRAY DEVICE, not by mere library
    availability: ``cubic.skimage`` calls ``cucim.skimage`` only when the
    array it receives is a CuPy (device) array, else it silently runs
    ``skimage`` on the CPU (see ``cubic/skimage.py`` ``func_wrapper``).
    Handing it a NumPy array therefore always computes on CPU — even with
    CuPy/CuCIM installed. So when a CUDA device is present we move the block
    to the GPU (``_ascupy``) before ``resize`` and bring it back
    (``_asnumpy``) after. The transfer is done one leading-axis (T) tile at
    a time: a full (T, C, Z, Y, X) block can be tens of GB, but a single-T
    slab plus the anti-alias/output temporaries stays a few GB. Per-T
    tiling is numerically identical to whole-block resize because only the
    last two axes are downsampled (T, C, Z are preserved, so no cross-tile
    blur).
    """
    if source_y_um == target_yx_um and source_x_um == target_yx_um:
        return block
    if source_y_um > target_yx_um or source_x_um > target_yx_um:
        raise ValueError(
            f"target_yx_pixel_size_um={target_yx_um} is finer than source "
            f"({source_y_um}, {source_x_um}) µm/px; refusing to upsample"
        )
    src_y, src_x = block.shape[-2], block.shape[-1]
    new_y = int(round(src_y * source_y_um / target_yx_um))
    new_x = int(round(src_x * source_x_um / target_yx_um))

    def _resize_yx(arr: np.ndarray) -> np.ndarray:
        out_shape = (*arr.shape[:-2], new_y, new_x)
        return _ski_transform.resize(
            arr,
            out_shape,
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        )

    if CUDAManager().get_num_gpus() > 0:
        # Move to the GPU per leading-axis tile so cubic dispatches to
        # cucim.skimage; bounds device memory to one T-slab.
        tiles = [_asnumpy(_resize_yx(_ascupy(block[i]))) for i in range(block.shape[0])]
        resized = np.stack(tiles, axis=0)
    else:
        resized = _asnumpy(_resize_yx(block))
    return np.asarray(resized).astype(block.dtype, copy=False)


def _raw_only_channel_indices(channel_names: list[str]) -> list[int]:
    """Return channel indices to keep in the raw-only publication artifact.

    Drops any channel whose name ends in ``DECONVOLVED_SUFFIX``
    (e.g. ``Structure_deconvolved``). All other channels — passthrough
    (``Phase3D``, ``Brightfield``) and raw targets (``Structure``,
    ``Nuclei``, ``Membrane``) — are kept in their original order.
    """
    return [i for i, name in enumerate(channel_names) if not name.endswith(DECONVOLVED_SUFFIX)]


# NGFF-managed / recomputed zattr keys that must NOT be copied verbatim from
# the 4-channel internal store into the channel-subset raw-only store. Under
# NGFF v0.5 the omero channel list lives in ``ome`` (copying it would make the
# 3-channel store report 4 channels — an out-of-bounds crash in the
# normalization pass), and ``normalization`` is recomputed for the kept
# channels below.
_NGFF_MANAGED_ZATTRS: tuple[str, ...] = ("ome", "normalization")


def _provenance_only_zattrs(zattrs) -> dict:
    """Copy of ``zattrs`` with NGFF-managed / recomputed keys removed."""
    attrs = dict(zattrs)
    for key in _NGFF_MANAGED_ZATTRS:
        attrs.pop(key, None)
    return attrs


def _derive_raw_only_store(
    internal_dir_zarr: Path,
    ozx_out_path: Path,
) -> bool:
    """Write a raw-only ``.ozx`` from an internal directory zarr.

    Channel-subsets the internal store, dropping every ``*_deconvolved``
    channel, into a temporary directory zarr, then packs it to
    ``ozx_out_path`` (RFC-9 archive). The internal store keeps all
    channels; only this published artifact is slimmed.

    Returns
    -------
    bool
        ``True`` if an ozx was written; ``False`` if subsetting would be a
        no-op (no deconvolved channels present) — caller may pack the
        internal store directly in that case.
    """
    with open_ome_zarr(internal_dir_zarr, mode="r", layout="hcs") as src:
        channel_names = list(src.channel_names)
    keep = _raw_only_channel_indices(channel_names)
    if len(keep) == len(channel_names):
        return False
    kept_names = [channel_names[i] for i in keep]

    tmp_dir = ozx_out_path.with_suffix(".rawonly.zarr.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if ozx_out_path.exists():
        ozx_out_path.unlink()
    ozx_out_path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        src = stack.enter_context(open_ome_zarr(internal_dir_zarr, mode="r", layout="hcs"))
        writer = stack.enter_context(
            open_ome_zarr(
                tmp_dir,
                mode="w",
                layout="hcs",
                channel_names=kept_names,
                version=_OME_NGFF_VERSION,
            )
        )
        writer.zattrs.update(_provenance_only_zattrs(src.zattrs))
        for pos_name, src_pos in src.positions():
            row, col, fov = pos_name.split("/")
            src_img = src_pos["0"]
            block = np.asarray(src_img.oindex[:, keep])
            scale = src_pos.scale
            spacing = tuple(scale[src_pos.get_axis_index(axis)] for axis in "zyx")
            out_pos = writer.create_position(row, col, fov)
            t_dim, c_dim, z_dim, y_dim, x_dim = block.shape
            z_chunk = min(_Z_CHUNK, z_dim)
            chunks = (1, 1, z_chunk, y_dim, x_dim)
            shards_ratio = (1, c_dim, (z_dim + z_chunk - 1) // z_chunk, 1, 1)
            out_pos.create_image(
                name="0",
                data=block.astype(np.float32, copy=False),
                chunks=chunks,
                shards_ratio=shards_ratio,
                transform=_scale_transform(spacing),
            )
            # Copy provenance zattrs but NOT the NGFF-managed keys: the source
            # is the 4-channel internal store, so its ``ome`` block describes 4
            # channels. ``create_image`` above already wrote the correct
            # 3-channel ``ome`` for ``kept_names``; copying the source's over it
            # would make ``channel_names`` report 4 channels for a 3-channel
            # array (out-of-bounds in the normalization pass).
            out_pos.zattrs.update(_provenance_only_zattrs(src_pos.zattrs))

    _write_heterogeneous_t_normalization(tmp_dir)
    pack_ozx(tmp_dir, ozx_out_path)
    shutil.rmtree(tmp_dir)
    return True


def _scale_transform(spacing: tuple[float, float, float]) -> list[TransformationMeta]:
    """OME-NGFF scale transform for a (T, C, Z, Y, X) array."""
    z, y, x = spacing
    return [
        TransformationMeta(
            type="scale",
            scale=[1.0, 1.0, float(z), float(y), float(x)],
        )
    ]


def _parse_crop_cfg(cfg: DictConfig, key: str, default: tuple[int, int] | None) -> tuple[int, int] | None:
    if key not in cfg:
        return default
    value = cfg[key]
    if value is None:
        return None
    crop_list = [int(v) for v in value]
    if len(crop_list) != 2 or any(v <= 0 for v in crop_list):
        raise ValueError(f"{key} must be [Y, X] with positive ints, got {value!r}")
    return (crop_list[0], crop_list[1])


def _parse_pixel_size_cfg(cfg: DictConfig, key: str, default: float | None) -> float | None:
    if key not in cfg:
        return default
    value = cfg[key]
    if value is None:
        return None
    px = float(value)
    if not (math.isfinite(px) and px > 0):
        raise ValueError(f"{key} must be a positive float or null, got {value!r}")
    return px


def _list_authored_plates(authoring_root: Path) -> list[str]:
    """Discover plate names from ``<authoring_root>/platemaps/*.yaml``."""
    platemaps_dir = authoring_root / "platemaps"
    return sorted(p.stem for p in platemaps_dir.glob("*.yaml"))


def _write_heterogeneous_t_normalization(zarr_dir: Path, *, grid_spacing: int = 32) -> None:
    """Compute and write VisCy-shaped normalization zattrs for a pool.

    Drop-in replacement for ``viscy_utils.generate_normalization_metadata``
    that handles heterogeneous T across positions. Pools combine plates
    with different ``native_t`` windows (e.g. one plate gives 10
    timepoints, another gives 7), and the upstream helper crashes on
    ``np.stack`` of varying-shape per-FOV samples. Here we never stack:

    - **fov_statistics**: computed per position from its own samples.
    - **dataset_statistics**: pooled over all positions by concatenating
      flattened samples (no leading-axis assumption).
    - **timepoint_statistics**: for each integer timepoint index ``t``,
      pooled over only the positions whose array has ``T > t``. Late
      timepoint indices contributed by a long-T plate are recorded over
      that plate's FOVs only.

    Schema mirrors the upstream output so VisCy's ``NormalizeSampled``
    transforms (``level=fov_statistics`` /
    ``level=dataset_statistics`` / ``level=timepoint_statistics``)
    work unchanged.
    """
    with open_ome_zarr(zarr_dir, mode="r+", layout="hcs") as plate:
        position_map = list(plate.positions())
        channel_names = list(plate.channel_names)
        for ch_idx, channel_name in enumerate(channel_names):
            print(f"  norm: channel {ch_idx + 1}/{len(channel_names)} {channel_name}")
            position_samples: list[np.ndarray] = []
            for _, pos in tqdm(position_map, desc="    positions", ncols=80):
                arr = pos["0"]
                # Grid-sample (T, Z, Y_sub, X_sub) for this channel.
                sample = np.asarray(
                    arr.oindex[
                        :,
                        [ch_idx],
                        :,
                        ::grid_spacing,
                        ::grid_spacing,
                    ]
                )[:, 0]  # drop channel axis → (T, Z, Y_sub, X_sub)
                position_samples.append(sample)

            # Per-FOV stats (write to each position).
            for (_, pos), sample in zip(position_map, position_samples):
                fov_stats = get_val_stats(sample)
                fov_timepoint_stats = {str(t): get_val_stats(sample[t]) for t in range(sample.shape[0])}
                _write_meta_field(
                    pos,
                    {
                        "fov_statistics": fov_stats,
                        "timepoint_statistics": fov_timepoint_stats,
                    },
                    "normalization",
                    channel_name,
                )

            # Dataset stats: concat flattened across all FOVs (no stack).
            flat_all = np.concatenate([s.ravel() for s in position_samples])
            dataset_stats = get_val_stats(flat_all)

            # Per-timepoint dataset stats: at index t, include only FOVs
            # whose array has T > t.
            max_t = max(s.shape[0] for s in position_samples)
            dataset_timepoint_stats: dict[str, dict] = {}
            for t in range(max_t):
                contrib = [s[t].ravel() for s in position_samples if s.shape[0] > t]
                dataset_timepoint_stats[str(t)] = get_val_stats(np.concatenate(contrib))

            _write_meta_field(
                plate,
                {
                    "dataset_statistics": dataset_stats,
                    "timepoint_statistics": dataset_timepoint_stats,
                },
                "normalization",
                channel_name,
            )


class _PlateContribution:
    """Per-plate state needed to read positions for one (target, split) view."""

    def __init__(
        self,
        *,
        plate_name: str,
        platemap_path: Path,
        splits_path: Path,
        platemap: Platemap,
        plate_zarr_path: Path,
        spacing: tuple[float, float, float],
        grid_frames: list[GridFrame],
        native_frame_indices: np.ndarray,
        positions: list[tuple[str, str]],  # list of (well_id, fov)
    ):
        self.plate_name = plate_name
        self.platemap_path = platemap_path
        self.splits_path = splits_path
        self.platemap = platemap
        self.plate_zarr_path = plate_zarr_path
        self.spacing = spacing
        self.grid_frames = grid_frames
        self.native_frame_indices = native_frame_indices
        self.positions = positions


def _gather_plate_contributions(
    *,
    target: str,
    condition: str | None,
    split: str,
    plates: list[str],
    plate_zarr_root: Path,
    authoring_root: Path,
    stride_h: float,
    window: tuple[float, float],
    tail_tol_h: float,
    t_cap: int | None,
) -> list[_PlateContribution]:
    """Per-plate views for one pool: filter positions by target+condition+split."""
    contribs: list[_PlateContribution] = []
    for plate_name in plates:
        platemap_path = authoring_root / "platemaps" / f"{plate_name}.yaml"
        splits_path = authoring_root / "splits" / f"{plate_name}.yaml"
        if not platemap_path.is_file() or not splits_path.is_file():
            continue
        platemap = load_platemap(platemap_path)
        splits = load_splits(splits_path)
        # Combined targets carry no split of their own; positions come from
        # their constituent genes, which must share an identical lattice
        # (co-imaged). Single-gene targets resolve to themselves.
        constituent_genes = resolve_target_genes(target)
        constituent_splits = [splits.targets[g] for g in constituent_genes if g in splits.targets]
        if len(constituent_splits) != len(constituent_genes):
            # Not every constituent gene is authored for this plate.
            continue
        split_position_sets = [list(getattr(s, split)) for s in constituent_splits]
        if any(set(p) != set(split_position_sets[0]) for p in split_position_sets):
            raise ValueError(
                f"combined target {target!r} on plate {plate_name}: constituent "
                f"genes {list(constituent_genes)} have divergent {split} position "
                f"lattices; combined targets require co-imaged (identical) FOVs"
            )
        split_positions = split_position_sets[0]
        if not split_positions:
            continue

        kept: list[tuple[str, str]] = []
        for pos_name in split_positions:
            well_id, fov = _parse_position(pos_name)
            if well_id not in platemap.wells:
                raise ValueError(f"Position {pos_name} not in platemap wells ({sorted(platemap.wells)})")
            well_cond = platemap.wells[well_id].condition
            if condition is not None and well_cond != condition:
                continue
            kept.append((well_id, fov))
        if not kept:
            continue

        grid_frames = build_grid(
            native_delta_t_min=platemap.native_delta_t_min,
            native_t=platemap.native_t,
            hpi_start=platemap.hpi_start,
            stride_h=stride_h,
            window=window,
            tail_tol_h=tail_tol_h,
        )
        if t_cap is not None and t_cap < len(grid_frames):
            grid_frames = grid_frames[:t_cap]
        native_frame_indices = np.array([f.native_idx for f in grid_frames], dtype=np.int64)

        plate_zarr_path = _plate_zarr_path(plate_zarr_root, platemap)
        spacing = _read_source_spacing(plate_zarr_path)

        contribs.append(
            _PlateContribution(
                plate_name=plate_name,
                platemap_path=platemap_path,
                splits_path=splits_path,
                platemap=platemap,
                plate_zarr_path=plate_zarr_path,
                spacing=spacing,
                grid_frames=grid_frames,
                native_frame_indices=native_frame_indices,
                positions=kept,
            )
        )
    return contribs


def _assembly_inputs_hash(
    *,
    contribs: list[_PlateContribution],
    target: str,
    condition: str | None,
    output_channel_names: list[str],
    center_crop_yx: tuple[int, int] | None,
    target_yx_pixel_size_um: float | None,
) -> str:
    """Pool-level sha256 over the inputs that determine assembled bytes.

    Identical hash → identical assembled-pool content (modulo zarr-side
    write timestamps). When this hash matches between two re-assembly
    runs but the bytes differ, that's a determinism bug in the writer,
    not in the inputs.

    Components hashed (in order, with explicit separators so the
    concatenation is unambiguous):

    - target gene key, condition slug (or ``"all"``)
    - per contributing plate (sorted by plate name): platemap YAML
      bytes, splits YAML bytes, NGFF spacing tuple, native frame indices
      selected by the grid, list of contributing positions
    - output channel names (post-rename)
    - center-crop YX (post-resample)
    - target lateral pixel size (only when a resample is requested;
      omitted when null)

    Excluded on purpose:

    - source zarr path (path strings can change without content
      changing; the spacing read from the source NGFF is a stronger
      proxy for "did the source change in a way that matters")
    - source array bytes (re-reading hundreds of GB to hash would
      negate the assembler's I/O savings)
    """
    h = hashlib.sha256()
    h.update(b"TARGET\n")
    h.update(target.encode("utf-8"))
    h.update(b"\nCONDITION\n")
    h.update((condition or "all").encode("utf-8"))
    for c in sorted(contribs, key=lambda c: c.plate_name):
        h.update(b"\nPLATE\n")
        h.update(c.plate_name.encode("utf-8"))
        h.update(b"\nPLATEMAP\n")
        h.update(c.platemap_path.read_bytes())
        h.update(b"\nSPLITS\n")
        h.update(c.splits_path.read_bytes())
        h.update(b"\nSPACING\n")
        h.update(json.dumps(list(c.spacing), sort_keys=True).encode("utf-8"))
        h.update(b"\nFRAMES\n")
        h.update(json.dumps(c.native_frame_indices.tolist()).encode("utf-8"))
        h.update(b"\nPOSITIONS\n")
        h.update(json.dumps([f"{w}/{f}" for w, f in c.positions]).encode("utf-8"))
    h.update(b"\nCHANNELS\n")
    h.update(json.dumps(output_channel_names).encode("utf-8"))
    h.update(b"\nCROP_YX\n")
    h.update(json.dumps(list(center_crop_yx) if center_crop_yx is not None else None).encode("utf-8"))
    if target_yx_pixel_size_um is not None:
        h.update(b"\nTARGET_YX_PIXEL_SIZE_UM\n")
        h.update(json.dumps(float(target_yx_pixel_size_um)).encode("utf-8"))
    return h.hexdigest()


def _assemble_one_pool(
    *,
    target: str,
    condition: str | None,
    split: str,
    plates: list[str],
    plate_zarr_root: Path,
    authoring_root: Path,
    output_root: Path,
    stride_h: float,
    window: tuple[float, float],
    tail_tol_h: float,
    overwrite: bool,
    crop_yx: tuple[int, int] | None,
    target_yx_pixel_size_um: float | None,
    t_cap: int | None,
    output_format: str,
    emit_raw_only_ozx: bool = False,
) -> bool:
    """Walk all plates and write one pooled HCS store for (target, condition, split).

    ``condition=None`` aggregates all conditions into a ``<TARGET>_all``
    pool — used on the train side as the single-store training corpus.

    Returns True if the pool produced an output store, False if no
    plate contributed any positions for the requested filter (caller
    can count skips).
    """
    contribs = _gather_plate_contributions(
        target=target,
        condition=condition,
        split=split,
        plates=plates,
        plate_zarr_root=plate_zarr_root,
        authoring_root=authoring_root,
        stride_h=stride_h,
        window=window,
        tail_tol_h=tail_tol_h,
        t_cap=t_cap,
    )
    if not contribs:
        return False

    final_path = _pool_output_path(output_root, split, target, condition, output_format)
    sidecar_path = _provenance_sidecar_path(final_path)
    if not overwrite and final_path.exists():
        raise FileExistsError(
            f"Output store {final_path} already exists; refusing to overwrite. Set overwrite=true to replace."
        )
    if overwrite and final_path.exists():
        if final_path.is_dir():
            shutil.rmtree(final_path)
        else:
            final_path.unlink()
    if overwrite and sidecar_path.exists():
        sidecar_path.unlink()

    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve channel selection up front from the first plate. All
    # contributing plates must agree on output_channel_names because
    # the merged store carries a single channel_names list.
    first = contribs[0]
    with open_ome_zarr(first.plate_zarr_path, mode="r", layout="hcs") as src:
        sel = resolve_channels(
            src.channel_names,
            first.platemap.wells[first.positions[0][0]].gene_channel_map,
            target,
        )
        output_channel_names = sel.output_names

    inputs_hash = _assembly_inputs_hash(
        contribs=contribs,
        target=target,
        condition=condition,
        output_channel_names=output_channel_names,
        center_crop_yx=crop_yx,
        target_yx_pixel_size_um=target_yx_pixel_size_um,
    )

    label = condition if condition is not None else "all"
    print(f"Assembling pool {GENE_TO_FILENAME[target]}_{label} ({split})")
    print(f"  output: {final_path}")
    print(f"  contributing plates: {len(contribs)}")
    total_positions = sum(len(c.positions) for c in contribs)
    print(f"  total positions: {total_positions}")
    print(f"  crop_yx={crop_yx}, target_yx_pixel_size_um={target_yx_pixel_size_um}")
    per_plate_t = sorted({len(c.grid_frames) for c in contribs})
    print(f"  per-plate T: {per_plate_t} (heterogeneous T preserved)")

    # Choose whether to write directly to <final> or to a temp dir-zarr
    # that we pack into ozx after close. Pack rewrites entries in BFS
    # order (RFC-9 SHOULD), which a direct OzxStore write would not
    # guarantee.
    if output_format == "ozx":
        dir_path = final_path.with_suffix(".zarr.tmp")
        if dir_path.exists():
            shutil.rmtree(dir_path)
    else:
        dir_path = final_path

    provenance: dict[str, dict] = {}
    written_count = 0

    with contextlib.ExitStack() as stack:
        writer = stack.enter_context(
            open_ome_zarr(
                dir_path,
                mode="w",
                layout="hcs",
                channel_names=output_channel_names,
                version=_OME_NGFF_VERSION,
            )
        )

        # Open every contributing plate's source zarr exactly once.
        sources: dict[str, object] = {}
        for c in contribs:
            sources[c.plate_name] = stack.enter_context(open_ome_zarr(c.plate_zarr_path, mode="r", layout="hcs"))

        global_index = 0
        for c in contribs:
            src = sources[c.plate_name]
            native_channels: list[str] = src.channel_names

            actual_t = next(iter(src.positions()))[1]["0"].shape[0]
            if actual_t != c.platemap.native_t:
                raise ValueError(
                    f"native_t mismatch on {c.plate_name}: platemap says {c.platemap.native_t}, zarr has {actual_t}"
                )

            if target_yx_pixel_size_um is not None:
                yx = float(target_yx_pixel_size_um)
                eff_spacing = (c.spacing[0], yx, yx)
            else:
                eff_spacing = c.spacing
            transform = _scale_transform(eff_spacing)

            sel_by_well: dict[str, ChannelSelection] = {}
            for well_id, fov in tqdm(c.positions, desc=f"  {c.plate_name}", ncols=80):
                if well_id not in sel_by_well:
                    sel_by_well[well_id] = resolve_channels(
                        native_channels,
                        c.platemap.wells[well_id].gene_channel_map,
                        target,
                    )
                selection = sel_by_well[well_id]
                if selection.output_names != output_channel_names:
                    raise ValueError(
                        f"Output channel names drift for "
                        f"{c.plate_name}/{well_id}/{fov}: "
                        f"{selection.output_names} vs {output_channel_names}"
                    )

                source_arr = src[f"{well_id}/{fov}"]["0"]
                native_block = np.asarray(source_arr.oindex[c.native_frame_indices, selection.input_indices])
                if target_yx_pixel_size_um is not None:
                    native_block = _resample_yx_to_pixel_size(
                        native_block,
                        source_y_um=c.spacing[1],
                        source_x_um=c.spacing[2],
                        target_yx_um=float(target_yx_pixel_size_um),
                    )
                native_block = _apply_center_crop_yx(native_block, crop_yx)

                pool_fov = f"fov{global_index:04d}"
                pool_pos_name = f"0/0/{pool_fov}"
                out_pos = writer.create_position("0", "0", pool_fov)
                t_dim, c_dim, z_dim, y_dim, x_dim = native_block.shape
                z_chunk = min(_Z_CHUNK, z_dim)
                chunks = (1, 1, z_chunk, y_dim, x_dim)
                # Per-T shard: each shard holds (1, C, Z, Y, X), ~700 MB
                # uncompressed for our shapes. Stays safely under the 4 GB
                # zip-member limit `pack_ozx` enforces.
                shards_ratio = (
                    1,
                    c_dim,
                    (z_dim + z_chunk - 1) // z_chunk,
                    1,
                    1,
                )
                out_pos.create_image(
                    name="0",
                    data=native_block.astype(np.float32, copy=False),
                    chunks=chunks,
                    shards_ratio=shards_ratio,
                    transform=transform,
                )
                out_pos.zattrs.update(_provenance_attrs(c.platemap, well_id, fov, c.grid_frames, stride_h))
                provenance[pool_pos_name] = {
                    "plate_id": c.platemap.experiment,
                    "well_id": well_id,
                    "fov": fov,
                    "source_position": f"{well_id}/{fov}",
                    "condition": c.platemap.wells[well_id].condition,
                    "hpi_start": c.platemap.hpi_start,
                    "native_delta_t_min": c.platemap.native_delta_t_min,
                    "native_frame_indices": [int(f.native_idx) for f in c.grid_frames],
                    "spacing_zyx_source_um": list(c.spacing),
                    "spacing_zyx_effective_um": list(eff_spacing),
                }
                global_index += 1
                written_count += 1

        # Pin pool-level metadata to the store root: the input fingerprint
        # and the run-shaping knobs that produced these bytes.
        writer.zattrs.update(
            {
                "assembly_inputs_sha256": inputs_hash,
                "assembly_target": target,
                "assembly_condition": condition or "all",
                "assembly_split": split,
                "assembly_center_crop_yx": (list(crop_yx) if crop_yx is not None else None),
                "assembly_target_yx_pixel_size_um": target_yx_pixel_size_um,
                "assembly_t_cap": t_cap,
                "assembly_contributing_plates": [c.plate_name for c in contribs],
            }
        )

    # Normalization stats must be computed before any OZX packing because
    # ``.ozx`` archives are immutable on close (see iohub.core.ozx). Run
    # on the directory store (which is the final path for format=zarr,
    # or the tmp dir for format=ozx).
    print(f"  normalization stats: {dir_path}")
    _write_heterogeneous_t_normalization(dir_path)

    # Post-close: pack the tmp directory zarr into the final .ozx and
    # remove the tmp dir. Skip when the target is plain zarr.
    if output_format == "ozx":
        print(f"  packing ozx: {final_path}")
        pack_ozx(dir_path, final_path)
        shutil.rmtree(dir_path)

    # Publication artifact: raw-only .ozx colocated with the internal
    # store, dropping any *_deconvolved channel. Only meaningful when the
    # internal store is a directory zarr (format=zarr) that we can read
    # back post-close; for a format=ozx internal store the dir was already
    # removed. Targets with no deconvolved channel (e.g. dual_nucl_memb)
    # have nothing to drop, so the mirror is the internal store packed as
    # is — still emit it so the publication .ozx exists for every target.
    if emit_raw_only_ozx and output_format == "zarr":
        raw_only_path = final_path.with_suffix(".ozx")
        if raw_only_path.exists():
            raw_only_path.unlink()
        print(f"  deriving raw-only ozx: {raw_only_path}")
        wrote = _derive_raw_only_store(final_path, raw_only_path)
        if not wrote:
            print(f"  packing internal store as raw-only ozx: {raw_only_path}")
            pack_ozx(final_path, raw_only_path)

    sidecar_path.write_text(
        json.dumps(
            {
                "store_path": str(final_path),
                "target": target,
                "condition": condition or "all",
                "split": split,
                "assembly_inputs_sha256": inputs_hash,
                "positions": provenance,
            },
            indent=2,
            sort_keys=False,
        )
    )
    print(f"  provenance sidecar: {sidecar_path}")
    print(f"  wrote {written_count} positions across {len(contribs)} plate(s)")
    return True


def assemble_pool(cfg: DictConfig) -> None:
    """Assemble per-organelle × per-condition pooled stores.

    Walks every plate authored under ``authoring_root`` (or the requested
    subset via ``cfg.plates``). For each (target, condition, split) tuple
    in the matrix, writes one HCS store at
    ``<output_root>/<split>/<TARGET>_<CONDITION>.<ext>`` (extension =
    ``ozx`` by default for per-condition stores).

    On the train side, also emits ``<output_root>/train/<TARGET>_all.zarr``
    when ``emit_train_pool_all`` is true (default). The ``_all`` store
    is plain zarr (not ozx) so downstream code that expects an
    appendable directory store can use it directly.

    Parameters
    ----------
    cfg : DictConfig
        OmegaConf config with keys:

        - ``targets`` (list[str]): subset of {sec61b, tomm20, h2b, caax}
          plus the combined key ``dual_nucl_memb`` (both v2 fluor genes
          → one ``[Phase3D, Brightfield, Nuclei, Membrane]`` pool).
        - ``conditions`` (list[str]): subset of {mock, ZIKV, DENV}.
        - ``output_root`` (Path): destination tree. Pools land at
          ``<output_root>/<split>/<TARGET>_<CONDITION>.<ext>``.
        - ``plate_zarr_root`` (Path): root of per-plate source zarrs.
        - ``grid`` (``stride_h`` / ``window`` / ``tail_tol_h``).

        Optional:

        - ``plates`` (list[str]): restrict the contributing plate set.
          Defaults to every plate authored under ``authoring_root``.
        - ``authoring_root`` (Path): defaults to packaged
          ``_configs/datasets/a549-mantis/authoring/``.
        - ``overwrite`` (bool, default False).
        - ``splits`` (list[str], default ``[train, test]``).
        - ``t_cap`` (int | None, default 10).
        - ``train_center_crop_yx`` / ``test_center_crop_yx``.
        - ``train_target_yx_pixel_size_um`` /
          ``test_target_yx_pixel_size_um``.
        - ``condition_format`` (default ``zarr``): per-condition stores.
          ``zarr`` keeps the internal per-condition store as a directory
          zarr (required to also emit the raw-only ozx below); ``ozx``
          packs each per-condition store directly as an archive.
        - ``pool_all_format`` (default ``zarr``): ``_all`` train store.
        - ``emit_train_pool_all`` (bool, default True).
        - ``emit_raw_only_ozx`` (bool, default False): when the internal
          per-condition store is a directory zarr, also derive a colocated
          raw-only ``.ozx`` (drops every ``*_deconvolved`` channel) as the
          publication artifact. No-op for genes without a deconvolved
          channel (nuclei/membrane).
    """
    targets = [str(t) for t in cfg.targets]
    conditions = [str(c) for c in cfg.conditions]
    for c in conditions:
        if c not in VALID_CONDITIONS:
            raise ValueError(f"condition {c!r} not in {VALID_CONDITIONS}")
    for t in targets:
        if t not in GENE_TO_FILENAME:
            raise ValueError(f"target {t!r} not in {sorted(GENE_TO_FILENAME)}")
        # Validate combined-target constituents resolve (raises on unknown).
        resolve_target_genes(t)

    output_root = Path(cfg.output_root)
    plate_zarr_root = Path(cfg.plate_zarr_root)
    authoring_root = Path(cfg.authoring_root) if cfg.get("authoring_root") else _default_authoring_root()
    overwrite = bool(cfg.get("overwrite", False))

    plates_cfg = cfg.get("plates")
    if not plates_cfg:
        plates = _list_authored_plates(authoring_root)
    else:
        plates = [str(p) for p in plates_cfg]

    splits_cfg = cfg.get("splits", list(_SPLIT_NAMES))
    splits = tuple(str(s) for s in splits_cfg)
    if not splits:
        raise ValueError("splits must contain at least one of train/test")
    invalid = [s for s in splits if s not in _SPLIT_NAMES]
    if invalid:
        raise ValueError(f"splits contains unknown name(s) {invalid!r}; expected {_SPLIT_NAMES}")

    condition_format = str(cfg.get("condition_format") or _DEFAULT_CONDITION_FORMAT)
    pool_all_format = str(cfg.get("pool_all_format") or _DEFAULT_POOL_ALL_FORMAT)
    for fmt, key in (
        (condition_format, "condition_format"),
        (pool_all_format, "pool_all_format"),
    ):
        if fmt not in _VALID_FORMATS:
            raise ValueError(f"{key}={fmt!r} not in {_VALID_FORMATS}")

    t_cap_cfg = cfg.get("t_cap", _DEFAULT_T_CAP)
    if t_cap_cfg is None:
        t_cap: int | None = None
    else:
        t_cap = int(t_cap_cfg)
        if t_cap <= 0:
            raise ValueError(f"t_cap must be a positive int or null, got {t_cap_cfg!r}")

    crop_defaults: dict[str, tuple[int, int] | None] = {
        "train": _DEFAULT_TRAIN_CENTER_CROP_YX,
        "test": _DEFAULT_TEST_CENTER_CROP_YX,
    }
    crop_by_split: dict[str, tuple[int, int] | None] = {
        split: _parse_crop_cfg(cfg, f"{split}_center_crop_yx", crop_defaults[split]) for split in _SPLIT_NAMES
    }
    pixel_defaults: dict[str, float | None] = {
        "train": _DEFAULT_TRAIN_TARGET_YX_PIXEL_SIZE_UM,
        "test": _DEFAULT_TEST_TARGET_YX_PIXEL_SIZE_UM,
    }
    target_yx_pixel_size_by_split: dict[str, float | None] = {
        split: _parse_pixel_size_cfg(cfg, f"{split}_target_yx_pixel_size_um", pixel_defaults[split])
        for split in _SPLIT_NAMES
    }

    emit_train_pool_all = bool(cfg.get("emit_train_pool_all", _DEFAULT_EMIT_TRAIN_POOL_ALL))
    emit_raw_only_ozx = bool(cfg.get("emit_raw_only_ozx", _DEFAULT_EMIT_RAW_ONLY_OZX))

    _check_output_safety(output_root, plate_zarr_root)

    stride_h = float(cfg.grid.stride_h)
    window = (float(cfg.grid.window[0]), float(cfg.grid.window[1]))
    tail_tol_h = float(cfg.grid.tail_tol_h)

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Assembling pools: targets={targets} conditions={conditions} splits={list(splits)}")
    print(f"  output root:        {output_root}")
    print(f"  authoring root:     {authoring_root}")
    print(f"  plate_zarr_root:    {plate_zarr_root}")
    print(f"  contributing plates: {len(plates)}")
    print(f"  condition_format:   {condition_format}")
    print(f"  pool_all_format:    {pool_all_format}")
    print(f"  t_cap:              {t_cap}")
    for split in splits:
        print(
            f"  {split}: target_yx_pixel_size_um="
            f"{target_yx_pixel_size_by_split[split]}, "
            f"center_crop_yx={crop_by_split[split]}"
        )
    print(f"  emit_train_pool_all: {emit_train_pool_all}")
    print(f"  emit_raw_only_ozx:  {emit_raw_only_ozx}")

    written = 0
    skipped = 0
    for target in targets:
        for split in splits:
            for condition in conditions:
                ok = _assemble_one_pool(
                    target=target,
                    condition=condition,
                    split=split,
                    plates=plates,
                    plate_zarr_root=plate_zarr_root,
                    authoring_root=authoring_root,
                    output_root=output_root,
                    stride_h=stride_h,
                    window=window,
                    tail_tol_h=tail_tol_h,
                    overwrite=overwrite,
                    crop_yx=crop_by_split[split],
                    target_yx_pixel_size_um=target_yx_pixel_size_by_split[split],
                    t_cap=t_cap,
                    output_format=condition_format,
                    emit_raw_only_ozx=emit_raw_only_ozx,
                )
                if ok:
                    written += 1
                else:
                    skipped += 1
                    print(f"  skip pool {GENE_TO_FILENAME[target]}_{condition} ({split}): no contributing positions")

        if emit_train_pool_all and "train" in splits:
            ok = _assemble_one_pool(
                target=target,
                condition=None,
                split="train",
                plates=plates,
                plate_zarr_root=plate_zarr_root,
                authoring_root=authoring_root,
                output_root=output_root,
                stride_h=stride_h,
                window=window,
                tail_tol_h=tail_tol_h,
                overwrite=overwrite,
                crop_yx=crop_by_split["train"],
                target_yx_pixel_size_um=target_yx_pixel_size_by_split["train"],
                t_cap=t_cap,
                output_format=pool_all_format,
            )
            if ok:
                written += 1
            else:
                skipped += 1
                print(f"  skip pool {GENE_TO_FILENAME[target]}_all (train): no contributing positions")

    print(f"Done. {written} pool(s) written, {skipped} skipped.")
