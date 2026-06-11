"""In-focus z-slice detection and ``focus_slice`` metadata (DynaCLR-compatible).

Centering a 2-D projection slab on the *in-focus* plane (instead of a fixed depth
fraction) is what keeps a max-Z projection from being dominated by out-of-focus
caps. The focal plane is estimated with
:func:`waveorder.focus.focus_from_transverse_band` — the same midband
spatial-frequency-power estimator that ``qc.FocusSliceMetric`` wraps — computed on
the **phase** channel (the label-free VS input present in every GT store), so the
plane is organelle-independent and shared by GT + prediction.

The written ``focus_slice`` zattrs layout matches what DynaCLR's ``z_range="auto"``
reads (``focus_slice[<channel>].dataset_statistics.z_focus_mean`` on the plate, plus
``fov_statistics`` / ``per_timepoint`` per position), so the metadata is
interoperable. The estimator and the zattrs writer are pulled from external/package
deps (``waveorder``, ``viscy_utils``) — never from the ``qc`` application — to keep
the dependency graph ``applications/ → packages/`` only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig, OmegaConf
from waveorder.focus import focus_from_transverse_band

from viscy_utils.meta_utils import write_meta_field

FOCUS_FIELD = "focus_slice"
MIDBAND_FRACTIONS: tuple[float, float] = (0.125, 0.25)


@dataclass(frozen=True)
class FocusSlabConfig:
    """Resolved ``feature_metrics.focus_slab`` settings (only when enabled).

    Attributes
    ----------
    channel_name : str
        GT phase channel whose ``focus_slice`` zattrs supply the focus plane.
    halfwidth : int
        Planes on each side of the focus plane; the slab spans
        ``2*halfwidth + 1`` planes.
    """

    channel_name: str
    halfwidth: int


def read_focus_slab_config(config: DictConfig) -> FocusSlabConfig | None:
    """Resolve ``feature_metrics.focus_slab`` from a config, or ``None`` when off.

    Returns ``None`` when the block is absent or ``enabled`` is false (the
    default), so every call site shares one source of truth for the toggle and
    the ``channel_name`` / ``halfwidth`` defaults.
    """
    cfg = OmegaConf.select(config, "feature_metrics.focus_slab", default=None)
    if cfg is None or not bool(OmegaConf.select(cfg, "enabled", default=False)):
        return None
    halfwidth = int(OmegaConf.select(cfg, "halfwidth", default=2))
    if halfwidth < 0:
        raise ValueError(
            f"feature_metrics.focus_slab.halfwidth must be >= 0, got {halfwidth} "
            "(a negative halfwidth yields an empty slab and crashes the max-Z projection)."
        )
    return FocusSlabConfig(
        channel_name=str(OmegaConf.select(cfg, "channel_name", default="Phase3D")),
        halfwidth=halfwidth,
    )


@dataclass(frozen=True)
class FocusComputeConfig:
    """Resolved ``focus`` block: physical params for computing the focus plane.

    Shared by the deep-feature slab path and the instance-seg ``slice_selection=focus``
    path so both estimate the plane identically. ``pixel_size`` defaults to the lateral
    spacing (``pixel_metrics.spacing[-1]``); ``na_det`` / ``lambda_ill`` are the mantis
    acquisition defaults.

    Attributes
    ----------
    channel_name : str
        Phase channel the focus plane is estimated from.
    na_det, lambda_ill, pixel_size : float
        Detection NA, illumination wavelength, object-space lateral pixel size.
    device : str
        Torch device for the estimator (``"cpu"`` or ``"cuda"``).
    """

    channel_name: str
    na_det: float
    lambda_ill: float
    pixel_size: float
    device: str


def read_focus_compute_config(config: DictConfig, *, channel_name: str | None = None) -> FocusComputeConfig:
    """Resolve the ``focus`` compute block, defaulting ``pixel_size`` to the lateral spacing.

    ``channel_name`` overrides ``focus.channel_name`` (e.g. the instance path passes
    ``segmentation.focus_channel_name``). Used for the eval-time compute-if-absent path,
    so focus works on read-only published ``.ozx`` stores that carry no ``focus_slice``
    zattrs (the plane is derived from the phase channel already inside the store).
    """
    pixel_size = OmegaConf.select(config, "focus.pixel_size", default=None)
    if pixel_size is None:
        pixel_size = float(config.pixel_metrics.spacing[-1])
    return FocusComputeConfig(
        channel_name=channel_name or str(OmegaConf.select(config, "focus.channel_name", default="Phase3D")),
        na_det=float(OmegaConf.select(config, "focus.na_det", default=1.35)),
        lambda_ill=float(OmegaConf.select(config, "focus.lambda_ill", default=0.450)),
        pixel_size=float(pixel_size),
        device=str(OmegaConf.select(config, "focus.device", default="cpu")),
    )


def estimate_focus_plane(
    zyx: np.ndarray, *, na_det: float, lambda_ill: float, pixel_size: float, device: str = "cpu"
) -> int:
    """Return the best-focus z index of a ``(Z, Y, X)`` volume.

    Wraps :func:`waveorder.focus.focus_from_transverse_band` (midband
    transverse-band power). ``na_det`` / ``lambda_ill`` / ``pixel_size`` are the
    detection NA, illumination wavelength, and object-space pixel size (same
    length units).
    """
    return int(
        focus_from_transverse_band(
            torch.as_tensor(np.asarray(zyx), device=device),
            NA_det=na_det,
            lambda_ill=lambda_ill,
            pixel_size=pixel_size,
            midband_fractions=MIDBAND_FRACTIONS,
        )
    )


def focus_slab_from_plane(z_focus: int, z_total: int, halfwidth: int) -> slice:
    """Return a ``slice`` of ``2*halfwidth + 1`` planes centered on ``z_focus``.

    Clipped to ``[0, z_total)``. ``halfwidth=0`` selects the single focus plane.
    """
    return slice(max(0, z_focus - halfwidth), min(z_total, z_focus + halfwidth + 1))


def build_focus_slabs(
    position,
    *,
    halfwidth: int,
    t_count: int,
    compute: FocusComputeConfig,
    cache_dir: str | Path | None = None,
    pos_name: str | None = None,
) -> list[slice]:
    """Per-timepoint in-focus slabs for ``position``, centered on the focus plane.

    The plane is resolved by :func:`resolve_focus_planes` (precomputed zattrs →
    ``gt_cache_dir`` cache → compute-from-phase + cache), then a ``2*halfwidth + 1``
    plane slab is centered on each. ``position`` is the **GT** position
    (``(T, C, Z, Y, X)``); the same slabs apply to the prediction, which maps
    slice-by-slice.
    """
    z_total = position.data.shape[2]
    planes = resolve_focus_planes(position, t_count=t_count, compute=compute, cache_dir=cache_dir, pos_name=pos_name)
    return [focus_slab_from_plane(plane, z_total, halfwidth) for plane in planes]


def _planes_from_zattrs(position, channel_name: str, t_count: int) -> list[int] | None:
    """Per-timepoint focus planes from a position's ``focus_slice`` zattrs, or ``None``.

    Returns ``None`` when no ``focus_slice[channel_name]`` metadata exists (caller then
    computes from the phase channel). Falls back to the dataset-mean plane for any
    timepoint missing from ``per_timepoint`` (DynaCLR ``z_range="auto"`` interop).
    """
    focus_meta = position.zattrs.get(FOCUS_FIELD, {}).get(channel_name)
    if focus_meta is None:
        return None
    per_t = focus_meta.get("per_timepoint") or {}
    fallback = focus_meta.get("dataset_statistics", {}).get("z_focus_mean")
    planes: list[int] = []
    for t in range(t_count):
        if str(t) in per_t:
            planes.append(int(per_t[str(t)]))
        elif fallback is not None:
            planes.append(int(round(fallback)))
        else:
            return None
    return planes


def _focus_cache_path(cache_dir: str | Path, channel_name: str, pos_name: str) -> Path:
    """Per-position focus-plane cache file under ``<cache_dir>/focus_planes/<channel>/``."""
    return Path(cache_dir) / "focus_planes" / channel_name / f"{pos_name.replace('/', '__')}.json"


def _read_focus_cache(
    cache_dir: str | Path,
    channel_name: str,
    pos_name: str,
    t_count: int,
    na_det: float,
    lambda_ill: float,
    pixel_size: float,
) -> list[int] | None:
    """Read cached focus planes, or ``None`` on miss / param-mismatch / short cache."""
    path = _focus_cache_path(cache_dir, channel_name, pos_name)
    if not path.is_file():
        return None
    rec = json.loads(path.read_text())
    params = rec.get("params", {})
    if (params.get("na_det"), params.get("lambda_ill"), params.get("pixel_size")) != (na_det, lambda_ill, pixel_size):
        return None
    planes = rec.get("planes", [])
    if len(planes) < t_count:
        return None
    return [int(p) for p in planes[:t_count]]


def _write_focus_cache(
    cache_dir: str | Path,
    channel_name: str,
    pos_name: str,
    planes: list[int],
    na_det: float,
    lambda_ill: float,
    pixel_size: float,
) -> None:
    """Atomically persist focus planes (tmp + ``os.replace``) so parallel evals don't tear writes."""
    path = _focus_cache_path(cache_dir, channel_name, pos_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {"na_det": na_det, "lambda_ill": lambda_ill, "pixel_size": pixel_size},
        "planes": [int(p) for p in planes],
    }
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def resolve_focus_planes(
    position,
    *,
    t_count: int,
    compute: FocusComputeConfig,
    cache_dir: str | Path | None = None,
    pos_name: str | None = None,
) -> list[int]:
    """Per-timepoint focus planes for ``position``. Source precedence:

    1. ``focus_slice`` zattrs in the store (precomputed by ``precompute-gt build.focus``;
       DynaCLR-compatible) — the iPSC writable-store fast path,
    2. the ``gt_cache_dir`` focus cache (computed-and-persisted) — lets focus-aware eval
       run on **read-only published ``.ozx``** that carry no zattrs,
    3. compute from the position's phase (``compute.channel_name``) volume + persist.

    Computing from phase is cheap and deterministic, so (2)/(3) reproduce the same planes
    anyone could derive from the published data — no need to fork the immutable store.
    """
    channel_name = compute.channel_name
    planes = _planes_from_zattrs(position, channel_name, t_count)
    if planes is not None:
        return planes
    if cache_dir is not None and pos_name is not None:
        cached = _read_focus_cache(
            cache_dir, channel_name, pos_name, t_count, compute.na_det, compute.lambda_ill, compute.pixel_size
        )
        if cached is not None:
            return cached
    channel_index = list(position.channel_names).index(channel_name)
    tzyx = np.asarray(position.data[:, channel_index])
    planes = [
        estimate_focus_plane(
            tzyx[t],
            na_det=compute.na_det,
            lambda_ill=compute.lambda_ill,
            pixel_size=compute.pixel_size,
            device=compute.device,
        )
        for t in range(t_count)
    ]
    if cache_dir is not None and pos_name is not None:
        _write_focus_cache(
            cache_dir, channel_name, pos_name, planes, compute.na_det, compute.lambda_ill, compute.pixel_size
        )
    return planes


def write_focus_slice_metadata(
    plate_path: str,
    *,
    channel_name: str,
    na_det: float,
    lambda_ill: float,
    pixel_size: float,
    device: str = "cpu",
) -> dict:
    """Compute per-(position, timepoint) focus planes and write ``focus_slice`` zattrs.

    Mirrors ``qc.FocusSliceMetric`` + ``generate_qc_metadata``: writes
    ``focus_slice[channel_name].dataset_statistics`` on the plate and
    ``{fov_statistics, per_timepoint, dataset_statistics}`` on each position.

    The store is opened ``mode="r+"`` — the target must be writable. Packed
    ``.ozx`` stores are read-only; estimate against the unpacked OME-Zarr and
    repackage, or run this on a writable copy.

    Returns the dataset-level statistics dict.
    """
    with open_ome_zarr(plate_path, mode="r+") as plate:
        channel_index = plate.channel_names.index(channel_name)
        per_position: list[tuple[object, list[int]]] = []
        all_planes: list[int] = []
        for _, pos in plate.positions():
            tzyx = np.asarray(pos.data[:, channel_index])
            planes = [
                estimate_focus_plane(
                    tzyx[t], na_det=na_det, lambda_ill=lambda_ill, pixel_size=pixel_size, device=device
                )
                for t in range(tzyx.shape[0])
            ]
            per_position.append((pos, planes))
            all_planes.extend(planes)

        arr = np.asarray(all_planes, dtype=float)
        dataset_stats = {
            "z_focus_mean": float(arr.mean()),
            "z_focus_std": float(arr.std()),
            "z_focus_min": int(arr.min()),
            "z_focus_max": int(arr.max()),
        }
        write_meta_field(plate, {"dataset_statistics": dataset_stats}, FOCUS_FIELD, channel_name)
        for pos, planes in per_position:
            a = np.asarray(planes, dtype=float)
            write_meta_field(
                pos,
                {
                    "fov_statistics": {"z_focus_mean": float(a.mean()), "z_focus_std": float(a.std())},
                    "per_timepoint": {str(t): int(v) for t, v in enumerate(planes)},
                    "dataset_statistics": dataset_stats,
                },
                FOCUS_FIELD,
                channel_name,
            )
        return dataset_stats
