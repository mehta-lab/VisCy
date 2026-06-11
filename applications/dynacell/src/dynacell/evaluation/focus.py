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

from dataclasses import dataclass

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
    return FocusSlabConfig(
        channel_name=str(OmegaConf.select(cfg, "channel_name", default="Phase3D")),
        halfwidth=int(OmegaConf.select(cfg, "halfwidth", default=2)),
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


def build_focus_slabs(position, *, channel_name: str, halfwidth: int, t_count: int) -> list[slice]:
    """Per-timepoint in-focus slabs for ``position`` from its ``focus_slice`` zattrs.

    Reads the cached focus plane per timepoint (written by ``dynacell precompute-gt
    build.focus=true``) and centers a ``2*halfwidth + 1`` plane slab on each. Raises
    if the metadata is absent — fail loud rather than silently fall back to a
    full-volume projection while ``focus_slab`` is enabled.

    ``position`` is the **GT** position (``(T, C, Z, Y, X)``); the same slabs apply
    to the prediction, which maps slice-by-slice.
    """
    z_total = position.data.shape[2]
    slabs: list[slice] = []
    for t in range(t_count):
        plane = read_focus_plane(position, channel_name, t)
        if plane is None:
            raise ValueError(
                f"feature_metrics.focus_slab is enabled but no focus_slice[{channel_name!r}] "
                "metadata exists at this GT position. Run `dynacell precompute-gt build.focus=true` "
                "first, or set feature_metrics.focus_slab.enabled=false."
            )
        slabs.append(focus_slab_from_plane(plane, z_total, halfwidth))
    return slabs


def read_focus_plane(position, channel_name: str, t: int) -> int | None:
    """Read the cached per-timepoint focus plane from a position's zattrs.

    Returns ``None`` when no ``focus_slice`` metadata exists for ``channel_name``
    (caller falls back to a fixed-fraction plane). Mirrors the layout written by
    :func:`write_focus_slice_metadata`.
    """
    focus_meta = position.zattrs.get(FOCUS_FIELD, {}).get(channel_name)
    if focus_meta is None:
        return None
    per_t = focus_meta.get("per_timepoint")
    if per_t is not None and str(t) in per_t:
        return int(per_t[str(t)])
    return int(round(focus_meta["dataset_statistics"]["z_focus_mean"]))


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
