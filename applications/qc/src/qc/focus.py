"""In-focus z-slice detection using midband spatial frequency power."""

from pathlib import Path

import numpy as np
import torch
from iohub import open_ome_zarr
from waveorder.focus import focus_from_transverse_band

from qc.qc_metrics import QCMetric


class FocusSliceMetric(QCMetric):
    """In-focus z-slice detection using midband spatial frequency power.

    Parameters
    ----------
    NA_det : float
        Detection numerical aperture.
    lambda_ill : float
        Illumination wavelength (same units as pixel_size).
    pixel_size : float
        Object-space pixel size (camera pixel size / magnification).
    channel_names : list[str]
        Channel names to compute focus for.
    midband_fractions : tuple[float, float]
        Inner and outer fractions of cutoff frequency.
    device : str
        Torch device for FFT computation (e.g. "cpu", "cuda").
    """

    field_name = "focus_slice"

    def __init__(
        self,
        NA_det: float,
        lambda_ill: float,
        pixel_size: float,
        channel_names: list[str],
        midband_fractions: tuple[float, float] = (0.125, 0.25),
        device: str = "cpu",
    ):
        self.NA_det = NA_det
        self.lambda_ill = lambda_ill
        self.pixel_size = pixel_size
        self.channel_names = channel_names
        self.midband_fractions = midband_fractions
        self.device = torch.device(device)

    def channels(self) -> list[str]:
        """Return the channels this metric is configured for."""
        return self.channel_names

    def __call__(self, position, channel_name, channel_index, num_workers=4):
        """Compute focus-slice index per timepoint for one channel of ``position``."""
        # Tensorstore concurrency is configured on the plate at
        # open-time (see qc_metrics.generate_qc_metadata); num_workers
        # is retained here only to match the QCMetric abstract interface.
        del num_workers
        tzyx = position["0"].native[:, channel_index].read().result()

        T = tzyx.shape[0]
        focus_indices = np.empty(T, dtype=int)

        for t in range(T):
            zyx = torch.as_tensor(np.asarray(tzyx[t]), device=self.device)
            focus_indices[t] = focus_from_transverse_band(
                zyx,
                NA_det=self.NA_det,
                lambda_ill=self.lambda_ill,
                pixel_size=self.pixel_size,
                midband_fractions=self.midband_fractions,
            )

        per_timepoint = {str(t): int(idx) for t, idx in enumerate(focus_indices)}
        fov_stats = {
            "z_focus_mean": float(np.mean(focus_indices)),
            "z_focus_std": float(np.std(focus_indices)),
        }
        return {
            "fov_statistics": fov_stats,
            "per_timepoint": per_timepoint,
        }

    def aggregate_dataset(self, all_results: list[dict]) -> dict:
        """Compute dataset-level focus statistics across all positions.

        Parameters
        ----------
        all_results : list[dict]
            List of dicts returned by ``__call__`` for each position.

        Returns
        -------
        dict
            Dataset-level z-focus statistics.
        """
        all_values = []
        for result in all_results:
            all_values.extend(result["per_timepoint"].values())
        arr = np.array(all_values, dtype=float)
        return {
            "z_focus_mean": float(np.mean(arr)),
            "z_focus_std": float(np.std(arr)),
            "z_focus_min": int(np.min(arr)),
            "z_focus_max": int(np.max(arr)),
        }


def audit_focus_slice(zarr_path: str | Path, channel_name: str) -> dict:
    """Audit already-written ``focus_slice`` metadata for suspect detections.

    Reads each FOV's ``focus_slice.{channel_name}.per_timepoint`` indices (written
    by :class:`FocusSliceMetric`) and flags timepoints where detection landed at a
    stack edge (``z == 0`` or ``z == Z - 1``) — the signature of a failed in-focus
    search on a 3D stack. The check is **Z-depth-aware**: for a 2D acquisition
    (``Z == 1``) ``focus_from_transverse_band`` returns the trivial slice ``0`` for
    every timepoint, so z=0 is expected and nothing is flagged.

    Parameters
    ----------
    zarr_path : str or Path
        OME-Zarr HCS plate whose positions carry ``focus_slice`` zattrs.
    channel_name : str
        Channel whose focus indices to audit (must be present in the metadata).

    Returns
    -------
    dict
        Summary with keys:

        ``z_depth`` : int
            Z size of the stack.
        ``is_2d`` : bool
            ``True`` when ``z_depth == 1`` (edge check is skipped, no flags).
        ``n_fovs`` : int
            FOVs carrying focus metadata for ``channel_name``.
        ``n_timepoints_total`` : int
            Total (fov, timepoint) focus indices audited.
        ``n_suspect`` : int
            Count of edge (``0`` or ``Z-1``) indices — always ``0`` when 2D.
        ``fovs_affected`` : dict[str, int]
            ``{fov_name: suspect_timepoint_count}`` for FOVs with any suspect index.
        ``valid_focus`` : dict
            ``min``/``max``/``mean``/``median`` over non-edge indices (``None`` if 2D
            or no valid index exists).
    """
    per_fov: dict[str, np.ndarray] = {}
    z_depth: int | None = None
    with open_ome_zarr(zarr_path, mode="r") as plate:
        for name, pos in plate.positions():
            focus_meta = pos.zattrs.get("focus_slice", {}).get(channel_name)
            if focus_meta is None:
                continue
            if z_depth is None:
                z_depth = int(pos["0"].shape[-3])
            per_tp = focus_meta["per_timepoint"]
            per_fov[name] = np.array([per_tp[str(t)] for t in range(len(per_tp))], dtype=int)

    if not per_fov:
        raise KeyError(f"No focus_slice metadata for channel {channel_name!r} in {zarr_path}")

    is_2d = z_depth == 1
    all_idx = np.concatenate(list(per_fov.values()))
    if is_2d:
        suspect_mask_by_fov = {name: np.zeros_like(idx, dtype=bool) for name, idx in per_fov.items()}
    else:
        suspect_mask_by_fov = {name: (idx == 0) | (idx == z_depth - 1) for name, idx in per_fov.items()}

    fovs_affected = {name: int(mask.sum()) for name, mask in suspect_mask_by_fov.items() if mask.any()}
    n_suspect = sum(fovs_affected.values())

    valid = None if is_2d else all_idx[(all_idx != 0) & (all_idx != z_depth - 1)]
    valid_focus = None
    if valid is not None and valid.size:
        valid_focus = {
            "min": int(valid.min()),
            "max": int(valid.max()),
            "mean": float(valid.mean()),
            "median": float(np.median(valid)),
        }

    return {
        "z_depth": z_depth,
        "is_2d": is_2d,
        "n_fovs": len(per_fov),
        "n_timepoints_total": int(all_idx.size),
        "n_suspect": n_suspect,
        "fovs_affected": fovs_affected,
        "valid_focus": valid_focus,
    }
