#!/usr/bin/env python
r"""Extract an in-focus slab OME-Zarr from a 3D train store (DynaCell 2D VS track).

Phase A of the 2D in-focus VS model track. Given a 3D OME-Zarr train store and a
phase channel name, this writes a ``<name>_focus.zarr`` where every position holds
the ``2*halfwidth + 1``-plane in-focus slab of **all** channels (phase + targets),
centered on the per-FOV/per-timepoint waveorder focus plane. A ``z_window_size=1``
2D training run then reads the slab as ``2*halfwidth + 1`` in-focus windows per
FOV/timepoint.

Design
------
Focus reuse
    The focus plane is estimated with
    :func:`dynacell.evaluation.focus.estimate_focus_plane` (which wraps
    ``waveorder.focus.focus_from_transverse_band`` on the ``midband`` transverse
    band) on the ``Phase3D`` ``(Z, Y, X)`` volume per FOV/timepoint. The estimator
    is *imported*, never reimplemented, so the training slab lands on the same
    in-focus plane the focus-aware eval derives (train/eval cannot diverge). The
    ``midband_fractions`` band is the module constant ``MIDBAND_FRACTIONS`` — not a
    parameter — so it already matches the eval.
Pixel size
    ``na_det`` / ``lambda_ill`` / ``pixel_size`` set the transverse-band cutoff.
    ``pixel_size`` defaults to the **store's lateral spacing** (``position.scale[-1]``)
    rather than a hardcoded value, because iPSC stores carry a different (sometimes
    placeholder) spacing than the A549 mantis ``0.1494`` — pass ``--pixel-size`` to
    override when a store's recorded scale is not the physical spacing.
Clamp-shift window
    Unlike the eval's :func:`~dynacell.evaluation.focus.focus_slab_from_plane`
    (which *clips* the slab at the stack caps, yielding fewer planes near an edge),
    this tool **clamp-shifts** the whole window inward into ``[hw, Z-1-hw]`` so every
    FOV/timepoint yields a uniform ``2*hw+1`` depth. That keeps the derived zarr
    rectangular and the ``SlidingWindowDataset`` window count uniform per FOV.
Channels & zattrs
    Every channel is copied at the per-timepoint slab z-range. The ``omero``
    rendering metadata and all custom position/plate zattrs (``normalization``,
    provenance, condition, ...) are preserved verbatim. The source ``normalization``
    stats are **full-stack** (per-FOV over all Z); they are copied as-is (not
    recomputed over the thin slab) and a log line notes this — per-FOV
    ``NormalizeSampled`` stats are close enough on the in-focus band that re-use is
    the intended behavior.

Layout is zarr v3 / OME-Zarr v0.5 end to end via :func:`iohub.open_ome_zarr`.

Usage::

    python applications/dynacell/tools/extract_focus_slab_store.py \\
        --input  /hpc/.../a549/mantis/train/dual_nucl_memb_all.zarr \\
        --output /hpc/.../a549/mantis/train/dual_nucl_memb_all_focus.zarr \\
        --phase-channel Phase3D --halfwidth 2

Importing this module pulls in ``dynacell.evaluation.focus`` (cubic + waveorder at
import), so run it in the dynacell-eval env, not the bare ``.venv``. The focus
argmax itself runs on CPU.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from iohub.ngff import TransformationMeta, open_ome_zarr

from dynacell.evaluation.focus import (
    DEFAULT_LAMBDA_ILL,
    DEFAULT_NA_DET,
    MIDBAND_FRACTIONS,
    estimate_focus_plane,
)

logger = logging.getLogger(__name__)

# ``DEFAULT_NA_DET`` / ``DEFAULT_LAMBDA_ILL`` are imported from ``focus`` so the training
# slab shares the eval estimator's acquisition constants (single source of truth).
DEFAULT_HALFWIDTH: int = 2

# zattrs key iohub manages itself (multiscales / axes / omero). Copied metadata
# must never clobber it, so it is excluded from the verbatim zattr copy.
_OME_METADATA_KEY: str = "ome"


@dataclass
class ExtractionSummary:
    """Result of an :func:`extract_focus_slab_store` run.

    Attributes
    ----------
    output_store : str
        Absolute path of the written focus-slab store.
    n_planes : int
        Slab depth (``2*halfwidth + 1``); uniform across all positions/timepoints.
    pixel_size : float
        Lateral spacing used for focus estimation.
    focus_planes : dict of str to list of int
        Per-position list of estimated focus z-planes (one per timepoint).
    slab_starts : dict of str to list of int
        Per-position list of clamp-shifted slab start z-indices (one per timepoint).
    n_positions : int
        Number of positions written (derived: ``len(focus_planes)``).
    """

    output_store: str
    n_planes: int
    pixel_size: float
    focus_planes: dict[str, list[int]] = field(default_factory=dict)
    slab_starts: dict[str, list[int]] = field(default_factory=dict)

    @property
    def n_positions(self) -> int:
        """Number of positions written (one entry per position in ``focus_planes``)."""
        return len(self.focus_planes)


def clamp_shift_slab(z_focus: int, z_total: int, halfwidth: int) -> slice:
    """Return a ``slice`` of exactly ``2*halfwidth + 1`` planes centered on ``z_focus``.

    Unlike :func:`dynacell.evaluation.focus.focus_slab_from_plane` (which clips at the
    stack caps and can yield fewer planes near an edge), the whole window is
    **shifted inward** into ``[0, z_total - width]`` so the returned slab always spans
    ``2*halfwidth + 1`` planes. This keeps the derived zarr rectangular.

    Parameters
    ----------
    z_focus : int
        Estimated in-focus z-plane.
    z_total : int
        Number of z-planes in the source volume.
    halfwidth : int
        Planes on each side of the center; the slab spans ``2*halfwidth + 1``.

    Returns
    -------
    slice
        A z-slice of length ``2*halfwidth + 1``.

    Raises
    ------
    ValueError
        If ``halfwidth`` is negative, or the stack is too thin for a full slab
        (``z_total < 2*halfwidth + 1``).
    """
    if halfwidth < 0:
        raise ValueError(f"halfwidth must be >= 0, got {halfwidth}")
    width = 2 * halfwidth + 1
    if z_total < width:
        raise ValueError(
            f"stack has {z_total} planes but a halfwidth={halfwidth} slab needs {width}; "
            "cannot clamp-shift a full slab out of a shorter stack"
        )
    lo = min(max(z_focus - halfwidth, 0), z_total - width)
    return slice(lo, lo + width)


def _custom_zattrs(node) -> dict:
    """Return a node's custom zattrs (everything except the iohub-managed ``ome`` block)."""
    return {k: v for k, v in dict(node.zattrs).items() if k != _OME_METADATA_KEY}


def extract_focus_slab_store(
    input_path: str | Path,
    output_path: str | Path,
    *,
    phase_channel: str = "Phase3D",
    halfwidth: int = DEFAULT_HALFWIDTH,
    limit_positions: int | None = None,
    overwrite: bool = False,
    pixel_size: float | None = None,
    na_det: float = DEFAULT_NA_DET,
    lambda_ill: float = DEFAULT_LAMBDA_ILL,
    device: str = "cpu",
) -> ExtractionSummary:
    """Write a ``<name>_focus.zarr`` holding the in-focus slab of every channel.

    For each position and timepoint, the focus plane is estimated on the
    ``phase_channel`` volume with :func:`estimate_focus_plane`, then a
    :func:`clamp_shift_slab` window of ``2*halfwidth + 1`` planes is cut from **all**
    channels. ``omero`` and all custom zattrs are preserved.

    Parameters
    ----------
    input_path : str or Path
        Source 3D OME-Zarr HCS store.
    output_path : str or Path
        Destination focus-slab store.
    phase_channel : str, optional
        Channel used to estimate the focus plane (default ``"Phase3D"``).
    halfwidth : int, optional
        Half the slab depth; the slab spans ``2*halfwidth + 1`` planes (default 2).
    limit_positions : int or None, optional
        If set, only the first ``limit_positions`` positions are extracted (smoke runs).
    overwrite : bool, optional
        Replace an existing ``output_path`` (default False → raise if it exists).
    pixel_size : float or None, optional
        Lateral spacing for the focus estimator. ``None`` (default) reads the store's
        ``position.scale[-1]``.
    na_det, lambda_ill : float, optional
        Detection NA and illumination wavelength for the transverse-band estimator.
    device : str, optional
        Torch device for the focus argmax (default ``"cpu"``).

    Returns
    -------
    ExtractionSummary
        Written store path, counts, and per-position focus planes / slab starts.

    Raises
    ------
    FileExistsError
        If ``output_path`` exists and ``overwrite`` is False.
    ValueError
        If ``phase_channel`` is absent, ``limit_positions < 1``, or a stack is too
        thin for a full slab.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if limit_positions is not None and limit_positions < 1:
        raise ValueError(f"limit_positions must be >= 1, got {limit_positions}")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass overwrite=True to replace it")
        shutil.rmtree(output_path)

    width = 2 * halfwidth + 1

    with open_ome_zarr(input_path, mode="r") as src:
        if phase_channel not in src.channel_names:
            raise ValueError(f"phase channel {phase_channel!r} not in store channels {list(src.channel_names)}")
        phase_idx = src.channel_names.index(phase_channel)
        positions = list(src.positions())
        if limit_positions is not None:
            positions = positions[:limit_positions]
        if not positions:
            raise ValueError(f"{input_path} has no positions to extract")

        resolved_pixel_size = float(positions[0][1].scale[-1]) if pixel_size is None else float(pixel_size)
        summary = ExtractionSummary(output_store=str(output_path), n_planes=width, pixel_size=resolved_pixel_size)
        logger.info(
            "focus estimator: channel=%s pixel_size=%g (%s) na_det=%g lambda_ill=%g midband=%s device=%s",
            phase_channel,
            resolved_pixel_size,
            "store scale[-1]" if pixel_size is None else "override",
            na_det,
            lambda_ill,
            MIDBAND_FRACTIONS,
            device,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open_ome_zarr(
            output_path, layout="hcs", mode="w-", channel_names=list(src.channel_names), version="0.5"
        ) as dst:
            plate_custom = _custom_zattrs(src)
            if plate_custom:
                dst.zattrs.update(plate_custom)
            dst.zattrs.update(
                {
                    "focus_slab_extraction": {
                        "source_store": str(input_path),
                        "phase_channel": phase_channel,
                        "halfwidth": halfwidth,
                        "n_planes": width,
                        "pixel_size": resolved_pixel_size,
                        "na_det": na_det,
                        "lambda_ill": lambda_ill,
                        "midband_fractions": list(MIDBAND_FRACTIONS),
                    }
                }
            )

            norm_note_emitted = False
            for name, pos in positions:
                data = pos.data
                t_count, _, z_total = data.shape[:3]
                if z_total < width:
                    raise ValueError(
                        f"{name}: stack has {z_total} planes but a halfwidth={halfwidth} slab needs {width}"
                    )
                phase_tzyx = np.asarray(data[:, phase_idx])
                planes = [
                    estimate_focus_plane(
                        phase_tzyx[t],
                        na_det=na_det,
                        lambda_ill=lambda_ill,
                        pixel_size=resolved_pixel_size,
                        device=device,
                    )
                    for t in range(t_count)
                ]
                slabs = [clamp_shift_slab(p, z_total, halfwidth) for p in planes]
                slab_stack = np.stack([np.asarray(data[t, :, slabs[t]]) for t in range(t_count)])
                if np.isnan(slab_stack).any():
                    raise ValueError(f"{name}: NaN encountered in extracted slab")

                row, col, fov = name.split("/")
                new_pos = dst.create_position(row, col, fov)
                chunks = (1, 1, width, slab_stack.shape[-2], slab_stack.shape[-1])
                new_pos.create_image(
                    "0",
                    slab_stack,
                    chunks=chunks,
                    transform=[TransformationMeta(type="scale", scale=list(pos.scale))],
                )
                new_pos.metadata.omero = pos.metadata.omero
                new_pos.dump_meta()

                pos_custom = _custom_zattrs(pos)
                if "normalization" in pos_custom and not norm_note_emitted:
                    logger.warning(
                        "copying full-stack `normalization` zattrs verbatim onto the %d-plane focus slab "
                        "(per-FOV stats NOT recomputed over the slab); re-use is intended per plan R3",
                        width,
                    )
                    norm_note_emitted = True
                if pos_custom:
                    new_pos.zattrs.update(pos_custom)

                summary.focus_planes[name] = planes
                summary.slab_starts[name] = [s.start for s in slabs]
                logger.info("wrote %s: focus planes=%s slab starts=%s", name, planes, summary.slab_starts[name])

    logger.info(
        "done: %d positions (%d planes each) -> %s", summary.n_positions, summary.n_planes, summary.output_store
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: extract an in-focus slab store from a 3D train store."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="source 3D OME-Zarr HCS store (absolute path)")
    ap.add_argument("--output", required=True, type=Path, help="destination <name>_focus.zarr (absolute path)")
    ap.add_argument("--phase-channel", default="Phase3D", help="channel for focus estimation (default: Phase3D)")
    ap.add_argument("--halfwidth", type=int, default=DEFAULT_HALFWIDTH, help="slab half-depth; slab = 2*hw+1 (def: 2)")
    ap.add_argument("--limit-positions", type=int, default=None, help="only extract the first N positions (smoke)")
    ap.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="lateral spacing for the focus estimator (default: read from store scale[-1])",
    )
    ap.add_argument("--na-det", type=float, default=DEFAULT_NA_DET, help=f"detection NA (default: {DEFAULT_NA_DET})")
    ap.add_argument(
        "--lambda-ill",
        type=float,
        default=DEFAULT_LAMBDA_ILL,
        help=f"illumination wavelength (def: {DEFAULT_LAMBDA_ILL})",
    )
    ap.add_argument("--device", default="cpu", help="torch device for the focus argmax (default: cpu)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output store")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    extract_focus_slab_store(
        args.input,
        args.output,
        phase_channel=args.phase_channel,
        halfwidth=args.halfwidth,
        limit_positions=args.limit_positions,
        overwrite=args.overwrite,
        pixel_size=args.pixel_size,
        na_det=args.na_det,
        lambda_ill=args.lambda_ill,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
