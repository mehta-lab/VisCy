#!/usr/bin/env python
r"""Build a timepoint-subset OME-Zarr from an A549 pooled train store (Phase 17).

Phase 17 of the A549 campaign asks whether the benchmark's **temporal diversity**
carries training signal at a **fixed frame budget**. The comparison is two FNet3D
arms that differ in *nothing* but which timepoints they see:

``--mode early``
    The first ``--n-timepoints`` frames of every position. On the A549 pools that is
    hpi 5 and 7 — a two-hour window at the start of infection.
``--mode spread``
    ``--n-timepoints`` frames per position, spread over that position's own range by
    a deterministic rule (no RNG, see :func:`spread_timepoints`): plate-order
    ordinal ``i`` with ``T_i`` frames takes ``t_a = i % T_i`` and
    ``t_b = (t_a + T_i // 2) % T_i``. Each position therefore contributes widely
    separated frames, and the arm collectively covers the whole time course.

Why a derived store instead of a datamodule flag
------------------------------------------------
Timepoints are the ``T`` axis *inside* a position, not separate FOVs, and there is
no timepoint filter anywhere in ``HCSDataModule`` / ``SlidingWindowDataset``:
``SlidingWindowDataset._read_img_window`` decodes ``t`` arithmetically from a
contiguous ``T x Z`` window index, so restricting ``T`` would mean rewriting that
hot path for every model in the benchmark. Subsetting on disk keeps the training
config delta down to a single ``data.init_args.data_path`` line, which is what makes
the ablation clean.

Equal budgets fall out of the construction: both arms keep **every** position and
take the **same** number of frames from each, so the (position, timepoint) count,
the pooled mock/DENV/ZIKV condition mix, and — because ``HCSDataModule`` splits by
position under ``seed_everything: 0`` — the train/val position split are all
identical across arms.

Metadata handling
-----------------
Copied verbatim
    ``normalization.<channel>.fov_statistics``. Both arms then normalize
    identically, which removes a confound; the information "leaked" from unseen
    frames is two scalars per channel. Recomputing per-subset stats would instead
    make the arms differ in normalization as well as in temporal coverage.
Re-keyed
    ``normalization.<channel>.timepoint_statistics`` is a ``{t_index: stats}`` map
    that ``SlidingWindowDataset._resolve_timepoint_norm_meta`` indexes with the
    **new** ``t``. Copied verbatim it would silently point frame 0 of a ``spread``
    store at the parent's frame-0 stats. It is re-keyed to the new indices.
Subset
    Per-timepoint temporal metadata (``hpi_values``, ``native_frame_indices``,
    ``tick_hpi_values``) is filtered to the kept indices, so ``hpi_values`` keeps
    naming the real hours-post-infection of the frames actually present.

Layout is zarr v3 / OME-Zarr v0.5 end to end via :func:`iohub.open_ome_zarr`, and
source chunking is preserved.

Usage::

    python applications/dynacell/tools/build_temporal_subset_zarr.py \\
        --source /hpc/.../a549/mantis_v1/train/H2B_all.zarr \\
        --dest   /hpc/.../a549/mantis_v1/train/H2B_t01.zarr \\
        --channels Phase3D Nuclei --mode early --n-timepoints 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from iohub.ngff import TransformationMeta, open_ome_zarr

logger = logging.getLogger(__name__)

# zattrs key iohub manages itself (multiscales / axes / omero); never clobber it.
_OME_METADATA_KEY: str = "ome"

# Position zattrs whose entries are per-timepoint and must be filtered to the kept
# indices rather than copied whole.
_PER_TIMEPOINT_KEYS: tuple[str, ...] = ("hpi_values", "native_frame_indices", "tick_hpi_values")


@dataclass
class SubsetSummary:
    """Result of a :func:`build_temporal_subset_zarr` run.

    Attributes
    ----------
    dest_store : str
        Absolute path of the written subset store.
    mode : str
        Selection mode used (``"early"`` or ``"spread"``).
    kept : dict of str to list of int
        Per-position source timepoint indices that were copied, in order.
    n_positions : int
        Number of positions written (derived).
    n_pairs : int
        Total (position, timepoint) pairs written (derived).
    """

    dest_store: str
    mode: str
    kept: dict[str, list[int]] = field(default_factory=dict)

    @property
    def n_positions(self) -> int:
        """Number of positions written."""
        return len(self.kept)

    @property
    def n_pairs(self) -> int:
        """Total (position, timepoint) pairs written."""
        return sum(len(v) for v in self.kept.values())


def early_timepoints(n_frames: int, n_keep: int) -> list[int]:
    """Return the first ``n_keep`` timepoint indices of a position.

    Parameters
    ----------
    n_frames : int
        Number of timepoints the source position has.
    n_keep : int
        Number of timepoints to keep.

    Returns
    -------
    list of int
        ``[0, 1, ..., n_keep - 1]``.

    Raises
    ------
    ValueError
        If the position has fewer than ``n_keep`` timepoints.
    """
    if n_frames < n_keep:
        raise ValueError(f"position has {n_frames} timepoints, cannot keep {n_keep}")
    return list(range(n_keep))


def spread_timepoints(ordinal: int, n_frames: int, n_keep: int) -> list[int]:
    """Return ``n_keep`` timepoint indices spread over a position's own range.

    Deterministic by design — the arm must be reproducible from the plate order
    alone, with no RNG and no seed to record. Position ``ordinal`` starts its
    sequence at ``ordinal % n_frames`` and steps by ``n_frames // n_keep``, wrapping.
    Stepping by a fixed fraction of the position's own length maximizes within-
    position temporal separation, while the per-position offset makes the arm cover
    the whole time course collectively.

    Parameters
    ----------
    ordinal : int
        Zero-based index of the position in plate order.
    n_frames : int
        Number of timepoints the source position has.
    n_keep : int
        Number of timepoints to keep.

    Returns
    -------
    list of int
        Sorted timepoint indices, ``len == n_keep``. Distinct by construction: the
        largest offset ``(n_keep - 1) * (n_frames // n_keep)`` is strictly below
        ``n_frames``, so the picks cannot wrap onto each other.

    Raises
    ------
    ValueError
        If the position has fewer than ``n_keep`` timepoints.
    """
    if n_frames < n_keep:
        raise ValueError(f"position has {n_frames} timepoints, cannot keep {n_keep}")
    step = n_frames // n_keep
    start = ordinal % n_frames
    return sorted((start + k * step) % n_frames for k in range(n_keep))


def _custom_zattrs(node) -> dict:
    """Return a node's custom zattrs (everything except the iohub-managed ``ome`` block)."""
    return {k: v for k, v in dict(node.zattrs).items() if k != _OME_METADATA_KEY}


def _subset_position_zattrs(custom: dict, kept: list[int], n_frames: int) -> dict:
    """Filter per-timepoint zattrs and re-key ``timepoint_statistics`` to new indices.

    Parameters
    ----------
    custom : dict
        The source position's custom zattrs (no ``ome`` block).
    kept : list of int
        Source timepoint indices being copied, in destination order.
    n_frames : int
        Source timepoint count, used to recognize per-timepoint lists by length.

    Returns
    -------
    dict
        zattrs for the destination position.

    Raises
    ------
    KeyError
        If a ``timepoint_statistics`` block is missing an entry for a kept index —
        that means the source metadata does not describe the frames being copied,
        and silently dropping it would let the arm train under wrong statistics.
    """
    out = dict(custom)
    for key in _PER_TIMEPOINT_KEYS:
        values = out.get(key)
        # Only filter when the entry really is one-per-timepoint; a scalar or a
        # differently-shaped list (e.g. figure axis ticks) is copied untouched.
        if isinstance(values, list) and len(values) == n_frames:
            out[key] = [values[t] for t in kept]
    norm = out.get("normalization")
    if norm is not None:
        out["normalization"] = {
            channel: {
                level: (
                    {str(new): stats[str(old)] for new, old in enumerate(kept)}
                    if level == "timepoint_statistics"
                    else stats
                )
                for level, stats in levels.items()
            }
            for channel, levels in norm.items()
        }
    return out


def build_temporal_subset_zarr(
    source_path: str | Path,
    dest_path: str | Path,
    *,
    channels: list[str],
    mode: str,
    n_timepoints: int,
) -> SubsetSummary:
    """Write a timepoint-subset copy of an HCS store, keeping every position.

    Parameters
    ----------
    source_path : str or Path
        Source pooled train store (read-only).
    dest_path : str or Path
        Destination store. Must not exist — these feed training runs, so an
        accidental overwrite of a store a fit is reading is not offered.
    channels : list of str
        Channel names to copy, in destination order (e.g. ``["Phase3D", "Nuclei"]``).
    mode : {"early", "spread"}
        Timepoint selection rule.
    n_timepoints : int
        Timepoints to keep per position.

    Returns
    -------
    SubsetSummary
        Destination path and the per-position source indices copied.

    Raises
    ------
    FileExistsError
        If ``dest_path`` already exists.
    ValueError
        If ``mode`` is unknown, ``n_timepoints < 1``, a requested channel is absent,
        the source has no positions, or a copied volume contains NaN.
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    if mode not in ("early", "spread"):
        raise ValueError(f"mode must be 'early' or 'spread', got {mode!r}")
    if n_timepoints < 1:
        raise ValueError(f"n_timepoints must be >= 1, got {n_timepoints}")
    if dest_path.exists():
        raise FileExistsError(f"{dest_path} already exists; refusing to overwrite a training store")

    summary = SubsetSummary(dest_store=str(dest_path), mode=mode)

    with open_ome_zarr(source_path, mode="r") as src:
        missing = [c for c in channels if c not in src.channel_names]
        if missing:
            raise ValueError(f"channels {missing} not in {source_path} channels {list(src.channel_names)}")
        ch_idx = [src.channel_names.index(c) for c in channels]
        positions = list(src.positions())
        if not positions:
            raise ValueError(f"{source_path} has no positions")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open_ome_zarr(dest_path, layout="hcs", mode="w-", channel_names=list(channels), version="0.5") as dst:
            plate_custom = _custom_zattrs(src)
            if plate_custom:
                dst.zattrs.update(plate_custom)
            dst.zattrs.update(
                {
                    "temporal_subset": {
                        "source_store": str(source_path),
                        "mode": mode,
                        "n_timepoints": n_timepoints,
                        "channels": list(channels),
                    }
                }
            )

            for ordinal, (name, pos) in enumerate(positions):
                n_frames = pos.data.shape[0]
                kept = (
                    early_timepoints(n_frames, n_timepoints)
                    if mode == "early"
                    else spread_timepoints(ordinal, n_frames, n_timepoints)
                )
                subset = np.stack([np.asarray(pos.data[t, ch_idx]) for t in kept])
                if np.isnan(subset).any():
                    raise ValueError(f"{name}: NaN encountered in subset array")

                row, col, fov = name.split("/")
                new_pos = dst.create_position(row, col, fov)
                src_chunks = tuple(pos.data.chunks)
                new_pos.create_image(
                    "0",
                    subset,
                    chunks=src_chunks,
                    transform=[TransformationMeta(type="scale", scale=list(pos.scale))],
                )
                pos_custom = _subset_position_zattrs(_custom_zattrs(pos), kept, n_frames)
                pos_custom["temporal_subset"] = {
                    "source_store": str(source_path),
                    "mode": mode,
                    "source_timepoints": kept,
                    "source_n_timepoints": n_frames,
                }
                new_pos.zattrs.update(pos_custom)
                # omero.channels is what `channel_names` reads back, so it must be
                # narrowed to the copied channels — assigning the source's block
                # verbatim would advertise channels the array does not have.
                source_omero = pos.metadata.omero
                if source_omero is not None:
                    new_pos.metadata.omero = source_omero.model_copy(
                        update={"channels": [source_omero.channels[i] for i in ch_idx]}
                    )
                    new_pos.dump_meta()

                summary.kept[name] = kept
                hpi = pos_custom.get("hpi_values")
                logger.info("wrote %s: source t=%s hpi=%s", name, kept, hpi)

    _write_provenance(source_path, dest_path, channels, summary, n_timepoints)
    logger.info(
        "done: %d positions, %d (position, timepoint) pairs -> %s",
        summary.n_positions,
        summary.n_pairs,
        summary.dest_store,
    )
    return summary


def _write_provenance(
    source_path: Path,
    dest_path: Path,
    channels: list[str],
    summary: SubsetSummary,
    n_timepoints: int,
) -> Path:
    """Write the colocated ``.provenance.json`` sidecar and return its path."""
    sidecar = dest_path.with_suffix(".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "kind": "temporal_subset",
                "source_store": str(source_path),
                "dest_store": str(dest_path),
                "mode": summary.mode,
                "n_timepoints": n_timepoints,
                "channels": list(channels),
                "n_positions": summary.n_positions,
                "n_pairs": summary.n_pairs,
                "positions": {name: kept for name, kept in summary.kept.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return sidecar


def verify_temporal_subset(source_path: str | Path, dest_path: str | Path, summary: SubsetSummary) -> None:
    """Assert every destination frame is byte-identical to its source frame.

    The subset is a pure copy, so equality is exact — ``np.array_equal``, not a
    tolerance. Checking every frame (not a sample) is affordable because the arms
    are small, and it is the only thing standing between a mis-indexed ``spread``
    rule and a silently wrong ablation.

    Parameters
    ----------
    source_path, dest_path : str or Path
        The two stores to compare.
    summary : SubsetSummary
        The mapping of destination position to source timepoint indices.

    Raises
    ------
    ValueError
        On any position-set, channel, shape, or content mismatch.
    """
    with open_ome_zarr(source_path, mode="r") as src, open_ome_zarr(dest_path, mode="r") as dst:
        dst_names = {name for name, _ in dst.positions()}
        if dst_names != set(summary.kept):
            raise ValueError(f"destination positions {sorted(dst_names)} != written {sorted(summary.kept)}")
        src_ch = [src.channel_names.index(c) for c in dst.channel_names]
        for name, kept in summary.kept.items():
            src_arr = src[name].data
            dst_arr = dst[name].data
            if dst_arr.shape[0] != len(kept):
                raise ValueError(f"{name}: destination has {dst_arr.shape[0]} frames, expected {len(kept)}")
            for new_t, old_t in enumerate(kept):
                expected = np.asarray(src_arr[old_t, src_ch])
                got = np.asarray(dst_arr[new_t])
                if not np.array_equal(expected, got):
                    raise ValueError(f"{name}: destination t={new_t} != source t={old_t}")
    logger.info("verified: %d (position, timepoint) pairs byte-identical to source", summary.n_pairs)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build one timepoint-subset store and verify it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="source pooled train store (absolute path)")
    ap.add_argument("--dest", required=True, type=Path, help="destination subset store (absolute path)")
    ap.add_argument(
        "--channels",
        required=True,
        nargs="+",
        help="channels to copy, in destination order (e.g. Phase3D Nuclei)",
    )
    ap.add_argument("--mode", required=True, choices=("early", "spread"), help="timepoint selection rule")
    ap.add_argument("--n-timepoints", type=int, default=2, help="timepoints to keep per position (default: 2)")
    ap.add_argument("--skip-verify", action="store_true", help="skip the byte-equality re-read pass")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = build_temporal_subset_zarr(
        args.source,
        args.dest,
        channels=args.channels,
        mode=args.mode,
        n_timepoints=args.n_timepoints,
    )
    if not args.skip_verify:
        verify_temporal_subset(args.source, args.dest, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
