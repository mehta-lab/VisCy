#!/usr/bin/env python
"""Add ``otsu_threshold`` to a training store WITHOUT rewriting its other statistics.

``viscy preprocess --compute_otsu`` is the obvious way to get the Otsu thresholds
Spotlight needs. It is the wrong way for the DynaCell iPSC training stores, because
it always calls ``generate_normalization_metadata``, which recomputes and REPLACES
``mean``/``std``/``median``/``iqr`` at three levels (``fov_statistics``,
``dataset_statistics``, every ``timepoint_statistics`` entry) on both the position
nodes and the plate node. Every published iPSC model normalizes against those values.

That replacement is only harmless if recomputation is bit-exact, and on
``cell_focus.zarr`` it is NOT. Measured on ``4/10615/1304``, channel ``Nuclei``:

    stored (inherited)   mean 417.3512  std 18.0563  max 690.0
    recomputed on slab   mean 419.8103  std 20.8461  max 635.0

The focus store was carved from the full-Z ``cell.zarr`` by
``extract_focus_slab_store.py`` and inherited its parent's ``normalization`` block
verbatim -- byte-identical on all 500 of 500 positions. So the stored statistics
describe 44 planes while the store holds 5. Running the stock CLI would silently
re-base the target normalization of every 2D arm that trains on this store, which
would confound any spotlight-vs-baseline comparison with a normalization change.

This tool computes the SAME threshold the stock path computes -- identical grid
sample, identical median filter, identical ``threshold_otsu`` call, identical
constant-input guard -- and writes only that one key, preserving every other
statistic byte-for-byte. ``generate_fg_masks`` reads
``normalization.<channel>.fov_statistics.otsu_threshold`` and so runs unmodified
afterwards.

Rollback is the inverse and is exact: delete the one key per (position, channel).
``--undo`` does it.
"""

from __future__ import annotations

import argparse
import sys

from iohub import ngff
from iohub.core.config import TensorStoreConfig
from skimage.filters import threshold_otsu
from tqdm import tqdm

from viscy_utils.meta_utils import _grid_sample, smooth_median, write_meta_field

_OTSU_KEY = "otsu_threshold"


def compute_otsu_threshold(position: ngff.Position, channel_index: int, grid_spacing: int) -> float:
    """Compute one FOV's Otsu threshold exactly as ``generate_normalization_metadata`` does.

    Parameters
    ----------
    position : ngff.Position
        Position node to sample.
    channel_index : int
        Index of the channel within the store's channel list.
    grid_spacing : int
        Stride of the sampling grid. Denser than the statistics grid to capture
        inter-cell gaps; the stock default is 8.

    Returns
    -------
    float
        The Otsu threshold, or the constant value itself for a constant input
        (Otsu is undefined there, and returning the constant makes
        ``generate_fg_masks`` mark the whole FOV as foreground-free).
    """
    samples = _grid_sample(position, grid_spacing, channel_index)
    smoothed = smooth_median(samples, size=(1, 1, 3, 3))
    flat = smoothed.ravel()
    if flat.min() == flat.max():
        return float(flat.min())
    return float(threshold_otsu(flat))


def main(argv: list[str] | None = None) -> int:
    """Write (or remove) ``otsu_threshold`` across every position of a store."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="Path to the HCS OME-Zarr training store.")
    ap.add_argument("--channel", action="append", required=True, dest="channels", help="Target channel (repeatable).")
    ap.add_argument("--otsu-grid-spacing", type=int, default=8, help="Sampling stride; stock default 8.")
    ap.add_argument("--num-workers", type=int, default=8, help="tensorstore data_copy_concurrency.")
    ap.add_argument("--dry-run", action="store_true", help="Compute and report; write nothing.")
    ap.add_argument("--undo", action="store_true", help="Delete the key instead of writing it.")
    args = ap.parse_args(argv)

    mode = "r" if args.dry_run else "r+"
    with ngff.open_ome_zarr(
        args.store,
        mode=mode,
        implementation="tensorstore",
        implementation_config=TensorStoreConfig(data_copy_concurrency=args.num_workers),
    ) as plate:
        missing = [c for c in args.channels if c not in plate.channel_names]
        if missing:
            print(f"[FAIL] channels not in store: {missing}; have {plate.channel_names}", file=sys.stderr)
            return 2
        indices = {c: plate.channel_names.index(c) for c in args.channels}
        positions = list(plate.positions())

        # Refuse a partial store outright. _collate_norm_meta takes its stat-key set
        # from sample 0 of the batch and indexes every peer with it, so a store where
        # only SOME positions carry otsu_threshold raises KeyError only on batches
        # whose sample 0 happens to have it -- intermittently, hours into a run, and
        # on BASELINE fits too, not just spotlight ones.
        present = sum(
            _OTSU_KEY in pos.zattrs["normalization"][c]["fov_statistics"] for _, pos in positions for c in args.channels
        )
        total = len(positions) * len(args.channels)
        print(args.store)
        print(f"  {len(positions)} positions x {len(args.channels)} channels; {present}/{total} carry {_OTSU_KEY}")
        if not args.undo and present == total:
            print("  nothing to do: already complete.")
            return 0
        if args.undo and present == 0:
            print("  nothing to do: already absent.")
            return 0

        written = 0
        for name, pos in tqdm(positions, desc="undo" if args.undo else "otsu"):
            for channel, index in indices.items():
                # write_meta_field merges at the CHANNEL level but replaces the whole
                # sub-dict it is handed, so pass fov_statistics back in full: that is
                # what makes this additive. dataset_statistics, timepoint_statistics,
                # the plate node and the `ome` block are never touched.
                stats = dict(pos.zattrs["normalization"][channel]["fov_statistics"])
                if args.undo:
                    if stats.pop(_OTSU_KEY, None) is None:
                        continue
                else:
                    stats[_OTSU_KEY] = compute_otsu_threshold(pos, index, args.otsu_grid_spacing)
                if args.dry_run:
                    if written < 3:
                        print(f"  [dry-run] {name}/{channel}: {_OTSU_KEY}={stats.get(_OTSU_KEY)}")
                else:
                    write_meta_field(pos, {"fov_statistics": stats}, "normalization", channel)
                written += 1

    verb = "would change" if args.dry_run else ("removed" if args.undo else "written")
    print(f"[ok] {written} (position, channel) pairs {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
