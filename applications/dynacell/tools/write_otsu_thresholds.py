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

``--threshold-from PARENT`` computes each threshold on a different store's
matching position instead of this one's. ``cell_focus.zarr`` needs it: the recipe
smooths across Z with a 5x5x5 median and a sigma=5 Gaussian, and a 5-plane slab
has no Z to accumulate over, so estimating there inverts the recipe's effect.
Measured over 60 positions, mean foreground on the focus store's own masks:

                                  Nuclei          Membrane
    on disk (old recipe)          0.3319          0.2496 (2 FOVs under 2%)
    new recipe on the 5 planes    0.3317          0.2189 (2 FOVs under 2%)
    new recipe on the parent      0.3846          0.1861 (0 FOVs under 2%)

The slab is a verbatim voxel copy of the parent, so the parent is the same
specimen on the same intensity scale -- and sharing one estimate keeps the 2D
arms (which train on the focus store) and the 3D arms (which train on the
parent) from being handed masks built by differently-behaving operators.

Rollback is the inverse and is exact: delete the one key per (position, channel).
``--undo`` does it.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack

from iohub import ngff
from iohub.core.config import TensorStoreConfig
from tqdm import tqdm

from viscy_utils.meta_utils import _grid_sample, otsu_threshold_from_volume, write_meta_field

_OTSU_KEY = "otsu_threshold"


def compute_otsu_threshold(position: ngff.Position, channel_index: int) -> float:
    """Compute one FOV's Otsu threshold exactly as ``generate_normalization_metadata`` does.

    Delegates to :func:`viscy_utils.meta_utils.otsu_threshold_from_volume` rather
    than reimplementing the recipe, so this tool cannot drift from the stock
    preprocess pass -- the whole point of the tool is that the value it writes is
    the one the pipeline would have written.

    Parameters
    ----------
    position : ngff.Position
        Position node to read.
    channel_index : int
        Index of the channel within the store's channel list.

    Returns
    -------
    float
        The Otsu threshold, or the constant value itself for a constant input
        (Otsu is undefined there).
    """
    return otsu_threshold_from_volume(_grid_sample(position, 1, channel_index))


def _resolve_source(plate, parent, name, pos):
    """Return the position the threshold is computed on, and its channel list.

    Raises rather than falling back when the parent lacks the position: a silent
    per-position fallback would give the store two different threshold recipes,
    which is the exact asymmetry --threshold-from exists to remove.
    """
    if parent is None:
        return pos, plate.channel_names
    if name not in _parent_names(parent):
        raise KeyError(f"position {name!r} is absent from the --threshold-from store")
    return parent[name], parent.channel_names


def _parent_names(parent):
    """Position names of the parent plate, cached on the plate object.

    iohub's Plate.__contains__ returns False for a valid nested "row/col/fov"
    path that indexes fine, so membership has to be tested against the list.
    """
    if not hasattr(parent, "_cached_position_names"):
        parent._cached_position_names = {n for n, _ in parent.positions()}
    return parent._cached_position_names


def main(argv: list[str] | None = None) -> int:
    """Write (or remove) ``otsu_threshold`` across every position of a store."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="Path to the HCS OME-Zarr training store.")
    ap.add_argument("--channel", action="append", required=True, dest="channels", help="Target channel (repeatable).")
    ap.add_argument("--num-workers", type=int, default=8, help="tensorstore data_copy_concurrency.")
    ap.add_argument("--dry-run", action="store_true", help="Compute and report; write nothing.")
    ap.add_argument("--undo", action="store_true", help="Delete the key instead of writing it.")
    ap.add_argument(
        "--threshold-from",
        default=None,
        help="Compute each threshold on this store's matching position instead of the target's.",
    )
    args = ap.parse_args(argv)

    mode = "r" if args.dry_run else "r+"
    config = TensorStoreConfig(data_copy_concurrency=args.num_workers)
    with ExitStack() as stack:
        plate = stack.enter_context(
            ngff.open_ome_zarr(args.store, mode=mode, implementation="tensorstore", implementation_config=config)
        )
        parent = None
        if args.threshold_from is not None:
            parent = stack.enter_context(
                ngff.open_ome_zarr(
                    args.threshold_from, mode="r", implementation="tensorstore", implementation_config=config
                )
            )
            print(f"  thresholds computed on {args.threshold_from}")
        missing = [c for c in args.channels if c not in plate.channel_names]
        if missing:
            print(f"[FAIL] channels not in store: {missing}; have {plate.channel_names}", file=sys.stderr)
            return 2
        indices = {c: plate.channel_names.index(c) for c in args.channels}
        if parent is not None:
            missing_parent = [c for c in args.channels if c not in parent.channel_names]
            if missing_parent:
                print(f"[FAIL] channels not in --threshold-from store: {missing_parent}", file=sys.stderr)
                return 2
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
            for channel in indices:
                # write_meta_field merges at the CHANNEL level but replaces the whole
                # sub-dict it is handed, so pass fov_statistics back in full: that is
                # what makes this additive. dataset_statistics, timepoint_statistics,
                # the plate node and the `ome` block are never touched.
                stats = dict(pos.zattrs["normalization"][channel]["fov_statistics"])
                if args.undo:
                    if stats.pop(_OTSU_KEY, None) is None:
                        continue
                else:
                    source, source_channels = _resolve_source(plate, parent, name, pos)
                    stats[_OTSU_KEY] = compute_otsu_threshold(source, source_channels.index(channel))
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
