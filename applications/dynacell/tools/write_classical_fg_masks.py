r"""Write ``fg_mask`` built with the eval's classical ER / mito binarizer.

This wrote the ``fg_mask`` arrays in the iPSC training stores
``/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/train/SEC61B.zarr``
(423 positions, 2026-10-01) and ``.../train/TOMM20.zarr`` (500 positions,
2026-10-01/02), which the ER and mito Spotlight-v2 arms
(``generate_spotlight_v2_leaves.py``, ``fg_mask_key: fg_mask``) train on. That
run predates the provenance attributes below, so those arrays carry none, and
it used ``segment`` as it stood before #519 moved its ER/mito path from
aicssegmentation to cubic.

Same array layout as ``viscy_utils.meta_utils.generate_fg_masks`` (uint8,
``(T, C, Z, Y, X)``, source chunking capped at 512, non-target channels filled
with 1) so ``HCSDataModule(fg_mask_key="fg_mask")`` reads it unchanged. Only the
binarizer differs: ``dynacell.evaluation.segmentation.segment``, the exact
operator the Dice readout applies to GT and prediction. Every array written
records its provenance in its zarr attributes under ``provenance``. Touches no
``normalization`` metadata, and refuses to overwrite an existing mask array.

Needs the eval venv. Shard over positions with ``--shard i/n``; each shard
writes disjoint positions.

Run::

    uv run --no-sync python applications/dynacell/tools/write_classical_fg_masks.py \
        <store.zarr> --target-name er --shard 0/8
"""

import argparse
import datetime
import time
from importlib.metadata import version

import numpy as np
from iohub import open_ome_zarr

from dynacell.evaluation.segmentation import segment

WRITER = "applications/dynacell/tools/write_classical_fg_masks.py"
BINARIZER = "dynacell.evaluation.segmentation.segment"


def main(argv: list[str] | None = None) -> int:
    """Write the classical-binarizer ``fg_mask`` for every position of a shard.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("--target-name", required=True, choices=["er", "mitochondria"])
    ap.add_argument("--channel", default="Structure")
    ap.add_argument("--fg-mask-key", default="fg_mask")
    ap.add_argument("--shard", default="0/1", help="i/n: process positions[i::n]")
    ap.add_argument("--dry-run", action="store_true", help="Segment and report; write nothing.")
    args = ap.parse_args(argv)
    i, n = (int(v) for v in args.shard.split("/"))
    if n <= 0 or not 0 <= i < n:
        raise ValueError(f"--shard {args.shard!r}: need 0 <= i < n")
    provenance = {
        "writer": WRITER,
        "binarizer": BINARIZER,
        "target_name": args.target_name,
        "channel": args.channel,
        "cubic_version": version("cubic"),
        "date": datetime.date.today().isoformat(),
    }

    with open_ome_zarr(args.store, mode="r" if args.dry_run else "r+") as plate:
        ch_idx = plate.channel_names.index(args.channel)
        positions = list(plate.positions())[i::n]
        for k, (pos_name, pos) in enumerate(positions):
            if not args.dry_run and args.fg_mask_key in pos:
                raise FileExistsError(f"'{args.fg_mask_key}' already exists at {pos_name}")
            img = pos["0"]
            t_total, c_total, *zyx = img.shape
            t0 = time.perf_counter()
            masks = [
                np.asarray(segment(np.asarray(img[t, ch_idx], dtype=np.float32), args.target_name))
                for t in range(t_total)
            ]
            frac = float(np.mean([m.mean() for m in masks]))
            if not args.dry_run:
                src = img.chunks
                arr = pos.create_zeros(
                    args.fg_mask_key,
                    shape=(t_total, c_total, *zyx),
                    dtype=np.uint8,
                    chunks=(1, 1, min(src[2], zyx[0]), min(src[3], 512), min(src[4], 512)),
                )
                arr.native.attrs["provenance"] = provenance
                for c in sorted(set(range(c_total)) - {ch_idx}):
                    arr[:, c] = 1
                for t, m in enumerate(masks):
                    arr[t, ch_idx] = m.astype(np.uint8)
            print(
                f"[{i}/{n}] {k + 1}/{len(positions)} {pos_name} fg={frac:.4f} {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
