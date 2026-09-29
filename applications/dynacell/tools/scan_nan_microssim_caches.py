#!/usr/bin/env python
r"""List eval caches stamped with a given cubic whose MicroMS3IM column is entirely NaN.

``provenance.CUBIC_VERSIONS_EQUIVALENT_TO`` lets 0.9.0a2 reuse 0.9.0a1 caches. That
holds only where the 0.9.0a1 cache holds a value. When the leaf-level MicroSSIM
calibration ran out of memory under 0.9.0a1, ``_calibrate_microssim``'s caller
wrote ``MicroMS3IM = NaN`` for every FOV. 0.9.0a2's bounded-memory fit would score
those leaves, but the reused cache keeps the NaN. This scan finds them so they can
be re-run with ``force_recompute.final_metrics=true``.

Mito A549 mock caches are all-NaN for a reason that predates this bump (the
MicroSSIM ``data_range`` case), so they are reported as known rather than
unexplained. Read-only: the tool never writes.

Exit status: 0 when no unexplained cache is found, 1 otherwise.

Usage::

    uv run --no-sync python applications/dynacell/tools/scan_nan_microssim_caches.py \
        --root /hpc/projects/virtual_staining/training/dynacell
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

from dynacell.evaluation.provenance import PROVENANCE_FILENAME

#: Save dirs whose all-NaN MicroMS3IM is the mito A549 mock ``data_range`` case:
#: canonical ``mito/<model>/<train>/a549__mock`` and legacy
#: ``a549/evaluations_with_embeddings/eval_<model>_mitochondria_mock``.
_KNOWN_ALL_NAN = re.compile(r"(/mito/[^/]+/[^/]+/a549__mock|/a549/[^/]+/eval_[^/]*mitochondria_mock)$")


def find_sidecars(root: Path, max_depth: int) -> list[Path]:
    """Return every provenance sidecar under ``root``, not descending into zarr stores."""
    found = []
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base
        dirnames[:] = [d for d in dirnames if not d.endswith((".zarr", ".ozx")) and depth < max_depth]
        if PROVENANCE_FILENAME in filenames:
            found.append(Path(dirpath) / PROVENANCE_FILENAME)
    return sorted(found)


def all_nan_microssim(sidecars: list[Path], cubic: str) -> list[Path]:
    """Return the save dirs among ``sidecars`` stamped ``cubic`` whose MicroMS3IM is entirely NaN."""
    hits = []
    for sidecar in sidecars:
        if json.loads(sidecar.read_text())["versions"]["cubic"] != cubic:
            continue
        column = pd.read_csv(sidecar.parent / "pixel_metrics.csv")["MicroMS3IM"]
        if column.isna().all():
            hits.append(sidecar.parent)
    return hits


def is_known_all_nan(save_dir: Path) -> bool:
    """Whether ``save_dir`` is the mito A549 mock case, all-NaN independent of cubic."""
    return _KNOWN_ALL_NAN.search(save_dir.as_posix()) is not None


def main(argv: list[str] | None = None) -> int:
    """Run the scan and print one line per all-NaN cache."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cubic", default="0.9.0a1", help="cubic version stamp to scan (default: 0.9.0a1)")
    parser.add_argument("--max-depth", type=int, default=6, help="directory depth below --root to search")
    args = parser.parse_args(argv)

    sidecars = find_sidecars(args.root, args.max_depth)
    hits = all_nan_microssim(sidecars, args.cubic)
    unexplained = [d for d in hits if not is_known_all_nan(d)]
    for save_dir in hits:
        print(f"{'known' if is_known_all_nan(save_dir) else 'UNEXPLAINED'}\t{save_dir}")
    print(
        f"{len(sidecars)} sidecars, {len(hits)} all-NaN MicroMS3IM stamped cubic {args.cubic}, "
        f"{len(unexplained)} unexplained"
    )
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
