"""Per-dataset MANIFEST.json emission for packed OZX trees.

The MANIFEST.json sits next to a dataset's packed OZX files and
records sha256 + size + ozx_version per archive. Croissant builders
read it to populate ``cr:FileObject.sha256`` + ``contentSize``.
"""

import dataclasses
import json
from pathlib import Path

from dynacell.distribution.ozx import PackResult


@dataclasses.dataclass(frozen=True)
class PackManifest:
    """In-memory representation of a per-dataset OZX MANIFEST.json."""

    dataset: str
    entries: list[dict]

    def to_json(self) -> str:
        """Serialize as deterministic, sort-keys JSON for git-friendly diffs."""
        return json.dumps(
            {"dataset": self.dataset, "entries": self.entries},
            sort_keys=True,
            indent=2,
        )


def write_pack_manifest(
    dataset: str,
    results: list[PackResult],
    output_path: Path,
) -> Path:
    """Write a per-dataset MANIFEST.json from a list of pack results.

    Parameters
    ----------
    dataset
        Registered dataset name (e.g. ``"aics-hipsc"``).
    results
        Output of :func:`pack_dataset`. One entry per packed file.
    output_path
        Where to write MANIFEST.json. Parent directory created if
        absent.

    Returns
    -------
    Path
        Resolved path to the written manifest.

    Notes
    -----
    Writing is a **merge**, not a full rebuild: a split-scoped pack
    (``pack --splits train`` then ``pack --splits test``) must not leave a
    manifest that omits the archive the earlier call produced. Prior entries
    survive only while their ``.ozx`` is still on disk, so a ledger whose whole
    job is checksums never keeps a row for an archive that has been deleted.
    Entries this call produced always win over a prior row for the same
    destination.
    """
    entries = [
        {
            "target": r.target,
            "split": r.split,
            "src_zarr_path": str(r.src_zarr_path),
            "dst_ozx_path": str(r.dst_ozx_path.resolve()),
            "bytes": r.bytes,
            "sha256": r.sha256,
            "ozx_version": r.ozx_version,
        }
        for r in results
    ]
    entries = _merge_with_existing(entries, output_path)
    manifest = PackManifest(dataset=dataset, entries=entries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json() + "\n")
    return output_path.resolve()


def _merge_with_existing(entries: list[dict], output_path: Path) -> list[dict]:
    """Fold surviving prior entries into ``entries`` and order deterministically.

    Parameters
    ----------
    entries
        Rows built from the current call's pack results.
    output_path
        MANIFEST.json being written. Absent on a first pack.

    Returns
    -------
    list of dict
        Merged rows sorted by ``(split, target)`` so a re-pack of the same
        archives is a no-op diff.
    """
    if output_path.exists():
        prior = json.loads(output_path.read_text())["entries"]
        fresh = {e["dst_ozx_path"] for e in entries}
        entries = [e for e in prior if e["dst_ozx_path"] not in fresh and Path(e["dst_ozx_path"]).exists()] + entries
    return sorted(entries, key=lambda e: (e["split"], e["target"]))
