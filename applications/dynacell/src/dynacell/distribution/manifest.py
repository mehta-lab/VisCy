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
    manifest = PackManifest(dataset=dataset, entries=entries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json() + "\n")
    return output_path.resolve()
