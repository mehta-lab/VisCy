"""Generate the package version table in ``docs/packages/index.md``.

Zensical 0.0.43 does not yet support build hooks, so the per-package version
matrix is produced by this standalone script instead. It reads the installed
distribution version of each workspace package (which uv-dynamic-versioning
derives from that package's ``viscy-<name>-vX.Y.Z`` git tag) and rewrites the
block between the ``versions:start`` / ``versions:end`` markers.

Run it before ``zensical build`` / ``zensical serve`` and in CI before deploy::

    uv run python docs/_gen_versions.py
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ["viscy-data", "viscy-models", "viscy-transforms", "viscy-utils"]

INDEX_PATH = Path(__file__).parent / "packages" / "index.md"
START_MARKER = "<!-- versions:start -->"
END_MARKER = "<!-- versions:end -->"


def render_table() -> str:
    """Render the package version table as a Markdown string.

    Returns
    -------
    str
        A Markdown table with one row per workspace package.
    """
    rows = ["| Package | Version | Install |", "|---------|---------|---------|"]
    for name in PACKAGES:
        try:
            ver = version(name)
        except PackageNotFoundError:
            ver = "—"
        rows.append(f"| [`{name}`]({name}.md) | `{ver}` | `pip install {name}` |")
    return "\n".join(rows)


def main() -> None:
    """Rewrite the version block in the packages overview page."""
    text = INDEX_PATH.read_text()
    if START_MARKER not in text or END_MARKER not in text:
        raise ValueError(f"Markers {START_MARKER!r}/{END_MARKER!r} not found in {INDEX_PATH}")
    before, rest = text.split(START_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    block = f"{START_MARKER}\n\n{render_table()}\n\n{END_MARKER}"
    INDEX_PATH.write_text(f"{before}{block}{after}")
    print(f"Updated version table in {INDEX_PATH}")


if __name__ == "__main__":
    main()
