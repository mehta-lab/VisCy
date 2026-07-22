"""CSV metadata sidecar for zarr stores that cannot be written to.

Mirrors the ``.zattrs["normalization"]`` / ``.zattrs["focus_slice"]``
metadata tree that ``viscy preprocess`` / ``qc run`` normally write in
place. When a store is mounted read-only, callers pass a ``csv_dir``
instead: one CSV file is written per zarr store, and the same statistics
can later be read back by ``preprocess_cell_index`` in place of ``.zattrs``.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from iohub import open_ome_zarr

__all__ = [
    "backfill_zattrs_to_csv",
    "build_provenance_fields",
    "csv_path_for_store",
    "metadata_to_rows",
    "read_meta_rows_csv",
    "write_meta_rows_csv",
]

_FIELD_NAMES = ("normalization", "focus_slice")

_KEY_COLUMNS = ["position_path", "channel_name", "field_name", "scope", "timepoint"]


def csv_path_for_store(csv_dir: str | Path, store_path: str | Path) -> Path:
    """Resolve the sidecar CSV path for a zarr store under ``csv_dir``.

    The filename is the store's own name without the ``.zarr`` suffix
    (e.g. ``2025_01_24_A549_G3BP1_DENV.zarr`` ->
    ``2025_01_24_A549_G3BP1_DENV.csv``). The full store path is still
    recorded in each row's ``store_path`` column, so provenance is
    preserved even though the filename is short. Two distinct stores that
    share the same name would collide onto one CSV; within a single
    ``csv_dir`` fleet, store names are expected to be unique.
    """
    return Path(csv_dir) / f"{Path(store_path).stem}.csv"


def build_provenance_fields() -> dict:
    """Timestamp + best-effort git commit + CLI invocation for sidecar rows."""
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit = "unknown"
    return {
        "written_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "cli_invocation": " ".join(sys.argv),
    }


def metadata_to_rows(
    metadata: dict,
    store_path: str | Path,
    position_path: str | None,
    channel_name: str,
    field_name: str,
) -> list[dict]:
    """Flatten a ``write_meta_field``-shaped metadata dict into sidecar rows.

    One row per (scope, timepoint): ``dataset_statistics``/``fov_statistics``
    each produce a single row with ``timepoint=None``; ``timepoint_statistics``
    (dict of dicts, as used for normalization) or ``per_timepoint`` (dict of
    scalars, as used for focus_slice) produce one row per timepoint, with the
    scalar case stored under a ``value`` column.

    Parameters
    ----------
    metadata : dict
        Same shape as would be passed to ``write_meta_field``.
    store_path : str or Path
        Path to the zarr store this metadata belongs to.
    position_path : str or None
        FOV path (e.g. ``"A/1/0"``), or None for dataset/plate-level rows.
    channel_name : str
        Channel this metadata was computed for.
    field_name : str
        Metadata field, e.g. ``"normalization"`` or ``"focus_slice"``.

    Returns
    -------
    list[dict]
        Rows ready to be assembled into a DataFrame by ``write_meta_rows_csv``.
    """
    rows = []
    for meta_key, scope in (("dataset_statistics", "dataset"), ("fov_statistics", "fov")):
        stats = metadata.get(meta_key)
        if stats:
            rows.append(
                {
                    "store_path": str(store_path),
                    "position_path": position_path,
                    "channel_name": channel_name,
                    "field_name": field_name,
                    "scope": scope,
                    "timepoint": None,
                    **stats,
                }
            )
    for meta_key in ("timepoint_statistics", "per_timepoint"):
        per_timepoint = metadata.get(meta_key)
        if per_timepoint:
            for t_str, value in per_timepoint.items():
                stats = value if isinstance(value, dict) else {"value": value}
                rows.append(
                    {
                        "store_path": str(store_path),
                        "position_path": position_path,
                        "channel_name": channel_name,
                        "field_name": field_name,
                        "scope": "timepoint",
                        "timepoint": int(t_str),
                        **stats,
                    }
                )
    return rows


def write_meta_rows_csv(csv_dir: str | Path, store_path: str | Path, rows: list[dict]) -> None:
    """Upsert ``rows`` into the per-store sidecar CSV under ``csv_dir``.

    Rows are keyed on ``(position_path, channel_name, field_name, scope,
    timepoint)``; a new row overwrites any existing row with the same key.
    Written atomically (temp file + rename) so an interrupted write can't
    truncate a previously valid sidecar.

    Parameters
    ----------
    csv_dir : str or Path
        Root directory for per-store sidecar CSVs.
    store_path : str or Path
        Path to the zarr store this metadata belongs to.
    rows : list[dict]
        Rows produced by ``metadata_to_rows``, with provenance fields
        (``written_at``, ``git_commit``, ``cli_invocation``) merged in.
    """
    if not rows:
        return
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_path_for_store(csv_dir, store_path)

    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=_KEY_COLUMNS, keep="last").reset_index(drop=True)

    tmp_path = csv_path.with_suffix(".csv.tmp")
    combined.to_csv(tmp_path, index=False)
    tmp_path.replace(csv_path)


def read_meta_rows_csv(csv_dir: str | Path, store_path: str | Path) -> pd.DataFrame | None:
    """Read the per-store sidecar CSV, or None if it doesn't exist."""
    csv_path = csv_path_for_store(csv_dir, store_path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def backfill_zattrs_to_csv(zarr_dir: str | Path, csv_dir: str | Path, dry_run: bool = False) -> bool:
    """Mirror existing ``.zattrs`` normalization/focus_slice metadata into a CSV sidecar.

    For stores that were already preprocessed somewhere with write access
    (e.g. on Bruno, then copied elsewhere read-only), this avoids
    recomputing anything — it's a plain read of the already-computed
    values, reshaped to match the CSV sidecar contract. Only reads the
    store (``mode="r"``); never requires write access to it.

    Parameters
    ----------
    zarr_dir : str or Path
        Path to the zarr store to read from.
    csv_dir : str or Path
        Root directory for per-store sidecar CSVs.
    dry_run : bool
        If True, report whether there's anything to backfill without
        actually writing to the sidecar.

    Returns
    -------
    bool
        True if ``.zattrs`` had any normalization/focus_slice metadata to
        mirror (and, unless ``dry_run``, it was written to the sidecar);
        False if there was nothing to backfill.
    """
    rows = []
    with open_ome_zarr(zarr_dir, mode="r") as plate:
        for field_name in _FIELD_NAMES:
            for channel_name, metadata in plate.zattrs.get(field_name, {}).items():
                rows.extend(metadata_to_rows(metadata, zarr_dir, None, channel_name, field_name))
        for pos_name, pos in plate.positions():
            for field_name in _FIELD_NAMES:
                for channel_name, metadata in pos.zattrs.get(field_name, {}).items():
                    rows.extend(metadata_to_rows(metadata, zarr_dir, pos_name, channel_name, field_name))

    if not rows:
        return False
    if dry_run:
        return True
    provenance = build_provenance_fields()
    for row in rows:
        row.update(provenance)
    write_meta_rows_csv(csv_dir, zarr_dir, rows)
    return True
