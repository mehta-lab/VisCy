"""Parquet-based cell observation index — one row per cell, built once, reused everywhere.

Provides:

* ``CELL_INDEX_SCHEMA`` — canonical pyarrow schema for the parquet contract.
* ``validate_cell_index`` / ``read_cell_index`` / ``write_cell_index`` — I/O utilities.
* ``build_timelapse_cell_index`` — builder from an experiment registry YAML + tracking CSVs.
* ``build_ops_cell_index`` — builder from OPS zarr + per-well label tables.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from iohub.ngff import open_ome_zarr
from tqdm import tqdm

from viscy_data._typing import (
    CELL_INDEX_CORE_COLUMNS,
    CELL_INDEX_GROUPING_COLUMNS,
    CELL_INDEX_OPS_COLUMNS,
    CELL_INDEX_TIMELAPSE_COLUMNS,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "CELL_INDEX_SCHEMA",
    "build_ops_cell_index",
    "build_timelapse_cell_index",
    "read_cell_index",
    "validate_cell_index",
    "write_cell_index",
]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CELL_INDEX_SCHEMA = pa.schema(
    [
        ("cell_id", pa.string()),
        ("experiment", pa.string()),
        ("store_path", pa.string()),
        ("tracks_path", pa.string()),
        ("fov", pa.string()),
        ("well", pa.string()),
        ("y", pa.float32()),
        ("x", pa.float32()),
        ("z", pa.int16()),
        ("source_channels", pa.string()),
        ("condition", pa.string()),
        ("channel_name", pa.string()),
        ("t", pa.int32()),
        ("track_id", pa.int32()),
        ("global_track_id", pa.string()),
        ("lineage_id", pa.string()),
        ("parent_track_id", pa.int32()),
        ("hours_post_perturbation", pa.float32()),
        ("gene_name", pa.string()),
        ("reporter", pa.string()),
        ("sgRNA", pa.string()),
        ("microscope", pa.string()),
    ]
)

_REQUIRED_COLUMNS = set(CELL_INDEX_CORE_COLUMNS + CELL_INDEX_GROUPING_COLUMNS)
_ALL_COLUMNS = set(
    CELL_INDEX_CORE_COLUMNS + CELL_INDEX_GROUPING_COLUMNS + CELL_INDEX_TIMELAPSE_COLUMNS + CELL_INDEX_OPS_COLUMNS
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_cell_index(df: pd.DataFrame, *, strict: bool = False) -> list[str]:
    """Validate a cell index DataFrame against the canonical schema.

    Parameters
    ----------
    df : pd.DataFrame
        Cell index to validate.
    strict : bool
        If ``True``, require **all** schema columns (not just core + grouping).

    Returns
    -------
    list[str]
        Warnings (e.g. nullable columns that are entirely null).

    Raises
    ------
    ValueError
        If required columns are missing or ``cell_id`` is not unique.
    """
    required = _ALL_COLUMNS if strict else _REQUIRED_COLUMNS
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["cell_id"].duplicated().any():
        n_dup = df["cell_id"].duplicated().sum()
        raise ValueError(f"cell_id must be unique, found {n_dup} duplicates")

    warnings: list[str] = []
    for col in _ALL_COLUMNS & set(df.columns):
        if df[col].isna().all() and len(df) > 0:
            warnings.append(f"column '{col}' is all null")
    return warnings


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_cell_index(
    df: pd.DataFrame,
    path: str | Path,
    *,
    validate: bool = True,
) -> None:
    """Write a cell index DataFrame to parquet with the canonical schema.

    Missing nullable columns are added as ``None`` before writing.

    Parameters
    ----------
    df : pd.DataFrame
        Cell index to write.
    path : str | Path
        Output parquet path.
    validate : bool
        Run :func:`validate_cell_index` before writing.
    """
    # Add any missing schema columns as None
    for field in CELL_INDEX_SCHEMA:
        if field.name not in df.columns:
            df[field.name] = None

    if validate:
        validate_cell_index(df)

    table = pa.Table.from_pandas(df, schema=CELL_INDEX_SCHEMA, preserve_index=False)
    pq.write_table(table, str(path))


def read_cell_index(path: str | Path) -> pd.DataFrame:
    """Read a cell index parquet into a pandas DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to parquet file.

    Returns
    -------
    pd.DataFrame
        Cell index with correct dtypes.
    """
    table = pq.read_table(str(path), schema=CELL_INDEX_SCHEMA)
    return table.to_pandas()


# ---------------------------------------------------------------------------
# Lineage reconstruction (standalone, reused by time-lapse builder)
# ---------------------------------------------------------------------------


def _reconstruct_lineage(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add ``lineage_id`` column linking daughters to root ancestor.

    Each track's ``lineage_id`` is set to the ``global_track_id`` of its root
    ancestor.  Tracks without a ``parent_track_id`` (or whose parent is not
    present in the data) are their own root.

    Parameters
    ----------
    tracks : pd.DataFrame
        Must contain ``global_track_id``, ``experiment``, ``fov``, ``track_id``.
        Optionally ``parent_track_id``.

    Returns
    -------
    pd.DataFrame
        Input with ``lineage_id`` column added/overwritten.
    """
    if tracks.empty:
        tracks["lineage_id"] = pd.Series(dtype=str)
        return tracks

    tracks["lineage_id"] = tracks["global_track_id"].copy()

    if "parent_track_id" not in tracks.columns:
        return tracks

    lineage_series = tracks["lineage_id"].copy()

    groups = list(tracks.groupby(["experiment", "fov"]))
    for (exp, fov), group in tqdm(groups, desc="Reconstructing lineages", unit="fov"):
        tid_to_gtid: dict[int, str] = dict(zip(group["track_id"], group["global_track_id"]))

        parent_map: dict[str, str] = {}
        for _, row in group.drop_duplicates("track_id").iterrows():
            ptid = row["parent_track_id"]
            if pd.notna(ptid) and int(ptid) in tid_to_gtid:
                parent_map[row["global_track_id"]] = tid_to_gtid[int(ptid)]

        def _find_root(gtid: str) -> str:
            visited: set[str] = set()
            current = gtid
            while current in parent_map and current not in visited:
                visited.add(current)
                current = parent_map[current]
            return current

        gtid_to_root = {gtid: _find_root(gtid) for gtid in group["global_track_id"].unique()}
        lineage_series.loc[group.index] = group["global_track_id"].map(gtid_to_root)

    tracks["lineage_id"] = lineage_series
    return tracks


# ---------------------------------------------------------------------------
# Time-lapse builder
# ---------------------------------------------------------------------------


def _build_experiment_tracks(
    exp: object,
    collection_source_channels: list,
    include_wells: list[str] | None,
    exclude_fovs: list[str] | None,
) -> pd.DataFrame:
    """Build track rows for a single experiment.

    Parameters
    ----------
    exp :
        Experiment object from collection.
    collection_source_channels :
        Source channel list from collection.
    include_wells : list[str] | None
        Well filter passed from caller.
    exclude_fovs : list[str] | None
        FOV exclusion list passed from caller.

    Returns
    -------
    pd.DataFrame
        Track rows for this experiment.
    """
    condition_wells = exp.condition_wells
    declared_wells = {w for wells in condition_wells.values() for w in wells}

    all_exclude = set(exp.exclude_fovs)
    if exclude_fovs is not None:
        all_exclude.update(exclude_fovs)

    source_channel_names = [
        sc.per_experiment[exp.name] for sc in collection_source_channels if exp.name in sc.per_experiment
    ]
    fluorescence_ch = source_channel_names[1] if len(source_channel_names) > 1 else ""

    exp_tracks: list[pd.DataFrame] = []

    with open_ome_zarr(exp.data_path, mode="r") as plate:
        fovs = list(plate.positions())

    for _pos_path, position in tqdm(fovs, desc=exp.name, leave=False, unit="fov"):
        fov_name = position.zgroup.name.strip("/")
        parts = fov_name.split("/")
        well_name = "/".join(parts[:2])

        if declared_wells and well_name not in declared_wells:
            continue
        if include_wells is not None and well_name not in include_wells:
            continue
        if all_exclude and fov_name in all_exclude:
            continue

        condition = _resolve_condition(condition_wells, well_name)

        tracks_dir = Path(exp.tracks_path) / fov_name
        csv_files = list(tracks_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No tracking CSV in {tracks_dir}")
        if len(csv_files) > 1:
            raise ValueError(f"Expected exactly one tracking CSV in {tracks_dir}, found: {csv_files}")
        tracks_df = pd.read_csv(csv_files[0])

        tracks_df["cell_id"] = (
            exp.name + "_" + fov_name + "_" + tracks_df["track_id"].astype(str) + "_" + tracks_df["t"].astype(str)
        )
        tracks_df["experiment"] = exp.name
        tracks_df["store_path"] = str(exp.data_path)
        tracks_df["tracks_path"] = str(exp.tracks_path)
        tracks_df["fov"] = fov_name
        tracks_df["well"] = well_name
        tracks_df["condition"] = condition
        tracks_df["channel_name"] = fluorescence_ch
        tracks_df["source_channels"] = json.dumps(source_channel_names)
        tracks_df["global_track_id"] = exp.name + "_" + fov_name + "_" + tracks_df["track_id"].astype(str)
        tracks_df["hours_post_perturbation"] = exp.start_hpi + tracks_df["t"] * exp.interval_minutes / 60.0
        tracks_df["microscope"] = exp.microscope

        if "z" not in tracks_df.columns:
            tracks_df["z"] = 0

        exp_tracks.append(tracks_df)

    return pd.concat(exp_tracks, ignore_index=True) if exp_tracks else pd.DataFrame()


def build_timelapse_cell_index(
    collection_path: str | Path,
    output_path: str | Path,
    include_wells: list[str] | None = None,
    exclude_fovs: list[str] | None = None,
    num_workers: int = -1,
) -> pd.DataFrame:
    """Build a cell index parquet from a collection YAML.

    Parameters
    ----------
    collection_path : str | Path
        Path to collection YAML file.
    output_path : str | Path
        Destination parquet path.
    include_wells : list[str] | None
        If given, only include positions from these wells (e.g. ``["A/1"]``).
    exclude_fovs : list[str] | None
        If given, skip these FOV paths (e.g. ``["A/1/0"]``).
    num_workers : int
        Number of parallel worker processes. ``-1`` (default) uses all
        available CPUs. ``1`` runs sequentially.

    Returns
    -------
    pd.DataFrame
        The written cell index.
    """
    import os

    from viscy_data.collection import load_collection

    collection = load_collection(collection_path)
    experiments = collection.experiments
    n_workers = os.cpu_count() if num_workers == -1 else num_workers

    print(f"Building cell index: {len(experiments)} experiments, {n_workers} workers")

    all_tracks: list[pd.DataFrame] = []

    if n_workers == 1:
        for exp in tqdm(experiments, desc="Experiments", unit="exp"):
            df = _build_experiment_tracks(exp, collection.source_channels, include_wells, exclude_fovs)
            if not df.empty:
                all_tracks.append(df)
                print(f"  {exp.name}: {len(df):,} rows")
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for exp in experiments:
                future = executor.submit(
                    _build_experiment_tracks,
                    exp,
                    collection.source_channels,
                    include_wells,
                    exclude_fovs,
                )
                futures[future] = exp.name

            with tqdm(total=len(futures), desc="Experiments", unit="exp") as pbar:
                for future in as_completed(futures):
                    exp_name = futures[future]
                    df = future.result()
                    if not df.empty:
                        all_tracks.append(df)
                        print(f"  {exp_name}: {len(df):,} rows")
                    pbar.update(1)

    if not all_tracks:
        df = pd.DataFrame(columns=list(_ALL_COLUMNS))
    else:
        df = pd.concat(all_tracks, ignore_index=True)
        df = _reconstruct_lineage(df)

    for col in CELL_INDEX_OPS_COLUMNS:
        df[col] = None

    write_cell_index(df, output_path)
    print(f"Wrote {len(df):,} rows to {output_path}")
    return df


# ---------------------------------------------------------------------------
# OPS builder
# ---------------------------------------------------------------------------


def build_ops_cell_index(
    store_path: str | Path,
    labels_path: str | Path,
    experiment_name: str,
    output_path: str | Path,
    wells: list[str] | None = None,
    channel_column: str = "channel",
    gene_column: str = "gene_name",
    reporter_column: str | None = "reporter",
    sgRNA_column: str | None = "sgRNA",
    bbox_column: str = "bbox",
    segmentation_id_column: str = "segmentation_id",
    min_bbox_size: int = 5,
    source_channels: list[str] | None = None,
    condition_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Build a cell index parquet from OPS data.

    Parameters
    ----------
    store_path : str | Path
        Path to the OME-Zarr data store.
    labels_path : str | Path
        Directory containing per-well label files
        (``{well_flat}_linked_pheno_iss.{csv,parquet}``).
    experiment_name : str
        Name for this experiment.
    output_path : str | Path
        Destination parquet path.
    wells : list[str] | None
        Specific wells to process (e.g. ``["A/1"]``). None = all.
    channel_column : str
        Column name for channel/reporter in the labels file.
    gene_column : str
        Column name for gene perturbation target.
    reporter_column : str | None
        Column name for reporter. None to skip.
    sgRNA_column : str | None
        Column name for guide RNA. None to skip.
    bbox_column : str
        Column name for bounding box string ``"(ymin, xmin, ymax, xmax)"``.
    segmentation_id_column : str
        Column name for segmentation ID.
    min_bbox_size : int
        Minimum bbox side length; smaller cells are dropped.
    source_channels : list[str] | None
        Channel names for ``source_channels`` field. None uses zarr metadata.
    condition_map : dict[str, list[str]] | None
        ``{condition: [well, ...]}`` mapping. None defaults to ``"unknown"``.

    Returns
    -------
    pd.DataFrame
        The written cell index.
    """
    store_path = Path(store_path)
    labels_path = Path(labels_path)

    plate = open_ome_zarr(store_path, mode="r")
    all_rows: list[pd.DataFrame] = []

    # Discover wells from zarr
    discovered_wells: set[str] = set()
    for pos_path, _position in plate.positions():
        well = "/".join(pos_path.split("/")[:2])
        discovered_wells.add(well)

    target_wells = wells if wells is not None else sorted(discovered_wells)

    # Resolve source channel names from zarr if not provided
    if source_channels is None:
        first_pos = next(iter(plate.positions()))[1]
        source_channels = list(first_pos.channel_names)

    for well in target_wells:
        well_flat = well.replace("/", "")
        # Find labels file
        label_file = None
        for ext in ("parquet", "csv"):
            candidate = labels_path / f"{well_flat}_linked_pheno_iss.{ext}"
            if candidate.exists():
                label_file = candidate
                break
        if label_file is None:
            _logger.warning("No label file for well %s, skipping", well)
            continue

        # Read labels
        if label_file.suffix == ".parquet":
            labels_df = pd.read_parquet(label_file)
        else:
            labels_df = pd.read_csv(label_file)

        # Drop rows with NaN segmentation ID
        labels_df = labels_df.dropna(subset=[segmentation_id_column])

        # Parse bbox → centroid and filter by size
        if bbox_column in labels_df.columns:
            centroids = labels_df[bbox_column].apply(_parse_bbox_to_centroid)
            labels_df["y"] = centroids.apply(lambda c: c[0])
            labels_df["x"] = centroids.apply(lambda c: c[1])

            # Filter small bboxes
            sizes = labels_df[bbox_column].apply(_parse_bbox_min_size)
            labels_df = labels_df[sizes >= min_bbox_size].copy()

        # Fill NaN gene_name → "NTC"
        if gene_column in labels_df.columns:
            labels_df[gene_column] = labels_df[gene_column].fillna("NTC")

        # Discover FOVs for this well
        well_fovs = []
        for pos_path, _pos in plate.positions():
            if pos_path.startswith(well + "/"):
                well_fovs.append(pos_path)

        fov = well_fovs[0] if well_fovs else well + "/0"

        # Build cell index rows
        labels_df["cell_id"] = (
            experiment_name + "_" + fov + "_" + labels_df[segmentation_id_column].astype(int).astype(str)
        )
        labels_df["experiment"] = experiment_name
        labels_df["store_path"] = str(store_path)
        labels_df["tracks_path"] = ""
        labels_df["fov"] = fov
        labels_df["well"] = well
        labels_df["z"] = 0
        labels_df["source_channels"] = json.dumps(source_channels)
        labels_df["channel_name"] = labels_df[channel_column] if channel_column in labels_df.columns else ""

        # Condition from map
        if condition_map is not None:
            labels_df["condition"] = _resolve_condition(condition_map, well)
        else:
            labels_df["condition"] = "unknown"

        # OPS-specific columns
        labels_df["gene_name"] = labels_df[gene_column] if gene_column in labels_df.columns else None
        if reporter_column and reporter_column in labels_df.columns:
            labels_df["reporter"] = labels_df[reporter_column]
        else:
            labels_df["reporter"] = None
        if sgRNA_column and sgRNA_column in labels_df.columns:
            labels_df["sgRNA"] = labels_df[sgRNA_column]
        else:
            labels_df["sgRNA"] = None

        # Time-lapse columns → None
        for col in CELL_INDEX_TIMELAPSE_COLUMNS:
            labels_df[col] = None

        all_rows.append(labels_df)

    if not all_rows:
        df = pd.DataFrame(columns=list(_ALL_COLUMNS))
    else:
        df = pd.concat(all_rows, ignore_index=True)

    write_cell_index(df, output_path)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_condition(condition_wells: dict[str, list[str]], well_name: str) -> str:
    """Map well_name to condition label from a condition→wells dict."""
    for condition_label, wells_list in condition_wells.items():
        if well_name in wells_list:
            return condition_label
    return "unknown"


def _parse_bbox_to_centroid(bbox_str: str) -> tuple[float, float]:
    """Parse bbox string ``"(ymin, xmin, ymax, xmax)"`` → centroid ``(y, x)``."""
    nums = [float(s.strip()) for s in bbox_str.strip("()").split(",")]
    ymin, xmin, ymax, xmax = nums[0], nums[1], nums[2], nums[3]
    return ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)


def _parse_bbox_min_size(bbox_str: str) -> float:
    """Parse bbox string and return the minimum side length."""
    nums = [float(s.strip()) for s in bbox_str.strip("()").split(",")]
    ymin, xmin, ymax, xmax = nums[0], nums[1], nums[2], nums[3]
    return min(ymax - ymin, xmax - xmin)
