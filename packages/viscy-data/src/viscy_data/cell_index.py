"""Parquet-based cell observation index — one row per cell, built once, reused everywhere.

Provides:

* ``CELL_INDEX_SCHEMA`` — canonical pyarrow schema for the parquet contract.
* ``validate_cell_index`` / ``read_cell_index`` / ``write_cell_index`` — I/O utilities.
* ``build_timelapse_cell_index`` — builder from an experiment registry YAML + tracking CSVs.
* ``build_ops_cell_index`` — builder from OPS zarr + per-well label tables.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from iohub.ngff import open_ome_zarr
from tqdm import tqdm

from viscy_data._typing import (
    CELL_INDEX_BIOLOGY_COLUMNS,
    CELL_INDEX_CORE_COLUMNS,
    CELL_INDEX_GROUPING_COLUMNS,
    CELL_INDEX_IMAGING_COLUMNS,
    CELL_INDEX_NORMALIZATION_COLUMNS,
    CELL_INDEX_OPS_COLUMNS,
    CELL_INDEX_TIMELAPSE_COLUMNS,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "CELL_INDEX_SCHEMA",
    "build_ops_cell_index",
    "build_timelapse_cell_index",
    "convert_ops_parquet",
    "preprocess_cell_index",
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
        ("perturbation", pa.string()),
        ("channel_name", pa.string()),
        ("t", pa.int32()),
        ("track_id", pa.int32()),
        ("global_track_id", pa.string()),
        ("lineage_id", pa.string()),
        ("parent_track_id", pa.int32()),
        ("hours_post_perturbation", pa.float32()),
        ("interval_minutes", pa.float32()),
        ("gene_name", pa.string()),
        ("reporter", pa.string()),
        ("sgRNA", pa.string()),
        ("microscope", pa.string()),
        ("marker", pa.string()),
        ("organelle", pa.string()),
        ("pixel_size_xy_um", pa.float32()),
        ("pixel_size_z_um", pa.float32()),
        ("T_shape", pa.int32()),
        ("C_shape", pa.int32()),
        ("Z_shape", pa.int32()),
        ("Y_shape", pa.int32()),
        ("X_shape", pa.int32()),
        ("z_focus_mean", pa.float32()),
        ("norm_mean", pa.float32()),
        ("norm_std", pa.float32()),
        ("norm_median", pa.float32()),
        ("norm_iqr", pa.float32()),
        ("norm_max", pa.float32()),
        ("norm_min", pa.float32()),
    ]
)

_REQUIRED_COLUMNS = set(CELL_INDEX_CORE_COLUMNS + CELL_INDEX_GROUPING_COLUMNS)
_ALL_COLUMNS = set(
    CELL_INDEX_CORE_COLUMNS
    + CELL_INDEX_GROUPING_COLUMNS
    + CELL_INDEX_BIOLOGY_COLUMNS
    + CELL_INDEX_TIMELAPSE_COLUMNS
    + CELL_INDEX_OPS_COLUMNS
    + CELL_INDEX_IMAGING_COLUMNS
    + CELL_INDEX_NORMALIZATION_COLUMNS
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
        If required columns are missing or ``(cell_id, channel_name)`` is not unique.
    """
    required = _ALL_COLUMNS if strict else _REQUIRED_COLUMNS
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dup_key = ["cell_id", "channel_name"]
    if df.duplicated(subset=dup_key).any():
        n_dup = df.duplicated(subset=dup_key).sum()
        raise ValueError(f"(cell_id, channel_name) must be unique, found {n_dup} duplicates")

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
# Preprocessing (clean up an existing cell index parquet)
# ---------------------------------------------------------------------------


def preprocess_cell_index(
    parquet_path: str | Path,
    output_path: str | Path | None = None,
    focus_channel: str | None = None,
) -> pd.DataFrame:
    """Add normalization stats, focus slice, and remove invalid rows.

    Reads precomputed metadata from each FOV's ``zattrs`` (written by
    ``viscy preprocess``) and writes them as parquet columns:

    - ``norm_mean``, ``norm_std``, ``norm_median``, ``norm_iqr``,
      ``norm_max``, ``norm_min`` — per-timepoint, per-channel statistics
    - ``z_focus_mean`` — per-FOV focus plane from ``focus_slice``

    Drops rows where timepoint stats are missing or ``norm_max == 0.0``
    (empty frames).

    Parameters
    ----------
    parquet_path : str | Path
        Path to the cell index parquet to preprocess.
    output_path : str | Path | None
        Destination path. When ``None``, overwrites *parquet_path* in place.
    focus_channel : str | None
        Channel name for ``focus_slice`` lookup (e.g. ``"Phase3D"``).
        When ``None``, uses the first channel_name in each FOV's group.

    Returns
    -------
    pd.DataFrame
        The preprocessed cell index with normalization and focus columns.

    Raises
    ------
    ValueError
        If a FOV has no normalization metadata (run ``viscy preprocess`` first).
    """
    if output_path is None:
        output_path = parquet_path

    df = read_cell_index(parquet_path)
    n_before = len(df)

    fov_col = "fov" if "fov" in df.columns else "fov_name"

    # Build lookups from zarr zattrs (one open per unique FOV)
    stat_lookup: dict[tuple[str, str, str, int], dict[str, float]] = {}
    focus_lookup: dict[tuple[str, str], float] = {}

    for (store_path, fov), group in df.groupby(["store_path", fov_col]):
        fov_path = f"{group['well'].iloc[0]}/{fov}" if "/" not in str(fov) else str(fov)
        with open_ome_zarr(f"{store_path}/{fov_path}", mode="r") as pos:
            norm_meta = pos.zattrs.get("normalization", None)
            focus_meta = pos.zattrs.get("focus_slice", {})
        if norm_meta is None:
            raise ValueError(
                f"FOV '{fov_path}' in store '{store_path}' has no normalization metadata. "
                "Run `viscy preprocess` on this dataset first."
            )
        for ch_name, ch_stats in norm_meta.items():
            for t_str, tp_stats in ch_stats.get("timepoint_statistics", {}).items():
                stat_lookup[(str(store_path), str(fov), ch_name, int(t_str))] = tp_stats

        fc = focus_channel or group["channel_name"].iloc[0]
        ch_focus = focus_meta.get(fc, {})
        fov_stats = ch_focus.get("fov_statistics", {})
        z_focus = fov_stats.get("z_focus_mean")
        if z_focus is not None:
            focus_lookup[(str(store_path), str(fov))] = float(z_focus)

    # Vectorized lookup: build norm + focus column arrays
    stat_keys = ["mean", "std", "median", "iqr", "max", "min"]
    store_arr = df["store_path"].astype(str).to_numpy()
    fov_arr = df[fov_col].astype(str).to_numpy()
    ch_arr = df["channel_name"].astype(str).to_numpy()
    t_arr = df["t"].astype(int).to_numpy()

    norm_arrays = {stat: np.full(len(df), float("nan"), dtype=np.float32) for stat in stat_keys}
    focus_arr = np.full(len(df), float("nan"), dtype=np.float32)
    valid_mask = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        tp_stats = stat_lookup.get((store_arr[i], fov_arr[i], ch_arr[i], t_arr[i]))
        if tp_stats is None or tp_stats.get("max", 1.0) == 0.0:
            valid_mask[i] = False
            continue
        for stat in stat_keys:
            norm_arrays[stat][i] = float(tp_stats[stat])
        z_focus = focus_lookup.get((store_arr[i], fov_arr[i]))
        if z_focus is not None:
            focus_arr[i] = z_focus

    for stat in stat_keys:
        df[f"norm_{stat}"] = norm_arrays[stat]
    df["z_focus_mean"] = focus_arr

    df = df[valid_mask].reset_index(drop=True)
    n_dropped = n_before - len(df)

    write_cell_index(df, output_path)
    if n_dropped > 0:
        _logger.info("Dropped %d invalid rows (%.1f%%).", n_dropped, 100 * n_dropped / n_before)
    print(f"Wrote {len(df):,} rows to {output_path} (dropped {n_dropped:,}, added norm + focus columns)")
    return df


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
    include_wells: list[str] | None,
    exclude_fovs: list[str] | None,
) -> pd.DataFrame:
    """Build flat track rows for a single experiment (one row per cell x timepoint x channel).

    Parameters
    ----------
    exp :
        Experiment entry with ``channels`` list and ``perturbation_wells``.
    include_wells : list[str] | None
        Well filter passed from caller.
    exclude_fovs : list[str] | None
        FOV exclusion list passed from caller.

    Returns
    -------
    pd.DataFrame
        Track rows for this experiment, one per (cell, timepoint, channel).
    """
    perturbation_wells = exp.perturbation_wells
    declared_wells = {w for wells in perturbation_wells.values() for w in wells}

    all_exclude = set(exp.exclude_fovs)
    if exclude_fovs is not None:
        all_exclude.update(exclude_fovs)

    # Channel-marker pairs from per-experiment channels list
    channel_marker_pairs = [(ch.name, ch.marker) for ch in exp.channels]

    exp_tracks: list[pd.DataFrame] = []

    with open_ome_zarr(exp.data_path, mode="r") as plate:
        fovs = list(plate.positions())

    for _pos_path, position in tqdm(fovs, desc=exp.name, leave=False, unit="fov"):
        fov_path = position.zgroup.name.strip("/")
        parts = fov_path.split("/")
        well_name = "/".join(parts[:2])
        fov_name = parts[2]

        if declared_wells and well_name not in declared_wells:
            continue
        if include_wells is not None and well_name not in include_wells:
            continue
        if all_exclude and fov_path in all_exclude:
            continue

        perturbation = _resolve_perturbation(perturbation_wells, well_name)

        tracks_dir = Path(exp.tracks_path) / fov_path
        csv_files = list(tracks_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No tracking CSV in {tracks_dir}")
        if len(csv_files) > 1:
            raise ValueError(f"Expected exactly one tracking CSV in {tracks_dir}, found: {csv_files}")
        tracks_df = pd.read_csv(csv_files[0])

        # TCZYX shape from zarr metadata (same for all positions in a well)
        img_arr = position["0"]
        t_shape, c_shape, z_shape, y_shape, x_shape = img_arr.shape

        # Base columns (shared across channel rows)
        tracks_df["cell_id"] = (
            exp.name + "_" + fov_path + "_" + tracks_df["track_id"].astype(str) + "_" + tracks_df["t"].astype(str)
        )
        tracks_df["experiment"] = exp.name
        tracks_df["store_path"] = str(exp.data_path)
        tracks_df["tracks_path"] = str(exp.tracks_path)
        tracks_df["fov"] = fov_name
        tracks_df["well"] = well_name
        tracks_df["perturbation"] = perturbation
        tracks_df["global_track_id"] = exp.name + "_" + fov_path + "_" + tracks_df["track_id"].astype(str)
        tracks_df["hours_post_perturbation"] = exp.start_hpi + tracks_df["t"] * exp.interval_minutes / 60.0
        tracks_df["interval_minutes"] = exp.interval_minutes
        tracks_df["microscope"] = exp.microscope
        tracks_df["organelle"] = exp.organelle
        tracks_df["pixel_size_xy_um"] = exp.pixel_size_xy_um
        tracks_df["pixel_size_z_um"] = exp.pixel_size_z_um
        tracks_df["T_shape"] = t_shape
        tracks_df["C_shape"] = c_shape
        tracks_df["Z_shape"] = z_shape
        tracks_df["Y_shape"] = y_shape
        tracks_df["X_shape"] = x_shape

        if "z" not in tracks_df.columns:
            tracks_df["z"] = 0

        # Explode: one row per channel
        for zarr_ch, marker in channel_marker_pairs:
            ch_df = tracks_df.copy()
            ch_df["channel_name"] = zarr_ch
            ch_df["marker"] = marker
            exp_tracks.append(ch_df)

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
    from viscy_data.collection import load_collection

    collection = load_collection(collection_path)
    experiments = collection.experiments
    n_workers = os.cpu_count() if num_workers == -1 else num_workers

    print(f"Building cell index: {len(experiments)} experiments, {n_workers} workers")

    all_tracks: list[pd.DataFrame] = []

    if n_workers == 1:
        for exp in tqdm(experiments, desc="Experiments", unit="exp"):
            df = _build_experiment_tracks(exp, include_wells, exclude_fovs)
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
    perturbation_map: dict[str, list[str]] | None = None,
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
    perturbation_map : dict[str, list[str]] | None
        ``{perturbation: [well, ...]}`` mapping. None defaults to ``"unknown"``.

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

    # Read pixel sizes from zarr
    first_pos = next(iter(plate.positions()))[1]
    scale = first_pos.scale  # [T, C, Z, Y, X]
    pixel_size_xy_um = scale[3] if len(scale) > 3 else None
    pixel_size_z_um = scale[2] if len(scale) > 2 else None

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

        fov_path = well_fovs[0] if well_fovs else well + "/0"
        fov_name = fov_path.split("/")[-1]

        # Build cell index rows
        labels_df["cell_id"] = (
            experiment_name + "_" + fov_path + "_" + labels_df[segmentation_id_column].astype(int).astype(str)
        )
        labels_df["experiment"] = experiment_name
        labels_df["store_path"] = str(store_path)
        labels_df["tracks_path"] = ""
        labels_df["fov"] = fov_name
        labels_df["well"] = well
        labels_df["z"] = 0
        labels_df["channel_name"] = labels_df[channel_column] if channel_column in labels_df.columns else ""
        labels_df["marker"] = labels_df[channel_column] if channel_column in labels_df.columns else ""
        labels_df["pixel_size_xy_um"] = pixel_size_xy_um
        labels_df["pixel_size_z_um"] = pixel_size_z_um

        # Perturbation from map
        if perturbation_map is not None:
            labels_df["perturbation"] = _resolve_perturbation(perturbation_map, well)
        else:
            labels_df["perturbation"] = "unknown"

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
# OPS parquet converter
# ---------------------------------------------------------------------------


def convert_ops_parquet(
    ops_parquet_path: str | Path,
    output_path: str | Path,
    store_root: str = "/hpc/projects/icd.fast.ops",
    store_suffix: str = "3-assembly/phenotyping_v3.zarr",
) -> pd.DataFrame:
    """Convert an OPS merged parquet to the canonical flat cell index schema.

    Supports multi-experiment parquets: each unique ``store_key`` in the
    input becomes a separate experiment in the output.

    Parameters
    ----------
    ops_parquet_path : str | Path
        Path to the OPS merged parquet file (e.g. ``373genes_filtered.parquet``).
    output_path : str | Path
        Destination parquet path.
    store_root : str
        Root directory for OPS zarr stores. Default: ``"/hpc/projects/icd.fast.ops"``.
    store_suffix : str
        Suffix appended after ``store_key`` to form ``store_path``.
        Default: ``"3-assembly/phenotyping_v3.zarr"``.

    Returns
    -------
    pd.DataFrame
        The written cell index.
    """
    df = pd.read_parquet(Path(ops_parquet_path))

    out = pd.DataFrame()

    # Use store_key as experiment name (supports multi-experiment parquets)
    out["experiment"] = df["store_key"]

    # store_path from store_key
    out["store_path"] = df["store_key"].apply(lambda k: f"{store_root}/{k}/{store_suffix}")

    # OPS 'well' column is the position path (e.g. "A/1/0")
    # Split into well (A/1) and fov (0)
    out["fov"] = df["well"].apply(lambda w: w.rsplit("/", 1)[1] if "/" in w else w)
    out["well"] = df["well"].apply(lambda w: w.rsplit("/", 1)[0])

    # Centroid from bbox
    centroids = df["bbox"].apply(_parse_bbox_to_centroid)
    out["y"] = centroids.apply(lambda c: c[0]).astype("float32")
    out["x"] = centroids.apply(lambda c: c[1]).astype("float32")
    out["z"] = pd.array([0] * len(df), dtype="int16")

    # Per-row channel name and marker (the zarr channel this cell was imaged in)
    out["channel_name"] = df["channel"] if "channel" in df.columns else ""
    out["marker"] = df["reporter"] if "reporter" in df.columns else out["channel_name"]
    out["organelle"] = None

    # OPS-specific columns
    out["gene_name"] = df["gene_name"].fillna("NTC") if "gene_name" in df.columns else None
    out["reporter"] = df["reporter"] if "reporter" in df.columns else None
    out["sgRNA"] = df["sgRNA"] if "sgRNA" in df.columns else None

    # condition = gene_name for perturbation mode
    out["perturbation"] = out["gene_name"] if "gene_name" in df.columns else "unknown"

    # OPS is single-timepoint
    out["t"] = pd.array([0] * len(df), dtype="Int32")
    id_series = df["total_index"].astype(str) if "total_index" in df.columns else pd.Series(range(len(df))).astype(str)
    out["track_id"] = (
        df["total_index"].astype("Int32") if "total_index" in df.columns else pd.array(range(len(df)), dtype="Int32")
    )
    # cell_id, global_track_id, lineage_id: each cell is its own lineage
    out["cell_id"] = df["store_key"].astype(str) + "_" + id_series
    out["global_track_id"] = df["store_key"].astype(str) + "_" + id_series
    out["lineage_id"] = df["store_key"].astype(str) + "_" + id_series
    out["parent_track_id"] = pd.array([-1] * len(df), dtype="Int32")
    out["hours_post_perturbation"] = pd.array([0.0] * len(df), dtype="float32")

    out["tracks_path"] = ""
    out["interval_minutes"] = pd.array([0.0] * len(df), dtype="float32")
    out["microscope"] = ""
    out["pixel_size_xy_um"] = None
    out["pixel_size_z_um"] = None

    write_cell_index(out, output_path)
    n_experiments = df["store_key"].nunique()
    _logger.info("Converted %d OPS cells (%d experiments) to %s", len(out), n_experiments, output_path)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_perturbation(perturbation_wells: dict[str, list[str]], well_name: str) -> str:
    """Map well_name to perturbation label from a perturbation→wells dict."""
    for perturbation_label, wells_list in perturbation_wells.items():
        if well_name in wells_list:
            return perturbation_label
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
