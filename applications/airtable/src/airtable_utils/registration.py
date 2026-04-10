"""Register zarr positions as per-FOV records in Airtable.

The atomic unit is a single ``ngff.Position`` path, e.g.::

    /data/dataset.zarr/A/1/000000

Shell globbing handles batch registration::

    register /data/dataset.zarr/*/*/*
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from iohub import open_ome_zarr

from airtable_utils.database import AirtableDatasets, MarkerRegistryEntry
from airtable_utils.schemas import MAX_CHANNELS, DatasetRecord, parse_channel_name, parse_position_name

logger = logging.getLogger(__name__)
DIM_NAMES = ("t_shape", "c_shape", "z_shape", "y_shape", "x_shape")
WELL_TEMPLATE_FIELDS = (
    "cell_type",
    "cell_state",
    "cell_line",
    "marker",
    "organelle",
    "perturbation",
    "hours_post_perturbation",
    "moi",
    "time_interval_min",
    "seeding_density",
    "treatment_concentration_nm",
    "fluorescence_modality",
)


@dataclass
class RegisterResult:
    """Result of registering one or more positions."""

    dataset: str
    created: list[dict] = field(default_factory=list)
    updated: list[dict] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    channel_names: list[str] = field(default_factory=list)
    pixel_size_xy_um: float | None = None
    pixel_size_z_um: float | None = None


def parse_position_path(position_path: Path) -> tuple[Path, str]:
    """Split a position path into zarr root and position name.

    Parameters
    ----------
    position_path : Path
        Full path to a position, e.g.
        ``/data/dataset.zarr/A/1/000000``.

    Returns
    -------
    tuple[Path, str]
        ``(zarr_root, pos_name)`` — e.g.
        ``(Path("/data/dataset.zarr"), "A/1/000000")``.

    Raises
    ------
    ValueError
        If the path does not contain a ``.zarr`` component.
    """
    parts = position_path.parts
    zarr_idx = None
    for i, part in enumerate(parts):
        if part.endswith(".zarr"):
            zarr_idx = i
            break
    if zarr_idx is None:
        raise ValueError(f"No .zarr component found in path: {position_path}")

    zarr_root = Path(*parts[: zarr_idx + 1])
    pos_name = "/".join(parts[zarr_idx + 1 :])
    return zarr_root, pos_name


def zarr_fields_for_position(
    zarr_path: Path,
    pos_name: str,
    channel_names: list[str],
    shape: tuple[int, ...],
    scale: tuple[float, ...] | None = None,
) -> dict:
    """Build Airtable field dict from zarr position data.

    Parameters
    ----------
    zarr_path : Path
        Root zarr store path.
    pos_name : str
        Position name within the zarr (e.g. ``"B/1/000000"``).
    channel_names : list[str]
        Channel names from the zarr store.
    shape : tuple[int, ...]
        Array shape ``(T, C, Z, Y, X)``.
    scale : tuple[float, ...] or None
        Physical scale ``(T, C, Z, Y, X)`` in micrometers from
        the zarr coordinate transforms.

    Returns
    -------
    dict
        Airtable fields derived from the zarr position.
    """
    fields: dict = {"data_path": str(zarr_path / pos_name)}
    for i, ch_name in enumerate(channel_names[:MAX_CHANNELS]):
        fields[f"channel_{i}_name"] = ch_name
    for dim_name, dim_val in zip(DIM_NAMES, shape):
        fields[dim_name] = dim_val
    if scale is not None and len(scale) >= 5:
        z_um, y_um, x_um = scale[2], scale[3], scale[4]
        if not (z_um == 1.0 and y_um == 1.0 and x_um == 1.0):
            if abs(x_um - y_um) > 0.001:
                logger.warning("X pixel size (%.4f) != Y (%.4f) for %s — using Y", x_um, y_um, pos_name)
            fields["pixel_size_xy_um"] = y_um
            fields["pixel_size_z_um"] = z_um
        else:
            logger.warning("Scale is (1,1,1) for %s — skipping pixel sizes (likely uncalibrated)", pos_name)
    return fields


def derive_channel_marker(
    channel_names: list[str],
    marker_entries: list[MarkerRegistryEntry],
) -> dict[str, str]:
    """Derive channel marker annotations from Marker Registry entries.

    For each channel name, finds the first registry entry whose aliases
    contain a substring match, and returns the protein marker name.

    Parameters
    ----------
    channel_names : list[str]
        Ordered channel names from the zarr store.
    marker_entries : list[MarkerRegistryEntry]
        Registry entries linked to the well record.

    Returns
    -------
    dict[str, str]
        Mapping of ``"channel_{i}_marker"`` -> marker label
        for channels that matched a registry entry.
    """
    result: dict[str, str] = {}
    for i, ch_name in enumerate(channel_names[:MAX_CHANNELS]):
        parsed = parse_channel_name(ch_name)
        ch_type = parsed.get("channel_type", "")

        if ch_type == "labelfree":
            result[f"channel_{i}_marker"] = ch_name
            continue

        if ch_type == "virtual_stain":
            result[f"channel_{i}_marker"] = ch_name
            continue

        for entry in marker_entries:
            if any(alias in ch_name for alias in entry.channel_name_aliases):
                result[f"channel_{i}_marker"] = entry.marker
                break
    return result


def copy_well_template_fields(template: DatasetRecord) -> dict:
    """Copy biologist-provided fields from a well template record.

    Parameters
    ----------
    template : DatasetRecord
        Well-level record with marker metadata.

    Returns
    -------
    dict
        Non-None metadata fields from the template.
    """
    fields: dict = {}
    for key in WELL_TEMPLATE_FIELDS:
        val = getattr(template, key)
        if val is not None:
            fields[key] = val
    for i in range(MAX_CHANNELS):
        marker_val = getattr(template, f"channel_{i}_marker", None)
        if marker_val is not None:
            fields[f"channel_{i}_marker"] = marker_val
    return fields


def build_validation_table(
    dataset_name: str,
    channel_names: list[str],
    records: list[DatasetRecord],
) -> str:
    """Build markdown validation table for channel / marker pairing.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    channel_names : list[str]
        Ordered channel names from the zarr.
    records : list[DatasetRecord]
        Airtable records (first record used for marker lookup).

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [
        "| dataset | idx | channel_name | type | filter_cube | marker (scientist) |",
        "|---------|-----|--------------|------|-------------|---------------------|",
    ]

    rec = records[0] if records else None

    for i, ch_name in enumerate(channel_names):
        parsed = parse_channel_name(ch_name)
        ch_type = parsed.get("channel_type", "—")
        filter_cube = parsed.get("filter_cube", "—")
        marker = "—"
        if rec and i < MAX_CHANNELS:
            marker_val = getattr(rec, f"channel_{i}_marker", None)
            if marker_val:
                marker = marker_val
        lines.append(f"| {dataset_name} | {i} | {ch_name} | {ch_type} | {filter_cube} | {marker} |")

    return "\n".join(lines)


def format_register_summary(result: RegisterResult, dry_run: bool = False) -> str:
    """Format registration results as markdown.

    Parameters
    ----------
    result : RegisterResult
        Output of :func:`register_fovs`.
    dry_run : bool
        Whether this was a dry run.

    Returns
    -------
    str
        Markdown summary string.
    """
    status = "dry_run" if dry_run else "executed"
    xy = f"{result.pixel_size_xy_um:.4f}" if result.pixel_size_xy_um is not None else "—"
    z = f"{result.pixel_size_z_um:.4f}" if result.pixel_size_z_um is not None else "—"
    lines = [
        f"\n## Register Summary — {result.dataset}\n",
        "| metric | value |",
        "|--------|-------|",
        f"| created | {len(result.created)} |",
        f"| updated | {len(result.updated)} |",
        f"| unmatched | {len(result.unmatched)} |",
        f"| pixel_size_xy_um | {xy} |",
        f"| pixel_size_z_um | {z} |",
        f"| status | {status} |",
        "",
    ]

    if result.unmatched:
        lines.append("### Unmatched positions (no well template)\n")
        for pos in result.unmatched[:20]:
            lines.append(f"- `{pos}`")
        if len(result.unmatched) > 20:
            lines.append(f"- ... and {len(result.unmatched) - 20} more")
        lines.append("")

    return "\n".join(lines)


# Fields required for a complete flat parquet cell index.
# "zarr" = written by register, "platemap" = biologist fills in Airtable.
PARQUET_REQUIRED_FIELDS: list[tuple[str, str]] = [
    ("data_path", "zarr"),
    ("tracks_path", "platemap"),
    ("channel_0_name", "zarr"),
    ("channel_0_marker", "zarr"),
    ("pixel_size_xy_um", "zarr"),
    ("pixel_size_z_um", "zarr"),
    ("perturbation", "platemap"),
    ("time_interval_min", "platemap"),
    ("hours_post_perturbation", "platemap"),
    ("cell_type", "platemap"),
]


def build_completeness_report(
    dataset_name: str,
    records: list[DatasetRecord],
) -> str:
    """Check a representative record for missing fields needed by the parquet pipeline.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    records : list[DatasetRecord]
        Airtable FOV records for the dataset.

    Returns
    -------
    str
        Markdown report with missing fields and suggested actions.
    """
    if not records:
        return ""

    rec = records[0]
    missing: list[tuple[str, str]] = []
    for field_name, source in PARQUET_REQUIRED_FIELDS:
        val = getattr(rec, field_name, None)
        if val is None or val == "" or val == []:
            missing.append((field_name, source))

    if not missing:
        return f"\n## Parquet Readiness — {dataset_name}\n\nAll required fields populated.\n"

    lines = [
        f"\n## Parquet Readiness — {dataset_name}\n",
        f"**{len(missing)} field(s) still needed** before building a flat parquet:\n",
        "| missing field | source | action |",
        "|---------------|--------|--------|",
    ]
    for field_name, source in missing:
        if source == "zarr":
            action = "re-run `register` (should have been filled — check zarr metadata)"
        else:
            action = "fill in Airtable platemap or use MCP bulk update"
        lines.append(f"| `{field_name}` | {source} | {action} |")
    lines.append("")

    return "\n".join(lines)


def register_fovs(
    position_paths: list[Path],
    db: AirtableDatasets | None = None,
    dataset_name: str | None = None,
) -> RegisterResult:
    """Compute per-FOV records to create/update for the given positions.

    Parameters
    ----------
    position_paths : list[Path]
        Paths to individual zarr positions, e.g.
        ``[Path("/data/ds.zarr/A/1/000000"), ...]``.
        All must belong to the same zarr store.
    db : AirtableDatasets or None
        Airtable interface. Created from env vars if None.
    dataset_name : str or None
        Airtable dataset name to look up. Defaults to the zarr
        store's stem (e.g. ``"my_dataset"`` from
        ``my_dataset.zarr``).

    Returns
    -------
    RegisterResult
        Computed creates, updates, and unmatched positions.

    Raises
    ------
    ValueError
        If no Airtable records exist for the dataset, or paths
        span multiple zarr stores.
    """
    if db is None:
        db = AirtableDatasets()

    if not position_paths:
        raise ValueError("No position paths provided.")

    zarr_root, first_pos = parse_position_path(position_paths[0])
    if dataset_name is None:
        dataset_name = zarr_root.stem

    # Validate all paths belong to the same zarr
    pos_names: list[str] = [first_pos]
    for p in position_paths[1:]:
        root, pos = parse_position_path(p)
        if root != zarr_root:
            raise ValueError(f"All positions must belong to the same zarr store. Got {zarr_root} and {root}.")
        pos_names.append(pos)

    existing_records = db.get_dataset_records(dataset_name)
    if not existing_records:
        raise ValueError(
            f"No Airtable records for dataset '{dataset_name}'. Ensure the platemap has been filled first."
        )

    # Fetch Marker Registry once — keyed by Airtable record ID
    registry = db.get_marker_registry()
    logger.info("Loaded %d Marker Registry entries", len(registry))

    well_templates: dict[str, DatasetRecord] = {}
    fov_records: dict[tuple[str, str], DatasetRecord] = {}
    for rec in existing_records:
        if rec.fov:
            fov_records[(rec.well_id, rec.fov)] = rec
        else:
            well_templates[rec.well_id] = rec

    logger.info(
        "Found %d well templates, %d existing FOV records for '%s'",
        len(well_templates),
        len(fov_records),
        dataset_name,
    )

    result = RegisterResult(dataset=dataset_name)

    # Filter to directories only — glob("*/*/*") also picks up zarr.json, .zattrs, .zgroup files
    pos_names = [p for p in pos_names if (zarr_root / p).is_dir()]

    with open_ome_zarr(str(zarr_root), mode="r") as plate:
        result.channel_names = plate.channel_names

        if len(plate.channel_names) > MAX_CHANNELS:
            logger.warning(
                "Zarr has %d channels but Airtable schema supports %d. Channels %d+ will not be recorded.",
                len(plate.channel_names),
                MAX_CHANNELS,
                MAX_CHANNELS,
            )

        # Read pixel scale from the first position (uniform across plate)
        first_pos = plate[pos_names[0]]
        scale = tuple(first_pos.scale) if hasattr(first_pos, "scale") else None
        if scale is not None and len(scale) >= 5:
            z_um, y_um = scale[2], scale[3]
            if not (z_um == 1.0 and y_um == 1.0):
                result.pixel_size_xy_um = y_um
                result.pixel_size_z_um = z_um

        for pos_name in pos_names:
            well_id, fov = parse_position_name(pos_name)
            pos = plate[pos_name]
            shape = pos.data.shape

            zarr_fields = zarr_fields_for_position(zarr_root, pos_name, result.channel_names, shape, scale=scale)

            # Resolve cell_line linked records -> registry entries -> marker
            rec_for_marker = fov_records.get((well_id, fov)) or well_templates.get(well_id)
            if rec_for_marker is not None and rec_for_marker.cell_line:
                marker_entries = [registry[rid] for rid in rec_for_marker.cell_line if rid in registry]
                marker_fields = derive_channel_marker(result.channel_names, marker_entries)
                zarr_fields.update(marker_fields)

            existing = fov_records.get((well_id, fov))
            if existing is not None:
                if existing.record_id:
                    result.updated.append({"id": existing.record_id, "fields": zarr_fields})
                continue

            template = well_templates.get(well_id)
            if template is None:
                result.unmatched.append(pos_name)
                continue

            fields = {
                "dataset": dataset_name,
                "well_id": well_id,
                "fov": fov,
                **zarr_fields,
                **copy_well_template_fields(template),
            }
            result.created.append({"fields": fields})

    return result
