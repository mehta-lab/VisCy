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

from airtable_utils.database import AirtableDatasets, CellLineEntry
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
    return fields


def _normalize_biology(biology: str) -> str:
    """Normalize biology label to canonical form."""
    if biology.lower() in ("sensor", "viral sensor"):
        return "viral_sensor"
    return biology


def derive_channel_biology(
    channel_names: list[str],
    cell_line_entries: list[CellLineEntry],
) -> dict[str, str]:
    """Derive channel biology annotations from Cell Line Registry entries.

    For each channel name, finds the first registry entry whose aliases
    contain a substring match, and returns the biology label.

    Parameters
    ----------
    channel_names : list[str]
        Ordered channel names from the zarr store.
    cell_line_entries : list[CellLineEntry]
        Registry entries linked to the well record.

    Returns
    -------
    dict[str, str]
        Mapping of ``"channel_{i}_biology"`` -> biology label
        for channels that matched a registry entry.
    """
    result: dict[str, str] = {}
    for i, ch_name in enumerate(channel_names[:MAX_CHANNELS]):
        for entry in cell_line_entries:
            if any(alias in ch_name for alias in entry.channel_name_aliases):
                result[f"channel_{i}_biology"] = _normalize_biology(entry.biology)
                break
    return result


def copy_well_template_fields(template: DatasetRecord) -> dict:
    """Copy biologist-provided fields from a well template record.

    Parameters
    ----------
    template : DatasetRecord
        Well-level record with biology metadata.

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
        bio_val = getattr(template, f"channel_{i}_biology", None)
        if bio_val is not None:
            fields[f"channel_{i}_biology"] = bio_val
    return fields


def build_validation_table(
    dataset_name: str,
    channel_names: list[str],
    records: list[DatasetRecord],
) -> str:
    """Build markdown validation table for channel / biology pairing.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    channel_names : list[str]
        Ordered channel names from the zarr.
    records : list[DatasetRecord]
        Airtable records (first record used for biology lookup).

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [
        "| dataset | idx | channel_name | type | filter_cube | biology (scientist) |",
        "|---------|-----|--------------|------|-------------|---------------------|",
    ]

    rec = records[0] if records else None

    for i, ch_name in enumerate(channel_names):
        parsed = parse_channel_name(ch_name)
        ch_type = parsed.get("channel_type", "—")
        filter_cube = parsed.get("filter_cube", "—")
        biology = "—"
        if rec and i < MAX_CHANNELS:
            bio_val = getattr(rec, f"channel_{i}_biology", None)
            if bio_val:
                biology = bio_val
        lines.append(f"| {dataset_name} | {i} | {ch_name} | {ch_type} | {filter_cube} | {biology} |")

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
    lines = [
        f"\n## Register Summary — {result.dataset}\n",
        "| metric | count |",
        "|--------|-------|",
        f"| created | {len(result.created)} |",
        f"| updated | {len(result.updated)} |",
        f"| unmatched | {len(result.unmatched)} |",
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


def register_fovs(
    position_paths: list[Path],
    db: AirtableDatasets | None = None,
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

    # Fetch Cell Line Registry once — keyed by Airtable record ID
    registry = db.get_cell_line_registry()
    logger.info("Loaded %d Cell Line Registry entries", len(registry))

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

    # Filter to directories only — glob("*/*/*") also picks up .zattrs/.zgroup files
    pos_names = [p for p in pos_names if not Path(zarr_root / p).name.startswith(".")]

    with open_ome_zarr(str(zarr_root), mode="r") as plate:
        result.channel_names = plate.channel_names

        if len(plate.channel_names) > MAX_CHANNELS:
            logger.warning(
                "Zarr has %d channels but Airtable schema supports %d. Channels %d+ will not be recorded.",
                len(plate.channel_names),
                MAX_CHANNELS,
                MAX_CHANNELS,
            )

        for pos_name in pos_names:
            well_id, fov = parse_position_name(pos_name)
            pos = plate[pos_name]
            shape = pos.data.shape

            zarr_fields = zarr_fields_for_position(zarr_root, pos_name, result.channel_names, shape)

            # Resolve cell_line linked records -> registry entries -> biology
            rec_for_biology = fov_records.get((well_id, fov)) or well_templates.get(well_id)
            if rec_for_biology is not None and rec_for_biology.cell_line:
                cell_line_entries = [registry[rid] for rid in rec_for_biology.cell_line if rid in registry]
                biology_fields = derive_channel_biology(result.channel_names, cell_line_entries)
                zarr_fields.update(biology_fields)

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
