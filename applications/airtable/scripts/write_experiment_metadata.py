"""Manage experiment metadata between Airtable and OME-Zarr datasets.

Two subcommands:

    register  — expand well-level Airtable records to per-FOV records
                using zarr position data (zarr → Airtable)
    write     — write experiment_metadata to zarr .zattrs from Airtable
                per-FOV records (Airtable → zarr)

Usage
-----
    uv run --package airtable-utils \
        applications/airtable/scripts/write_experiment_metadata.py \
        register /path/to/dataset.zarr [--dry-run]

    uv run --package airtable-utils \
        applications/airtable/scripts/write_experiment_metadata.py \
        write /path/to/dataset.zarr [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from iohub import open_ome_zarr

from airtable_utils.database import AirtableDatasets
from airtable_utils.schemas import DatasetRecord, parse_channel_name, parse_position_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACKING_CSV = Path("experiment_metadata_tracking.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_tracking_csv() -> set[str]:
    """Return set of dataset names already processed successfully."""
    if not TRACKING_CSV.exists():
        return set()
    done = set()
    with open(TRACKING_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "success":
                done.add(row["dataset"])
    return done


def _append_tracking_csv(row: dict) -> None:
    """Append a row to the tracking CSV."""
    write_header = not TRACKING_CSV.exists()
    with open(TRACKING_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "zarr_path",
                "num_fovs",
                "status",
                "error_message",
                "timestamp",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _build_validation_table(
    dataset_name: str,
    channel_names: list[str],
    records: list[DatasetRecord],
) -> str:
    """Build markdown validation table for channel / biology pairing."""
    lines = [
        "| dataset | idx | channel_name | type | filter_cube | biology (scientist) |",
        "|---------|-----|--------------|------|------------- |---------------------|",
    ]

    rec = records[0] if records else None

    for i, ch_name in enumerate(channel_names):
        parsed = parse_channel_name(ch_name)
        ch_type = parsed.get("channel_type", "—")
        filter_cube = parsed.get("filter_cube", "—")
        biology = "—"
        if rec and i <= 3:
            bio_val = getattr(rec, f"channel_{i}_biology", None)
            if bio_val:
                biology = bio_val
        lines.append(f"| {dataset_name} | {i} | {ch_name} | {ch_type} | {filter_cube} | {biology} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# register: zarr → Airtable (well records → per-FOV records)
# ---------------------------------------------------------------------------


def register(zarr_path: Path, dry_run: bool = False) -> None:
    """Expand well-level Airtable records into per-FOV records using zarr."""
    dataset_name = zarr_path.stem
    logger.info("Registering FOVs for dataset: %s", dataset_name)

    db = AirtableDatasets()
    existing_records = db.get_dataset_records(dataset_name)

    if not existing_records:
        logger.error(
            "No Airtable records found for dataset '%s'. Ensure the platemap has been filled first.",
            dataset_name,
        )
        return

    # Build lookup: well_id → well record (records without fov are well-level)
    well_lookup: dict[str, DatasetRecord] = {}
    existing_fovs: set[tuple[str, str]] = set()
    for rec in existing_records:
        if rec.fov:
            existing_fovs.add((rec.well_id, rec.fov))
        else:
            well_lookup[rec.well_id] = rec

    if not well_lookup:
        # All records already have FOV — maybe they're already per-FOV
        logger.info(
            "All %d existing records already have FOVs set. Building lookup from per-FOV records instead.",
            len(existing_records),
        )
        for rec in existing_records:
            well_lookup.setdefault(rec.well_id, rec)

    logger.info(
        "Found %d well templates, %d existing FOV records",
        len(well_lookup),
        len(existing_fovs),
    )

    # Build lookup: (well_id, fov) → existing record for updates
    existing_record_lookup: dict[tuple[str, str], DatasetRecord] = {}
    for rec in existing_records:
        if rec.fov:
            existing_record_lookup[(rec.well_id, rec.fov)] = rec

    plate = open_ome_zarr(str(zarr_path), mode="r")
    position_list = list(plate.positions())
    channel_names = plate.channel_names
    dim_names = ("t_shape", "c_shape", "z_shape", "y_shape", "x_shape")

    new_records: list[dict] = []
    airtable_updates: list[dict] = []
    unmatched = []

    for pos_name, pos in position_list:
        well_path, fov = parse_position_name(pos_name)
        shape = pos.data.shape
        expected_data_path = str(zarr_path / pos_name)

        # Zarr-derived fields common to both create and update
        zarr_fields: dict = {"data_path": expected_data_path}
        for i, ch_name in enumerate(channel_names):
            if i <= 3:
                zarr_fields[f"channel_{i}_name"] = ch_name
        for dim_name, dim_val in zip(dim_names, shape):
            zarr_fields[dim_name] = dim_val

        existing_rec = existing_record_lookup.get((well_path, fov))
        if existing_rec is not None:
            # Update existing FOV record with zarr-derived fields
            if existing_rec.record_id:
                airtable_updates.append({"id": existing_rec.record_id, "fields": zarr_fields})
            continue

        # New FOV — need a well template to copy platemap metadata
        well_rec = well_lookup.get(well_path)
        if well_rec is None:
            unmatched.append(pos_name)
            continue

        fields: dict = {
            "dataset": dataset_name,
            "well_id": well_path,
            "fov": fov,
            **zarr_fields,
        }

        for key in (
            "cell_type",
            "cell_state",
            "cell_line",
            "organelle",
            "perturbation",
            "hours_post_perturbation",
            "moi",
            "time_interval_min",
            "seeding_density",
            "treatment_concentration_nm",
            "fluorescence_modality",
        ):
            val = getattr(well_rec, key)
            if val is not None:
                fields[key] = val

        for i in range(4):
            bio_val = getattr(well_rec, f"channel_{i}_biology", None)
            if bio_val is not None:
                fields[f"channel_{i}_biology"] = bio_val

        new_records.append({"fields": fields})

    plate.close()

    if unmatched:
        logger.warning(
            "No well record found for %d positions: %s",
            len(unmatched),
            unmatched[:10],
        )

    logger.info(
        "FOVs to create: %d | existing to update: %d | unmatched: %d",
        len(new_records),
        len(airtable_updates),
        len(unmatched),
    )

    if dry_run:
        for rec in new_records[:5]:
            logger.info("[DRY RUN] Would create: %s", rec["fields"])
        if len(new_records) > 5:
            logger.info("  ... and %d more", len(new_records) - 5)
        for upd in airtable_updates[:5]:
            logger.info("[DRY RUN] Would update %s: %s", upd["id"], upd["fields"])
        if len(airtable_updates) > 5:
            logger.info("  ... and %d more", len(airtable_updates) - 5)
    else:
        if new_records:
            db.batch_create(new_records)
            logger.info("Created %d per-FOV records in Airtable", len(new_records))
        if airtable_updates:
            db.batch_update(airtable_updates)
            logger.info(
                "Updated %d existing records (channel names, shapes, data_path)",
                len(airtable_updates),
            )

    # Print channel validation table
    validation = _build_validation_table(dataset_name, channel_names, existing_records)
    print(f"\n## Channel Validation — {dataset_name}\n")
    print(validation)
    print()


# ---------------------------------------------------------------------------
# write: Airtable → zarr (per-FOV records → .zattrs)
# ---------------------------------------------------------------------------


def write(zarr_path: Path, dry_run: bool = False) -> None:
    """Write experiment_metadata from per-FOV Airtable records to zarr."""
    dataset_name = zarr_path.stem
    logger.info("Writing experiment metadata for dataset: %s", dataset_name)

    db = AirtableDatasets()
    all_records = db.get_dataset_records(dataset_name)

    # Only use records that have fov set (per-FOV)
    fov_records = [r for r in all_records if r.fov]
    if not fov_records:
        logger.error(
            "No per-FOV records found for dataset '%s'. Run 'register' first to expand well records.",
            dataset_name,
        )
        return

    # Build lookup: (well_id, fov) → record
    record_lookup: dict[tuple[str, str], DatasetRecord] = {}
    for rec in fov_records:
        record_lookup[(rec.well_id, rec.fov)] = rec

    logger.info("Found %d per-FOV records", len(fov_records))

    plate = open_ome_zarr(str(zarr_path), mode="r+" if not dry_run else "r")
    position_list = list(plate.positions())
    channel_names = plate.channel_names

    airtable_updates: list[dict] = []
    dim_names = ("t_shape", "c_shape", "z_shape", "y_shape", "x_shape")

    fov_count = 0
    for pos_name, pos in position_list:
        well_path, fov = parse_position_name(pos_name)

        rec = record_lookup.get((well_path, fov))
        if rec is None:
            logger.warning(
                "No Airtable record for %s (well=%s, fov=%s), skipping",
                pos_name,
                well_path,
                fov,
            )
            continue

        # Read shape from this FOV's array
        shape = pos.data.shape

        # Enrich the record with channel names from zarr before writing zattrs
        for i, ch_name in enumerate(channel_names):
            if i <= 3:
                setattr(rec, f"channel_{i}_name", ch_name)

        channel_annotation = rec.to_channel_annotation()
        experiment_metadata = rec.to_experiment_metadata()

        # Build Airtable update: channel names, shapes, data_path
        airtable_fields: dict = {}
        if rec.record_id:
            for i, ch_name in enumerate(channel_names):
                if i <= 3:
                    airtable_fields[f"channel_{i}_name"] = ch_name
            for dim_name, dim_val in zip(dim_names, shape):
                airtable_fields[dim_name] = dim_val
            expected_data_path = str(zarr_path / pos_name)
            if rec.data_path != expected_data_path:
                airtable_fields["data_path"] = expected_data_path

        if dry_run:
            logger.info(
                "[DRY RUN] %s\n  channel_annotation: %s\n  experiment_metadata: %s\n  airtable: %s",
                pos_name,
                channel_annotation,
                experiment_metadata,
                airtable_fields,
            )
        else:
            pos.zattrs["channel_annotation"] = channel_annotation
            pos.zattrs["experiment_metadata"] = experiment_metadata
            fov_count += 1

        if airtable_fields and rec.record_id:
            airtable_updates.append({"id": rec.record_id, "fields": airtable_fields})

    # Write plate-level channel_annotation (use first record's annotation)
    if not dry_run and fov_records:
        first_rec = fov_records[0]
        for i, ch_name in enumerate(channel_names):
            if i <= 3:
                setattr(first_rec, f"channel_{i}_name", ch_name)
        plate.zattrs["channel_annotation"] = first_rec.to_channel_annotation()

    plate.close()

    # Batch-update Airtable with zarr-derived fields
    if airtable_updates and not dry_run:
        db.batch_update(airtable_updates)
        logger.info(
            "Updated %d Airtable records (channel names, shapes, data_path)",
            len(airtable_updates),
        )

    result = {
        "dataset": dataset_name,
        "zarr_path": str(zarr_path),
        "num_fovs": fov_count,
        "status": "dry_run" if dry_run else "success",
        "error_message": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        _append_tracking_csv(result)

    # Print summary
    print("\n## Experiment Metadata Write Summary\n")
    print("| dataset | zarr_path | num_fovs | status |")
    print("|---------|-----------|----------|--------|")
    print(f"| {result['dataset']} | {result['zarr_path']} | {result['num_fovs']} | {result['status']} |")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Manage experiment metadata between Airtable and OME-Zarr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register subcommand
    reg_parser = subparsers.add_parser(
        "register",
        help="Expand well-level Airtable records to per-FOV using zarr positions",
    )
    reg_parser.add_argument("zarr_path", type=Path, help="Path to the OME-Zarr dataset")
    reg_parser.add_argument("--dry-run", action="store_true", help="Log what would happen without writing")

    # write subcommand
    write_parser = subparsers.add_parser(
        "write",
        help="Write experiment_metadata from Airtable per-FOV records to zarr .zattrs",
    )
    write_parser.add_argument("zarr_path", type=Path, help="Path to the OME-Zarr dataset")
    write_parser.add_argument("--dry-run", action="store_true", help="Log what would happen without writing")

    args = parser.parse_args()

    if args.command == "register":
        register(args.zarr_path, dry_run=args.dry_run)
    elif args.command == "write":
        write(args.zarr_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
