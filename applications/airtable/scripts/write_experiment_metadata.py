"""Manage experiment metadata between Airtable and OME-Zarr datasets.

Two subcommands:

    register  — expand well-level Airtable records to per-FOV records
                using zarr position data (zarr -> Airtable)
    write     — write experiment_metadata to zarr .zattrs from Airtable
                per-FOV records (Airtable -> zarr)

Both operate at the position level. Use shell globbing for batch::

    uv run --package airtable-utils \
        applications/airtable/scripts/write_experiment_metadata.py \
        register /path/to/dataset.zarr/A/1/000000       # single position

    uv run --package airtable-utils \
        applications/airtable/scripts/write_experiment_metadata.py \
        register /path/to/dataset.zarr/*/*/*             # all positions

    uv run --package airtable-utils \
        applications/airtable/scripts/write_experiment_metadata.py \
        write /path/to/dataset.zarr/*/*/*                # write zattrs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from iohub import open_ome_zarr

from airtable_utils.database import AirtableDatasets
from airtable_utils.registration import (
    build_completeness_report,
    build_validation_table,
    format_register_summary,
    parse_position_path,
    register_fovs,
)
from airtable_utils.schemas import MAX_CHANNELS, DatasetRecord, parse_position_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# register: zarr -> Airtable (well records -> per-FOV records)
# ---------------------------------------------------------------------------


def register(position_paths: list[Path], dry_run: bool = False, dataset: str | None = None) -> None:
    """Register zarr positions as per-FOV records in Airtable."""
    db = AirtableDatasets()
    result = register_fovs(position_paths, db=db, dataset_name=dataset)

    logger.info(
        "FOVs to create: %d | existing to update: %d | unmatched: %d",
        len(result.created),
        len(result.updated),
        len(result.unmatched),
    )

    if not dry_run:
        if result.created:
            db.batch_create(result.created)
            logger.info("Created %d per-FOV records in Airtable", len(result.created))
        if result.updated:
            db.batch_update(result.updated)
            logger.info("Updated %d existing records", len(result.updated))

    print(format_register_summary(result, dry_run=dry_run))

    all_records = db.get_dataset_records(result.dataset)
    validation = build_validation_table(result.dataset, result.channel_names, all_records)
    print(f"## Channel Validation — {result.dataset}\n")
    print(validation)
    print()

    fov_records = [r for r in all_records if r.fov]
    completeness = build_completeness_report(result.dataset, fov_records)
    print(completeness)


# ---------------------------------------------------------------------------
# write: Airtable -> zarr (per-FOV records -> .zattrs)
# ---------------------------------------------------------------------------


def write(position_paths: list[Path], dry_run: bool = False) -> None:
    """Write experiment_metadata from per-FOV Airtable records to zarr."""
    zarr_root, first_pos = parse_position_path(position_paths[0])
    dataset_name = zarr_root.stem

    pos_names: list[str] = [first_pos]
    for p in position_paths[1:]:
        root, pos = parse_position_path(p)
        if root != zarr_root:
            raise ValueError(f"All positions must belong to the same zarr store. Got {zarr_root} and {root}.")
        pos_names.append(pos)

    logger.info(
        "Writing experiment metadata for %d positions in %s",
        len(pos_names),
        dataset_name,
    )

    db = AirtableDatasets()
    all_records = db.get_dataset_records(dataset_name)

    fov_records = [r for r in all_records if r.fov]
    if not fov_records:
        raise ValueError(
            f"No per-FOV records for dataset '{dataset_name}'. Run 'register' first to expand well records."
        )

    record_lookup: dict[tuple[str, str], DatasetRecord] = {}
    for rec in fov_records:
        record_lookup[(rec.well_id, rec.fov)] = rec

    logger.info("Found %d per-FOV records in Airtable", len(fov_records))

    fov_count = 0
    with open_ome_zarr(str(zarr_root), mode="r+" if not dry_run else "r") as plate:
        channel_names = plate.channel_names

        for pos_name in pos_names:
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

            for i, ch_name in enumerate(channel_names[:MAX_CHANNELS]):
                setattr(rec, f"channel_{i}_name", ch_name)

            channels_metadata = rec.to_channels_metadata()
            experiment_metadata = rec.to_experiment_metadata()

            if dry_run:
                logger.info(
                    "[DRY RUN] %s\n  channels_metadata: %s\n  experiment_metadata: %s",
                    pos_name,
                    channels_metadata,
                    experiment_metadata,
                )
            else:
                pos = plate[pos_name]
                pos.zattrs["channels_metadata"] = channels_metadata
                pos.zattrs["experiment_metadata"] = experiment_metadata
                fov_count += 1

        # Write plate-level channels_metadata
        if not dry_run and fov_records:
            first_rec = fov_records[0]
            for i, ch_name in enumerate(channel_names[:MAX_CHANNELS]):
                setattr(first_rec, f"channel_{i}_name", ch_name)
            plate.zattrs["channels_metadata"] = first_rec.to_channels_metadata()

    status = "dry_run" if dry_run else "success"
    print("\n## Experiment Metadata Write Summary\n")
    print("| dataset | zarr_path | num_fovs | status |")
    print("|---------|-----------|----------|--------|")
    print(f"| {dataset_name} | {zarr_root} | {fov_count} | {status} |")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description="Manage experiment metadata between Airtable and OME-Zarr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reg_parser = subparsers.add_parser(
        "register",
        help="Register zarr positions as per-FOV Airtable records",
    )
    reg_parser.add_argument(
        "positions",
        type=Path,
        nargs="+",
        help="Position path(s), e.g. /data/ds.zarr/A/1/000000 or /data/ds.zarr/*/*/*",
    )
    reg_parser.add_argument("--dry-run", action="store_true", help="Log what would happen without writing")
    reg_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Airtable dataset name override (default: zarr stem). Use when zarr stem doesn't match.",
    )

    write_parser = subparsers.add_parser(
        "write",
        help="Write experiment_metadata from Airtable per-FOV records to zarr .zattrs",
    )
    write_parser.add_argument(
        "positions",
        type=Path,
        nargs="+",
        help="Position path(s), e.g. /data/ds.zarr/A/1/000000 or /data/ds.zarr/*/*/*",
    )
    write_parser.add_argument("--dry-run", action="store_true", help="Log what would happen without writing")

    args = parser.parse_args()

    if args.command == "register":
        register(args.positions, dry_run=args.dry_run, dataset=args.dataset)
    elif args.command == "write":
        write(args.positions, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
