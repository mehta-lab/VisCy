"""Backfill cell_line_2 from cell_line in the Datasets table.

Reads all Datasets records that have cell_line set, matches each value
against the Cell Line Registry by name, and sets cell_line_2 to the
linked record IDs.

Usage
-----
    uv run --package airtable-utils \
        applications/airtable/scripts/migrate_cell_line_2.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import os

from pyairtable import Api

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REGISTRY_TABLE_ID = "tblmP8l2GmpCeERyD"
DATASETS_TABLE_ID = "tblaFzrDMlVZHPZIj"


def main(dry_run: bool = False) -> None:  # noqa: D103
    api_key = os.environ["AIRTABLE_API_KEY"]
    base_id = os.environ["AIRTABLE_BASE_ID"]
    api = Api(api_key)

    registry_table = api.table(base_id, REGISTRY_TABLE_ID)
    datasets_table = api.table(base_id, DATASETS_TABLE_ID)

    # Build lookup: cell_line name -> registry record ID
    logger.info("Fetching Cell Line Registry...")
    registry_records = registry_table.all(fields=["cell_line"])
    registry_lut: dict[str, str] = {}
    for rec in registry_records:
        name = rec["fields"].get("cell_line", "")
        if name:
            registry_lut[name] = rec["id"]
    logger.info("Registry has %d entries", len(registry_lut))

    # Fetch all Datasets records that have cell_line set
    logger.info("Fetching Datasets records with cell_line...")
    raw_records = datasets_table.all(fields=["cell_line", "cell_line_2"])

    updates: list[dict] = []
    unmatched: set[str] = set()

    for rec in raw_records:
        fields = rec["fields"]
        cell_line_values = fields.get("cell_line", [])
        if not cell_line_values:
            continue

        # Normalize: cell_line may be list of strings or list of dicts
        names: list[str] = []
        for v in cell_line_values:
            if isinstance(v, dict):
                names.append(v.get("name", ""))
            else:
                names.append(v)

        # Match each name to a registry record ID
        linked_ids: list[str] = []
        for name in names:
            if name in registry_lut:
                linked_ids.append(registry_lut[name])
            else:
                unmatched.add(name)

        if not linked_ids:
            continue

        # Skip if cell_line_2 already set to the same values
        existing = fields.get("cell_line_2", [])
        existing_ids = {r["id"] if isinstance(r, dict) else r for r in existing}
        if existing_ids == set(linked_ids):
            continue

        updates.append(
            {
                "id": rec["id"],
                "fields": {"cell_line_2": linked_ids},
            }
        )

    logger.info(
        "Records to update: %d | unmatched cell_line values: %d",
        len(updates),
        len(unmatched),
    )

    if unmatched:
        logger.warning(
            "No registry entry found for: %s\nAdd these to the Cell Line Registry table before re-running.",
            sorted(unmatched),
        )

    if dry_run:
        logger.info("[DRY RUN] Would update %d records", len(updates))
        for upd in updates[:5]:
            logger.info("  %s -> cell_line_2: %s", upd["id"], upd["fields"]["cell_line_2"])
        if len(updates) > 5:
            logger.info("  ... and %d more", len(updates) - 5)
        return

    # Batch update in chunks of 10 (Airtable limit)
    for i in range(0, len(updates), 10):
        batch = updates[i : i + 10]
        datasets_table.batch_update(batch)
        logger.info("Updated records %d-%d", i + 1, i + len(batch))

    logger.info("Done. Updated %d records.", len(updates))

    if unmatched:
        print("\n## Unmatched cell_line values (not in registry)\n")
        for name in sorted(unmatched):
            print(f"- `{name}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill cell_line_2 from cell_line")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
