"""Migrate channel_N_marker values from organelle names to protein markers.

Reads all Datasets records that have cell_line links, resolves each
cell_line record ID against the Marker Registry (which has the
canonical ``marker`` protein name and ``channel_name_aliases``), and
updates ``channel_N_marker`` fields in the Datasets table.

Logic per channel slot (N=0..7):

- If ``channel_N_name`` exists: use ``parse_channel_name`` to classify.
  - labelfree  -> set marker = channel_N_name
  - virtual_stain -> set marker = channel_N_name
  - fluorescence -> match against cell_line aliases -> set marker from registry
- If ``channel_N_name`` is absent but ``channel_N_marker`` exists:
  the old marker is an organelle name. Use the FOV's cell_line link to
  look up the registry ``marker`` for the first linked construct. Only
  update fluorescence-like slots (skip slots whose old marker is
  "brightfield", "labelfree", or starts with "virtual-stain").

Usage
-----
    uv run --package airtable-utils \
        applications/airtable/scripts/migrate_channel_markers.py --dry-run

    uv run --package airtable-utils \
        applications/airtable/scripts/migrate_channel_markers.py
"""

from __future__ import annotations

import argparse
import logging
import os

from pyairtable import Api

from viscy_data.channel_utils import parse_channel_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REGISTRY_TABLE_ID = "tblmP8l2GmpCeERyD"
DATASETS_TABLE_ID = "tblaFzrDMlVZHPZIj"
MAX_CHANNELS = 8

LABELFREE_MARKERS = frozenset({"brightfield", "labelfree"})


def _is_labelfree_or_virtual(marker_value: str) -> bool:
    """Return True if the old marker value is labelfree or virtual-stain."""
    lower = marker_value.lower()
    return lower in LABELFREE_MARKERS or lower.startswith("virtual-stain") or lower == "nucleus"


def _match_alias(channel_name: str, aliases: list[str]) -> bool:
    """Return True if channel_name (lowercased) contains any alias (lowercased)."""
    name_lower = channel_name.lower()
    return any(alias.lower() in name_lower for alias in aliases)


def main(dry_run: bool = False, limit: int = 0) -> None:
    """Run the migration.

    Parameters
    ----------
    dry_run : bool
        If True, print changes without writing.
    limit : int
        Max number of changes to print in dry-run mode (0 = all).
    """
    api_key = os.environ["AIRTABLE_API_KEY"]
    base_id = os.environ["AIRTABLE_BASE_ID"]
    api = Api(api_key)

    registry_table = api.table(base_id, REGISTRY_TABLE_ID)
    datasets_table = api.table(base_id, DATASETS_TABLE_ID)

    # Build Marker Registry lookup: record_id -> {marker_fluorophore, aliases, marker}
    logger.info("Fetching Marker Registry...")
    registry_raw = registry_table.all(fields=["marker-fluorophore", "channel_name_aliases", "marker"])
    registry: dict[str, dict] = {}
    for rec in registry_raw:
        fields = rec.get("fields", {})
        marker_fluor = fields.get("marker-fluorophore", "")
        aliases_raw = fields.get("channel_name_aliases", "")
        aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]
        marker = fields.get("marker", "")
        if marker_fluor and marker:
            registry[rec["id"]] = {
                "marker_fluorophore": marker_fluor,
                "aliases": aliases,
                "marker": marker,
            }
    logger.info("Registry has %d entries with marker values", len(registry))

    # Fetch all Datasets fields we need
    channel_fields = []
    for i in range(MAX_CHANNELS):
        channel_fields.extend([f"channel_{i}_name", f"channel_{i}_marker"])
    fetch_fields = ["cell_line", "dataset", "well_id", "fov"] + channel_fields

    logger.info("Fetching Datasets records...")
    raw_records = datasets_table.all(fields=fetch_fields)
    logger.info("Fetched %d records", len(raw_records))

    updates: list[dict] = []
    no_cell_line = 0
    no_change = 0
    unmatched_channels: list[str] = []

    for rec in raw_records:
        fields = rec["fields"]
        cell_line_ids = fields.get("cell_line", [])
        if not cell_line_ids:
            no_cell_line += 1
            continue

        # Resolve cell_line IDs to registry entries
        entries = [registry[rid] for rid in cell_line_ids if rid in registry]
        if not entries:
            continue

        new_fields: dict[str, str] = {}

        for i in range(MAX_CHANNELS):
            ch_name = fields.get(f"channel_{i}_name")
            old_marker = fields.get(f"channel_{i}_marker")

            if ch_name is not None:
                # Have channel name: use parse_channel_name
                parsed = parse_channel_name(ch_name)
                ch_type = parsed.get("channel_type", "unknown")

                if ch_type == "labelfree":
                    new_marker = ch_name
                elif ch_type == "virtual_stain":
                    new_marker = ch_name
                elif ch_type == "fluorescence" or ch_type == "unknown":
                    # Match against registry aliases
                    matched = False
                    for entry in entries:
                        if _match_alias(ch_name, entry["aliases"]):
                            new_marker = entry["marker"]
                            matched = True
                            break
                    if not matched:
                        fov_id = f"{fields.get('dataset', '?')}_{fields.get('well_id', '?')}_{fields.get('fov', '')}"
                        unmatched_channels.append(f"{fov_id} ch{i}={ch_name}")
                        continue
                else:
                    continue

                if old_marker != new_marker:
                    new_fields[f"channel_{i}_marker"] = new_marker

            elif old_marker is not None:
                # No channel name but have old marker (organelle name).
                # Skip labelfree/virtual-stain markers.
                if _is_labelfree_or_virtual(old_marker):
                    continue
                # For fluorescence slots: use the first cell_line entry's marker
                # (most FOVs have a single construct)
                new_marker = entries[0]["marker"]
                if old_marker != new_marker:
                    new_fields[f"channel_{i}_marker"] = new_marker

        if new_fields:
            updates.append({"id": rec["id"], "fields": new_fields})
        else:
            no_change += 1

    logger.info(
        "Records to update: %d | no cell_line: %d | no change needed: %d | unmatched channels: %d",
        len(updates),
        no_cell_line,
        no_change,
        len(unmatched_channels),
    )

    if dry_run:
        show = updates[:limit] if limit > 0 else updates
        print(f"\n## Dry Run: {len(updates)} records to update (showing {len(show)})\n")
        print("| record_id | field | old | new |")
        print("|-----------|-------|-----|-----|")
        for upd in show:
            rec_id = upd["id"]
            # Look up old values from original records
            original = next(r for r in raw_records if r["id"] == rec_id)
            for field_name, new_val in upd["fields"].items():
                old_val = original["fields"].get(field_name, "(empty)")
                print(f"| {rec_id} | {field_name} | {old_val} | {new_val} |")
        if limit > 0 and len(updates) > limit:
            print(f"\n... and {len(updates) - limit} more records")

        if unmatched_channels:
            print(f"\n## Unmatched fluorescence channels ({len(unmatched_channels)})\n")
            for entry in unmatched_channels[:20]:
                print(f"- `{entry}`")
            if len(unmatched_channels) > 20:
                print(f"- ... and {len(unmatched_channels) - 20} more")
        return

    # Batch update in chunks of 10 (Airtable API limit)
    for i in range(0, len(updates), 10):
        batch = updates[i : i + 10]
        datasets_table.batch_update(batch)
        logger.info("Updated records %d-%d of %d", i + 1, i + len(batch), len(updates))

    logger.info("Done. Updated %d records.", len(updates))

    if unmatched_channels:
        print(f"\n## Unmatched fluorescence channels ({len(unmatched_channels)})\n")
        for entry in unmatched_channels[:20]:
            print(f"- `{entry}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate channel_N_marker from organelle names to protein markers")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing to Airtable")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of changes to show in dry-run mode (0 = all)",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, limit=args.limit)
