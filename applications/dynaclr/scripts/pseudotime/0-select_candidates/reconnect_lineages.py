"""Inspect lineage reconnection on a candidate-set's productive cohort.

Reads the productive cohort CSV (output of :mod:`select_candidates` or
:mod:`manual_candidates`), reconnects lineages via ``parent_track_id``,
and prints a summary of:

- lineages with two or more tracks (mother + daughter chains)
- distribution of ``divides ∈ {none, pre, during, post}``
- per-lineage track lists for visual inspection

This is read-only — it does not rewrite the cohort CSV. It exists to
verify Phase 1 lineage reconnection is doing what we expect before
running the rest of the pipeline.

Usage::

    cd applications/dynaclr/scripts/pseudotime/0-select_candidates
    uv run python reconnect_lineages.py \
        --candidate-set zikv_productive_07_24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATES_DIR = SCRIPT_DIR / "candidates"


def main() -> None:
    """Print a per-cohort lineage report (multi-track lineages + divides distribution)."""
    parser = argparse.ArgumentParser(description="Inspect lineage reconnection on a candidate set.")
    parser.add_argument("--candidate-set", required=True, help="Candidate-set name")
    parser.add_argument(
        "--cohort",
        default="productive",
        choices=["productive", "bystander", "abortive", "mock"],
        help="Cohort CSV to inspect (default: productive)",
    )
    args = parser.parse_args()

    csv_path = CANDIDATES_DIR / f"{args.candidate_set}_{args.cohort}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run select_candidates.py or manual_candidates.py first.")

    df = pd.read_csv(csv_path)
    print(f"# Lineage report — {csv_path.name}\n")
    print(f"- {len(df):,} rows")
    print(f"- {df.groupby(['dataset_id', 'fov_name', 'track_id']).ngroups:,} unique tracks")
    print(f"- {df['lineage_id'].nunique():,} unique lineages")
    print()

    # Multi-track lineages — these are the mother+daughter chains.
    track_counts = df.groupby("lineage_id")["track_id"].nunique()
    multi_track = track_counts[track_counts > 1].sort_values(ascending=False)
    print(f"## Multi-track lineages ({len(multi_track):,}/{len(track_counts):,})\n")
    if len(multi_track):
        for lineage_id, n_tracks in multi_track.head(20).items():
            sub = df[df["lineage_id"] == lineage_id]
            track_ids = sorted(sub["track_id"].unique())
            divides = sub["divides"].iloc[0]
            ds = sub["dataset_id"].iloc[0]
            fov = sub["fov_name"].iloc[0]
            print(f"- lineage {lineage_id} | {ds} {fov} | tracks {track_ids} | divides={divides}")
        if len(multi_track) > 20:
            print(f"- ... ({len(multi_track) - 20} more)")
    else:
        print("(none — every lineage is a single track)")
    print()

    # Divides distribution per cohort.
    divides_by_lineage = df.drop_duplicates(["lineage_id"])[["lineage_id", "divides"]]
    print("## Divides distribution\n")
    counts = divides_by_lineage["divides"].value_counts()
    for k in ("none", "pre", "during", "post"):
        print(f"- {k}: {counts.get(k, 0)}")


if __name__ == "__main__":
    main()
