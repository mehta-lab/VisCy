"""Verify FlexibleBatchSampler composition for batch_group_by=marker + stratify_by=experiment.

Loads the production cell index, configures a sampler that mirrors the
proposed DynaCLR-2D-MIP single-marker override (marker batches stratified
by experiment), draws a handful of batches, and prints a marker x experiment
cross-tab per batch.

Run before committing a sampler config change to confirm batches compose
the way the config promises.

Usage
-----
    uv run python applications/dynaclr/scripts/dataloader_inspection/check_marker_experiment_stratify.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

CELL_INDEX_PARQUET = (
    "/hpc/projects/organelle_phenotyping/models/collections/DynaCLR-2D-MIP-BagOfChannels-v3.parquet"
)
BATCH_SIZE = 256
N_BATCHES_TO_SHOW = 16
SEED = 42


def _config(label: str, batch_group_by, stratify_by, group_weights=None) -> dict:
    return {
        "label": label,
        "batch_group_by": batch_group_by,
        "stratify_by": stratify_by,
        "group_weights": group_weights,
    }


# Uniform weights matching the v3 single-marker override (9 markers after BF
# and Retardance are dropped from the v3 collection).
UNIFORM_WEIGHTS = {
    "Phase3D": 1.0,
    "pAL10": 1.0,
    "viral_sensor": 1.0,
    "G3BP1": 1.0,
    "SEC61B": 1.0,
    "TOMM20": 1.0,
    "CAAX": 1.0,
    "HIST2H2BE": 1.0,
    "DIC": 1.0,
}

CONFIGS = [
    _config("current (stratify_by=null)", batch_group_by="marker", stratify_by=None),
    _config("proposed (stratify_by=experiment)", batch_group_by="marker", stratify_by="experiment"),
    _config(
        "proposed + uniform group_weights",
        batch_group_by="marker",
        stratify_by="experiment",
        group_weights=UNIFORM_WEIGHTS,
    ),
]


def main() -> None:
    from viscy_data.sampler import FlexibleBatchSampler

    print(f"Loading parquet: {CELL_INDEX_PARQUET}")
    df = pd.read_parquet(CELL_INDEX_PARQUET)
    print(f"  rows={len(df):,}  unique markers={df['marker'].nunique()}  unique experiments={df['experiment'].nunique()}")
    print()

    # FlexibleBatchSampler expects valid_anchors with the relevant columns;
    # for sampler-composition QC we don't need _real_ anchor validity, just
    # representative rows. Use the full parquet directly.
    valid_anchors = df

    for cfg in CONFIGS:
        print("=" * 80)
        print(
            f"## {cfg['label']}: batch_group_by={cfg['batch_group_by']!r}, "
            f"stratify_by={cfg['stratify_by']!r}, "
            f"group_weights={'set' if cfg.get('group_weights') else 'None'}"
        )
        print("=" * 80)

        sampler = FlexibleBatchSampler(
            valid_anchors=valid_anchors,
            batch_size=BATCH_SIZE,
            batch_group_by=cfg["batch_group_by"],
            stratify_by=cfg["stratify_by"],
            group_weights=cfg.get("group_weights"),
            leaky=0.0,
            seed=SEED,
        )

        # Collect first N batches.
        marker_counts: Counter = Counter()
        for i, batch_indices in enumerate(sampler):
            if i >= N_BATCHES_TO_SHOW:
                break
            batch_rows = valid_anchors.iloc[batch_indices]
            markers = batch_rows["marker"].unique()
            experiments = batch_rows["experiment"].value_counts()
            primary_marker = batch_rows["marker"].mode().iloc[0]
            marker_counts[primary_marker] += 1

            print(
                f"batch {i:>2}: marker={primary_marker!s:<14} "
                f"unique_markers={len(markers)} "
                f"unique_experiments={len(experiments)}"
            )
            # If marker integrity holds, len(markers) should be 1.
            if len(markers) > 1:
                print(f"   WARN: batch contains MULTIPLE markers: {sorted(markers)}")
            # Show top 3 experiments in the batch
            for exp_name, count in experiments.head(3).items():
                print(f"   {exp_name:<60s} {count:>4d}")
            if len(experiments) > 3:
                print(f"   ... +{len(experiments) - 3} more experiments")

        print()
        print(f"Marker selection across {N_BATCHES_TO_SHOW} batches:")
        for m, n in marker_counts.most_common():
            print(f"  {m:<20s} {n}")
        print()


if __name__ == "__main__":
    main()
