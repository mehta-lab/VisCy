"""Lineage overview: count tracks by division and infection state.

Loads annotated datasets from the multi_template config and reports
track counts per combination of division state and infection class.
Also reports whether division occurs before or after infection onset
(first infected timepoint) for dividing+transitioning tracks.

Outputs one CSV per dataset and a combined summary CSV.

Usage::

    uv run python \
        applications/dynaclr/scripts/pseudotime/0-build_templates/lineage_overview.py \
        --config applications/dynaclr/configs/pseudotime/multi_template.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from build_templates import _classify_tracks, _find_zarr, _load_annotations_with_tracking

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _summarize_dataset(ds: dict, emb_pattern: str) -> pd.DataFrame:
    """Load one dataset and return a track-level summary DataFrame."""
    dataset_id = ds["dataset_id"]
    zarr_path = _find_zarr(ds["pred_dir"], emb_pattern)
    adata = ad.read_zarr(zarr_path)
    annotations = _load_annotations_with_tracking(ds["annotations_path"], adata)

    # Scope to this dataset's FOV pattern (same scoping align_tracks applies)
    fov_pattern = ds.get("fov_pattern")
    if fov_pattern is not None:
        annotations = annotations[annotations["fov_name"].astype(str).str.contains(fov_pattern, regex=True)]

    classified = _classify_tracks(annotations)

    # One row per track — division_timing already computed by _classify_tracks
    # n_annotated_timepoints: only timepoints with a non-null infection_state label,
    # matching what align_tracks actually uses for the min_track_timepoints filter.
    classified["_is_annotated"] = classified["infection_state"].notna()
    track_df = (
        classified.groupby(["fov_name", "track_id"])
        .agg(
            divides=("divides", "first"),
            infection_class=("infection_class", "first"),
            division_timing=("division_timing", "first"),
            n_timepoints=("t", "nunique"),
            n_annotated_timepoints=("_is_annotated", "sum"),
        )
        .reset_index()
    )

    track_df.insert(0, "dataset_id", dataset_id)
    return track_df


def _plot_survival_curve(
    combined: pd.DataFrame,
    frame_intervals: dict[str, float],
    min_track_minutes_values: list[int],
    output_dir: Path,
) -> None:
    """Plot track survival curve around the config min_track_minutes thresholds.

    Parameters
    ----------
    combined : pd.DataFrame
        Track-level DataFrame with n_timepoints, infection_class, divides, dataset_id.
    frame_intervals : dict[str, float]
        dataset_id -> frame_interval_minutes.
    min_track_minutes_values : list[int]
        Threshold values from the config templates (used to set x-axis range).
    output_dir : Path
        Where to save the PNG.
    """
    # Use annotated timepoints only — matches what align_tracks filters on
    combined = combined.copy()
    combined["track_minutes"] = combined.apply(
        lambda r: r["n_annotated_timepoints"] * frame_intervals.get(r["dataset_id"], 1.0), axis=1
    )

    ref = min_track_minutes_values[0] if min_track_minutes_values else 300
    x_min = ref * 0.2
    x_max = ref * 2.0
    cutoffs = np.linspace(x_min, x_max, 120)

    fig, ax = plt.subplots(figsize=(9, 5))

    # transitioning non-dividing — the clean template case
    grp = combined[(combined["infection_class"] == "transitioning") & (~combined["divides"])]
    counts = [(grp["track_minutes"] >= c).sum() for c in cutoffs]
    ax.plot(cutoffs, counts, label="transitioning / non-dividing")

    # transitioning + divides, split by when division occurs
    for timing, label in [
        ("before", "transitioning / divides before infection"),
        ("after", "transitioning / divides after infection"),
    ]:
        grp = combined[
            (combined["infection_class"] == "transitioning")
            & combined["divides"]
            & (combined["division_timing"] == timing)
        ]
        counts = [(grp["track_minutes"] >= c).sum() for c in cutoffs]
        ax.plot(cutoffs, counts, linestyle="--", label=label)

    # uninfected_only non-dividing — pure cell cycle reference
    grp = combined[(combined["infection_class"] == "uninfected_only") & (~combined["divides"])]
    counts = [(grp["track_minutes"] >= c).sum() for c in cutoffs]
    ax.plot(cutoffs, counts, linestyle=":", label="uninfected_only / non-dividing")

    for v in min_track_minutes_values:
        ax.axvline(v, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(v + 2, ax.get_ylim()[1] * 0.95, f"{v} min", fontsize=8, va="top")

    ax.set_xlabel("Min track length (minutes)")
    ax.set_ylabel("Number of tracks surviving")
    ax.set_title("Track survival by min length cutoff")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    path = output_dir / "track_survival_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _logger.info(f"Saved survival curve: {path}")


def main() -> None:
    """Run lineage overview across all datasets in config."""
    parser = argparse.ArgumentParser(description="Lineage overview")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "lineage_overview"
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_pattern = config["embeddings"]["sensor"]
    frame_intervals = {ds["dataset_id"]: ds["frame_interval_minutes"] for ds in config["datasets"]}
    min_track_minutes_values = sorted(
        {
            tmpl_cfg["min_track_minutes"]
            for tmpl_cfg in config.get("templates", {}).values()
            if "min_track_minutes" in tmpl_cfg
        }
    )

    all_summaries = []

    for ds in config["datasets"]:
        dataset_id = ds["dataset_id"]
        _logger.info(f"Processing {dataset_id}")
        track_df = _summarize_dataset(ds, emb_pattern)

        # Per-dataset CSV
        per_ds_path = output_dir / f"{dataset_id}_lineages.csv"
        track_df.to_csv(per_ds_path, index=False)
        _logger.info(f"  Saved {per_ds_path}")

        # Print summary table (exclude unknown and infected_only)
        counts = (
            track_df[~track_df["infection_class"].isin(["unknown", "infected_only"])]
            .groupby(["infection_class", "divides", "division_timing"])
            .size()
            .reset_index(name="n_tracks")
            .sort_values(["infection_class", "divides", "division_timing"])
        )
        _logger.info(f"\n## {dataset_id}\n\n{counts.to_string(index=False)}\n")

        all_summaries.append(track_df)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined = combined[~combined["infection_class"].isin(["unknown", "infected_only"])]
    combined_path = output_dir / "combined_lineages.csv"
    combined.to_csv(combined_path, index=False)
    _logger.info(f"Combined saved: {combined_path}")

    # Print combined summary
    combined_counts = (
        combined.groupby(["infection_class", "divides", "division_timing"])
        .size()
        .reset_index(name="n_tracks")
        .sort_values(["infection_class", "divides", "division_timing"])
    )
    print(f"\n## Combined lineage overview\n\n{combined_counts.to_string(index=False)}\n")

    _plot_survival_curve(combined, frame_intervals, min_track_minutes_values, output_dir)


if __name__ == "__main__":
    main()
