"""Compare empty annotations to cell fluorescence within their own frame.

This is an exploratory annotation audit, never a classifier evaluation with
unmarked observations treated as known positives. Run on a compute node after
the phenotyping aggregate tables have been generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def render_examples(attempt: Path, selected: pd.DataFrame, target: Path) -> None:
    """Show low and high ranked annotated cells with cell boundaries and raw units."""
    import matplotlib.pyplot as plt
    import zarr
    from skimage.segmentation import find_boundaries

    from dynaclr.phenotyping.io import read_position

    measured = selected[selected.annotation_match_status == "matched"].sort_values(
        "mean_intensity_within_frame_percentile"
    )
    keys = ["dataset", "fov", "track_id"]
    examples = pd.concat(
        [
            measured.drop_duplicates(keys).head(6),
            measured.iloc[::-1].drop_duplicates(keys).head(6),
        ]
    ).drop_duplicates(keys + ["t"])
    if examples.empty:
        return
    config = json.loads((attempt / "manifest.json").read_text())["scientific"]["config"]
    ncols = 4
    nrows = (len(examples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for axis in axes:
        axis.axis("off")
    for axis, row in zip(axes, examples.itertuples()):
        position = read_position(Path(config["source"]) / row.fov)
        status = json.loads((attempt / "frames" / row.fov / f"t{row.t:06d}" / "status.json").read_text())
        half = 256
        y = max(0, min(int(row.y) - half, position.shape[-2] - 2 * half))
        x = max(0, min(int(row.x) - half, position.shape[-1] - 2 * half))
        ys, xs = slice(y, y + 2 * half), slice(x, x + 2 * half)
        pixels = np.asarray(
            position.open()[
                row.t,
                position.channels.index(row.channel_name),
                slice(status["z_start"], status["z_stop"]),
                ys,
                xs,
            ]
        ).max(axis=0)
        labels = np.asarray(
            zarr.open_group(str(attempt / "segmentations.ome.zarr"), mode="r")[row.fov]["0"][row.t, 0, 0, ys, xs]
        )
        # Fixed scale across this montage; no per-crop stretch that brightens weak cells.
        high = max(float(measured.p90_intensity.quantile(0.95)) * 2, 1.0)
        rgb = np.repeat(np.clip(pixels / high, 0, 1)[..., None], 3, axis=-1)
        rgb[find_boundaries(labels == row.cell_id)] = [1, 0, 1]
        axis.imshow(rgb)
        axis.plot(row.x - x, row.y - y, "o", mfc="none", mec="yellow", ms=10)
        axis.set_title(
            f"{row.fov} track {row.track_id}, t{row.t}\n"
            f"mean={row.mean_intensity:.1f}; frame percentile={row.mean_intensity_within_frame_percentile:.1f}",
            fontsize=8,
        )
    fig.suptitle("Annotated empty cells: magenta cell boundary; yellow tracked nucleus; common raw intensity scale")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2)
    fig.savefig(target / "annotated_examples.png", dpi=160)
    plt.close(fig)
    examples.to_csv(target / "montage_examples.csv", index=False)


def review(attempt: Path, marker: str, review_name: str = "annotation_review") -> dict:
    """Write within-frame annotation ranks without deriving exclusion labels."""
    from dynaclr.phenotyping.io import assert_compute_node

    assert_compute_node()
    script_source = Path(__file__).read_bytes()
    if Path(review_name).name != review_name:
        raise ValueError("Review name must be a single directory name")
    observations = pd.read_parquet(attempt / "observations.parquet")
    annotations = pd.read_parquet(attempt / "annotation_comparison.parquet")
    manifest = json.loads((attempt / "manifest.json").read_text())
    snapshot = pd.read_parquet(attempt / "provenance" / "annotation_obs.parquet")
    expected_annotations = snapshot[
        snapshot.annotated_empty.fillna(False) & (snapshot.dataset == manifest["scientific"]["config"]["dataset"])
    ]
    selected = annotations[annotations.annotated_empty.fillna(False) & (annotations.marker == marker)].copy()
    population = observations[
        observations.valid_signal.fillna(False) & (observations.marker == marker)
    ].drop_duplicates(["dataset", "fov", "t", "cell_id"])
    groups = {key: group for key, group in population.groupby(["dataset", "fov", "t"])}
    for metric in ("mean_intensity", "median_intensity", "p90_intensity"):
        ranks = []
        counts = []
        for row in selected.itertuples():
            group = groups.get((row.dataset, row.fov, row.t))
            values = np.sort(group[metric].dropna().to_numpy()) if group is not None else np.array([])
            value = getattr(row, metric)
            valid = row.annotation_match_status == "matched" and np.isfinite(value) and len(values)
            rank = (
                50
                * (np.searchsorted(values, value, side="left") + np.searchsorted(values, value, side="right"))
                / len(values)
                if valid
                else np.nan
            )
            ranks.append(rank)
            counts.append(len(values))
        selected[f"{metric}_within_frame_percentile"] = ranks
        selected["reference_cells_in_frame"] = counts
    selected["interpretation"] = "exploratory; unmarked cells unreviewed; shared cells counted once in reference"
    target = attempt / "qc" / review_name
    target.mkdir(parents=True, exist_ok=False)
    selected.to_parquet(target / "annotated_empty_within_frame.parquet", index=False)
    # Highest ranks are disagreements worth reviewing, not false-positive labels.
    selected.sort_values("mean_intensity_within_frame_percentile", ascending=False).to_csv(
        target / "review_high_signal_annotations.csv", index=False
    )
    summary = []
    for (dataset, confidence), expected in expected_annotations.groupby(["dataset", "annotation_confidence"]):
        group = selected[(selected.dataset == dataset) & (selected.annotation_confidence == confidence)]
        ranks = group.mean_intensity_within_frame_percentile.dropna()
        summary.append(
            {
                "dataset": dataset,
                "confidence": confidence,
                "marker": marker,
                "annotated_empty_rows": len(expected),
                "processed_annotation_rows": len(group),
                "measured_rows": len(ranks),
                "measurement_coverage": len(ranks) / len(expected) if len(expected) else None,
                "median_within_frame_percentile": float(ranks.median()) if len(ranks) else None,
                "fraction_in_lowest_quarter_of_frame": float((ranks <= 25).mean()) if len(ranks) else None,
                "fraction_above_frame_median": float((ranks > 50).mean()) if len(ranks) else None,
                "reference_status": "unreviewed population; not confirmed signal-positive controls",
            }
        )
    report = {
        "groups": summary,
        "active_threshold": None,
        "note": (
            "Ranks control for FOV and time, but cannot establish annotation accuracy "
            "without independently reviewed masks and signal-positive controls."
        ),
        "script_sha256": hashlib.sha256(script_source).hexdigest(),
    }
    cells = pd.read_parquet(attempt / "cells.parquet")
    interior = cells[(cells.nuclei_in_cell > 0) & ~cells.touches_border]
    report["cell_size_population"] = {
        "all_segmented_cells": len(cells),
        "nucleus_supported_nonborder_cells": len(interior),
        "description": "Each cell/frame counted once; border-truncated masks excluded from primary size estimates",
    }
    report["interior_cell_size"] = {}
    for column in ("area_um2", "bbox_width_px", "bbox_height_px"):
        if len(interior):
            report["interior_cell_size"][column] = {
                "mean": float(interior[column].mean()),
                **{f"q{int(q * 100)}": float(interior[column].quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)},
            }
    anchors = observations[
        (observations.assignment_status == "matched") & ~observations.touches_border.fillna(True)
    ].drop_duplicates(["dataset", "fov", "t", "track_id"])
    required = anchors.nuclear_centered_patch_side_px.dropna()
    report["nuclear_centered_native_patch_coverage"] = {
        str(side): float((required <= side).mean()) if len(required) else None
        for side in (128, 160, 192, 224, 256, 384, 512, 768)
    }
    (target / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (target / "review_cell_signal.py").write_bytes(script_source)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for (dataset, confidence), group in selected.groupby(["dataset", "annotation_confidence"]):
        ranks = group.mean_intensity_within_frame_percentile.dropna()
        if len(ranks):
            ax.hist(
                ranks, bins=np.linspace(0, 100, 21), alpha=0.5, label=f"{dataset[:10]} {confidence}, n={len(ranks)}"
            )
    ax.set(
        xlabel="Cytoplasmic mean intensity percentile within the same frame",
        ylabel="Annotated empty observations",
        title=f"{marker}: independent signal audit",
    )
    if ax.patches:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target / "within_frame_ranks.png", dpi=170)
    plt.close(fig)
    render_examples(attempt, selected, target)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--marker", default="TOMM20")
    parser.add_argument("--review-name", default="annotation_review")
    args = parser.parse_args()
    print(json.dumps(review(args.attempt, args.marker, args.review_name), indent=2))
