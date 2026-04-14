r"""Summarize the cell/track filtering funnel across all pipeline stages.

Collects counts post-hoc from existing outputs without re-running the pipeline:

- Stage 0: total annotated tracks per dataset (from template zarr attrs)
- Stage 1: tracks after class filter (from template zarr attrs)
- Stage 2: tracks after min_track_timepoints (from template zarr attrs)
- Stage 3: tracks after DTW alignment — all and finite-cost (from alignments.parquet)
- Stage 4: tracks used in evaluation (from evaluation_summary.parquet)

Usage::

    uv run python cell_count_funnel.py --templates-dir 0-build_templates/templates \\
        --alignments 1-align_cells/alignments/alignments.parquet \\
        --evaluation 2-evaluate_dtw/evaluation/evaluation_summary.parquet \\
        --config 0-build_templates/multi_template.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _load_template_attrs(templates_dir: Path) -> dict[str, dict]:
    """Load attrs from all template zarr stores.

    Returns
    -------
    dict[str, dict]
        {template_name: attrs_dict}
    """
    result = {}
    for zarr_path in sorted(templates_dir.glob("template_*.zarr")):
        store = zarr.open(str(zarr_path), mode="r")
        attrs = dict(store.attrs)
        name = attrs.get("template_name", zarr_path.stem.removeprefix("template_"))
        result[name] = attrs
    return result


def main() -> None:
    """Print and save the cell/track filtering funnel."""
    parser = argparse.ArgumentParser(description="Summarize filtering funnel across pipeline stages")
    parser.add_argument("--config", required=True, help="Path to YAML config (multi_template.yaml)")
    parser.add_argument(
        "--templates-dir",
        default=None,
        help="Path to templates directory (default: relative to config)",
    )
    parser.add_argument(
        "--alignments",
        default=None,
        help="Path to alignments.parquet (default: relative to config)",
    )
    parser.add_argument(
        "--evaluation",
        default=None,
        help="Path to evaluation_summary.parquet (default: relative to config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: funnel_summary.csv next to config)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    pseudotime_dir = config_path.parent.parent  # scripts/pseudotime/

    templates_dir = (
        Path(args.templates_dir) if args.templates_dir else pseudotime_dir / "0-build_templates" / "templates"
    )
    alignments_path = (
        Path(args.alignments)
        if args.alignments
        else pseudotime_dir / "1-align_cells" / "alignments" / "alignments.parquet"
    )
    evaluation_path = (
        Path(args.evaluation)
        if args.evaluation
        else pseudotime_dir / "2-evaluate_dtw" / "evaluation" / "evaluation_summary.parquet"
    )
    output_path = Path(args.output) if args.output else config_path.parent / "funnel_summary.csv"

    # --- Stage 0-2: per-template filter funnel (from zarr attrs) ---
    template_attrs = _load_template_attrs(templates_dir)
    stage1_rows = []
    for template_name, attrs in template_attrs.items():
        n_input = attrs.get("n_input_tracks", np.nan)
        per_dataset = attrs.get("track_counts_per_dataset", {})
        if per_dataset:
            for dataset_id, counts in per_dataset.items():
                stage1_rows.append(
                    {
                        "template": template_name,
                        "dataset_id": dataset_id,
                        "n_annotated": counts.get("n_annotated", np.nan),
                        "n_after_class_filter": counts.get("n_after_class_filter", np.nan),
                        "n_after_min_timepoints": counts.get("n_after_min_timepoints", np.nan),
                        "n_into_dba": n_input,
                    }
                )
                _logger.info(
                    f"Stage 1 | template={template_name} dataset={dataset_id}: "
                    f"{counts.get('n_annotated')} annotated -> "
                    f"{counts.get('n_after_class_filter')} after class filter -> "
                    f"{counts.get('n_after_min_timepoints')} after min_timepoints"
                )
        else:
            # Old zarr without per-dataset breakdown — only total available
            stage1_rows.append(
                {
                    "template": template_name,
                    "dataset_id": None,
                    "n_annotated": np.nan,
                    "n_after_class_filter": np.nan,
                    "n_after_min_timepoints": np.nan,
                    "n_into_dba": n_input,
                }
            )
            _logger.info(f"Stage 1 | template={template_name}: {n_input} tracks into DBA (no per-dataset breakdown)")
    stage1 = pd.DataFrame(stage1_rows)

    # --- Stage 3 & 4: tracks from alignments.parquet ---
    if not alignments_path.exists():
        _logger.warning(f"alignments.parquet not found at {alignments_path}, skipping stages 3-4")
        stage2 = pd.DataFrame()
    else:
        alignments = pd.read_parquet(alignments_path)

        # All aligned tracks (any DTW cost)
        all_tracks = (
            alignments.groupby("dataset_id")[["fov_name", "track_id"]]
            .apply(lambda g: g.drop_duplicates().shape[0])
            .reset_index()
            .rename(columns={0: "n_tracks_aligned_all"})
        )
        all_cells = alignments.groupby("dataset_id").size().reset_index(name="n_cells_aligned_all")

        # Finite-cost tracks only
        finite = alignments[np.isfinite(alignments["dtw_cost"])]
        finite_tracks = (
            finite.groupby("dataset_id")[["fov_name", "track_id"]]
            .apply(lambda g: g.drop_duplicates().shape[0])
            .reset_index()
            .rename(columns={0: "n_tracks_finite_cost"})
        )
        finite_cells = finite.groupby("dataset_id").size().reset_index(name="n_cells_finite_cost")

        stage2 = (
            all_tracks.merge(all_cells, on="dataset_id")
            .merge(finite_tracks, on="dataset_id")
            .merge(finite_cells, on="dataset_id")
        )
        for _, row in stage2.iterrows():
            _logger.info(
                f"Stage 2-3 | {row['dataset_id']}: "
                f"{row['n_tracks_aligned_all']} aligned tracks "
                f"({row['n_tracks_finite_cost']} finite cost)"
            )

    # --- Stage 5: tracks used in evaluation ---
    if not evaluation_path.exists():
        _logger.warning(f"evaluation_summary.parquet not found at {evaluation_path}, skipping stage 5")
        stage3 = pd.DataFrame()
    else:
        eval_df = pd.read_parquet(evaluation_path)
        stage3 = eval_df[["dataset_id", "n_tracks", "n_cells"]].rename(
            columns={"n_tracks": "n_tracks_evaluated", "n_cells": "n_cells_evaluated"}
        )
        for _, row in stage3.iterrows():
            _logger.info(
                f"Stage 4 | {row['dataset_id']}: "
                f"{row['n_tracks_evaluated']} evaluated tracks, {row['n_cells_evaluated']} cells"
            )

    # --- Print funnel summary ---
    print("\n## Filtering Funnel Summary\n")

    if len(stage1) > 0:
        funnel = stage1.copy()
        if len(stage2) > 0:
            funnel = funnel.merge(stage2, on="dataset_id", how="left")
        if len(stage3) > 0:
            funnel = funnel.merge(stage3, on="dataset_id", how="left")
        print(funnel.to_markdown(index=False))
        funnel.to_csv(output_path, index=False)
        _logger.info(f"Saved funnel summary to {output_path}")


if __name__ == "__main__":
    main()
