"""Orchestrated linear classifiers evaluation from a single embeddings zarr.

Reads the combined embeddings.zarr produced by the predict step, filters by
experiment and marker, joins per-experiment annotation CSVs, and trains one
logistic regression classifier per (task, marker_filter) combination.

Outputs a metrics_summary.csv to the output directory. No W&B logging.
For standalone training with W&B use ``dynaclr train-linear-classifier``.

Usage
-----
dynaclr run-linear-classifiers -c linear_classifiers.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
import pandas as pd

from viscy_utils.cli_utils import format_markdown_table, load_config
from viscy_utils.evaluation.annotation import load_annotation_anndata
from viscy_utils.evaluation.linear_classifier import train_linear_classifier

if TYPE_CHECKING:
    import anndata as ad

    from dynaclr.evaluation.evaluate_config import LinearClassifiersStepConfig


def run_linear_classifiers(
    embeddings_path: Path,
    config: LinearClassifiersStepConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Train linear classifiers for each (task, marker_filter) combination.

    Parameters
    ----------
    embeddings_path : Path
        Path to the combined embeddings zarr (AnnData format). Must have
        experiment and marker columns in obs (added by the predict step).
    config : LinearClassifiersStepConfig
        Configuration with annotations list and task specs.
    output_dir : Path
        Directory to write metrics_summary.csv.

    Returns
    -------
    pd.DataFrame
        One row per (task, marker_filter) with accuracy, F1, AUROC, etc.
    """
    import anndata as ad

    click.echo(f"Loading embeddings from {embeddings_path}")
    adata = ad.read_zarr(embeddings_path)
    click.echo(f"  {adata.n_obs} cells, {adata.n_vars} features")

    missing = [col for col in ["experiment", "marker"] if col not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"embeddings.zarr obs is missing columns: {missing}. "
            "Re-run the predict step with the updated pipeline to include metadata."
        )

    all_metrics: list[dict] = []

    for task_spec in config.tasks:
        task = task_spec.task
        # Expand marker_filters: None → [None] (one run, all markers); list → one run per marker
        runs: list[str | None] = task_spec.marker_filters if task_spec.marker_filters is not None else [None]

        for marker_filter in runs:
            label = f"{task}" + (f" (marker={marker_filter})" if marker_filter else " (all markers)")
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Task: {label}")
            click.echo("=" * 60)

            # Filter by marker if specified
            if marker_filter is not None:
                adata_task = adata[adata.obs["marker"] == marker_filter]
                click.echo(f"  Filtered to {adata_task.n_obs} cells with marker={marker_filter}")
            else:
                adata_task = adata

            if adata_task.n_obs == 0:
                click.echo(f"  No cells found for marker_filter={marker_filter!r}, skipping.")
                continue

            # Join annotation CSVs per experiment and collect annotated subsets
            annotated_parts: list[ad.AnnData] = []
            for ann_src in config.annotations:
                exp_mask = adata_task.obs["experiment"] == ann_src.experiment
                n_exp = int(exp_mask.sum())
                if n_exp == 0:
                    click.echo(f"  Experiment {ann_src.experiment!r}: no matching cells, skipping.")
                    continue

                adata_exp = adata_task[exp_mask].copy()
                ann_path = Path(ann_src.path)
                if not ann_path.exists():
                    raise FileNotFoundError(f"Annotation CSV not found: {ann_src.path}")

                try:
                    adata_exp = load_annotation_anndata(adata_exp, str(ann_path), task)
                except KeyError:
                    click.echo(f"  Experiment {ann_src.experiment!r}: task {task!r} not in {ann_path.name}, skipping.")
                    continue

                valid_mask = adata_exp.obs[task].notna() & (adata_exp.obs[task] != "unknown")
                n_valid = int(valid_mask.sum())
                if n_valid == 0:
                    click.echo(f"  Experiment {ann_src.experiment!r}: no valid labels for {task!r}, skipping.")
                    continue

                annotated_parts.append(adata_exp[valid_mask])
                click.echo(f"  Experiment {ann_src.experiment!r}: {n_valid}/{n_exp} labeled cells")

            if not annotated_parts:
                click.echo(f"  No annotated data found for task {task!r}, skipping.")
                continue

            combined = annotated_parts[0] if len(annotated_parts) == 1 else ad.concat(annotated_parts, join="outer")
            class_dist = combined.obs[task].value_counts().to_dict()
            click.echo(f"  Total: {combined.n_obs} cells, class distribution: {class_dist}")

            classifier_params = {
                "max_iter": config.max_iter,
                "class_weight": config.class_weight,
                "solver": config.solver,
                "random_state": config.random_seed,
            }

            _, metrics = train_linear_classifier(
                adata=combined,
                task=task,
                use_scaling=config.use_scaling,
                use_pca=config.use_pca,
                n_pca_components=config.n_pca_components,
                classifier_params=classifier_params,
                split_train_data=config.split_train_data,
                random_seed=config.random_seed,
            )

            row = {
                "task": task,
                "marker_filter": marker_filter,
                "n_samples": combined.n_obs,
                **metrics,
            }
            all_metrics.append(row)

    if not all_metrics:
        click.echo("\nNo classifiers trained — check annotations and marker filters.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics_summary.csv"
    results_df.to_csv(summary_path, index=False)
    click.echo(f"\nMetrics summary written to {summary_path}")

    _print_summary(results_df)
    return results_df


def _print_summary(results_df: pd.DataFrame) -> None:
    """Print a markdown summary table of key metrics."""
    click.echo("\n## Linear Classifier Results\n")

    summary_cols = ["task", "marker_filter", "n_samples", "val_accuracy", "val_weighted_f1", "val_auroc"]
    display = results_df[[c for c in summary_cols if c in results_df.columns]].copy()

    float_cols = [c for c in display.columns if c not in ("task", "marker_filter")]
    for col in float_cols:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "N/A")

    rows = display.to_dict(orient="records")
    click.echo(format_markdown_table(rows, headers=list(display.columns)))


class _RunLinearClassifiersConfig:
    """Config container for the run-linear-classifiers CLI."""

    def __init__(self, raw: dict):
        from dynaclr.evaluation.evaluate_config import LinearClassifiersStepConfig

        self.embeddings_path = Path(raw["embeddings_path"])
        self.output_dir = Path(raw["output_dir"])
        self.lc_config = LinearClassifiersStepConfig(
            **{k: v for k, v in raw.items() if k not in ("embeddings_path", "output_dir")}
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path) -> None:
    """Run linear classifiers on a combined embeddings zarr from the evaluation orchestrator."""
    raw = load_config(config)
    cfg = _RunLinearClassifiersConfig(raw)
    run_linear_classifiers(cfg.embeddings_path, cfg.lc_config, cfg.output_dir)


if __name__ == "__main__":
    main()
