"""Orchestrated linear classifiers evaluation from a single embeddings zarr.

Reads the combined embeddings.zarr produced by the predict step, filters by
experiment and marker, joins per-experiment annotation CSVs, and trains one
logistic regression classifier per (task, marker_filter) combination.

Outputs a metrics_summary.csv and a summary PDF to the output directory.
No W&B logging. For standalone training with W&B use ``dynaclr train-linear-classifier``.

Usage
-----
dynaclr run-linear-classifiers -c linear_classifiers.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from viscy_utils.cli_utils import format_markdown_table, load_config
from viscy_utils.evaluation.annotation import load_annotation_anndata
from viscy_utils.evaluation.linear_classifier import train_linear_classifier

matplotlib.use("Agg")

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
    if embeddings_path.is_dir() and not str(embeddings_path).endswith(".zarr"):
        zarr_paths = sorted(embeddings_path.glob("*.zarr"))
        if not zarr_paths:
            raise FileNotFoundError(f"No .zarr files found in {embeddings_path}")
        parts = [ad.read_zarr(p) for p in zarr_paths]
        adata = ad.concat(parts, join="outer")
        adata.obs_names_make_unique()
        click.echo(f"  Loaded {len(zarr_paths)} per-experiment zarrs")
    else:
        adata = ad.read_zarr(embeddings_path)
    click.echo(f"  {adata.n_obs} cells, {adata.n_vars} features")

    missing = [col for col in ["experiment", "marker"] if col not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"embeddings.zarr obs is missing columns: {missing}. "
            "Re-run the predict step with the updated pipeline to include metadata."
        )

    all_metrics: list[dict] = []
    all_val_outputs: list[dict[str, Any]] = []

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

            _, metrics, val_outputs = train_linear_classifier(
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
            all_val_outputs.append({"task": task, "marker_filter": marker_filter, **val_outputs})

    if not all_metrics:
        click.echo("\nNo classifiers trained — check annotations and marker filters.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics_summary.csv"
    results_df.to_csv(summary_path, index=False)
    click.echo(f"\nMetrics summary written to {summary_path}")

    _print_summary(results_df)
    _save_summary_plots(results_df, all_val_outputs, output_dir)
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


def _save_summary_plots(
    results_df: pd.DataFrame,
    all_val_outputs: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save a PDF with bar charts and ROC curves for quick visual assessment.

    Parameters
    ----------
    results_df : pd.DataFrame
        Metrics summary (one row per task/marker_filter).
    all_val_outputs : list[dict]
        Raw validation outputs per classifier run. Each entry has keys
        ``task``, ``marker_filter``, ``y_val``, ``y_val_proba``, ``classes``.
    output_dir : Path
        Directory to write ``metrics_summary.pdf``.
    """

    pdf_path = output_dir / "metrics_summary.pdf"

    with PdfPages(pdf_path) as pdf:
        _plot_metrics_bar(pdf, results_df)
        for vo in all_val_outputs:
            if vo["y_val"] is not None and vo["y_val_proba"] is not None:
                _plot_roc_curves(pdf, vo["task"], vo["marker_filter"], vo["y_val"], vo["y_val_proba"], vo["classes"])

    click.echo(f"Summary plots written to {pdf_path}")


def _plot_metrics_bar(pdf: PdfPages, results_df: pd.DataFrame) -> None:
    """Bar chart of AUROC, accuracy, and weighted F1 across all classifiers."""
    metric_cols = ["val_auroc", "val_accuracy", "val_weighted_f1"]
    present = [c for c in metric_cols if c in results_df.columns]
    if not present:
        return

    labels = []
    for _, row in results_df.iterrows():
        label = str(row["task"])
        if pd.notna(row.get("marker_filter")):
            label += f"\n({row['marker_filter']})"
        labels.append(label)

    x = np.arange(len(labels))
    n_metrics = len(present)
    width = 0.8 / n_metrics

    metric_display = {"val_auroc": "AUROC", "val_accuracy": "Accuracy", "val_weighted_f1": "Weighted F1"}
    colors = ["#0072B2", "#E69F00", "#009E73"]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    for i, col in enumerate(present):
        vals = results_df[col].fillna(0).values
        ax.bar(x + i * width, vals, width, label=metric_display.get(col, col), color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="Random (0.5)")
    ax.set_ylabel("Score")
    ax.set_title("Linear Classifier Performance Summary")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curves(
    pdf: PdfPages,
    task: str,
    marker_filter: str | None,
    y_val: np.ndarray,
    y_val_proba: np.ndarray,
    classes: list[str],
) -> None:
    """One-vs-rest ROC curves for a single classifier."""
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    title = task + (f" (marker={marker_filter})" if marker_filter else "")

    # Colorblind-friendly palette (Wong 2011)
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(f"ROC Curves: {title}", fontsize=11)

    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_val, y_val_proba[:, 1], pos_label=classes[1])
        auroc = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, color=palette[0], linewidth=2, label=f"{classes[1]} (AUROC={auroc:.3f})")
    else:
        y_bin = label_binarize(y_val, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_val_proba[:, i])
            auroc = float(np.trapezoid(tpr, fpr))
            ax.plot(fpr, tpr, color=palette[i % len(palette)], linewidth=1.5, label=f"{cls} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


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
