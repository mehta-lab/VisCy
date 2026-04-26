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

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

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
    # val_outputs_by_task: task → list of per-marker dicts for plotting
    val_outputs_by_task: dict[str, list[dict[str, Any]]] = {}
    # Saved pipelines for append-predictions step. When publish_dir is set,
    # we stage here and atomically promote to a versioned registry dir at
    # the end of training. Otherwise legacy behavior: write in place under
    # output_dir/pipelines/.
    pipelines_dir = output_dir / "pipelines"
    pipelines_dir.mkdir(parents=True, exist_ok=True)
    pipeline_manifest: list[dict] = []
    # Collect trained (task, marker, pipeline) tuples for publish_dir promotion.
    trained_pipelines: list[tuple[str, str, Any]] = []

    for task_spec in config.tasks:
        task = task_spec.task
        # Expand marker_filters: None → all unique markers; list → one run per specified marker
        runs: list[str] = (
            task_spec.marker_filters
            if task_spec.marker_filters is not None
            else sorted(adata.obs["marker"].unique().tolist())
        )
        val_outputs_by_task[task] = []

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

            try:
                pipeline, metrics, val_outputs = train_linear_classifier(
                    adata=combined,
                    task=task,
                    use_scaling=config.use_scaling,
                    use_pca=config.use_pca,
                    n_pca_components=config.n_pca_components,
                    classifier_params=classifier_params,
                    split_train_data=config.split_train_data,
                    random_seed=config.random_seed,
                )
            except ValueError as exc:
                click.echo(f"  Skipping {label}: {exc}")
                continue

            # Save pipeline for append-predictions step. Always write to the
            # local staging dir; promotion to publish_dir (if configured) happens
            # atomically after all classifiers finish training.
            pipeline_filename = f"{task}_{marker_filter}.joblib"
            joblib.dump(pipeline, pipelines_dir / pipeline_filename)
            pipeline_manifest.append({"task": task, "marker_filter": marker_filter, "path": pipeline_filename})
            trained_pipelines.append((task, marker_filter, pipeline))
            click.echo(f"  Pipeline saved: {pipeline_filename}")

            # Replay the same split to recover val obs (hours_post_perturbation)
            y_full = combined.obs[task].to_numpy(dtype=object)
            val_hours: np.ndarray | None = None
            if config.split_train_data < 1.0 and "hours_post_perturbation" in combined.obs.columns:
                try:
                    idx = np.arange(len(combined))
                    _, idx_val = train_test_split(
                        idx,
                        train_size=config.split_train_data,
                        random_state=config.random_seed,
                        stratify=y_full,
                        shuffle=True,
                    )
                    val_hours = combined.obs["hours_post_perturbation"].to_numpy()[idx_val]
                except ValueError:
                    click.echo("  Could not replay stratified split for val_hours; F1-over-time plot skipped.")

            row = {
                "task": task,
                "marker_filter": marker_filter,
                "n_samples": combined.n_obs,
                **metrics,
            }
            all_metrics.append(row)
            val_outputs_by_task[task].append(
                {
                    "marker_filter": marker_filter,
                    "val_hours": val_hours,
                    **val_outputs,
                }
            )

    if not all_metrics:
        click.echo("\nNo classifiers trained — check annotations and marker filters.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics_summary.csv"
    results_df.to_csv(summary_path, index=False)
    click.echo(f"\nMetrics summary written to {summary_path}")

    # New-format manifest: dict with trained_at + pipelines list.
    # Model identity (feature_space) and version are carried by the directory
    # structure: {registry_root}/{model_name}/v{N}/. No need to duplicate here.
    manifest_dict = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "pipelines": pipeline_manifest,
    }
    manifest_path = pipelines_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)
    click.echo(f"Pipeline manifest written to {manifest_path}")

    # Promote to central LC registry if publish_dir is configured.
    publish_dir_str = getattr(config, "publish_dir", None)
    if publish_dir_str:
        new_dir = _publish_atomically(
            publish_dir=Path(publish_dir_str),
            trained=trained_pipelines,
            manifest_dict=manifest_dict,
        )
        click.echo(f"Published LC bundle to {new_dir} (latest -> {new_dir.name})")

    _print_summary(results_df)
    for task, task_val_outputs in val_outputs_by_task.items():
        task_df = results_df[results_df["task"] == task]
        _save_task_plots(task, task_df, task_val_outputs, output_dir)
    return results_df


def _publish_atomically(
    publish_dir: Path,
    trained: list[tuple[str, str, Any]],
    manifest_dict: dict,
) -> Path:
    """Atomically publish a new versioned LC bundle under ``publish_dir``.

    Writes pipelines + manifest.json to a staging directory, renames it to
    ``vN/`` (where N is max existing version + 1), then swaps the ``latest``
    symlink to point at the new version. Crash-safe: partial bundles never
    appear as ``vN/`` because the rename is atomic.

    Parameters
    ----------
    publish_dir : Path
        Model registry root (e.g.,
        ``/hpc/projects/.../linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/``).
        Created if it does not exist.
    trained : list of (task, marker_filter, pipeline)
        Fitted pipelines to persist.
    manifest_dict : dict
        Manifest content to write as ``manifest.json`` inside the new
        version directory.

    Returns
    -------
    Path
        Absolute path of the newly published ``vN/`` directory.
    """
    publish_dir.mkdir(parents=True, exist_ok=True)

    # Pick next version number by scanning existing v* dirs.
    existing = sorted(int(p.name[1:]) for p in publish_dir.glob("v*") if p.is_dir() and p.name[1:].isdigit())
    next_v = (max(existing) + 1) if existing else 1
    new_dir = publish_dir / f"v{next_v}"

    # Stage everything in a temp dir under publish_dir (same filesystem for
    # atomic rename). If we crash here, nothing named vN/ appears.
    staging = Path(tempfile.mkdtemp(prefix=f".v{next_v}.stage.", dir=publish_dir))
    for task, marker_filter, pipeline in trained:
        joblib.dump(pipeline, staging / f"{task}_{marker_filter}.joblib")
    with open(staging / "manifest.json", "w") as f:
        json.dump(manifest_dict, f, indent=2)

    # Atomic rename: staging -> vN.
    os.rename(staging, new_dir)

    # Atomic symlink swap: write latest.new, then rename over latest.
    # Relative target ("vN") so the symlink stays valid if the registry
    # root is ever moved.
    latest = publish_dir / "latest"
    latest_new = publish_dir / "latest.new"
    if latest_new.is_symlink() or latest_new.exists():
        latest_new.unlink()
    os.symlink(new_dir.name, latest_new)
    os.replace(latest_new, latest)

    return new_dir


def _print_summary(results_df: pd.DataFrame) -> None:
    """Print a markdown summary table of key metrics."""
    click.echo("\n## Linear Classifier Results\n")

    per_class_f1_cols = sorted(c for c in results_df.columns if c.startswith("val_") and c.endswith("_f1"))
    summary_cols = [
        "task",
        "marker_filter",
        "n_samples",
        "val_accuracy",
        "val_weighted_f1",
        "val_auroc",
    ] + per_class_f1_cols
    display = results_df[[c for c in summary_cols if c in results_df.columns]].copy()

    float_cols = [c for c in display.columns if c not in ("task", "marker_filter")]
    for col in float_cols:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "N/A")

    rows = display.to_dict(orient="records")
    click.echo(format_markdown_table(rows, headers=list(display.columns)))


def _save_task_plots(
    task: str,
    task_df: pd.DataFrame,
    task_val_outputs: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save one PDF per task with bar chart, ROC curves, and F1-over-time plots.

    Parameters
    ----------
    task : str
        Task name (used in filename and titles).
    task_df : pd.DataFrame
        Rows from metrics_summary.csv for this task (one row per marker).
    task_val_outputs : list[dict]
        Per-marker val outputs. Each entry has keys ``marker_filter``,
        ``y_val``, ``y_val_proba``, ``classes``, ``val_hours``.
    output_dir : Path
        Directory to write ``{task}_summary.pdf``.
    """
    pdf_path = output_dir / f"{task}_summary.pdf"

    with PdfPages(pdf_path) as pdf:
        _plot_metrics_bar(pdf, task, task_df)
        for vo in task_val_outputs:
            if vo["y_val"] is None or vo["y_val_proba"] is None:
                continue
            _plot_roc_curves(pdf, task, vo["marker_filter"], vo["y_val"], vo["y_val_proba"], vo["classes"])
            if vo["val_hours"] is not None:
                _plot_f1_over_time(
                    pdf, task, vo["marker_filter"], vo["y_val"], vo["y_val_proba"], vo["classes"], vo["val_hours"]
                )

    click.echo(f"Plots written to {pdf_path}")


def _plot_metrics_bar(pdf: PdfPages, task: str, task_df: pd.DataFrame) -> None:
    """Bar chart of AUROC, accuracy, and weighted F1 per marker for one task."""
    metric_cols = ["val_auroc", "val_accuracy", "val_weighted_f1"]
    present = [c for c in metric_cols if c in task_df.columns]
    if not present:
        return

    labels = task_df["marker_filter"].fillna("all").tolist()
    x = np.arange(len(labels))
    n_metrics = len(present)
    width = 0.8 / n_metrics

    metric_display = {"val_auroc": "AUROC", "val_accuracy": "Accuracy", "val_weighted_f1": "Weighted F1"}
    colors = ["#0072B2", "#E69F00", "#009E73"]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    for i, col in enumerate(present):
        vals = task_df[col].fillna(0).values
        ax.bar(x + i * width, vals, width, label=metric_display.get(col, col), color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="Random (0.5)")
    ax.set_ylabel("Score")
    ax.set_title(f"{task} — classifier performance per marker")
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
    """One-vs-rest ROC curves for a single (task, marker) classifier."""
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    # Colorblind-friendly palette (Wong 2011)
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(f"ROC — {task} ({marker_filter})", fontsize=11)

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


def _plot_f1_over_time(
    pdf: PdfPages,
    task: str,
    marker_filter: str | None,
    y_val: np.ndarray,
    y_val_proba: np.ndarray,
    classes: list[str],
    val_hours: np.ndarray,
) -> None:
    """Per-class F1 at each unique timepoint for a single (task, marker) classifier."""
    from sklearn.metrics import f1_score

    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]

    y_pred = np.array(classes)[np.argmax(y_val_proba, axis=1)]
    timepoints = sorted(np.unique(val_hours[~np.isnan(val_hours)]))

    # (n_timepoints, n_classes)
    f1_per_time = np.full((len(timepoints), len(classes)), np.nan)
    for ti, t in enumerate(timepoints):
        mask = val_hours == t
        if mask.sum() < 2:
            continue
        f1s = f1_score(y_val[mask], y_pred[mask], labels=classes, average=None, zero_division=0)
        f1_per_time[ti] = f1s

    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, cls in enumerate(classes):
        ax.plot(timepoints, f1_per_time[:, ci], marker="o", color=palette[ci % len(palette)], linewidth=2, label=cls)

    ax.set_xlabel("Hours post perturbation")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"F1 over time — {task} ({marker_filter})")
    ax.legend(fontsize=9)
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
