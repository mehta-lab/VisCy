"""Leave-one-dataset-out cross-validation for training dataset impact analysis.

Trains linear classifiers on subsets of the training pool (dropping one dataset
at a time) and evaluates on the held-out test set. Produces raw results,
aggregated summaries, and a PDF report identifying which training datasets
help, hurt, or have uncertain impact.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from viscy.representation.evaluation import load_annotation_anndata
from viscy.representation.evaluation.linear_classifier import (
    load_and_combine_datasets,
    predict_with_classifier,
    train_linear_classifier,
)

from .evaluate_dataset import (
    DatasetEvalConfig,
    _find_channel_zarrs,
    _get_available_tasks,
    _resolve_task_channels,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_n_features(config: DatasetEvalConfig) -> int:
    """Read n_features from the first available training zarr.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration.

    Returns
    -------
    int
        Number of embedding dimensions (columns in X).
    """
    for model_spec in config.models.values():
        for train_ds in model_spec.train_datasets:
            zarrs = _find_channel_zarrs(
                train_ds.embeddings_dir, ["phase", "sensor", "organelle"]
            )
            for zarr_path in zarrs.values():
                adata = ad.read_zarr(zarr_path)
                return adata.X.shape[1]
    raise RuntimeError("Could not detect n_features: no training zarrs found.")


def _check_class_safety(
    datasets_for_combo: list[dict],
    task: str,
    min_class_samples: int,
) -> bool:
    """Check if the remaining dataset subset has enough samples per class.

    Parameters
    ----------
    datasets_for_combo : list[dict]
        Dataset dicts with 'embeddings' and 'annotations' keys.
    task : str
        Classification task column name.
    min_class_samples : int
        Minimum required samples for each class.

    Returns
    -------
    bool
        True if all classes meet the minimum threshold.
    """
    all_labels = []
    for ds in datasets_for_combo:
        ann = pd.read_csv(ds["annotations"])
        if task in ann.columns:
            valid = ann[task].dropna()
            valid = valid[valid != "unknown"]
            all_labels.extend(valid.tolist())

    if not all_labels:
        return False

    class_counts = pd.Series(all_labels).value_counts()
    return bool((class_counts >= min_class_samples).all())


def _get_dataset_name(train_ds) -> str:
    """Extract a human-readable dataset name from a TrainDataset."""
    return train_ds.annotations.parent.name


def _build_datasets_for_combo(
    train_datasets, channel: str, task: str
) -> list[tuple[Any, dict]]:
    """Build (train_ds, dataset_dict) pairs for a given channel and task.

    Returns
    -------
    list[tuple]
        Each element is (train_ds_object, {"embeddings": ..., "annotations": ...}).
    """
    result = []
    for train_ds in train_datasets:
        channel_zarrs = _find_channel_zarrs(train_ds.embeddings_dir, [channel])
        if channel not in channel_zarrs:
            continue
        available_tasks = _get_available_tasks(train_ds.annotations)
        if task not in available_tasks:
            continue
        result.append(
            (
                train_ds,
                {
                    "embeddings": str(channel_zarrs[channel]),
                    "annotations": str(train_ds.annotations),
                },
            )
        )
    return result


def _get_class_counts(datasets_for_combo: list[dict], task: str) -> dict[str, int]:
    """Count per-class samples across datasets."""
    all_labels = []
    for ds in datasets_for_combo:
        ann = pd.read_csv(ds["annotations"])
        if task in ann.columns:
            valid = ann[task].dropna()
            valid = valid[valid != "unknown"]
            all_labels.extend(valid.tolist())
    return dict(pd.Series(all_labels).value_counts())


# ---------------------------------------------------------------------------
# Core CV unit
# ---------------------------------------------------------------------------


def _train_and_evaluate(
    config: DatasetEvalConfig,
    model_label: str,
    task: str,
    channel: str,
    datasets_for_combo: list[dict],
    seed: int,
    excluded_dataset: str | None = None,
) -> dict[str, Any]:
    """Train on a subset and evaluate on test. Returns a flat result dict.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration.
    model_label : str
        Model key in config.models.
    task : str
        Classification task.
    channel : str
        Input channel.
    datasets_for_combo : list[dict]
        Training datasets (already filtered for channel/task).
    seed : int
        Random seed for this run.
    excluded_dataset : str or None
        Name of excluded dataset, or None for baseline.

    Returns
    -------
    dict
        Flat row with model, task, channel, excluded_dataset, seed, metrics.
    """
    model_spec = config.models[model_label]
    row: dict[str, Any] = {
        "model": model_label,
        "task": task,
        "channel": channel,
        "excluded_dataset": excluded_dataset or "baseline",
        "seed": seed,
        "n_train_datasets": len(datasets_for_combo),
    }

    # Per-dataset contribution counts
    for ds in datasets_for_combo:
        ds_name = Path(ds["annotations"]).stem.replace("_annotations", "")
        ann = pd.read_csv(ds["annotations"])
        if task in ann.columns:
            valid = ann[task].dropna()
            valid = valid[valid != "unknown"]
            row[f"n_samples_{ds_name}"] = len(valid)

    # Class counts
    class_counts = _get_class_counts(datasets_for_combo, task)
    for cls, cnt in class_counts.items():
        row[f"train_class_{cls}"] = cnt

    # Identify minority class
    if class_counts:
        minority_class = min(class_counts, key=class_counts.get)
        row["minority_class"] = minority_class
        row["minority_class_count"] = class_counts[minority_class]
    else:
        row["minority_class"] = None
        row["minority_class_count"] = 0

    try:
        combined_adata = load_and_combine_datasets(datasets_for_combo, task)

        classifier_params = {
            "max_iter": config.max_iter,
            "class_weight": config.class_weight,
            "solver": config.solver,
            "random_state": seed,
        }

        pipeline, metrics = train_linear_classifier(
            adata=combined_adata,
            task=task,
            use_scaling=config.use_scaling,
            use_pca=config.use_pca,
            n_pca_components=config.n_pca_components,
            classifier_params=classifier_params,
            split_train_data=config.split_train_data,
            random_seed=seed,
        )

        row.update(metrics)

        # Load test embeddings and predict
        test_channel_zarrs = _find_channel_zarrs(
            model_spec.test_embeddings_dir, [channel]
        )
        if channel not in test_channel_zarrs:
            row["auroc"] = np.nan
            row["error"] = "no test channel zarr"
            return row

        test_adata = ad.read_zarr(test_channel_zarrs[channel])
        test_adata = predict_with_classifier(test_adata, pipeline, task)

        # Join test annotations
        annotated = load_annotation_anndata(
            test_adata, str(config.test_annotations_csv), task
        )

        mask = annotated.obs[task].notna() & (annotated.obs[task] != "unknown")
        eval_subset = annotated[mask]

        if len(eval_subset) == 0:
            row["auroc"] = np.nan
            row["error"] = "no annotated test cells"
            return row

        pred_col = f"predicted_{task}"
        y_true = eval_subset.obs[task].values
        y_pred = eval_subset.obs[pred_col].values

        # AUROC
        proba_key = f"predicted_{task}_proba"
        classes_key = f"predicted_{task}_classes"
        if proba_key in annotated.obsm and classes_key in annotated.uns:
            y_proba = annotated[mask].obsm[proba_key]
            classes = annotated.uns[classes_key]
            n_classes = len(classes)

            if n_classes == 2:
                positive_idx = 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        auroc = roc_auc_score(y_true, y_proba[:, positive_idx])
                    except ValueError:
                        auroc = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        auroc = roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="macro"
                        )
                    except ValueError:
                        auroc = np.nan
            row["auroc"] = auroc
        else:
            row["auroc"] = np.nan

        # Classification report
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        row["test_accuracy"] = report["accuracy"]
        row["test_weighted_f1"] = report["weighted avg"]["f1-score"]
        row["test_weighted_precision"] = report["weighted avg"]["precision"]
        row["test_weighted_recall"] = report["weighted avg"]["recall"]
        row["test_n_samples"] = len(eval_subset)

        # Per-class metrics
        for class_name in sorted(set(y_true) | set(y_pred)):
            if class_name in report:
                row[f"test_{class_name}_f1"] = report[class_name]["f1-score"]
                row[f"test_{class_name}_precision"] = report[class_name]["precision"]
                row[f"test_{class_name}_recall"] = report[class_name]["recall"]

        # Minority class metrics
        if row.get("minority_class") and row["minority_class"] in report:
            mc = row["minority_class"]
            row["minority_f1"] = report[mc]["f1-score"]
            row["minority_recall"] = report[mc]["recall"]
            row["minority_precision"] = report[mc]["precision"]

    except Exception as e:
        row["auroc"] = np.nan
        row["error"] = str(e)
        logger.warning(f"CV fold failed: {excluded_dataset}, seed={seed}: {e}")

    return row


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------


def cross_validate_datasets(
    config: DatasetEvalConfig,
    ranking_metric: str = "auroc",
    n_bootstrap: int = 5,
    min_class_samples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-dataset-out cross-validation.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration (with ``wandb_logging=False`` recommended).
    ranking_metric : str
        Metric to use for impact ranking (default: "auroc").
    n_bootstrap : int
        Number of bootstrap seeds per fold.
    min_class_samples : int or None
        Minimum samples per class to consider a fold safe. When ``None``
        (default), auto-detected from the embedding dimensionality
        (``n_features``) so that each class has at least as many samples
        as there are features.

    Returns
    -------
    pd.DataFrame
        Raw results (one row per fold x seed).
    pd.DataFrame
        Aggregated summary with impact labels.
    """
    tc = _resolve_task_channels(config)
    if not tc:
        raise ValueError("No valid tasks found in test annotations CSV.")

    if min_class_samples is None:
        if config.use_pca and config.n_pca_components is not None:
            effective_dim = config.n_pca_components
            source = f"n_pca_components={config.n_pca_components}"
        else:
            effective_dim = _detect_n_features(config)
            source = f"{effective_dim}-dim embeddings"
        min_class_samples = effective_dim
        print(f"  Auto-detected min_class_samples={min_class_samples} (from {source})")

    base_seed = config.random_seed
    seeds = [base_seed + i for i in range(n_bootstrap)]

    all_rows: list[dict[str, Any]] = []

    for model_label, model_spec in config.models.items():
        print(f"\n## Cross-validation: {model_label} ({model_spec.name})")

        for task, channels in tc.items():
            for channel in channels:
                print(f"\n### {task} / {channel}")

                # Build full dataset list for this combo
                pairs = _build_datasets_for_combo(
                    model_spec.train_datasets, channel, task
                )
                if not pairs:
                    print("  No datasets available, skipping.")
                    continue

                ds_names = [_get_dataset_name(p[0]) for p in pairs]

                if len(pairs) <= 1:
                    logger.warning(
                        f"  Only {len(pairs)} dataset(s) for "
                        f"{task}/{channel} — cannot do leave-one-out CV."
                    )
                    continue

                # Baseline: all datasets
                all_ds_dicts = [p[1] for p in pairs]
                print(f"  Baseline: {len(all_ds_dicts)} datasets, {n_bootstrap} seeds")
                for seed in seeds:
                    row = _train_and_evaluate(
                        config,
                        model_label,
                        task,
                        channel,
                        all_ds_dicts,
                        seed,
                        excluded_dataset=None,
                    )
                    all_rows.append(row)

                # Leave-one-out folds
                for i, (train_ds, _) in enumerate(pairs):
                    ds_name = ds_names[i]
                    remaining = [p[1] for j, p in enumerate(pairs) if j != i]

                    # Safety check
                    safe = _check_class_safety(remaining, task, min_class_samples)
                    if not safe:
                        print(f"  Excluding {ds_name}: UNSAFE (class threshold)")
                        for seed in seeds:
                            unsafe_row = {
                                "model": model_label,
                                "task": task,
                                "channel": channel,
                                "excluded_dataset": ds_name,
                                "seed": seed,
                                "n_train_datasets": len(remaining),
                                "impact": "unsafe",
                                "auroc": np.nan,
                            }
                            all_rows.append(unsafe_row)
                        continue

                    print(
                        f"  Excluding {ds_name}: "
                        f"{len(remaining)} remaining, {n_bootstrap} seeds"
                    )
                    for seed in seeds:
                        row = _train_and_evaluate(
                            config,
                            model_label,
                            task,
                            channel,
                            remaining,
                            seed,
                            excluded_dataset=ds_name,
                        )
                        all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(all_rows)

    # Compute summary
    summary_df = _compute_summary(results_df, ranking_metric)

    # Save CSVs
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.output_dir / "cv_results.csv"
    summary_path = config.output_dir / "cv_summary.csv"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Raw results: {results_path}")
    print(f"  Summary: {summary_path}")

    # Print markdown summary
    _print_markdown_summary(summary_df, ranking_metric)

    return results_df, summary_df


def _compute_summary(
    results_df: pd.DataFrame,
    ranking_metric: str = "auroc",
) -> pd.DataFrame:
    """Aggregate raw CV results into per-fold summary with impact labels.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results from cross_validate_datasets.
    ranking_metric : str
        Metric column to compute deltas on.

    Returns
    -------
    pd.DataFrame
        Summary with columns: model, task, channel, excluded_dataset,
        mean_{metric}, std_{metric}, baseline_mean, delta, impact.
    """
    if results_df.empty:
        return pd.DataFrame()

    group_cols = ["model", "task", "channel"]

    summary_rows = []

    for group_key, group_df in results_df.groupby(group_cols):
        model, task, channel = group_key

        # Baseline stats
        baseline = group_df[group_df["excluded_dataset"] == "baseline"]
        baseline_mean = baseline[ranking_metric].mean()
        baseline_std = baseline[ranking_metric].std()

        # Per excluded dataset
        for exc_ds, fold_df in group_df.groupby("excluded_dataset"):
            fold_mean = fold_df[ranking_metric].mean()
            fold_std = fold_df[ranking_metric].std()

            if exc_ds == "baseline":
                delta = 0.0
                impact = "baseline"
            elif fold_df.get("impact", pd.Series()).eq("unsafe").any():
                delta = np.nan
                impact = "unsafe"
            else:
                # Delta = fold_mean - baseline_mean
                # Positive delta = performance improved when dataset removed
                #   → dataset HURTS
                # Negative delta = performance dropped when dataset removed
                #   → dataset HELPS
                delta = fold_mean - baseline_mean

                # Use pooled std for significance threshold
                pooled_std = (
                    np.sqrt((baseline_std**2 + fold_std**2) / 2)
                    if not (np.isnan(baseline_std) or np.isnan(fold_std))
                    else 0.0
                )

                if pooled_std == 0 or np.isnan(delta):
                    impact = "uncertain"
                elif delta > 0 and delta > pooled_std:
                    impact = "hurts"
                elif delta < 0 and abs(delta) > pooled_std:
                    impact = "helps"
                else:
                    impact = "uncertain"

            summary_rows.append(
                {
                    "model": model,
                    "task": task,
                    "channel": channel,
                    "excluded_dataset": exc_ds,
                    f"mean_{ranking_metric}": fold_mean,
                    f"std_{ranking_metric}": fold_std,
                    "baseline_mean": baseline_mean,
                    "delta": delta,
                    "impact": impact,
                }
            )

    return pd.DataFrame(summary_rows)


def _print_markdown_summary(summary_df: pd.DataFrame, ranking_metric: str) -> None:
    """Print a markdown-formatted summary table."""
    if summary_df.empty:
        print("\nNo cross-validation results to summarize.")
        return

    print("\n## Cross-Validation Impact Summary\n")

    for (model, task, channel), group in summary_df.groupby(
        ["model", "task", "channel"]
    ):
        print(f"\n### {model} / {task} / {channel}\n")
        print(f"| Excluded Dataset | Mean {ranking_metric.upper()} | Delta | Impact |")
        print("|------------------|------------|-------|--------|")

        for _, row in group.sort_values(
            "delta", ascending=False, na_position="last"
        ).iterrows():
            mean_val = row.get(f"mean_{ranking_metric}", np.nan)
            delta = row.get("delta", np.nan)
            impact = row.get("impact", "?")

            mean_str = f"{mean_val:.4f}" if not np.isnan(mean_val) else "N/A"
            delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"

            print(
                f"| {row['excluded_dataset']} | {mean_str} | {delta_str} | {impact} |"
            )


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------


def generate_cv_report(
    config: DatasetEvalConfig,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ranking_metric: str = "auroc",
) -> Path:
    """Generate a PDF cross-validation report.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration.
    results_df : pd.DataFrame
        Raw CV results.
    summary_df : pd.DataFrame
        Aggregated summary.
    ranking_metric : str
        Metric used for ranking.

    Returns
    -------
    Path
        Path to generated PDF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path = config.output_dir / "cv_report.pdf"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        # Page 1: Title + methodology
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.85,
            "Cross-Validation: Training Dataset Impact Analysis",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
        )
        methodology = (
            f"Method: Leave-one-dataset-out CV\n"
            f"Ranking metric: {ranking_metric}\n"
            f"Seeds per fold: {results_df['seed'].nunique()}\n"
            f"Models: {', '.join(summary_df['model'].unique())}\n\n"
            f"Impact classification:\n"
            f"  hurts: removing dataset improves {ranking_metric} "
            f"by > 1 pooled std\n"
            f"  helps: removing dataset decreases {ranking_metric} "
            f"by > 1 pooled std\n"
            f"  uncertain: delta within 1 pooled std\n"
            f"  unsafe: fold skipped (class threshold not met)"
        )
        ax.text(
            0.5,
            0.55,
            methodology,
            ha="center",
            va="top",
            fontsize=12,
            fontfamily="monospace",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Annotation inventory
        _render_annotation_inventory(pdf, results_df)

        # Page 3+: Impact heatmaps per model
        for model in summary_df["model"].unique():
            model_summary = summary_df[
                (summary_df["model"] == model)
                & (summary_df["excluded_dataset"] != "baseline")
            ]
            if model_summary.empty:
                continue
            _render_impact_heatmap(pdf, model_summary, model, ranking_metric)

        # Per-task detail pages
        for (model, task, channel), group in summary_df.groupby(
            ["model", "task", "channel"]
        ):
            non_baseline = group[group["excluded_dataset"] != "baseline"]
            if non_baseline.empty:
                continue
            _render_delta_bar_chart(
                pdf, non_baseline, f"{model} / {task} / {channel}", ranking_metric
            )

    print(f"\n  CV report saved: {output_path}")
    return output_path


def _render_annotation_inventory(pdf, results_df: pd.DataFrame) -> None:
    """Render annotation count table page."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Annotation Inventory (training class counts)", fontsize=14, pad=20)

    # Gather class count columns
    class_cols = [c for c in results_df.columns if c.startswith("train_class_")]
    if not class_cols:
        ax.text(0.5, 0.5, "No class count data available.", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)
        return

    baseline = results_df[results_df["excluded_dataset"] == "baseline"]
    if baseline.empty:
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Show one row per (model, task, channel) baseline
    display_cols = ["model", "task", "channel"] + class_cols
    summary = baseline.groupby(["model", "task", "channel"])[class_cols].first()
    summary = summary.reset_index()

    cell_text = []
    for _, row in summary.iterrows():
        cell_text.append([str(row[c]) for c in display_cols])

    col_labels = display_cols
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    pdf.savefig(fig)
    plt.close(fig)


def _render_impact_heatmap(
    pdf, model_summary: pd.DataFrame, model: str, ranking_metric: str
) -> None:
    """Render impact heatmap for one model."""
    import matplotlib.pyplot as plt

    pivot = model_summary.pivot_table(
        index="excluded_dataset",
        columns=["task", "channel"],
        values="delta",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(11, max(4, len(pivot) * 0.8 + 2)))
    ax.set_title(f"Impact Heatmap: {model}", fontsize=14)

    # Use blue-orange diverging colormap (colorblind-friendly)
    cmap = plt.cm.RdYlBu_r  # Blue = negative (helps), Orange/Red = positive (hurts)
    vmax = (
        max(
            abs(pivot.values[~np.isnan(pivot.values)].max()),
            abs(pivot.values[~np.isnan(pivot.values)].min()),
        )
        if pivot.values.size > 0 and not np.all(np.isnan(pivot.values))
        else 0.05
    )
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [f"{t}/{c}" for t, c in pivot.columns], rotation=45, ha="right", fontsize=9
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="gray")

    fig.colorbar(im, ax=ax, label=f"{ranking_metric} delta (positive = hurts)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_delta_bar_chart(
    pdf, group: pd.DataFrame, title: str, ranking_metric: str
) -> None:
    """Render per-fold delta bar chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(f"Dataset Impact: {title}", fontsize=13)

    sorted_group = group.sort_values("delta", ascending=True)
    datasets = sorted_group["excluded_dataset"].values
    deltas = sorted_group["delta"].values
    impacts = sorted_group["impact"].values

    colors = []
    for imp in impacts:
        if imp == "hurts":
            colors.append("#E69F00")  # orange
        elif imp == "helps":
            colors.append("#0072B2")  # blue
        elif imp == "unsafe":
            colors.append("#999999")  # gray
        else:
            colors.append("#56B4E9")  # light blue

    y_pos = range(len(datasets))
    ax.barh(y_pos, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_xlabel(f"{ranking_metric} delta (positive = removing helps)", fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#E69F00", edgecolor="black", label="hurts"),
        Patch(facecolor="#0072B2", edgecolor="black", label="helps"),
        Patch(facecolor="#56B4E9", edgecolor="black", label="uncertain"),
        Patch(facecolor="#999999", edgecolor="black", label="unsafe"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Leave-one-dataset-out cross-validation"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5,
        help="Number of bootstrap seeds per fold",
    )
    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=None,
        help="Minimum samples per class for a fold to be safe. "
        "Default: auto-detect from embedding dimensionality (n_features).",
    )
    parser.add_argument(
        "--ranking-metric",
        type=str,
        default="auroc",
        help="Metric for impact ranking",
    )
    parser.add_argument(
        "--n-pca-components",
        type=int,
        default=None,
        help="Number of PCA components. Enables PCA when set.",
    )
    parser.add_argument(
        "--no-scaling",
        action="store_true",
        default=False,
        help="Disable StandardScaler normalization",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip PDF report generation",
    )
    args = parser.parse_args()

    from .evaluate_dataset import build_default_config

    config = build_default_config()
    config.wandb_logging = False
    config.n_pca_components = args.n_pca_components
    config.use_scaling = not args.no_scaling

    print(f"Output: {config.output_dir}")
    print(f"PCA: {config.use_pca} (n_components={config.n_pca_components})")
    print(f"Scaling: {config.use_scaling}")
    for label, spec in config.models.items():
        print(f"  {label}: {len(spec.train_datasets)} training datasets")

    results_df, summary_df = cross_validate_datasets(
        config,
        ranking_metric=args.ranking_metric,
        n_bootstrap=args.n_bootstrap,
        min_class_samples=args.min_class_samples,
    )

    if not args.no_report and not results_df.empty:
        generate_cv_report(
            config, results_df, summary_df, ranking_metric=args.ranking_metric
        )
