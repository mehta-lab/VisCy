"""PDF report generation for linear classifier evaluation and cross-validation.

Provides two report generators:
- ``generate_comparison_report``: Evaluation report comparing models on a test set.
- ``generate_cv_report``: Cross-validation report with impact analysis.

Both are optional and gated behind the ``--report`` flag in the respective scripts.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

matplotlib.use("Agg")

# Colorblind-friendly palette (Wong 2011)
_COLOR_HELPS = "#0072B2"
_COLOR_HURTS = "#E69F00"
_COLOR_UNCERTAIN = "#56B4E9"
_COLOR_UNSAFE = "#999999"
_COLOR_BASELINE = "#000000"

_IMPACT_COLORS = {
    "helps": _COLOR_HELPS,
    "hurts": _COLOR_HURTS,
    "uncertain": _COLOR_UNCERTAIN,
    "unsafe": _COLOR_UNSAFE,
    "baseline": _COLOR_BASELINE,
}

_MODEL_COLORS = {"2D": "#1f77b4", "3D": "#ff7f0e"}
_EXTRA_COLORS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

_TEMPORAL_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#882255",
]


def _get_model_color(label: str, idx: int = 0) -> str:
    return _MODEL_COLORS.get(label, _EXTRA_COLORS[idx % len(_EXTRA_COLORS)])


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------


def generate_comparison_report(
    output_dir: Path,
    dataset_name: str,
    model_labels: list[str],
    tasks: list[str],
    channels: list[str],
    train_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
    eval_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> Path:
    """Generate a PDF comparing model performance on a held-out test set.

    Parameters
    ----------
    output_dir : Path
        Directory to save the report.
    dataset_name : str
        Name of the test dataset.
    model_labels : list[str]
        Model labels (e.g. ``["2D", "3D"]``).
    tasks : list[str]
        Classification tasks evaluated.
    channels : list[str]
        Input channels evaluated.
    train_results : dict
        ``model_label -> (task, channel) -> {"metrics": {...}, ...}``.
    eval_results : dict
        ``model_label -> (task, channel) -> {"metrics": {...}, "annotated_adata": ...}``.

    Returns
    -------
    Path
        Path to the generated PDF.
    """
    report_path = output_dir / f"{dataset_name}_comparison_report.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(report_path) as pdf:
        _eval_page_title(pdf, dataset_name, model_labels, tasks, channels, train_results)
        _eval_page_global_metrics(pdf, model_labels, tasks, channels, train_results, eval_results)
        for task in tasks:
            _eval_page_task_comparison(pdf, task, model_labels, channels, eval_results)
        for channel in channels:
            _eval_page_channel_comparison(pdf, channel, model_labels, tasks, train_results, eval_results)

    print(f"\nReport saved: {report_path}")
    return report_path


def _eval_page_title(pdf, dataset_name, model_labels, tasks, channels, train_results):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    lines = [
        "Linear Classifier Comparison Report",
        "",
        f"Test Dataset: {dataset_name}",
        "",
    ]
    for label in model_labels:
        n_combos = len(train_results.get(label, {}))
        lines.append(f"Model {label}: {n_combos} classifiers trained")
    lines.append("")
    lines.append(f"Channels: {', '.join(channels)}")
    lines.append(f"Tasks: {', '.join(tasks)}")

    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
    )
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _eval_page_global_metrics(pdf, model_labels, tasks, channels, train_results, eval_results):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.suptitle("Global Metrics Summary", fontsize=14, fontweight="bold")

    col_labels = ["Task", "Channel"]
    for label in model_labels:
        col_labels.extend([f"{label}\nVal Acc", f"{label}\nVal F1", f"{label}\nTest Acc", f"{label}\nTest F1"])

    table_data = []
    for task in tasks:
        for channel in channels:
            row = [task, channel]
            for label in model_labels:
                train_r = train_results.get(label, {}).get((task, channel))
                eval_r = eval_results.get(label, {}).get((task, channel))
                val_acc = f"{train_r['metrics']['val_accuracy']:.3f}" if train_r else "-"
                val_f1 = f"{train_r['metrics']['val_weighted_f1']:.3f}" if train_r else "-"
                test_acc = f"{eval_r['metrics']['test_accuracy']:.3f}" if eval_r else "-"
                test_f1 = f"{eval_r['metrics']['test_weighted_f1']:.3f}" if eval_r else "-"
                row.extend([val_acc, val_f1, test_acc, test_f1])
            table_data.append(row)

    if table_data:
        table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _eval_page_task_comparison(pdf, task, model_labels, channels, eval_results):
    n_models = len(model_labels)

    all_classes: set[str] = set()
    for label in model_labels:
        for ch in channels:
            r = eval_results.get(label, {}).get((task, ch))
            if r and "annotated_adata" in r:
                adata = r["annotated_adata"]
                if task in adata.obs.columns:
                    all_classes.update(adata.obs[task].dropna().unique())
    all_classes_sorted = sorted(all_classes)

    # F1 bar chart
    fig, ax_bar = plt.subplots(figsize=(11, 5))
    fig.suptitle(f"Task: {task} - Per-Class F1", fontsize=14, fontweight="bold")

    if all_classes_sorted:
        x = np.arange(len(all_classes_sorted))
        width = 0.8 / max(n_models, 1)
        for i, label in enumerate(model_labels):
            f1_values = []
            for cls in all_classes_sorted:
                f1s = []
                for ch in channels:
                    r = eval_results.get(label, {}).get((task, ch))
                    if r:
                        f1 = r["metrics"].get(f"test_{cls}_f1")
                        if f1 is not None:
                            f1s.append(f1)
                f1_values.append(np.mean(f1s) if f1s else 0)
            ax_bar.bar(
                x + i * width,
                f1_values,
                width,
                label=label,
                color=_get_model_color(label, i),
            )
        ax_bar.set_xticks(x + width * (n_models - 1) / 2)
        ax_bar.set_xticklabels(all_classes_sorted)
        ax_bar.set_ylabel("Test F1 (avg across channels)")
        ax_bar.legend()
        ax_bar.set_ylim(0, 1.05)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrices
    n_cols = len(channels)
    n_rows = n_models
    if n_cols == 0 or n_rows == 0:
        return

    fig_cm, cm_axes = plt.subplots(n_rows, max(n_cols, 1), figsize=(4 * max(n_cols, 1), 3.5 * n_rows))
    fig_cm.suptitle(f"Confusion Matrices: {task}", fontsize=14, fontweight="bold")

    if n_rows == 1 and n_cols == 1:
        cm_axes = [[cm_axes]]
    elif n_rows == 1:
        cm_axes = [cm_axes]
    elif n_cols == 1:
        cm_axes = [[row] for row in cm_axes]

    for i, label in enumerate(model_labels):
        for j, ch in enumerate(channels):
            ax = cm_axes[i][j]
            r = eval_results.get(label, {}).get((task, ch))
            if r and "annotated_adata" in r:
                adata = r["annotated_adata"]
                pred_col = f"predicted_{task}"
                mask = adata.obs[task].notna() & (adata.obs[task] != "unknown")
                subset = adata[mask]
                if len(subset) > 0 and pred_col in subset.obs.columns:
                    y_true = subset.obs[task].values
                    y_pred = subset.obs[pred_col].values
                    labels = sorted(set(y_true) | set(y_pred))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"{label} / {ch}", fontsize=10)

    fig_cm.tight_layout()
    pdf.savefig(fig_cm, bbox_inches="tight")
    plt.close(fig_cm)


def _eval_page_channel_comparison(pdf, channel, model_labels, tasks, train_results, eval_results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Channel: {channel}", fontsize=14, fontweight="bold")

    n_models = len(model_labels)
    x = np.arange(len(tasks))
    width = 0.8 / max(n_models, 1)

    ax = axes[0]
    for i, label in enumerate(model_labels):
        accs = []
        for task in tasks:
            r = eval_results.get(label, {}).get((task, channel))
            accs.append(r["metrics"]["test_accuracy"] if r else 0)
        ax.bar(
            x + i * width,
            accs,
            width,
            label=label,
            color=_get_model_color(label, i),
        )
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Test Accuracy")

    ax2 = axes[1]
    for i, label in enumerate(model_labels):
        val_accs, test_accs = [], []
        for task in tasks:
            tr = train_results.get(label, {}).get((task, channel))
            ev = eval_results.get(label, {}).get((task, channel))
            val_accs.append(tr["metrics"]["val_accuracy"] if tr else 0)
            test_accs.append(ev["metrics"]["test_accuracy"] if ev else 0)

        color = _get_model_color(label, i)
        ax2.bar(
            x + i * width - width / 4,
            val_accs,
            width / 2,
            label=f"{label} Val",
            color=color,
            alpha=0.5,
        )
        ax2.bar(
            x + i * width + width / 4,
            test_accs,
            width / 2,
            label=f"{label} Test",
            color=color,
            alpha=1.0,
        )

    ax2.set_xticks(x + width * (n_models - 1) / 2)
    ax2.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7)
    ax2.set_title("Val vs Test (Generalization)")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-validation report
# ---------------------------------------------------------------------------


def generate_cv_report(
    output_dir: Path,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    config_summary: dict[str, Any],
    ranking_metric: str = "auroc",
) -> Path:
    """Generate a PDF cross-validation report with impact analysis.

    Parameters
    ----------
    output_dir : Path
        Directory to save the report.
    results_df : pd.DataFrame
        Raw results (one row per fold x seed).
    summary_df : pd.DataFrame
        Aggregated summary with impact labels.
    config_summary : dict
        Summary of config parameters for the title page.
    ranking_metric : str
        Metric used for impact ranking.

    Returns
    -------
    Path
        Path to the generated PDF.
    """
    output_path = output_dir / "cv_report.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        _cv_page_title(pdf, config_summary, results_df, summary_df, ranking_metric)
        _cv_page_annotation_inventory(pdf, results_df)

        for model in summary_df["model"].unique():
            model_summary = summary_df[(summary_df["model"] == model) & (summary_df["excluded_dataset"] != "baseline")]
            if not model_summary.empty:
                _cv_page_impact_heatmap(pdf, model_summary, model, ranking_metric)

        for (model, task, channel), _ in results_df.groupby(["model", "task", "channel"]):
            _cv_page_auroc_distribution(pdf, results_df, summary_df, model, task, channel, ranking_metric)

        for (model, task, channel), _ in results_df.groupby(["model", "task", "channel"]):
            _cv_page_temporal_curves(pdf, results_df, summary_df, model, task, channel)

        for (model, task, channel), group in summary_df.groupby(["model", "task", "channel"]):
            non_baseline = group[group["excluded_dataset"] != "baseline"]
            if not non_baseline.empty:
                _cv_page_delta_bar_chart(
                    pdf,
                    non_baseline,
                    f"{model} / {task} / {channel}",
                    ranking_metric,
                )

    print(f"\n  CV report saved: {output_path}")
    return output_path


def _cv_page_title(pdf, config_summary, results_df, summary_df, ranking_metric):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(
        0.5,
        0.85,
        "Rotating CV: Training Dataset Impact Analysis",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )

    pca_str = (
        f"PCA: {config_summary.get('n_pca_components')} components"
        if config_summary.get("n_pca_components")
        else "PCA: disabled"
    )
    methodology = (
        f"Method: Rotating test-set leave-one-dataset-out CV\n"
        f"Ranking metric: {ranking_metric}\n"
        f"Seeds per fold: {results_df['seed'].nunique()}\n"
        f"Models: {', '.join(summary_df['model'].unique())}\n\n"
        f"Classifier training parameters:\n"
        f"  Scaling: {'StandardScaler' if config_summary.get('use_scaling', True) else 'disabled'}\n"
        f"  {pca_str}\n"
        f"  Solver: {config_summary.get('solver', 'liblinear')}\n"
        f"  Class weight: {config_summary.get('class_weight', 'balanced')}\n"
        f"  Max iter: {config_summary.get('max_iter', 1000)}\n"
        f"  Train/val split: {config_summary.get('split_train_data', 0.8)}\n\n"
        f"Impact classification:\n"
        f"  hurts: removing dataset improves {ranking_metric} by > 1 SEM\n"
        f"  helps: removing dataset decreases {ranking_metric} by > 1 SEM\n"
        f"  uncertain: delta within 1 SEM\n"
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


def _cv_page_annotation_inventory(pdf, results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Annotation Inventory (training class counts)", fontsize=14, pad=20)

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

    display_cols = ["model", "task", "channel"] + class_cols
    summary = baseline.groupby(["model", "task", "channel"])[class_cols].first()
    summary = summary.reset_index()

    cell_text = [[str(row[c]) for c in display_cols] for _, row in summary.iterrows()]

    table = ax.table(cellText=cell_text, colLabels=display_cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(list(range(len(display_cols))))
    table.scale(1.2, 1.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _cv_page_impact_heatmap(pdf, model_summary: pd.DataFrame, model: str, ranking_metric: str) -> None:
    pivot = model_summary.pivot_table(
        index="excluded_dataset",
        columns=["task", "channel"],
        values="delta",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(11, max(4, len(pivot) * 0.8 + 2)))
    ax.set_title(f"Impact Heatmap: {model}", fontsize=14)

    vals = pivot.values[~np.isnan(pivot.values)]
    vmax = max(abs(vals.max()), abs(vals.min())) if vals.size > 0 else 0.05
    im = ax.imshow(pivot.values, cmap="RdYlBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{t}/{c}" for t, c in pivot.columns], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text = f"{val:+.3f}" if not np.isnan(val) else "N/A"
            color = "gray" if np.isnan(val) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label=f"{ranking_metric} delta (positive = hurts)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _cv_page_auroc_distribution(pdf, results_df, summary_df, model, task, channel, ranking_metric) -> None:
    group = results_df[
        (results_df["model"] == model) & (results_df["task"] == task) & (results_df["channel"] == channel)
    ]
    if group.empty:
        return

    summary_group = summary_df[
        (summary_df["model"] == model) & (summary_df["task"] == task) & (summary_df["channel"] == channel)
    ]
    impact_map = dict(zip(summary_group["excluded_dataset"], summary_group["impact"]))

    conditions = sorted(group["excluded_dataset"].unique())
    if "baseline" in conditions:
        conditions.remove("baseline")
        conditions = ["baseline"] + conditions

    box_data = []
    colors = []
    labels = []
    for cond in conditions:
        vals = group[group["excluded_dataset"] == cond][ranking_metric].dropna().values
        box_data.append(vals)
        labels.append(cond)
        impact = impact_map.get(cond, "uncertain")
        colors.append(_IMPACT_COLORS.get(impact, _COLOR_UNCERTAIN))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(f"AUROC Distribution: {model} / {task} / {channel}", fontsize=13)

    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    if "baseline" in conditions:
        bl_vals = group[group["excluded_dataset"] == "baseline"][ranking_metric].dropna()
        if not bl_vals.empty:
            ax.axhline(
                y=bl_vals.mean(),
                color="black",
                linewidth=1,
                linestyle="--",
                label=f"Baseline mean ({bl_vals.mean():.3f})",
            )
            ax.legend(fontsize=9)

    ax.set_ylabel(ranking_metric.upper())
    ax.set_xlabel("Excluded dataset")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _cv_page_temporal_curves(pdf, results_df, summary_df, model, task, channel) -> None:
    group = results_df[
        (results_df["model"] == model) & (results_df["task"] == task) & (results_df["channel"] == channel)
    ]

    if "temporal_metrics" not in group.columns:
        return
    if not group["temporal_metrics"].notna().any():
        return

    conditions = sorted(group["excluded_dataset"].unique())
    if "baseline" in conditions:
        conditions.remove("baseline")
        conditions = ["baseline"] + conditions

    excl_conditions = [c for c in conditions if c != "baseline"]
    excl_color_map = {c: _TEMPORAL_PALETTE[i % len(_TEMPORAL_PALETTE)] for i, c in enumerate(excl_conditions)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(f"Temporal Metrics: {model} / {task} / {channel}", fontsize=13)

    for cond in conditions:
        cond_df = group[group["excluded_dataset"] == cond]
        temporal_jsons = cond_df["temporal_metrics"].dropna()
        if temporal_jsons.empty:
            continue

        parsed = [json.loads(s) for s in temporal_jsons]
        n_bins = len(parsed[0]["auroc"])
        bin_edges = parsed[0]["bin_edges"]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]

        is_baseline = cond == "baseline"
        linewidth = 2.5 if is_baseline else 1.2
        color = _COLOR_BASELINE if is_baseline else excl_color_map[cond]

        for ax_idx, metric_key in enumerate(["auroc", "f1_macro"]):
            ax = axes[ax_idx]
            all_vals = np.array([[v if v is not None else np.nan for v in p[metric_key]] for p in parsed])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                means = np.nanmean(all_vals, axis=0)
                stds = np.nanstd(all_vals, axis=0)

            ax.plot(
                bin_centers,
                means,
                label=cond,
                linewidth=linewidth,
                color=color,
            )
            ax.fill_between(
                bin_centers,
                means - stds,
                means + stds,
                alpha=0.15,
                color=color,
            )

    for ax, title in zip(axes, ["AUROC", "F1 Macro"]):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Normalized time")
        ax.set_ylabel(title)
        ax.axhline(y=0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _cv_page_delta_bar_chart(pdf, group: pd.DataFrame, title: str, ranking_metric: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(f"Dataset Impact: {title}", fontsize=13)

    sorted_group = group.sort_values("delta", ascending=True)
    datasets = sorted_group["excluded_dataset"].values
    deltas = sorted_group["delta"].values
    impacts = sorted_group["impact"].values

    colors = [_IMPACT_COLORS.get(imp, _COLOR_UNCERTAIN) for imp in impacts]

    y_pos = range(len(datasets))
    ax.barh(y_pos, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_xlabel(f"{ranking_metric} delta (positive = removing helps)", fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")

    legend_elements = [
        Patch(facecolor=_COLOR_HURTS, edgecolor="black", label="hurts"),
        Patch(facecolor=_COLOR_HELPS, edgecolor="black", label="helps"),
        Patch(facecolor=_COLOR_UNCERTAIN, edgecolor="black", label="uncertain"),
        Patch(facecolor=_COLOR_UNSAFE, edgecolor="black", label="unsafe"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
