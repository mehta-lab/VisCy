"""PDF report generation for 2D vs 3D linear classifier comparison."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Colorblind-friendly palette
COLOR_MAP = {
    "2D": "#1f77b4",  # Blue
    "3D": "#ff7f0e",  # Orange
}

# Fallback colors for additional models
_EXTRA_COLORS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def _get_model_color(label: str, idx: int = 0) -> str:
    return COLOR_MAP.get(label, _EXTRA_COLORS[idx % len(_EXTRA_COLORS)])


def generate_comparison_report(
    config,
    train_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
    eval_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> Path:
    """Generate a PDF comparing model performance.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration.
    train_results : dict
        Output from ``train_classifiers()``.
    eval_results : dict
        Output from ``evaluate_predictions()``.

    Returns
    -------
    Path
        Path to the generated PDF.
    """
    from applications.DynaCLR.evaluation.linear_classifiers.evaluate_dataset import (
        _resolve_tasks,
    )

    tasks = _resolve_tasks(config)
    model_labels = list(config.models.keys())
    report_path = config.output_dir / f"{config.dataset_name}_2d_vs_3d_report.pdf"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(report_path) as pdf:
        _page_title(pdf, config, model_labels, train_results, eval_results)
        _page_global_metrics(
            pdf, model_labels, tasks, config.channels, train_results, eval_results
        )

        for task in tasks:
            _page_task_comparison(
                pdf, task, model_labels, config.channels, eval_results
            )

        for channel in config.channels:
            _page_channel_comparison(
                pdf, channel, model_labels, tasks, train_results, eval_results
            )

        _page_test_data_summary(pdf, tasks, model_labels, config.channels, eval_results)

    print(f"\nReport saved: {report_path}")
    return report_path


def _page_title(pdf, config, model_labels, train_results, eval_results):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    lines = [
        "Linear Classifier Comparison Report",
        "",
        f"Test Dataset: {config.dataset_name}",
        "",
    ]

    for label in model_labels:
        spec = config.models[label]
        n_train = len(spec.train_datasets)
        n_combos = len(train_results.get(label, {}))
        lines.append(f"Model {label}: {spec.name} ({spec.version})")
        lines.append(f"  Training datasets: {n_train}")
        lines.append(f"  Trained classifiers: {n_combos}")
        lines.append("")

    lines.append(f"Channels: {', '.join(config.channels)}")

    from applications.DynaCLR.evaluation.linear_classifiers.evaluate_dataset import (
        _resolve_tasks,
    )

    tasks = _resolve_tasks(config)
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
    fig.suptitle("2D vs 3D Model Comparison", fontsize=16, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_global_metrics(
    pdf, model_labels, tasks, channels, train_results, eval_results
):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.suptitle("Global Metrics Summary", fontsize=14, fontweight="bold")

    col_labels = ["Task", "Channel"]
    for label in model_labels:
        col_labels.extend(
            [
                f"{label}\nVal Acc",
                f"{label}\nVal F1",
                f"{label}\nTest Acc",
                f"{label}\nTest F1",
            ]
        )

    table_data = []
    for task in tasks:
        for channel in channels:
            row = [task, channel]
            for label in model_labels:
                train_r = train_results.get(label, {}).get((task, channel))
                eval_r = eval_results.get(label, {}).get((task, channel))
                val_acc = (
                    f"{train_r['metrics']['val_accuracy']:.3f}" if train_r else "-"
                )
                val_f1 = (
                    f"{train_r['metrics']['val_weighted_f1']:.3f}" if train_r else "-"
                )
                test_acc = (
                    f"{eval_r['metrics']['test_accuracy']:.3f}" if eval_r else "-"
                )
                test_f1 = (
                    f"{eval_r['metrics']['test_weighted_f1']:.3f}" if eval_r else "-"
                )
                row.extend([val_acc, val_f1, test_acc, test_f1])
            table_data.append(row)

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_task_comparison(pdf, task, model_labels, channels, eval_results):
    n_models = len(model_labels)
    n_channels = len(channels)

    fig, axes = plt.subplots(
        1 + 1, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1, 2]}
    )
    fig.suptitle(f"Task: {task}", fontsize=14, fontweight="bold")

    # Top: grouped bar chart of F1-per-class
    ax_bar = axes[0]
    all_classes = set()
    for label in model_labels:
        r = eval_results.get(label, {}).get((task, channels[0]))
        if r:
            adata = r.get("annotated_adata")
            if adata is not None and task in adata.obs.columns:
                all_classes.update(adata.obs[task].dropna().unique())
    all_classes = sorted(all_classes)

    if all_classes:
        x = np.arange(len(all_classes))
        width = 0.8 / max(n_models, 1)

        for i, label in enumerate(model_labels):
            f1_values = []
            for cls in all_classes:
                # Average F1 across channels for this class
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
        ax_bar.set_xticklabels(all_classes)
        ax_bar.set_ylabel("Test F1 (avg across channels)")
        ax_bar.legend()
        ax_bar.set_ylim(0, 1.05)

    # Bottom: confusion matrices grid
    ax_cm = axes[1]
    ax_cm.axis("off")

    n_cols = n_channels
    n_rows = n_models
    fig_cm, cm_axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    fig_cm.suptitle(f"Confusion Matrices: {task}", fontsize=14, fontweight="bold")

    if n_rows == 1:
        cm_axes = [cm_axes]
    if n_cols == 1:
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
                if len(subset) > 0:
                    y_true = subset.obs[task].values
                    y_pred = subset.obs[pred_col].values
                    labels = sorted(set(y_true) | set(y_pred))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
                        ax=ax, cmap="Blues", colorbar=False
                    )
            ax.set_title(f"{label} / {ch}", fontsize=10)

    fig_cm.tight_layout()
    pdf.savefig(fig_cm, bbox_inches="tight")
    plt.close(fig_cm)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_channel_comparison(
    pdf, channel, model_labels, tasks, train_results, eval_results
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Channel: {channel}", fontsize=14, fontweight="bold")

    n_models = len(model_labels)
    x = np.arange(len(tasks))
    width = 0.8 / max(n_models, 1)

    # Left: test accuracy per task
    ax = axes[0]
    for i, label in enumerate(model_labels):
        accs = []
        for task in tasks:
            r = eval_results.get(label, {}).get((task, channel))
            accs.append(r["metrics"]["test_accuracy"] if r else 0)
        ax.bar(
            x + i * width, accs, width, label=label, color=_get_model_color(label, i)
        )
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Test Accuracy")

    # Right: train-val vs test accuracy (overfitting check)
    ax2 = axes[1]
    for i, label in enumerate(model_labels):
        val_accs = []
        test_accs = []
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


def _page_test_data_summary(pdf, tasks, model_labels, channels, eval_results):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.suptitle("Test Data Summary", fontsize=14, fontweight="bold")

    lines = []
    for task in tasks:
        lines.append(f"### {task}")
        for label in model_labels:
            for ch in channels:
                r = eval_results.get(label, {}).get((task, ch))
                if r and "annotated_adata" in r:
                    adata = r["annotated_adata"]
                    mask = adata.obs[task].notna() & (adata.obs[task] != "unknown")
                    subset = adata[mask]
                    counts = subset.obs[task].value_counts()
                    dist = ", ".join(f"{k}: {v}" for k, v in counts.items())
                    lines.append(f"  {label}/{ch}: n={len(subset)} ({dist})")
                    break  # Same annotations across channels
            break  # Same annotations across models
        lines.append("")

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
