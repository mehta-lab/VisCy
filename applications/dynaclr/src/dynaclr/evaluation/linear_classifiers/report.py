"""PDF report generation for linear classifier cross-validation.

Provides ``generate_cv_report`` for cross-validation reports with impact analysis.
This is optional and gated behind the ``--report`` flag in the cross-validation script.
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
