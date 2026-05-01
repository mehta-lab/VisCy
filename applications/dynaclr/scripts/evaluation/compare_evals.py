"""Compare evaluation results across multiple model runs.

Reads outputs produced by ``dynaclr evaluate`` from multiple model eval directories,
compares smoothness, linear classifier AUROC, and MMD activity z-scores side by side,
and writes summary CSVs and plots to a shared output directory.

Usage
-----
python compare_evals.py -c eval_registry.yml

Registry YAML format
--------------------
models:
  - name: DynaCLR-v3
    eval_dir: /path/to/eval_v3
  - name: DINOv3-MLP
    eval_dir: /path/to/eval_dino
output_dir: /path/to/comparison_output
fdr_threshold: 0.05   # optional, default 0.05
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

# Per-class metric columns are emitted by the LC trainer as
# ``val_<class>_<metric>`` (precision, recall, f1). ``val_weighted_*`` and
# ``val_<class>_auroc`` are excluded — only point-classification metrics here.
_PER_CLASS_METRIC_RE = re.compile(r"^val_(?P<cls>.+)_(?P<metric>precision|recall|f1)$")

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def _load_registry(path: Path) -> tuple[list[dict], Path, float]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    output_dir = Path(raw["output_dir"])
    fdr_threshold = float(raw.get("fdr_threshold", 0.05))
    return raw["models"], output_dir, fdr_threshold


def _build_model_palette(model_names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Map model name → RGBA color, stable across all plots in one run.

    Uses ``tab10`` for ≤10 models (10 visually distinct hues) and ``tab20``
    for 11–20 models. Colors are picked from discrete colormap indices so
    they don't blur into each other when many models are compared.
    """
    n = len(model_names)
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    return {name: cmap(i % cmap.N) for i, name in enumerate(sorted(model_names))}


# ---------------------------------------------------------------------------
# Smoothness
# ---------------------------------------------------------------------------


def _load_smoothness(models: list[dict]) -> pd.DataFrame | None:
    """Load per-marker smoothness CSVs from all evals and concat.

    Each eval writes one ``*_per_marker_smoothness.csv`` per (experiment, marker)
    with columns including ``smoothness_score`` and ``dynamic_range``. We
    concat all of them and tag with the model name; the plotting step
    aggregates (mean across experiments+markers) for the headline bar chart.
    """
    frames = []
    for entry in models:
        smoothness_dir = Path(entry["eval_dir"]) / "smoothness"
        csvs = list(smoothness_dir.glob("*_per_marker_smoothness.csv"))
        if not csvs:
            click.echo(f"[smoothness] No smoothness CSV found for {entry['name']}", err=True)
            continue
        per_csv = [pd.read_csv(c) for c in csvs]
        df = pd.concat(per_csv, ignore_index=True)
        df["model"] = entry["name"]
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _plot_smoothness(df: pd.DataFrame, output_dir: Path, model_color: dict) -> None:
    """Plot per-model smoothness as mean ± std across (experiment, marker) rows."""
    metrics = ["smoothness_score", "dynamic_range"]
    present = [m for m in metrics if m in df.columns]
    if not present:
        return

    # Aggregate across (experiment, marker) per model: mean ± std.
    agg = df.groupby("model")[present].agg(["mean", "std"])

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)
    for ax, metric in zip(axes[0], present):
        means = agg[(metric, "mean")]
        stds = agg[(metric, "std")].fillna(0.0)
        bar_colors = [model_color.get(m, "gray") for m in means.index]
        ax.bar(
            means.index,
            means.values,
            yerr=stds.values,
            capsize=4,
            color=bar_colors,
        )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out = output_dir / "smoothness_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"[smoothness] Saved: {out}", err=True)


# ---------------------------------------------------------------------------
# Linear classifiers
# ---------------------------------------------------------------------------


def _load_linear_classifiers(models: list[dict]) -> pd.DataFrame | None:
    frames = []
    for entry in models:
        csv = Path(entry["eval_dir"]) / "linear_classifiers" / "metrics_summary.csv"
        if not csv.exists():
            click.echo(f"[linear_classifiers] Not found for {entry['name']}: {csv}", err=True)
            continue
        df = pd.read_csv(csv)
        df["model"] = entry["name"]
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _plot_linear_classifiers(df: pd.DataFrame, output_dir: Path, model_color: dict) -> None:
    # The LC writer emits per-split metrics: train_auroc and val_auroc.
    # Plot val_auroc (held-out generalization) — that is the headline number
    # for cross-model comparison.
    auroc_col = "val_auroc"
    if auroc_col not in df.columns:
        return

    # Marker breakdown lives in `marker_filter` (per-marker LCs), not `marker`.
    marker_col = "marker_filter" if "marker_filter" in df.columns else "marker"

    tasks = sorted(df["task"].unique()) if "task" in df.columns else ["all"]
    ncols = min(4, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    models = sorted(df["model"].unique())

    for ax_idx, task in enumerate(tasks):
        ax = axes_flat[ax_idx]
        sub = df[df["task"] == task] if "task" in df.columns else df
        pivot = sub.pivot_table(
            index=marker_col if marker_col in sub.columns else sub.index,
            columns="model",
            values=auroc_col,
        )
        pivot = pivot.reindex(columns=models)

        x = np.arange(len(pivot))
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            if model not in pivot.columns:
                continue
            ax.bar(x + i * width, pivot[model].values, width, label=model, color=model_color[model])

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Validation AUROC")
        ax.set_title(task, fontsize=9)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 1.05)

    for ax in axes_flat[len(tasks) :]:
        ax.set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=model_color[m], label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), fontsize=8, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = output_dir / "linear_classifiers_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"[linear_classifiers] Saved: {out}", err=True)


def _discover_per_class_metrics(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map class name → list of available metrics (precision, recall, f1).

    Columns are auto-detected from ``val_<class>_<metric>`` names so the plot
    works across tasks with different label sets (infected/uninfected,
    interphase/mitosis, alive/dead, noremodel/remodel, etc.).
    """
    found: dict[str, set[str]] = {}
    for col in df.columns:
        m = _PER_CLASS_METRIC_RE.match(col)
        if m is None:
            continue
        cls, metric = m.group("cls"), m.group("metric")
        if cls in {"weighted", "macro"}:
            continue
        # Only include classes that have at least one non-null value in this slice —
        # avoids polluting a task panel with empty bars for classes from other tasks.
        if df[col].notna().any():
            found.setdefault(cls, set()).add(metric)
    return {cls: sorted(metrics) for cls, metrics in found.items()}


def _plot_linear_classifiers_per_class(df: pd.DataFrame, output_dir: Path, model_color: dict) -> None:
    """Per-class precision/recall/F1 grouped bars per (task, marker_filter).

    AUROC is prevalence-invariant and rewards good ranking, but the
    infectomics LC tasks are heavily imbalanced (cell_division_state ~99/1,
    organelle_state ~91/9). Per-class precision and recall expose whether
    the classifier is actually usable at the chosen decision threshold for
    the minority class — they are the metrics that move when imbalance bites.
    """
    marker_col = "marker_filter" if "marker_filter" in df.columns else "marker"
    if "task" not in df.columns:
        return

    models = sorted(df["model"].unique())

    # One subplot per (task, marker_filter) so SEC61B vs G3BP1 stay separate.
    # Normalize missing marker_filter to None so iteration semantics are
    # consistent (pandas yields float NaN which is not None and not equality-comparable).
    if marker_col in df.columns:
        seen: set[tuple[str, str | None]] = set()
        panels: list[tuple[str, str | None]] = []
        for _, row in df[["task", marker_col]].drop_duplicates().iterrows():
            mf = row[marker_col]
            mf_norm = None if pd.isna(mf) else mf
            key = (row["task"], mf_norm)
            if key not in seen:
                seen.add(key)
                panels.append(key)
        panels.sort(key=lambda p: (p[0], "" if p[1] is None else p[1]))
    else:
        panels = [(t, None) for t in sorted(df["task"].unique())]

    if not panels:
        return

    ncols = min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    any_panel_drawn = False
    for ax_idx, (task, marker_filter) in enumerate(panels):
        ax = axes_flat[ax_idx]
        sub = df[df["task"] == task]
        if marker_col in sub.columns:
            if marker_filter is None:
                sub = sub[sub[marker_col].isna()]
            else:
                sub = sub[sub[marker_col] == marker_filter]
        if sub.empty:
            ax.set_visible(False)
            continue

        per_class = _discover_per_class_metrics(sub)
        if not per_class:
            ax.set_visible(False)
            continue

        # x-axis groups: (class, metric) pairs; bars within each group: models.
        groups: list[tuple[str, str]] = []
        for cls in sorted(per_class):
            for metric in ["precision", "recall", "f1"]:
                if metric in per_class[cls]:
                    groups.append((cls, metric))

        x = np.arange(len(groups))
        width = 0.8 / max(len(models), 1)
        for i, model in enumerate(models):
            row = sub[sub["model"] == model]
            if row.empty:
                continue
            row = row.iloc[0]
            values = [row.get(f"val_{cls}_{metric}", np.nan) for cls, metric in groups]
            ax.bar(x + i * width, values, width, label=model, color=model_color[model])

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([f"{cls}\n{metric}" for cls, metric in groups], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        title = f"{task}" if marker_filter in (None, "") else f"{task} ({marker_col}={marker_filter})"

        # Annotate panel with minority-class N when val support is available.
        # Per-class support is identical across models (same dataset, same seed),
        # so we read it from the first non-null row.
        support_cols = {cls: f"val_{cls}_support" for cls in per_class}
        supports: dict[str, int] = {}
        for cls, col in support_cols.items():
            if col in sub.columns:
                vals = sub[col].dropna()
                if not vals.empty:
                    supports[cls] = int(vals.iloc[0])
        if supports:
            total_n = sum(supports.values())
            minority_cls = min(supports, key=supports.get)
            minority_n = supports[minority_cls]
            minority_pct = 100 * minority_n / total_n if total_n else 0.0
            title += f"\nval N={total_n} | minority {minority_cls}={minority_n} ({minority_pct:.1f}%)"
        ax.set_title(title, fontsize=9)
        ax.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
        any_panel_drawn = True

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    if not any_panel_drawn:
        plt.close(fig)
        click.echo("[linear_classifiers] No per-class metrics found in metrics_summary.csv", err=True)
        return

    handles = [plt.Rectangle((0, 0), 1, 1, color=model_color[m], label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), fontsize=8, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = output_dir / "linear_classifiers_per_class.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"[linear_classifiers] Saved: {out}", err=True)


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------


def _load_mmd(models: list[dict]) -> pd.DataFrame | None:
    frames = []
    for entry in models:
        mmd_root = Path(entry["eval_dir"]) / "mmd"
        if not mmd_root.exists():
            click.echo(f"[mmd] No mmd directory for {entry['name']}", err=True)
            continue
        for csv in sorted(mmd_root.rglob("mmd_results.csv")):
            block_name = csv.parent.name
            df = pd.read_csv(csv)
            df["model"] = entry["name"]
            df["block"] = block_name
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _plot_mmd_kinetics(df: pd.DataFrame, output_dir: Path, fdr_threshold: float, model_color: dict) -> None:
    temporal = df.dropna(subset=["hours_bin_start", "hours_bin_end"]).copy()
    if temporal.empty:
        click.echo("[mmd] No temporal rows — skipping kinetics plot", err=True)
        return

    temporal["hours_mid"] = (temporal["hours_bin_start"] + temporal["hours_bin_end"]) / 2
    markers = sorted(temporal["marker"].unique())
    models = sorted(temporal["model"].unique())
    labels = sorted(temporal["label"].unique())
    blocks = sorted(temporal["block"].unique())

    for block in blocks:
        sub_block = temporal[temporal["block"] == block]
        ncols = min(4, len(markers))
        nrows = int(np.ceil(len(markers) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        linestyles = ["-", "--", ":", "-."]
        label_ls = dict(zip(labels, linestyles[: len(labels)]))

        for ax_idx, marker in enumerate(markers):
            ax = axes_flat[ax_idx]
            sub = sub_block[sub_block["marker"] == marker]
            for model in models:
                for label in labels:
                    grp = sub[(sub["model"] == model) & (sub["label"] == label)].sort_values("hours_mid")
                    if grp.empty:
                        continue
                    ax.plot(
                        grp["hours_mid"],
                        grp["activity_zscore"],
                        color=model_color[model],
                        linestyle=label_ls[label],
                        linewidth=1.5,
                    )
                    if "q_value" in grp.columns:
                        sig = grp[grp["q_value"] < fdr_threshold]
                        ax.scatter(sig["hours_mid"], sig["activity_zscore"], color=model_color[model], s=30, zorder=5)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_title(marker, fontsize=9)
            ax.set_xlabel("Hours post perturbation")
            ax.set_ylabel("Activity z-score")

        for ax in axes_flat[len(markers) :]:
            ax.set_visible(False)

        legend_handles = [Line2D([0], [0], color=model_color[m], linewidth=2, label=m) for m in models]
        legend_handles += [
            Line2D([0], [0], color="black", linestyle=label_ls[lb], linewidth=1.5, label=lb) for lb in labels
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(models) + len(labels),
            fontsize=8,
            bbox_to_anchor=(0.5, 0),
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])

        out = output_dir / f"mmd_kinetics_{block}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        click.echo(f"[mmd] Saved: {out}", err=True)


def _plot_mmd_summary_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    blocks = sorted(summary["block"].unique())
    labels = sorted(summary["label"].unique())
    models = sorted(summary["model"].unique())

    for block in blocks:
        sub_block = summary[summary["block"] == block]
        ncols = len(labels)
        markers = sorted(sub_block["marker"].unique())
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, max(3, len(markers) * 0.5 + 1)), squeeze=False)
        for col_idx, label in enumerate(labels):
            ax = axes[0, col_idx]
            pivot = sub_block[sub_block["label"] == label].pivot_table(
                index="marker", columns="model", values="mean_activity_zscore", aggfunc="mean"
            )
            pivot = pivot.reindex(columns=models)
            vmax = np.nanpercentile(np.abs(pivot.values), 95) if pivot.values.size > 0 else 1.0
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_title(label, fontsize=9)
            plt.colorbar(im, ax=ax, label="Mean activity z-score")

        fig.tight_layout()
        out = output_dir / f"mmd_summary_heatmap_{block}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        click.echo(f"[mmd] Saved: {out}", err=True)


def _build_mmd_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["block", "model", "marker", "label"])["activity_zscore"]
        .agg(mean_activity_zscore="mean", n_bins="count")
        .reset_index()
        .sort_values(["block", "label", "marker", "mean_activity_zscore"], ascending=[True, True, True, False])
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "-c", "--config", required=True, type=click.Path(exists=True, path_type=Path), help="Path to eval_registry.yml"
)
def main(config: Path) -> None:
    """Compare evaluation results across model runs."""
    models, output_dir, fdr_threshold = _load_registry(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the model→color palette once from the registry's model list so
    # every plot in this run uses the same color for the same model.
    model_color = _build_model_palette([m["name"] for m in models])

    # Smoothness
    smoothness_df = _load_smoothness(models)
    if smoothness_df is not None:
        smoothness_df.to_csv(output_dir / "smoothness_comparison.csv", index=False)
        _plot_smoothness(smoothness_df, output_dir, model_color)
        click.echo("\n## Smoothness\n")
        click.echo(smoothness_df[["model", "smoothness_score", "dynamic_range"]].to_markdown(index=False))

    # Linear classifiers
    lc_df = _load_linear_classifiers(models)
    if lc_df is not None:
        lc_df.to_csv(output_dir / "linear_classifiers_comparison.csv", index=False)
        _plot_linear_classifiers(lc_df, output_dir, model_color)
        _plot_linear_classifiers_per_class(lc_df, output_dir, model_color)
        summary_cols = [c for c in ["model", "task", "marker", "auroc", "f1"] if c in lc_df.columns]
        click.echo("\n## Linear Classifiers\n")
        click.echo(lc_df[summary_cols].to_markdown(index=False))

    # MMD
    mmd_df = _load_mmd(models)
    if mmd_df is not None:
        mmd_summary = _build_mmd_summary(mmd_df)
        mmd_summary.to_csv(output_dir / "mmd_comparison.csv", index=False)
        _plot_mmd_kinetics(mmd_df, output_dir, fdr_threshold, model_color)
        _plot_mmd_summary_heatmap(mmd_summary, output_dir)
        click.echo("\n## MMD activity z-score\n")
        click.echo(mmd_summary.to_markdown(index=False))


if __name__ == "__main__":
    main()
