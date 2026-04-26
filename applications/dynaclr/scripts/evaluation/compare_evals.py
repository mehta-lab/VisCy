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

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def _load_registry(path: Path) -> tuple[list[dict], Path, float]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    output_dir = Path(raw["output_dir"])
    fdr_threshold = float(raw.get("fdr_threshold", 0.05))
    return raw["models"], output_dir, fdr_threshold


# ---------------------------------------------------------------------------
# Smoothness
# ---------------------------------------------------------------------------


def _load_smoothness(models: list[dict]) -> pd.DataFrame | None:
    frames = []
    for entry in models:
        smoothness_dir = Path(entry["eval_dir"]) / "smoothness"
        csvs = list(smoothness_dir.glob("*_smoothness_stats.csv"))
        if not csvs:
            click.echo(f"[smoothness] No smoothness CSV found for {entry['name']}", err=True)
            continue
        # Take the first (usually only) stats file — not per-group
        df = pd.read_csv(csvs[0])
        df["model"] = entry["name"]
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _plot_smoothness(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["smoothness_score", "dynamic_range"]
    present = [m for m in metrics if m in df.columns]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)
    for ax, metric in zip(axes[0], present):
        vals = df.set_index("model")[metric]
        ax.bar(vals.index, vals.values, color=plt.cm.tab10(np.arange(len(vals)) / len(vals)))
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


def _plot_linear_classifiers(df: pd.DataFrame, output_dir: Path) -> None:
    if "auroc" not in df.columns:
        return

    tasks = sorted(df["task"].unique()) if "task" in df.columns else ["all"]
    ncols = min(4, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    models = sorted(df["model"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_color = dict(zip(models, colors))

    for ax_idx, task in enumerate(tasks):
        ax = axes_flat[ax_idx]
        sub = df[df["task"] == task] if "task" in df.columns else df
        pivot = sub.pivot_table(
            index="marker" if "marker" in sub.columns else sub.index, columns="model", values="auroc"
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
        ax.set_ylabel("AUROC")
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


def _plot_mmd_kinetics(df: pd.DataFrame, output_dir: Path, fdr_threshold: float) -> None:
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

        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        linestyles = ["-", "--", ":", "-."]
        model_color = dict(zip(models, colors))
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

    # Smoothness
    smoothness_df = _load_smoothness(models)
    if smoothness_df is not None:
        smoothness_df.to_csv(output_dir / "smoothness_comparison.csv", index=False)
        _plot_smoothness(smoothness_df, output_dir)
        click.echo("\n## Smoothness\n")
        click.echo(smoothness_df[["model", "smoothness_score", "dynamic_range"]].to_markdown(index=False))

    # Linear classifiers
    lc_df = _load_linear_classifiers(models)
    if lc_df is not None:
        lc_df.to_csv(output_dir / "linear_classifiers_comparison.csv", index=False)
        _plot_linear_classifiers(lc_df, output_dir)
        summary_cols = [c for c in ["model", "task", "marker", "auroc", "f1"] if c in lc_df.columns]
        click.echo("\n## Linear Classifiers\n")
        click.echo(lc_df[summary_cols].to_markdown(index=False))

    # MMD
    mmd_df = _load_mmd(models)
    if mmd_df is not None:
        mmd_summary = _build_mmd_summary(mmd_df)
        mmd_summary.to_csv(output_dir / "mmd_comparison.csv", index=False)
        _plot_mmd_kinetics(mmd_df, output_dir, fdr_threshold)
        _plot_mmd_summary_heatmap(mmd_summary, output_dir)
        click.echo("\n## MMD activity z-score\n")
        click.echo(mmd_summary.to_markdown(index=False))


if __name__ == "__main__":
    main()
