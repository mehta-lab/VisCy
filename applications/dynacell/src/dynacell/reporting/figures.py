"""Paper-ready figures from evaluation outputs.

Generates matplotlib figures suitable for the NeurIPS paper.  All functions
return ``matplotlib.figure.Figure`` objects and optionally save to disk.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dynacell.reporting.tables import (
    MASK_METRICS,
    PIXEL_METRICS,
    load_and_aggregate,
)

logger = logging.getLogger(__name__)


def metric_comparison_barplot(
    model_results: dict[str, Path],
    metrics: list[str] | None = None,
    save_path: Path | None = None,
    pixel_csv: str = "pixel_metrics.csv",
    mask_csv: str = "mask_metrics.csv",
) -> plt.Figure:
    """Plot grouped bar chart comparing models across metrics.

    Parameters
    ----------
    model_results
        Mapping of model name to results directory.
    metrics
        Metric columns to plot. Default: PIXEL_METRICS + MASK_METRICS.
    save_path
        If set, save the figure as PDF.
    pixel_csv, mask_csv
        CSV filenames to load.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = PIXEL_METRICS + MASK_METRICS

    model_data = {}
    for name, results_dir in model_results.items():
        agg, available = load_and_aggregate(results_dir, metrics, pixel_csv=pixel_csv, mask_csv=mask_csv)
        if agg.empty:
            logger.warning(
                "Model %r has no evaluation results in %s — omitting from plot.",
                name,
                results_dir,
            )
            continue
        model_data[name] = {
            "mean": agg.loc["mean", available],
            "std": agg.loc["std", available],
        }

    if not model_data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    first_model = next(iter(model_data.values()))
    plot_metrics = list(first_model["mean"].index)
    n_models = len(model_data)
    n_metrics = len(plot_metrics)

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 5))
    x = range(n_metrics)
    width = 0.8 / n_models

    for i, (name, stats) in enumerate(model_data.items()):
        offsets = [xi + i * width - (n_models - 1) * width / 2 for xi in x]
        means = stats["mean"].reindex(plot_metrics)
        stds = stats["std"].reindex(plot_metrics)
        ax.bar(
            offsets,
            means.values,
            width,
            yerr=stds.values,
            label=name,
            capsize=3,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_metrics, rotation=45, ha="right")
    ax.legend()
    ax.set_ylabel("Metric Value")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
