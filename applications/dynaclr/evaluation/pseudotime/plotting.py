"""Plotting functions for pseudotime remodeling analysis.

All functions save to pdf+png and return the matplotlib Figure.

Ported from:
- .ed_planning/tmp/scripts/annotation_remodling.py (fraction curves, heatmaps, distributions)
- .ed_planning/tmp/scripts/multi_organelle_remodeling.py (distance curves)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


def _save_figure(fig: plt.Figure, output_dir: Path, filename_prefix: str) -> None:
    """Save figure in pdf and png formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{filename_prefix}.{ext}",
            dpi=300,
            bbox_inches="tight",
        )


def plot_response_curves(
    organelle_curves: dict[str, pd.DataFrame],
    organelle_configs: dict[str, dict],
    output_dir: Path,
    signal_type: Literal["fraction", "continuous"] = "fraction",
    min_cells_per_bin: int = 5,
    title: str = "Organelle remodeling after infection",
    filename_prefix: str = "response_curves",
) -> plt.Figure:
    """Two-panel plot: signal with CI/IQR bands (top) + N cells (bottom).

    Parameters
    ----------
    organelle_curves : dict[str, pd.DataFrame]
        Per-organelle output of metrics.aggregate_population.
    organelle_configs : dict[str, dict]
        Per-organelle config with "label" and "color" keys.
    output_dir : Path
        Directory for saving plots.
    signal_type : {"fraction", "continuous"}
        Determines which columns to plot and band type.
    min_cells_per_bin : int
        Minimum cells to include a bin in the plot.
    title : str
        Plot title.
    filename_prefix : str
        Filename prefix for saved files.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), height_ratios=[3, 1], sharex=True
    )

    if signal_type == "fraction":
        signal_col = "fraction"
        band_lower = "ci_lower"
        band_upper = "ci_upper"
        ylabel = "Fraction remodeling"
    else:
        signal_col = "mean"
        band_lower = "q25"
        band_upper = "q75"
        ylabel = "Distance from baseline"

    for organelle, curve_df in organelle_curves.items():
        config = organelle_configs[organelle]
        color = config["color"]
        label = config["label"]

        mask = curve_df["n_cells"] >= min_cells_per_bin
        plot_df = curve_df[mask]
        time_hours = plot_df["time_minutes"] / 60

        axes[0].plot(time_hours, plot_df[signal_col], color=color, label=label, lw=2)
        axes[0].fill_between(
            time_hours,
            plot_df[band_lower],
            plot_df[band_upper],
            color=color,
            alpha=0.2,
        )
        axes[1].plot(time_hours, plot_df["n_cells"], color=color, label=label, lw=1.5)

    axes[0].axvline(0, color="gray", ls="--", lw=1, label="Infection")
    axes[0].set_ylabel(ylabel)
    if signal_type == "fraction":
        axes[0].set_ylim(-0.02, 1.0)
    axes[0].legend(frameon=False)
    axes[0].set_title(title)

    axes[1].axvline(0, color="gray", ls="--", lw=1)
    axes[1].set_ylabel("N cells")
    axes[1].set_xlabel("Time relative to infection (hours)")

    plt.tight_layout()
    _save_figure(fig, output_dir, filename_prefix)

    return fig


def plot_cell_heatmap(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    signal_col: str = "signal",
    signal_type: Literal["fraction", "continuous"] = "fraction",
    organelle_label: str = "",
    output_dir: Path | None = None,
    filename_prefix: str = "cell_heatmap",
) -> plt.Figure:
    """Per-track heatmap sorted by signal onset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with signal, t_relative_minutes, fov_name, track_id.
    time_bins : np.ndarray
        Bin edges in minutes.
    signal_col : str
        Column containing signal values.
    signal_type : {"fraction", "continuous"}
        "fraction" uses a 3-state colormap (no data/negative/positive).
        "continuous" uses viridis.
    organelle_label : str
        Label for the plot title.
    output_dir : Path or None
        If provided, save the figure.
    filename_prefix : str
        Filename prefix for saved files.

    Returns
    -------
    plt.Figure
    """
    valid = df.dropna(subset=[signal_col]).copy()
    valid["time_bin"] = pd.cut(
        valid["t_relative_minutes"],
        bins=time_bins,
        labels=time_bins[:-1],
        right=False,
    )
    valid["time_bin"] = valid["time_bin"].astype(float)

    # Build per-track unique key
    group_cols = ["fov_name", "track_id"]
    if "experiment" in valid.columns:
        group_cols.append("experiment")
    valid["track_key"] = valid.groupby(group_cols).ngroup()

    if signal_type == "fraction":
        pivot = valid.pivot_table(
            index="track_key",
            columns="time_bin",
            values=signal_col,
            aggfunc="max",
        )
        # Sort by first positive timepoint
        first_positive = pivot.apply(
            lambda row: row.index[row == 1][0] if (row == 1).any() else np.inf,
            axis=1,
        )
    else:
        pivot = valid.pivot_table(
            index="track_key",
            columns="time_bin",
            values=signal_col,
            aggfunc="mean",
        )
        # Sort by time of max signal
        first_positive = pivot.apply(
            lambda row: (
                row.idxmax() if row.notna().any() and row.max() > 0 else np.inf
            ),
            axis=1,
        )

    pivot = pivot.loc[first_positive.sort_values().index]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.06)))

    bin_centers = pivot.columns.values
    bin_width = time_bins[1] - time_bins[0]
    bin_edges_hours = np.append(bin_centers, bin_centers[-1] + bin_width) / 60

    if signal_type == "fraction":
        plot_data = pivot.values.copy()
        plot_data = np.where(np.isnan(plot_data), -1, plot_data)
        cmap = ListedColormap(["#ffffff", "#c6dbef", "#08519c"])
        im = ax.pcolormesh(
            bin_edges_hours,
            np.arange(len(pivot) + 1),
            plot_data,
            cmap=cmap,
            vmin=-1,
            vmax=1,
        )
        cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(["No data", "No remodel", "Remodel"])
    else:
        plot_data = pivot.values.copy()
        im = ax.pcolormesh(
            bin_edges_hours,
            np.arange(len(pivot) + 1),
            plot_data,
            cmap="viridis",
        )
        plt.colorbar(im, ax=ax, label="Distance from baseline")

    ax.axvline(0, color="black", ls="--", lw=1, label="Infection")
    ax.set_xlabel("Time relative to infection (hours)")
    ax.set_ylabel("Cell tracks (sorted by onset)")
    ax.set_title(f"{organelle_label} — Per-track heatmap")
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    if output_dir is not None:
        _save_figure(fig, output_dir, filename_prefix)

    return fig


def plot_timing_distributions(
    track_timing_df: pd.DataFrame,
    organelle_configs: dict[str, dict],
    output_dir: Path,
    filename_prefix: str = "timing_distributions",
) -> plt.Figure:
    """Two-panel histogram: onset (left) and duration (right).

    Parameters
    ----------
    track_timing_df : pd.DataFrame
        Output of metrics.compute_track_timing with "organelle" column.
    organelle_configs : dict[str, dict]
        Per-organelle config with "label" and "color" keys.
    output_dir : Path
        Directory for saving plots.
    filename_prefix : str
        Filename prefix for saved files.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for organelle in track_timing_df["organelle"].unique():
        org_df = track_timing_df[track_timing_df["organelle"] == organelle]
        config = organelle_configs.get(organelle, {"color": "gray", "label": organelle})
        color = config["color"]
        label = config["label"]

        axes[0].hist(
            org_df["onset_minutes"] / 60,
            bins=30,
            alpha=0.6,
            color=color,
            label=label,
            edgecolor="white",
        )
        axes[1].hist(
            org_df["span_minutes"] / 60,
            bins=30,
            alpha=0.6,
            color=color,
            label=label,
            edgecolor="white",
        )

    axes[0].axvline(0, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("Remodeling onset relative to infection (hours)")
    axes[0].set_ylabel("N tracks")
    axes[0].set_title("When does remodeling start?")
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("Remodeling duration (hours)")
    axes[1].set_ylabel("N tracks")
    axes[1].set_title("How long does remodeling last?")
    axes[1].legend(frameon=False)

    plt.tight_layout()
    _save_figure(fig, output_dir, filename_prefix)

    return fig


def plot_onset_comparison(
    timing_metrics: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str = "onset_comparison",
) -> plt.Figure:
    """Bar chart comparing T_onset, T_50, T_peak across organelles.

    Parameters
    ----------
    timing_metrics : pd.DataFrame
        DataFrame with columns: organelle, T_onset_minutes, T_50_minutes,
        T_peak_minutes (and optionally color).
    output_dir : Path
        Directory for saving plots.
    filename_prefix : str
        Filename prefix for saved files.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    organelles = timing_metrics["organelle"].values
    x = np.arange(len(organelles))
    width = 0.25

    metrics_to_plot = []
    labels = []
    for col, label in [
        ("T_onset_minutes", "T_onset"),
        ("T_50_minutes", "T_50"),
        ("T_peak_minutes", "T_peak"),
    ]:
        if col in timing_metrics.columns:
            metrics_to_plot.append(col)
            labels.append(label)

    for i, (col, label) in enumerate(zip(metrics_to_plot, labels)):
        values_hours = timing_metrics[col].values / 60
        offset = (i - len(metrics_to_plot) / 2 + 0.5) * width
        ax.bar(x + offset, values_hours, width, label=label, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(organelles)
    ax.set_ylabel("Time relative to infection (hours)")
    ax.set_title("Timing metric comparison across organelles")
    ax.legend(frameon=False)
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    plt.tight_layout()
    _save_figure(fig, output_dir, filename_prefix)

    return fig
