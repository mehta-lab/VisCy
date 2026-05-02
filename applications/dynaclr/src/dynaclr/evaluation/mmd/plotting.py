"""Plots for MMD perturbation evaluation: kinetics curves and heatmaps."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests


def _bh_significance(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return boolean mask of BH-corrected significant p-values."""
    p_values = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p_values)
    sig = np.zeros(len(p_values), dtype=bool)
    if valid.sum() == 0:
        return sig
    _, corrected, _, _ = multipletests(p_values[valid], alpha=alpha, method="fdr_bh")
    sig[valid] = corrected
    return sig


def plot_mmd_kinetics(df: pd.DataFrame, output_path: Path) -> None:
    """Plot MMD kinetics curves (one line per marker over temporal bins).

    Parameters
    ----------
    df : pd.DataFrame
        MMD results for a single treatment group, with columns:
        marker, hours_bin_start, hours_bin_end, mmd2, p_value.
    output_path : Path
        Output file path. Format inferred from suffix (.pdf or .png).
    """
    df = df.copy().dropna(subset=["hours_bin_start", "hours_bin_end"])
    if df.empty:
        return
    df["bin_mid"] = (df["hours_bin_start"] + df["hours_bin_end"]) / 2

    markers = sorted(df["marker"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("tab10", n_colors=len(markers))

    for marker, color in zip(markers, palette):
        sub = df[df["marker"] == marker].sort_values("bin_mid")
        ax.plot(sub["bin_mid"], sub["mmd2"], marker="o", label=marker, color=color)
        # Stars for BH-significant bins
        sig = _bh_significance(sub["p_value"])
        for _, row, s in zip(range(len(sub)), sub.itertuples(), sig):
            if s:
                ax.text(row.bin_mid, row.mmd2, "*", ha="center", va="bottom", color=color, fontsize=12)

    ax.set_xlabel("Hours post perturbation (bin midpoint)")
    ax.set_ylabel("MMD²")
    ax.set_title(df["label"].iloc[0] if "label" in df.columns else "")
    ax.legend(title="Marker", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10, title_fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mmd_combined_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot combined cross-experiment MMD heatmap: markers × experiment pairs.

    One subplot per condition. Rows = markers, columns = exp_a vs exp_b pairs
    (averaged over temporal bins if present).

    Parameters
    ----------
    df : pd.DataFrame
        Combined MMD results with columns: marker, exp_a, exp_b, condition,
        hours_bin_start, hours_bin_end, mmd2, p_value.
    output_path : Path
        Output file path.
    """
    df = df.copy()
    df["exp_pair"] = (
        df["exp_a"].str.split("_").str[:3].str.join("_") + "\nvs\n" + df["exp_b"].str.split("_").str[:3].str.join("_")
    )
    conditions = sorted(df["condition"].unique())
    n_conds = len(conditions)

    fig, axes = plt.subplots(
        1, n_conds, figsize=(max(5 * n_conds, 6), max(4, df["marker"].nunique() * 0.7)), squeeze=False
    )

    for ax, condition in zip(axes[0], conditions):
        sub = df[df["condition"] == condition]
        pivot_mmd = sub.pivot_table(index="marker", columns="exp_pair", values="mmd2", aggfunc="mean")
        pivot_pval = sub.pivot_table(index="marker", columns="exp_pair", values="p_value", aggfunc="min")

        if pivot_mmd.empty or pivot_mmd.isna().all().all():
            ax.set_visible(False)
            continue

        sns.heatmap(pivot_mmd, ax=ax, cmap="viridis", linewidths=0.5, cbar_kws={"label": "MMD²"})

        sig = _bh_significance(pivot_pval.values.ravel())
        sig_matrix = sig.reshape(pivot_pval.shape)
        for r in range(sig_matrix.shape[0]):
            for c in range(sig_matrix.shape[1]):
                if sig_matrix[r, c]:
                    ax.text(
                        c + 0.5, r + 0.5, "*", ha="center", va="center", color="white", fontsize=10, fontweight="bold"
                    )

        ax.set_title(f"condition: {condition}")
        ax.set_xlabel("Experiment pair")
        ax.set_ylabel("Marker")
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Cross-experiment MMD — all markers", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mmd_multi_panel_kinetics(
    df: pd.DataFrame,
    output_path: Path,
    baseline_label: str | None = None,
    ncols: int = 4,
) -> None:
    """Plot per-marker MMD kinetics in a multi-panel grid with optional baseline band.

    One subplot per marker. Treatment comparisons are plotted as colored lines;
    if ``baseline_label`` is given, that comparison is shown as a gray dashed
    line with a shaded ±1 std band instead of a treatment line.

    Parameters
    ----------
    df : pd.DataFrame
        MMD results with columns: marker, label, hours_bin_start, hours_bin_end,
        mmd2, p_value.
    output_path : Path
        Output file path (.pdf or .png).
    baseline_label : str or None
        Label of the baseline comparison to render as a band. Default: None.
    ncols : int
        Number of columns in the panel grid. Default: 4.
    """
    df = df.copy().dropna(subset=["hours_bin_start", "hours_bin_end"])
    if df.empty:
        return
    df["bin_mid"] = (df["hours_bin_start"] + df["hours_bin_end"]) / 2

    markers = sorted(df["marker"].unique())
    treatment_labels = [lbl for lbl in df["label"].unique() if lbl != baseline_label]
    nrows = math.ceil(len(markers) / ncols)
    palette = sns.color_palette("tab10", n_colors=max(len(treatment_labels), 1))

    # Shared y-axis range
    treat_vals = df[df["label"].isin(treatment_labels)]["mmd2"].dropna()
    y_min = float(treat_vals.min()) if len(treat_vals) else 0.0
    y_max = float(treat_vals.max()) if len(treat_vals) else 1.0
    y_pad = (y_max - y_min) * 0.1 + 1e-6

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8), squeeze=False)

    for ax_idx, marker in enumerate(markers):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        sub = df[df["marker"] == marker]

        # Baseline band
        if baseline_label is not None:
            base = sub[sub["label"] == baseline_label].sort_values("bin_mid")
            if not base.empty:
                ax.axhline(base["mmd2"].mean(), color="gray", linewidth=1.0, linestyle="--", zorder=1)
                ax.fill_between(
                    base["bin_mid"],
                    base["mmd2"] - base["mmd2"].std(),
                    base["mmd2"] + base["mmd2"].std(),
                    color="gray",
                    alpha=0.2,
                    zorder=1,
                )

        # Treatment lines
        for lbl, color in zip(treatment_labels, palette):
            treat = sub[sub["label"] == lbl].sort_values("bin_mid")
            if treat.empty:
                continue
            sig = _bh_significance(treat["p_value"])
            ax.plot(treat["bin_mid"], treat["mmd2"], color=color, linewidth=1.2, label=lbl, zorder=2)
            sig_rows = treat[sig]
            if not sig_rows.empty:
                ax.scatter(
                    sig_rows["bin_mid"],
                    sig_rows["mmd2"],
                    color=color,
                    edgecolors="black",
                    linewidths=0.8,
                    s=40,
                    zorder=3,
                )

        ax.set_title(marker, fontsize=9)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.axhline(0, color="lightgray", linewidth=0.5, linestyle="--")
        sns.despine(ax=ax)

    # Hide unused axes
    for ax_idx in range(len(markers), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    # Shared legend
    handles, lbls = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, lbls, loc="lower center", ncol=len(treatment_labels), fontsize=9, bbox_to_anchor=(0.5, -0.02)
        )

    fig.supxlabel("Hours post perturbation (bin midpoint)", fontsize=10)
    fig.supylabel("MMD²", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_activity_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    linthresh: float = 1.0,
) -> None:
    """Plot activity z-score heatmap (markers × temporal bins).

    Uses symmetric log normalization so both small and large z-scores are
    visible. Significance stars mark FDR-corrected significant cells.

    Parameters
    ----------
    df : pd.DataFrame
        MMD results with columns: marker, label, hours_bin_start, hours_bin_end,
        activity_zscore, p_value.
    output_path : Path
        Output file path (.pdf or .png).
    linthresh : float
        Linear threshold for ``SymLogNorm``. Values within ``[-linthresh,
        linthresh]`` are rendered linearly; outside is log-scaled. Default: 1.0.
    """
    if "activity_zscore" not in df.columns or df["activity_zscore"].isna().all():
        return
    df = df.copy().dropna(subset=["hours_bin_start", "hours_bin_end", "activity_zscore"])
    if df.empty:
        return
    df["bin_label"] = df.apply(lambda r: f"{r.hours_bin_start:.0f}–{r.hours_bin_end:.0f}h", axis=1)

    labels = [lbl for lbl in df["label"].unique() if lbl]
    n_labels = len(labels)
    fig, axes = plt.subplots(
        1,
        n_labels,
        figsize=(max(5, len(df["bin_label"].unique()) * 1.0 * n_labels), max(4, df["marker"].nunique() * 0.6)),
        squeeze=False,
    )

    for ax, lbl in zip(axes[0], labels):
        sub = df[df["label"] == lbl]
        pivot_z = sub.pivot_table(index="marker", columns="bin_label", values="activity_zscore", aggfunc="mean")
        pivot_pval = sub.pivot_table(index="marker", columns="bin_label", values="p_value", aggfunc="min")
        bin_order = sub.drop_duplicates("bin_label").sort_values("hours_bin_start")["bin_label"].tolist()
        pivot_z = pivot_z.reindex(columns=bin_order)
        pivot_pval = pivot_pval.reindex(columns=bin_order)

        if pivot_z.empty or pivot_z.isna().all().all():
            ax.set_visible(False)
            continue

        vmax = float(np.nanmax(np.abs(pivot_z.values)))
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
        sns.heatmap(pivot_z, ax=ax, cmap="RdBu_r", norm=norm, linewidths=0.3, cbar_kws={"label": "Activity z-score"})

        sig = _bh_significance(pivot_pval.values.ravel())
        sig_matrix = sig.reshape(pivot_pval.shape)
        for r in range(sig_matrix.shape[0]):
            for c in range(sig_matrix.shape[1]):
                if sig_matrix[r, c]:
                    ax.text(
                        c + 0.5, r + 0.5, "*", ha="center", va="center", color="black", fontsize=10, fontweight="bold"
                    )

        ax.set_title(lbl)
        ax.set_xlabel("Temporal bin")
        ax.set_ylabel("Marker")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_paired_heatmaps(
    df: pd.DataFrame,
    condition_labels: list[str],
    value_col: str,
    output_path: Path,
    linthresh: float = 1.0,
) -> None:
    """Plot side-by-side heatmaps for two conditions sharing a colorbar.

    Parameters
    ----------
    df : pd.DataFrame
        MMD results. Must have columns: marker, label, hours_bin_start,
        hours_bin_end, ``value_col``, p_value.
    condition_labels : list[str]
        Exactly two comparison labels to plot side-by-side.
    value_col : str
        Column to use as heatmap values (e.g. ``"activity_zscore"``).
    output_path : Path
        Output file path.
    linthresh : float
        Linear threshold for ``SymLogNorm``. Default: 1.0.
    """
    if value_col not in df.columns or len(condition_labels) < 2:
        return
    df = df.copy().dropna(subset=["hours_bin_start", "hours_bin_end", value_col])
    if df.empty:
        return
    df["bin_label"] = df.apply(lambda r: f"{r.hours_bin_start:.0f}–{r.hours_bin_end:.0f}h", axis=1)
    bin_order = df.drop_duplicates("bin_label").sort_values("hours_bin_start")["bin_label"].tolist()

    all_vals = df[df["label"].isin(condition_labels)][value_col].dropna()
    if all_vals.empty:
        return
    vmax = float(np.nanmax(np.abs(all_vals)))
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(
        1, 2, figsize=(max(10, len(bin_order) * 2), max(4, df["marker"].nunique() * 0.6)), squeeze=False
    )

    for ax, lbl in zip(axes[0], condition_labels[:2]):
        sub = df[df["label"] == lbl]
        pivot_val = sub.pivot_table(index="marker", columns="bin_label", values=value_col, aggfunc="mean")
        pivot_pval = sub.pivot_table(index="marker", columns="bin_label", values="p_value", aggfunc="min")
        pivot_val = pivot_val.reindex(columns=bin_order)
        pivot_pval = pivot_pval.reindex(columns=bin_order)

        if pivot_val.empty or pivot_val.isna().all().all():
            ax.set_visible(False)
            continue

        im = ax.imshow(
            pivot_val.values,
            aspect="auto",
            norm=norm,
            cmap="YlOrRd",
            origin="upper",
        )
        ax.set_xticks(range(len(pivot_val.columns)))
        ax.set_xticklabels(pivot_val.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot_val.index)))
        ax.set_yticklabels(pivot_val.index, fontsize=8)
        ax.set_title(lbl)

        sig = _bh_significance(pivot_pval.values.ravel())
        sig_matrix = sig.reshape(pivot_pval.shape)
        for r in range(sig_matrix.shape[0]):
            for c in range(sig_matrix.shape[1]):
                val = pivot_val.values[r, c]
                if np.isfinite(val):
                    txt = f"{int(val)}" if abs(val) >= 1 else f"{val:.1f}"
                    if sig_matrix[r, c]:
                        txt += "*"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=7, color="black")

    plt.colorbar(im, ax=axes[0], label=value_col)
    fig.suptitle(f"{' vs '.join(condition_labels[:2])}", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mmd_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot MMD heatmap (markers x temporal bins or aggregate).

    Parameters
    ----------
    df : pd.DataFrame
        MMD results for a single treatment group.
    output_path : Path
        Output file path.
    """
    df = df.copy()
    has_bins = not df["hours_bin_start"].isna().all()

    if has_bins:
        df["bin_label"] = df.apply(lambda r: f"{r.hours_bin_start:.0f}–{r.hours_bin_end:.0f}h", axis=1)
        pivot_mmd = df.pivot_table(index="marker", columns="bin_label", values="mmd2", aggfunc="mean")
        pivot_pval = df.pivot_table(index="marker", columns="bin_label", values="p_value", aggfunc="min")
        # Order columns by bin start
        bin_order = df.drop_duplicates("bin_label").sort_values("hours_bin_start")["bin_label"].tolist()
        pivot_mmd = pivot_mmd.reindex(columns=bin_order)
        pivot_pval = pivot_pval.reindex(columns=bin_order)
        xlabel = "Temporal bin"
        figsize = (max(6, len(bin_order) * 0.8), max(4, len(pivot_mmd) * 0.6))
    else:
        pivot_mmd = df.set_index("marker")[["mmd2"]].rename(columns={"mmd2": "aggregate"})
        pivot_pval = df.set_index("marker")[["p_value"]].rename(columns={"p_value": "aggregate"})
        xlabel = ""
        figsize = (3, max(4, len(pivot_mmd) * 0.6))

    if pivot_mmd.empty or pivot_mmd.isna().all().all():
        return

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_mmd,
        ax=ax,
        cmap="viridis",
        annot=False,
        linewidths=0.5,
        cbar_kws={"label": "MMD²"},
    )

    # Add significance stars
    sig = _bh_significance(pivot_pval.values.ravel())
    sig_matrix = sig.reshape(pivot_pval.shape)
    for r in range(sig_matrix.shape[0]):
        for c in range(sig_matrix.shape[1]):
            if sig_matrix[r, c]:
                ax.text(c + 0.5, r + 0.5, "*", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    ax.set_title(f"MMD heatmap — {df['label'].iloc[0] if 'label' in df.columns else ''}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Marker")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
