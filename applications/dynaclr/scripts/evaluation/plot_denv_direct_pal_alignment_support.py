"""Plot supporting diagnostics for DENV direct-pAL temporal alignment.

The first figure shows the direct-pAL posterior before and after lineage-level
alignment, the physical-HPI distribution of sustained pAL onsets, and the
lineage support across relative time. The second figure exposes the complete
persistent organelle-onset rule: baseline window, stability interval, baseline
ceiling, and the first qualifying configured-length persistent run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_INPUT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_04_28_A549_SEC61B_DENV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_denv_pal40_phase3d_joint_tied_qc_nonblank_5dataset/"
    "direct_pal_lineage_native30_same_cohort_alignment"
)

DATASET_ORDER = (
    "04_14_SEC61B",
    "04_28_SEC61B",
    "08_11_SEC61B",
    "04_30_TOMM20",
    "04_21_G3BP1",
)
DATASET_LABELS = {
    "04_14_SEC61B": "SEC61B · 04_14",
    "04_28_SEC61B": "SEC61B · 04_28",
    "08_11_SEC61B": "SEC61B · 08_11",
    "04_30_TOMM20": "TOMM20 · 04_30",
    "04_21_G3BP1": "G3BP1 · 04_21",
}
DATASET_COLORS = {
    "04_14_SEC61B": "#56B4E9",
    "04_28_SEC61B": "#0072B2",
    "08_11_SEC61B": "#6A3D9A",
    "04_30_TOMM20": "#009E73",
    "04_21_G3BP1": "#CC79A7",
}
MARKER_ORDER = ("SEC61B", "TOMM20", "G3BP1")
MARKER_COLORS = {
    "SEC61B": "#0072B2",
    "TOMM20": "#009E73",
    "G3BP1": "#CC79A7",
}


def _load_lineage_time_states(input_path: Path) -> pd.DataFrame:
    parts = []
    for marker in MARKER_ORDER:
        frame = pd.read_parquet(
            input_path / f"{marker.lower()}_cell_manifest.parquet",
            columns=[
                "dataset",
                "marker",
                "fov_name",
                "lineage_id",
                "t",
                "hours_post_perturbation",
                "gmm_positive_posterior",
                "pal_onset_hpi",
                "tau",
                "perturbation",
            ],
        )
        frame = frame.loc[frame["perturbation"].astype(str).eq("DENV") & frame["pal_onset_hpi"].notna()]
        states = (
            frame.groupby(
                ["dataset", "marker", "fov_name", "lineage_id", "t"],
                sort=False,
                observed=True,
            )
            .agg(
                hours_post_perturbation=("hours_post_perturbation", "median"),
                gmm_positive_posterior=("gmm_positive_posterior", "median"),
                pal_onset_hpi=("pal_onset_hpi", "first"),
                tau=("tau", "median"),
            )
            .reset_index()
        )
        states["lineage_uid"] = (
            states["dataset"].astype(str)
            + "::"
            + states["fov_name"].astype(str)
            + "::"
            + states["lineage_id"].astype(str)
        )
        parts.append(states)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["dataset", "fov_name", "lineage_id", "t"]).any():
        raise ValueError("Duplicate lineage-time states after branch aggregation")
    return result


def _posterior_curve(states: pd.DataFrame, axis: str) -> pd.DataFrame:
    value = "hours_post_perturbation" if axis == "hpi" else "tau"
    working = states.loc[np.isfinite(states[value])].copy()
    working["axis_value"] = np.round(working[value].to_numpy(float) * 2.0) / 2.0
    return (
        working.groupby(["dataset", "marker", "axis_value"], observed=True)["gmm_positive_posterior"]
        .agg(
            posterior_q25=lambda x: float(np.quantile(x, 0.25)),
            posterior_median="median",
            posterior_q75=lambda x: float(np.quantile(x, 0.75)),
            n_lineages="size",
        )
        .reset_index()
        .assign(axis=axis)
    )


def _support_curve(states: pd.DataFrame) -> pd.DataFrame:
    working = states.loc[states["tau"].between(-10.0, 12.0)].copy()
    working["tau_center"] = np.round(working["tau"].to_numpy(float) * 2.0) / 2.0
    return (
        working.groupby(["dataset", "marker", "tau_center"], observed=True)["lineage_uid"]
        .nunique()
        .rename("n_lineages")
        .reset_index()
    )


def _onset_table(states: pd.DataFrame) -> pd.DataFrame:
    return states[["dataset", "marker", "lineage_uid", "pal_onset_hpi"]].drop_duplicates(["dataset", "lineage_uid"])


def _plot_alignment_qc(
    curves: pd.DataFrame,
    onsets: pd.DataFrame,
    support: pd.DataFrame,
    output: Path,
    perturbation: str = "DENV",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    hpi_ax, tau_ax, onset_ax, support_ax = axes.flat
    for axis, ax in (("hpi", hpi_ax), ("tau", tau_ax)):
        for dataset in DATASET_ORDER:
            group = curves.loc[curves["axis"].eq(axis) & curves["dataset"].eq(dataset)].sort_values("axis_value")
            x = group["axis_value"].to_numpy(float)
            ax.plot(
                x,
                group["posterior_median"],
                color=DATASET_COLORS[dataset],
                lw=2.2,
                label=DATASET_LABELS[dataset],
            )
            ax.fill_between(
                x,
                group["posterior_q25"].to_numpy(float),
                group["posterior_q75"].to_numpy(float),
                color=DATASET_COLORS[dataset],
                alpha=0.16,
                linewidth=0,
            )
        ax.axhline(0.8, color="0.25", ls=":", lw=1.2, label="positive gate (0.8)")
        ax.axhline(0.2, color="0.55", ls=":", lw=1.0, label="negative gate (0.2)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    hpi_ax.set(
        xlabel="Physical time (HPI)",
        ylabel="Direct-pAL GMM-positive posterior\nmedian and interquartile range",
        title="A  Same onset-eligible lineages before alignment",
    )
    tau_ax.axvline(0, color="black", ls="--", lw=1.3)
    tau_ax.set(
        xlabel="Hours relative to sustained pAL onset (τ)",
        ylabel="Direct-pAL GMM-positive posterior\nmedian and interquartile range",
        title="B  Direct-pAL response after lineage alignment",
        xlim=(-10, 12),
    )
    handles, labels = tau_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    tau_ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=9, ncol=2)

    onset_values = [
        onsets.loc[onsets["dataset"].eq(dataset), "pal_onset_hpi"].to_numpy(float) for dataset in DATASET_ORDER
    ]
    positions = np.arange(len(DATASET_ORDER))
    violins = onset_ax.violinplot(
        onset_values,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, dataset in zip(violins["bodies"], DATASET_ORDER, strict=True):
        body.set_facecolor(DATASET_COLORS[dataset])
        body.set_edgecolor(DATASET_COLORS[dataset])
        body.set_alpha(0.28)
    boxes = onset_ax.boxplot(
        onset_values,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for box, dataset in zip(boxes["boxes"], DATASET_ORDER, strict=True):
        box.set_facecolor(DATASET_COLORS[dataset])
        box.set_alpha(0.75)
    onset_ax.set_xticks(
        positions,
        [
            f"{DATASET_LABELS[dataset]}\n(n={len(values):,})"
            for dataset, values in zip(DATASET_ORDER, onset_values, strict=True)
        ],
        rotation=18,
        ha="right",
    )
    onset_ax.set(
        ylabel="Sustained pAL onset (HPI)",
        title="C  Physical timing of lineage pAL onset",
    )
    onset_ax.grid(axis="y", alpha=0.2)

    for dataset in DATASET_ORDER:
        group = support.loc[support["dataset"].eq(dataset)].sort_values("tau_center")
        support_ax.plot(
            group["tau_center"],
            group["n_lineages"],
            color=DATASET_COLORS[dataset],
            lw=2.0,
            label=DATASET_LABELS[dataset],
        )
    support_ax.axvline(0, color="black", ls="--", lw=1.3)
    support_ax.set(
        xlabel="Hours relative to sustained pAL onset (τ)",
        ylabel="Contributing lineages per τ bin",
        title="D  Lineage support across aligned time",
        xlim=(-10, 12),
    )
    support_ax.grid(alpha=0.2)
    fig.suptitle(
        f"Direct-pAL temporal-alignment diagnostic for {perturbation}\n"
        "Alignment sharpening is expected by construction and serves as a positive-control QC",
        fontsize=16,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_persistent_onsets(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
    persistent_bins: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.4),
        sharex=True,
        layout="constrained",
    )
    for ax, marker in zip(axes, MARKER_ORDER, strict=True):
        group = curves.loc[curves["marker"].eq(marker) & curves["axis"].eq("tau")].sort_values("axis_center")
        metric = metrics.loc[metrics["marker"].eq(marker) & metrics["axis"].eq("tau")].iloc[0]
        x = group["axis_center"].to_numpy(float)
        observed = group["null_standardized_mmd"].to_numpy(float)
        low = group["subsample_q025"].to_numpy(float)
        high = group["subsample_q975"].to_numpy(float)
        cutoff = float(metric["baseline_upper_z"])
        onset = float(metric["persistent_baseline_onset"])
        color = MARKER_COLORS[marker]
        ax.axvspan(-10, -6, color="0.75", alpha=0.28, label="baseline window")
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, observed, color=color, lw=2.2, marker="o", ms=3.0)
        ax.plot(x, low, color=color, lw=1.0, alpha=0.75, label="lower stability bound")
        ax.axhline(cutoff, color="0.25", ls=":", lw=1.4, label="baseline ceiling")
        ax.axvline(0, color="black", ls="--", lw=1.2, label="pAL onset")
        if np.isfinite(onset):
            width = float(group["bin_width"].iloc[0])
            ax.axvspan(
                onset - width / 2,
                onset + (persistent_bins - 0.5) * width,
                color="#F0E442",
                alpha=0.28,
                label=f"first {persistent_bins}-bin run",
            )
            ax.axvline(onset, color=color, ls="-.", lw=1.5)
        ax.set(
            xlabel="Hours relative to sustained pAL onset (τ)",
            title=f"{marker}: persistent onset = {onset:+g} h",
            xlim=(-10, 12),
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Null-standardized organelle MMD²\nvs date/HPI-matched controls")
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Persistent organelle-remodeling onset after direct-pAL alignment\n"
        f"Call = first {persistent_bins} consecutive 30-minute lower bounds "
        "above the early-baseline ceiling",
        fontsize=15,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    input_path = args.input
    states = _load_lineage_time_states(input_path)
    curves = pd.concat(
        [_posterior_curve(states, "hpi"), _posterior_curve(states, "tau")],
        ignore_index=True,
    )
    onsets = _onset_table(states)
    support = _support_curve(states)
    mmd_curves = pd.read_csv(input_path / "mmd_curves.csv")
    metrics = pd.read_csv(input_path / "alignment_curve_metrics.csv")

    curves.to_csv(input_path / "direct_pal_alignment_qc_curves.csv", index=False)
    onsets.to_csv(input_path / "direct_pal_alignment_qc_onsets.csv", index=False)
    support.to_csv(input_path / "direct_pal_alignment_qc_support.csv", index=False)
    _plot_alignment_qc(
        curves,
        onsets,
        support,
        input_path / "direct_pal_alignment_qc.png",
    )
    _plot_persistent_onsets(
        mmd_curves,
        metrics,
        input_path / "organelle_persistent_onset_audit.png",
        args.persistent_bins,
    )
    run_info = {
        "input": str(input_path),
        "candidate_population": "DENV lineages with a sustained direct-pAL onset",
        "branch_aggregation": "median posterior per lineage and frame",
        "posterior_summary": "median and interquartile range",
        "persistent_onset": {
            "baseline_tau_hours": [-10.0, -6.0],
            "rule": "lower stability bound exceeds baseline ceiling",
            "consecutive_bins": args.persistent_bins,
            "bin_width_hours": 0.5,
        },
        "interpretation_limit": (
            "Aligned pAL sharpening is expected by construction and is an alignment QC, "
            "not independent evidence of biological prediction"
        ),
        "outputs": [
            "direct_pal_alignment_qc.png",
            "organelle_persistent_onset_audit.png",
        ],
    }
    (input_path / "support_figure_run_info.json").write_text(json.dumps(run_info, indent=2))
    print(input_path / "direct_pal_alignment_qc.png")
    print(input_path / "organelle_persistent_onset_audit.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--persistent-bins", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
