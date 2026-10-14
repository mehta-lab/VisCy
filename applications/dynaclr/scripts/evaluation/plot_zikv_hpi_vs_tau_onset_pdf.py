"""Export vector ZIKV HPI-versus-pAL-aligned MMD figures with onset lines.

This is a plotting-only postprocessor. It reads the frozen native-30-minute,
same-lineage-cohort MMD curves and their persistent baseline-divergence calls;
it does not refit normalization, PCA, MMD, or onset detection.
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_INPUT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_zikv_pal17_phase3d_joint_tied_qc_nonblank/"
    "direct_pal_lineage_native30_same_cohort_alignment"
)
MARKERS = ("TOMM20", "SEC61B", "G3BP1")
COLORS = {"TOMM20": "#009E73", "SEC61B": "#0072B2", "G3BP1": "#CC79A7"}
SCOPE_TITLES = {
    "common_07_01": "Shared 07_01 acquisition",
    "replicate_augmented": ("Per-marker pooled datasets: 07_01 + 03_24 SEC61B / 04_10 TOMM20 / 04_08 G3BP1"),
}
SCOPE_FILENAMES = {
    "common_07_01": "common_07_01",
    "replicate_augmented": "marker_specific_pooled",
}
COMMON_DATASET = "07_01_SEC61B_TOMM20_G3BP1"
SCOPE_DATASETS = {
    "common_07_01": {marker: (COMMON_DATASET,) for marker in MARKERS},
    "replicate_augmented": None,
}
PERTURBATION = "ZIKV"
VIRUS_DISPLAY_NAME = "ZIKV"
OUTPUT_PREFIX = "zikv"


def _validate(curves: pd.DataFrame, metrics: pd.DataFrame) -> None:
    required_curves = {
        "marker",
        "axis",
        "scope",
        "axis_center",
        "bin_width",
        "null_standardized_mmd",
        "subsample_q025",
        "subsample_q975",
    }
    required_metrics = {
        "marker",
        "axis",
        "scope",
        "persistent_baseline_onset",
        "baseline_upper_z",
    }
    missing_curves = required_curves - set(curves.columns)
    missing_metrics = required_metrics - set(metrics.columns)
    if missing_curves or missing_metrics:
        raise ValueError(f"Missing columns: curves={sorted(missing_curves)}, metrics={sorted(missing_metrics)}")
    if not np.allclose(curves["bin_width"].to_numpy(float), 0.5):
        raise ValueError("Expected native 30-minute (0.5-hour) bins")


def _load_pal_curves(input_path: Path) -> pd.DataFrame:
    """Collapse descendants per lineage/time, then summarize pAL posterior."""
    parts: list[pd.DataFrame] = []
    for marker in MARKERS:
        frame = pd.read_parquet(
            input_path / f"{marker.lower()}_cell_manifest.parquet",
            columns=[
                "dataset",
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
        frame = frame.loc[frame["perturbation"].astype(str).eq(PERTURBATION) & frame["pal_onset_hpi"].notna()]
        states = (
            frame.groupby(
                ["dataset", "fov_name", "lineage_id", "t"],
                sort=False,
                observed=True,
            )
            .agg(
                hours_post_perturbation=("hours_post_perturbation", "median"),
                tau=("tau", "median"),
                gmm_positive_posterior=("gmm_positive_posterior", "median"),
            )
            .reset_index()
        )
        for scope in SCOPE_TITLES:
            scope_datasets = SCOPE_DATASETS[scope]
            scoped = states if scope_datasets is None else states.loc[states["dataset"].isin(scope_datasets[marker])]
            for axis_name, value_column in (
                ("hpi", "hours_post_perturbation"),
                ("tau", "tau"),
            ):
                working = scoped.loc[np.isfinite(scoped[value_column])].copy()
                working["axis_value"] = np.round(working[value_column].to_numpy(float) * 2.0) / 2.0
                curve = (
                    working.groupby("axis_value", observed=True)["gmm_positive_posterior"]
                    .agg(
                        posterior_q25=lambda x: float(np.quantile(x, 0.25)),
                        posterior_median="median",
                        posterior_q75=lambda x: float(np.quantile(x, 0.75)),
                        n_lineages="size",
                    )
                    .reset_index()
                )
                curve.insert(0, "axis", axis_name)
                curve.insert(0, "scope", scope)
                curve.insert(0, "marker", marker)
                parts.append(curve)
    return pd.concat(parts, ignore_index=True)


def _make_figure(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    pal_curves: pd.DataFrame,
    scope: str,
    persistent_bins: int,
) -> tuple[plt.Figure, list[dict[str, object]]]:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.8),
        sharey=True,
    )
    onset_rows: list[dict[str, object]] = []
    onset_lookup: dict[tuple[str, str], float] = {}
    for axis_plot, axis_name in zip(axes, ("hpi", "tau"), strict=True):
        pal_axis = axis_plot.twinx()
        subset = curves.loc[curves["axis"].eq(axis_name) & curves["scope"].eq(scope)]
        if axis_name == "tau":
            baseline_start, baseline_end = -10.0, -6.0
        else:
            baseline_start = float(subset["axis_center"].min())
            baseline_end = baseline_start + 4.0
        axis_plot.axvspan(
            baseline_start,
            baseline_end,
            color="0.75",
            alpha=0.22,
            linewidth=0,
            zorder=0,
        )
        for marker in MARKERS:
            group = subset.loc[subset["marker"].eq(marker)].sort_values("axis_center")
            if group.empty:
                raise ValueError(f"No {scope}/{axis_name}/{marker} curve")
            x = group["axis_center"].to_numpy(float)
            y = group["null_standardized_mmd"].to_numpy(float)
            low = group["subsample_q025"].to_numpy(float)
            high = group["subsample_q975"].to_numpy(float)
            axis_plot.fill_between(
                x,
                low,
                high,
                color=COLORS[marker],
                alpha=0.22,
                linewidth=0,
            )
            axis_plot.plot(
                x,
                y,
                color=COLORS[marker],
                lw=2.2,
                marker="o",
                ms=2.5,
                label=marker,
            )
            pal_group = pal_curves.loc[
                pal_curves["marker"].eq(marker)
                & pal_curves["axis"].eq(axis_name)
                & pal_curves["scope"].eq(scope)
                & pal_curves["axis_value"].between(float(x.min()), float(x.max()))
            ].sort_values("axis_value")
            if pal_group.empty:
                raise ValueError(f"No pAL curve for {scope}/{axis_name}/{marker}")
            pal_axis.plot(
                pal_group["axis_value"],
                pal_group["posterior_median"],
                color=COLORS[marker],
                linestyle="--",
                linewidth=1.5,
                alpha=0.72,
                zorder=2,
            )
            metric = metrics.loc[
                metrics["marker"].eq(marker) & metrics["axis"].eq(axis_name) & metrics["scope"].eq(scope)
            ]
            if len(metric) != 1:
                raise ValueError(f"Expected one onset row for {scope}/{axis_name}/{marker}; found {len(metric)}")
            onset = float(metric.iloc[0]["persistent_baseline_onset"])
            baseline_ceiling = float(metric.iloc[0]["baseline_upper_z"])
            onset_lookup[(axis_name, marker)] = onset
            axis_plot.axhline(
                baseline_ceiling,
                color=COLORS[marker],
                linestyle="-.",
                linewidth=1.0,
                alpha=0.78,
                zorder=3,
            )
            if np.isfinite(onset):
                axis_plot.axvline(
                    onset,
                    color=COLORS[marker],
                    linestyle=":",
                    linewidth=1.7,
                    alpha=0.95,
                    zorder=4,
                )
            onset_rows.append(
                {
                    "scope": scope,
                    "axis": axis_name,
                    "marker": marker,
                    "persistent_baseline_onset_hours": onset,
                }
            )

        axis_plot.axhline(0, color="0.35", lw=0.9)
        if axis_name == "tau":
            axis_plot.axvline(
                0,
                color="black",
                linestyle="--",
                linewidth=1.3,
                zorder=3,
            )
            axis_plot.set(
                title="Aligned response time (τ)",
                xlabel="Hours relative to sustained direct-pAL onset",
            )
        else:
            axis_plot.set(
                title="Unaligned physical time (HPI)",
                xlabel="Hours post infection",
            )
        axis_plot.grid(alpha=0.18)
        pal_axis.set_ylim(-0.03, 1.03)
        pal_axis.set_yticks([0.0, 0.2, 0.5, 0.8, 1.0])
        pal_axis.tick_params(axis="y", colors="0.35", labelsize=8)
        if axis_name == "tau":
            pal_axis.set_ylabel("Direct-pAL positive posterior", color="0.3")
        else:
            pal_axis.set_yticklabels([])
    axes[0].set_ylabel("Null-standardized MMD² vs exact-HPI matched controls")
    curve_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[marker],
            lw=2.2,
            label=(f"{marker}: {onset_lookup[('hpi', marker)]:g} HPI; {onset_lookup[('tau', marker)]:+g} τ"),
        )
        for marker in MARKERS
    ]
    reference_handles = [
        Line2D(
            [0],
            [0],
            color="0.3",
            lw=1.5,
            ls="--",
            label="direct-pAL posterior (right axis)",
        ),
        Patch(
            facecolor="0.4",
            alpha=0.22,
            label="95% family-subsample stability band",
        ),
        Patch(facecolor="0.75", alpha=0.22, label="early baseline window"),
        Line2D(
            [0],
            [0],
            color="0.3",
            lw=1.0,
            ls="-.",
            label="marker baseline ceiling",
        ),
        Line2D([0], [0], color="0.25", lw=1.7, ls=":", label="persistent organelle threshold"),
        Line2D([0], [0], color="black", lw=1.3, ls="--", label="pAL onset (τ=0)"),
    ]
    figure.legend(
        handles=curve_handles + reference_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        fontsize=7.5,
        ncol=3,
    )
    figure.suptitle(
        f"{VIRUS_DISPLAY_NAME} organelle divergence before and after direct-pAL alignment\n" + SCOPE_TITLES[scope],
        fontsize=15,
        y=0.97,
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.28, top=0.83, wspace=0.04)
    figure.text(
        0.5,
        0.025,
        "Colored shading = 95% family-subsample stability; gray = baseline window; "
        f"horizontal dash-dot = baseline ceiling; vertical dotted = first qualifying "
        f"{persistent_bins}-bin run.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.25",
    )
    return figure, onset_rows


def run(args: argparse.Namespace) -> None:
    curves = pd.read_csv(args.input / "mmd_curves.csv")
    metrics = pd.read_csv(args.input / "alignment_curve_metrics.csv")
    pal_curves = _load_pal_curves(args.input)
    run_info = json.loads((args.input / "run_info.json").read_text())
    persistent_bins = int(run_info.get("alignment_gain_postprocess", {}).get("persistent_bins", 3))
    _validate(curves, metrics)
    scopes = tuple(args.scopes)
    invalid = set(scopes) - set(SCOPE_TITLES)
    if invalid:
        raise ValueError(f"Unsupported scopes: {sorted(invalid)}")

    onset_rows: list[dict[str, object]] = []
    combined_path = args.input / f"{OUTPUT_PREFIX}_unaligned_vs_pal_aligned_onsets.pdf"
    with PdfPages(
        combined_path,
        metadata={
            "Title": f"{VIRUS_DISPLAY_NAME} organelle divergence before and after direct-pAL alignment",
            "Subject": "Native 30-minute, same-lineage-cohort MMD with persistent onset calls",
        },
    ) as pdf:
        for scope in scopes:
            figure, figure_onset_rows = _make_figure(curves, metrics, pal_curves, scope, persistent_bins)
            onset_rows.extend(figure_onset_rows)
            stem = f"{OUTPUT_PREFIX}_unaligned_vs_pal_aligned_onsets__{SCOPE_FILENAMES[scope]}"
            figure.savefig(args.input / f"{stem}.pdf", bbox_inches="tight")
            if args.png_preview:
                figure.savefig(args.input / f"{stem}.png", dpi=220, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    pd.DataFrame(onset_rows).to_csv(
        args.input / f"{OUTPUT_PREFIX}_unaligned_vs_pal_aligned_onsets.csv",
        index=False,
    )
    pal_curves.to_csv(
        args.input / f"{OUTPUT_PREFIX}_unaligned_vs_pal_aligned_pal_overlay.csv",
        index=False,
    )
    print(combined_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=list(SCOPE_TITLES),
        choices=tuple(SCOPE_TITLES),
    )
    parser.add_argument("--png-preview", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
