"""Compare ZIKV organelle dynamics per dataset and in the capped aggregate.

The upstream direct-pAL analysis stores fixed marker PCA coordinates, a fixed
marker bandwidth, and per-cell lineage metadata. This audit leaves those
artifacts frozen, reconstructs the stored 07_01 MMD curves as a code-contract
check, computes each marker-specific replicate by itself, and compares:

1. the common 07_01 dataset;
2. the marker-specific replicate alone; and
3. the existing per-dataset-capped aggregate.

The output is descriptive. Stability bands come from repeated 80% family
subsamples and are not biological-replicate confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import analyze_direct_pal_aligned_organelle_mmd as core
import matplotlib.pyplot as plt
import quantify_pal_alignment_gain as quantify

DEFAULT_INPUT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_zikv_pal17_phase3d_joint_tied_qc_nonblank/"
    "direct_pal_lineage_native30_same_cohort_alignment"
)

MARKER_ORDER = ("SEC61B", "TOMM20", "G3BP1")
COMMON_DATASET = "07_01_SEC61B_TOMM20_G3BP1"
REPLICATE_DATASETS = {
    "SEC61B": "03_24_SEC61B",
    "TOMM20": "04_10_TOMM20",
    "G3BP1": "04_08_G3BP1",
}
SCOPE_ORDER = ("common_07_01", "replicate_only", "replicate_augmented")
SCOPE_LABELS = {
    "common_07_01": "07_01 only",
    "replicate_only": "marker replicate only",
    "replicate_augmented": "per-dataset-capped aggregate",
}
SCOPE_COLORS = {
    "common_07_01": "#6C6C6C",
    "replicate_only": "#E69F00",
    "replicate_augmented": "#0072B2",
}
MARKER_COLORS = {
    "SEC61B": "#0072B2",
    "TOMM20": "#009E73",
    "G3BP1": "#CC79A7",
}
RECONSTRUCTION_COLUMNS = (
    "mmd2_unbiased",
    "null_mean",
    "null_std",
    "null_standardized_mmd",
    "subsample_q025",
    "subsample_median",
    "subsample_q975",
)


def _core_args(run_info: dict[str, object]) -> Namespace:
    return Namespace(**run_info["arguments"])


def _metric_args(run_info: dict[str, object]) -> Namespace:
    details = run_info.get("alignment_gain_postprocess", {})
    return Namespace(
        spline_knots=int(details.get("spline_knots", 6)),
        ridge_alpha=float(details.get("ridge_alpha", 0.001)),
        tau_baseline_start=float(details.get("tau_baseline", [-10.0, -6.0])[0]),
        tau_baseline_end=float(details.get("tau_baseline", [-10.0, -6.0])[1]),
        hpi_baseline_hours=float(details.get("hpi_baseline_hours_from_first_point", 4.0)),
        persistent_bins=int(details.get("persistent_bins", 3)),
    )


def _scope_contract() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        "common_07_01": {marker: (COMMON_DATASET,) for marker in MARKER_ORDER},
        "replicate_augmented": {marker: (COMMON_DATASET, REPLICATE_DATASETS[marker]) for marker in MARKER_ORDER},
        "replicate_only": {marker: (REPLICATE_DATASETS[marker],) for marker in MARKER_ORDER},
    }


def _bandwidths(existing: pd.DataFrame) -> dict[str, float]:
    result = {}
    for marker, group in existing.groupby("marker", observed=True):
        values = group["bandwidth"].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"{marker}: expected one frozen bandwidth, got {values}")
        result[str(marker)] = float(values[0])
    return result


def _reconstruction_row(
    marker: str,
    computed: pd.DataFrame,
    stored: pd.DataFrame,
) -> dict[str, float | int | str]:
    left = computed.sort_values("axis_center").reset_index(drop=True)
    right = stored.sort_values("axis_center").reset_index(drop=True)
    if not np.array_equal(left["axis_center"], right["axis_center"]):
        raise ValueError(f"{marker}: reconstructed 07_01 bin centers differ")
    row: dict[str, float | int | str] = {
        "marker": marker,
        "n_bins": len(left),
    }
    for column in RECONSTRUCTION_COLUMNS:
        error = float(np.nanmax(np.abs(left[column].to_numpy(float) - right[column].to_numpy(float))))
        row[f"max_abs_error__{column}"] = error
        if error > 1e-6:
            raise ValueError(f"{marker}: stored 07_01 reconstruction failed for {column}: {error}")
    return row


def _compute_replicate_curves(
    input_path: Path,
    existing: pd.DataFrame,
    run_info: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    args = _core_args(run_info)
    scopes = _scope_contract()
    core.SCOPE_DATASETS = scopes
    centers = np.arange(
        args.tau_min,
        args.tau_max + args.tau_bin_hours / 2,
        args.tau_bin_hours,
    )
    bandwidths = _bandwidths(existing)
    replicate_curves = []
    replicate_samples = []
    reconstruction = []
    for marker in MARKER_ORDER:
        scores = np.load(input_path / f"{marker.lower()}_pca80_scores.npy", mmap_mode="r")
        frame = pd.read_parquet(input_path / f"{marker.lower()}_cell_manifest.parquet")
        for scope in ("common_07_01", "replicate_only"):
            curve, samples = core._curve(
                scores,
                frame,
                marker=marker,
                axis="tau",
                scope=scope,
                datasets=scopes[scope][marker],
                centers=centers,
                width=args.tau_bin_hours,
                bandwidth=bandwidths[marker],
                args=args,
            )
            if scope == "common_07_01":
                stored = existing.loc[
                    existing["marker"].eq(marker) & existing["axis"].eq("tau") & existing["scope"].eq(scope)
                ]
                reconstruction.append(_reconstruction_row(marker, curve, stored))
            else:
                replicate_curves.append(curve)
                replicate_samples.append(samples)
    return (
        pd.concat(replicate_curves, ignore_index=True),
        pd.concat(replicate_samples, ignore_index=True),
        pd.DataFrame(reconstruction),
    )


def _metric_table(curves: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    rows = []
    for (marker, scope), group in curves.groupby(["marker", "scope"], observed=True):
        rows.append(
            {
                "marker": marker,
                "scope": scope,
                "dataset": (
                    REPLICATE_DATASETS[marker]
                    if scope == "replicate_only"
                    else COMMON_DATASET
                    if scope == "common_07_01"
                    else f"{COMMON_DATASET}+{REPLICATE_DATASETS[marker]}"
                ),
                "n_bins": len(group),
                **quantify._curve_metrics(group, args),
            }
        )
    return pd.DataFrame(rows)


def _normalized_curve(
    group: pd.DataFrame,
    args: Namespace,
    grid: np.ndarray,
) -> np.ndarray:
    group = group.sort_values("axis_center")
    x = group["axis_center"].to_numpy(float)
    y = group["null_standardized_mmd"].to_numpy(float)
    model = quantify._model(args.spline_knots, args.ridge_alpha).fit(x[:, None], y)
    fitted = model.predict(grid[:, None])
    baseline_mask = (grid >= args.tau_baseline_start) & (grid <= args.tau_baseline_end)
    baseline = float(np.median(fitted[baseline_mask]))
    amplitude = float(np.quantile(fitted, 0.95) - baseline)
    if amplitude <= 0:
        return np.full_like(fitted, np.nan)
    return (fitted - baseline) / amplitude


def _agreement_table(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    args: Namespace,
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    grid = np.linspace(-10.0, 12.0, 441)
    normalized = {}
    rows = []
    for marker in MARKER_ORDER:
        for scope in SCOPE_ORDER:
            group = curves.loc[curves["marker"].eq(marker) & curves["scope"].eq(scope)]
            normalized[(marker, scope)] = (
                grid,
                _normalized_curve(group, args, grid),
            )
        common = normalized[(marker, "common_07_01")][1]
        replicate = normalized[(marker, "replicate_only")][1]
        aggregate = normalized[(marker, "replicate_augmented")][1]
        calls = metrics.loc[metrics["marker"].eq(marker)].set_index("scope")["persistent_baseline_onset"].to_dict()
        transition_t10 = metrics.loc[metrics["marker"].eq(marker)].set_index("scope")["transition_t10"].to_dict()
        rows.append(
            {
                "marker": marker,
                "common_onset_tau": calls["common_07_01"],
                "replicate_onset_tau": calls["replicate_only"],
                "aggregate_onset_tau": calls["replicate_augmented"],
                "replicate_minus_common_onset_hours": (calls["replicate_only"] - calls["common_07_01"]),
                "common_transition_t10_tau": transition_t10["common_07_01"],
                "replicate_transition_t10_tau": transition_t10["replicate_only"],
                "aggregate_transition_t10_tau": transition_t10["replicate_augmented"],
                "replicate_minus_common_t10_hours": (transition_t10["replicate_only"] - transition_t10["common_07_01"]),
                "common_replicate_curve_pearson_r": float(np.corrcoef(common, replicate)[0, 1]),
                "common_replicate_normalized_rmse": float(np.sqrt(np.mean((common - replicate) ** 2))),
                "common_aggregate_curve_pearson_r": float(np.corrcoef(common, aggregate)[0, 1]),
                "replicate_aggregate_curve_pearson_r": float(np.corrcoef(replicate, aggregate)[0, 1]),
            }
        )
    return pd.DataFrame(rows), normalized


def _plot_decision_audit(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
    persistent_bins: int,
) -> None:
    fig, axes = plt.subplots(
        len(MARKER_ORDER),
        len(SCOPE_ORDER),
        figsize=(16, 12),
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    for row, marker in enumerate(MARKER_ORDER):
        for column, scope in enumerate(SCOPE_ORDER):
            ax = axes[row, column]
            group = curves.loc[curves["marker"].eq(marker) & curves["scope"].eq(scope)].sort_values("axis_center")
            metric = metrics.loc[metrics["marker"].eq(marker) & metrics["scope"].eq(scope)].iloc[0]
            x = group["axis_center"].to_numpy(float)
            observed = group["null_standardized_mmd"].to_numpy(float)
            low = group["subsample_q025"].to_numpy(float)
            high = group["subsample_q975"].to_numpy(float)
            onset = float(metric["persistent_baseline_onset"])
            ceiling = float(metric["baseline_upper_z"])
            color = MARKER_COLORS[marker]
            ax.axvspan(-10, -6, color="0.75", alpha=0.28)
            ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, observed, color=color, lw=2.0, marker="o", ms=2.5)
            ax.plot(x, low, color=color, lw=0.9, alpha=0.75)
            ax.axhline(ceiling, color="0.25", ls=":", lw=1.2)
            ax.axvline(0, color="black", ls="--", lw=1.1)
            if np.isfinite(onset):
                width = float(group["bin_width"].iloc[0])
                ax.axvspan(
                    onset - width / 2,
                    onset + (persistent_bins - 0.5) * width,
                    color="#F0E442",
                    alpha=0.28,
                )
                ax.axvline(onset, color=color, ls="-.", lw=1.4)
                onset_label = f"{onset:+g} h"
            else:
                onset_label = "no call"
            dataset_label = REPLICATE_DATASETS[marker] if scope == "replicate_only" else SCOPE_LABELS[scope]
            ax.set_title(f"{marker} · {dataset_label}\nonset {onset_label}")
            ax.grid(alpha=0.18)
            if row == len(MARKER_ORDER) - 1:
                ax.set_xlabel("Hours relative to sustained pAL onset (τ)")
            if column == 0:
                ax.set_ylabel("Null-standardized MMD²")
    fig.suptitle(
        "ZIKV organelle dynamics per dataset and in the capped aggregate\n"
        "Gray = baseline; yellow = first qualifying three-bin run; bands = family-subsample stability",
        fontsize=16,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_normalized_dynamics(
    normalized: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    metrics: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, layout="constrained")
    for ax, marker in zip(axes, MARKER_ORDER, strict=True):
        for scope in SCOPE_ORDER:
            grid, curve = normalized[(marker, scope)]
            ax.plot(
                grid,
                curve,
                color=SCOPE_COLORS[scope],
                lw=2.2,
                label=SCOPE_LABELS[scope],
            )
            onset = metrics.loc[
                metrics["marker"].eq(marker) & metrics["scope"].eq(scope),
                "persistent_baseline_onset",
            ].iloc[0]
            if np.isfinite(onset):
                ax.axvline(
                    onset,
                    color=SCOPE_COLORS[scope],
                    ls=":",
                    lw=1.1,
                    alpha=0.85,
                )
        ax.axvline(0, color="black", ls="--", lw=1.1)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set(
            title=marker,
            xlabel="Hours relative to sustained pAL onset (τ)",
            xlim=(-10, 12),
        )
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Baseline-to-peak normalized spline")
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Per-dataset versus aggregate trajectory shape\n"
        "Normalization removes MMD sample-size scale; dotted lines are persistent calls",
        fontsize=15,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    input_path = args.input
    run_info = json.loads((input_path / "run_info.json").read_text())
    existing = pd.read_csv(input_path / "mmd_curves.csv")
    existing_tau = existing.loc[
        existing["axis"].eq("tau") & existing["scope"].isin(("common_07_01", "replicate_augmented"))
    ].copy()
    replicate, samples, reconstruction = _compute_replicate_curves(input_path, existing, run_info)
    curves = pd.concat([existing_tau, replicate], ignore_index=True)
    metric_args = _metric_args(run_info)
    metrics = _metric_table(curves, metric_args)
    agreement, normalized = _agreement_table(curves, metrics, metric_args)

    curves.to_csv(input_path / "organelle_per_dataset_mmd_curves.csv", index=False)
    samples.to_parquet(input_path / "organelle_per_dataset_sample_manifest.parquet", index=False)
    metrics.to_csv(input_path / "organelle_per_dataset_metrics.csv", index=False)
    agreement.to_csv(input_path / "organelle_per_dataset_agreement.csv", index=False)
    reconstruction.to_csv(input_path / "organelle_per_dataset_reconstruction.csv", index=False)
    _plot_decision_audit(
        curves,
        metrics,
        input_path / "organelle_per_dataset_persistent_onset_audit.png",
        metric_args.persistent_bins,
    )
    _plot_normalized_dynamics(
        normalized,
        metrics,
        input_path / "organelle_per_dataset_normalized_dynamics.png",
    )

    output_info = {
        "input": str(input_path),
        "fixed_upstream": [
            "control normalization",
            "marker PCA coordinates",
            "marker bandwidth",
            "direct-pAL lineage onset",
        ],
        "scope_contract": _scope_contract(),
        "aggregate": (
            "one pooled MMD after sampling at most 180 matched families per dataset; "
            "not an average of independently estimated dataset curves"
        ),
        "stability_bands": "50 repeated 80% family subsamples; not confidence intervals",
        "reconstruction_tolerance": 1e-6,
        "outputs": [
            "organelle_per_dataset_mmd_curves.csv",
            "organelle_per_dataset_metrics.csv",
            "organelle_per_dataset_agreement.csv",
            "organelle_per_dataset_reconstruction.csv",
            "organelle_per_dataset_persistent_onset_audit.png",
            "organelle_per_dataset_normalized_dynamics.png",
        ],
    }
    (input_path / "organelle_per_dataset_run_info.json").write_text(json.dumps(output_info, indent=2))
    print(metrics[["marker", "scope", "persistent_baseline_onset"]].to_string(index=False))
    print(agreement.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
