"""Sweep confidence gates for the fixed pooled ZIKV pAL witness GMM.

The PCA transforms, control normalization scales, witness scores, and joint/tied
GMM posterior are reused from the canonical 0.8 run.  For every requested
posterior threshold, leave-one-dataset-out pAL->pAL and pAL->Phase classifiers
are retrained using only the resulting confident ZIKV-well pseudo-labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluate_pooled_zikv_pal17_phase3d import (
    DATASETS,
    KEYS,
    SENSORS,
    _evaluate_classifiers,
    _teacher_summary,
    _transform_all,
)

DEFAULT_BASE = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_zikv_pal17_phase3d_joint_tied"
)


def _threshold_name(threshold: float) -> str:
    return f"p{threshold:.2f}".replace(".", "p")


def _state_from_posterior(
    obs: pd.DataFrame,
    posterior: np.ndarray,
    threshold: float,
    *,
    gate_controls: bool = False,
) -> np.ndarray:
    perturbation = obs["perturbation"].astype(str).to_numpy()
    control = perturbation == "uninfected"
    zikv = perturbation == "ZIKV"
    state = np.full(len(obs), np.nan, dtype=float)
    if gate_controls:
        state[posterior >= threshold] = 1.0
        state[posterior <= 1.0 - threshold] = 0.0
        return state
    state[control] = 0.0
    state[zikv & (posterior >= threshold)] = 1.0
    state[zikv & (posterior <= 1.0 - threshold)] = 0.0
    return state


def _posterior_boundary(gmm, remod_component: int, threshold: float) -> float:
    means = gmm.means_.ravel()
    if gmm.covariance_type == "tied":
        stds = np.full(2, np.sqrt(gmm.covariances_[0, 0]))
    else:
        stds = np.sqrt(gmm.covariances_.reshape(2, -1)[:, 0])
    grid = np.linspace(float(np.min(means - 8 * stds)), float(np.max(means + 8 * stds)), 200_000)
    posterior = gmm.predict_proba(grid[:, None])[:, remod_component]
    return float(grid[np.argmin(np.abs(posterior - threshold))])


def _hpi_summary(
    obs: pd.DataFrame,
    posterior: np.ndarray,
    oof_phase: np.ndarray,
    threshold: float,
    bin_hours: float,
) -> pd.DataFrame:
    frame = obs[["dataset", "perturbation", "hours_post_perturbation"]].copy()
    frame["hpi_bin"] = np.round(frame["hours_post_perturbation"].to_numpy(float) / bin_hours) * bin_hours
    frame["teacher_positive"] = posterior >= threshold
    frame["teacher_confident"] = (posterior >= threshold) | (posterior <= 1.0 - threshold)
    frame["phase_positive"] = oof_phase >= 0.5
    frame["phase_available"] = np.isfinite(oof_phase)

    rows = []
    for keys, group in frame.groupby(["dataset", "perturbation", "hpi_bin"], observed=True, sort=True):
        confident = group["teacher_confident"].to_numpy(bool)
        phase_available = group["phase_available"].to_numpy(bool)
        rows.append(
            {
                "threshold": threshold,
                "dataset": keys[0],
                "perturbation": str(keys[1]),
                "hpi_bin": float(keys[2]),
                "n_cells": len(group),
                "teacher_positive_pct_all": 100.0 * float(group["teacher_positive"].mean()),
                "teacher_positive_pct_confident": 100.0 * float(group.loc[confident, "teacher_positive"].mean())
                if confident.any()
                else np.nan,
                "teacher_confident_coverage_pct": 100.0 * float(confident.mean()),
                "phase_positive_pct": 100.0 * float(group.loc[phase_available, "phase_positive"].mean())
                if phase_available.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_summary(teacher: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    datasets = [spec.name for spec in DATASETS] + ["pooled"]
    metric_datasets = [spec.name for spec in DATASETS] + ["pooled_oof"]
    thresholds = sorted(teacher["threshold"].unique())
    colors = dict(zip(thresholds, ("#0072B2", "#E69F00", "#CC79A7"), strict=True))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for threshold in thresholds:
        part = teacher[teacher["threshold"] == threshold].set_index("dataset")
        coverage = [100 * (1 - part.loc[d, "n_zikv_abstain"] / part.loc[d, "n_zikv"]) for d in datasets]
        axes[0, 0].plot(datasets, coverage, marker="o", color=colors[threshold], label=f"{threshold:.2f}")
        axes[1, 0].plot(
            datasets,
            100 * part.loc[datasets, "control_gmm_false_positive_rate"],
            marker="o",
            color=colors[threshold],
        )

        phase = metrics[(metrics["threshold"] == threshold) & (metrics["sensor"] == "Phase3D")].set_index(
            "evaluation_dataset"
        )
        axes[0, 1].plot(
            metric_datasets,
            100 * phase.loc[metric_datasets, "balanced_accuracy"],
            marker="o",
            color=colors[threshold],
        )
        axes[1, 1].plot(
            metric_datasets,
            phase.loc[metric_datasets, "auroc"],
            marker="o",
            color=colors[threshold],
        )

    titles = (
        "Confident ZIKV-label coverage",
        "pAL→Phase balanced accuracy",
        "Control teacher false-positive rate",
        "pAL→Phase AUROC",
    )
    ylabels = ("percent", "percent", "percent", "AUROC")
    for ax, title, ylabel in zip(axes.ravel(), titles, ylabels, strict=True):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(title="posterior gate", frameon=False)
    fig.suptitle("Fixed joint/tied GMM: posterior confidence-gate sweep")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_hpi(hpi: pd.DataFrame, output: Path) -> None:
    thresholds = sorted(hpi["threshold"].unique())
    colors = dict(zip(thresholds, ("#0072B2", "#E69F00", "#CC79A7"), strict=True))
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13, 3.2 * len(DATASETS)), squeeze=False)
    for row, spec in enumerate(DATASETS):
        subset = hpi[(hpi["dataset"] == spec.name) & (hpi["perturbation"] == "ZIKV")]
        for threshold in thresholds:
            part = subset[subset["threshold"] == threshold]
            axes[row, 0].plot(
                part["hpi_bin"], part["teacher_positive_pct_all"], color=colors[threshold], label=f"{threshold:.2f}"
            )
            axes[row, 1].plot(part["hpi_bin"], part["teacher_confident_coverage_pct"], color=colors[threshold])
        axes[row, 0].set_ylabel(f"{spec.name}\npercent")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    axes[0, 0].set_title("Teacher-positive ZIKV-well cells / all cells")
    axes[0, 1].set_title("Confident-label coverage in ZIKV wells")
    axes[0, 0].legend(title="posterior gate", frameon=False)
    axes[-1, 0].set_xlabel("HPI (2-hour bins)")
    axes[-1, 1].set_xlabel("HPI (2-hour bins)")
    fig.suptitle("Effect of broadening the GMM abstention interval")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_confusions(metrics: pd.DataFrame, output: Path) -> None:
    phase = metrics[(metrics["sensor"] == "Phase3D") & (metrics["evaluation_dataset"] == "pooled_oof")].sort_values(
        "threshold"
    )
    fig, axes = plt.subplots(1, len(phase), figsize=(4.2 * len(phase), 4), squeeze=False)
    for ax, (_, row) in zip(axes.ravel(), phase.iterrows(), strict=True):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=float)
        matrix /= matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{100 * matrix[i, j]:.1f}%", ha="center", va="center")
        ax.set_title(f"gate={row.threshold:.2f}\nn={int(row.n):,}")
        ax.set_xticks([0, 1], ["Phase −", "Phase +"])
        ax.set_yticks([0, 1], ["Teacher −", "Teacher +"])
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    fig.suptitle("Pooled out-of-fold pAL→Phase agreement on confident ZIKV cells")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    base = args.base_dir
    final_output = args.output_dir or base / "posterior_threshold_sweep"
    working = final_output.with_name(final_output.name + ".in_progress")
    if final_output.exists() or working.exists():
        raise FileExistsError(f"Refusing to overwrite {final_output} or {working}")
    working.mkdir(parents=True)

    canonical = pd.read_parquet(base / "leave_one_dataset_out_predictions.parquet")
    pcas = {sensor: joblib.load(base / "models" / f"pooled_{sensor}_pca80.joblib") for sensor in SENSORS}
    scale_file = np.load(base / "models" / "per_dataset_control_mad_scales.npz")
    scales = {sensor: {spec.name: scale_file[f"{spec.name}__{sensor}"] for spec in DATASETS} for sensor in SENSORS}
    transform_args = argparse.Namespace(
        smooth_sigma_timepoints=args.smooth_sigma_timepoints,
        scale_floor_quantile=args.scale_floor_quantile,
    )
    features, obs, _ = _transform_all(DATASETS, pcas, scales, transform_args)
    identity = ["dataset"] + KEYS
    if not canonical[identity].reset_index(drop=True).equals(obs[identity].reset_index(drop=True)):
        raise ValueError("Canonical predictions and reconstructed feature rows differ")
    posterior = canonical["gmm_positive_posterior"].to_numpy(float)
    witness = canonical["witness_score"].to_numpy(float)
    gmm = joblib.load(base / "models" / "pooled_pal17_witness_joint_tied_gmm.joblib")
    remod = int(np.argmin(gmm.means_.ravel()))

    metrics_parts = []
    teacher_parts = []
    hpi_parts = []
    boundary_rows = []
    prediction_columns = canonical[
        identity + ["perturbation", "hours_post_perturbation", "witness_score", "gmm_positive_posterior"]
    ].copy()
    for threshold_index, threshold in enumerate(args.thresholds):
        name = _threshold_name(threshold)
        print(f"Threshold {threshold:.2f}", flush=True)
        state = _state_from_posterior(obs, posterior, threshold, gate_controls=args.gate_controls)
        model_dir = working / "models" / name
        model_dir.mkdir(parents=True)
        classifier_args = argparse.Namespace(
            classifier_max_per_dataset_class=args.classifier_max_per_dataset_class,
            random_seed=args.random_seed + 10_000 * threshold_index,
            classifier_population="all_gmm_gated" if args.gate_controls else "zikv_only",
        )
        metrics, oof, manifest = _evaluate_classifiers(features, obs, state, DATASETS, classifier_args, model_dir)
        metrics.insert(0, "threshold", threshold)
        metrics_parts.append(metrics)
        manifest.to_csv(working / f"classifier_training_cells__{name}.csv", index=False)
        teacher = _teacher_summary(obs, witness, posterior, state, DATASETS, threshold)
        teacher.insert(0, "threshold", threshold)
        teacher_parts.append(teacher)
        hpi_parts.append(_hpi_summary(obs, posterior, oof["Phase3D"], threshold, args.hpi_bin_hours))
        prediction_columns[f"teacher_state__{name}"] = state
        prediction_columns[f"pal_to_pal_probability__{name}"] = oof["pAL17"]
        prediction_columns[f"pal_to_phase_probability__{name}"] = oof["Phase3D"]
        low = _posterior_boundary(gmm, remod, 1.0 - threshold)
        high = _posterior_boundary(gmm, remod, threshold)
        boundary_rows.append(
            {
                "threshold": threshold,
                "infected_boundary": high,
                "uninfected_boundary": low,
                "ambiguous_width": low - high,
                "ambiguous_midpoint": (low + high) / 2,
            }
        )

    metrics = pd.concat(metrics_parts, ignore_index=True)
    teacher = pd.concat(teacher_parts, ignore_index=True)
    hpi = pd.concat(hpi_parts, ignore_index=True)
    boundaries = pd.DataFrame(boundary_rows)
    metrics.to_csv(working / "classification_metrics_by_threshold.csv", index=False)
    teacher.to_csv(working / "teacher_label_summary_by_threshold.csv", index=False)
    hpi.to_csv(working / "hpi_summary_by_threshold.csv", index=False)
    boundaries.to_csv(working / "posterior_boundaries.csv", index=False)
    prediction_columns.to_parquet(working / "predictions_by_threshold.parquet", index=False)
    _plot_summary(teacher, metrics, working / "posterior_threshold_sweep_summary.png")
    _plot_hpi(hpi, working / "posterior_threshold_sweep_hpi.png")
    _plot_confusions(metrics, working / "posterior_threshold_sweep_phase_confusions.png")

    pooled_teacher = teacher[teacher["dataset"] == "pooled"].copy()
    pooled_teacher["zikv_confident_coverage"] = 1 - (pooled_teacher["n_zikv_abstain"] / pooled_teacher["n_zikv"])
    pooled_phase = metrics[(metrics["sensor"] == "Phase3D") & (metrics["evaluation_dataset"] == "pooled_oof")]
    summary = pooled_teacher.merge(
        pooled_phase[["threshold", "n", "balanced_accuracy", "auroc", "accuracy", "control_predicted_positive_rate"]],
        on="threshold",
        validate="one_to_one",
    ).merge(boundaries, on="threshold", validate="one_to_one")
    summary.to_csv(working / "pooled_threshold_comparison.csv", index=False)
    run_info = {
        "base_run": str(base),
        "thresholds": args.thresholds,
        "invariant": "PCA, normalization, witness scores, and joint/tied GMM posterior fixed from canonical run",
        "refit_per_threshold": "LODO pAL->pAL and pAL->Phase logistic classifiers",
        "evaluation_scope": "confident cells in ZIKV wells; control rates reported separately",
        "gate_controls": args.gate_controls,
        "hpi_bin_hours": args.hpi_bin_hours,
        "random_seed": args.random_seed,
    }
    (working / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    working.rename(final_output)
    print(f"Wrote {final_output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    parser.add_argument("--classifier-max-per-dataset-class", type=int, default=10_000)
    parser.add_argument("--smooth-sigma-timepoints", type=float, default=2.0)
    parser.add_argument("--scale-floor-quantile", type=float, default=0.05)
    parser.add_argument("--hpi-bin-hours", type=float, default=2.0)
    parser.add_argument("--gate-controls", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
