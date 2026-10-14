"""Quantify whether direct-pAL alignment sharpens organelle MMD trajectories.

This post-processes a native-timepoint, same-lineage-cohort run from
``analyze_direct_pal_aligned_organelle_mmd.py``. Metrics use raw MMD squared
within each marker; null-standardized MMD is retained only for the persistent
baseline-deviation rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_INPUT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_zikv_pal17_phase3d_joint_tied_qc_nonblank/"
    "direct_pal_lineage_native30_same_cohort_alignment"
)
COLORS = {"TOMM20": "#009E73", "SEC61B": "#0072B2", "G3BP1": "#CC79A7"}


def _cohort_audit(
    input_path: Path,
    markers: tuple[str, ...],
    perturbation: str,
    scopes: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    """Audit onset-eligible candidates and the IDs actually sampled per axis."""
    samples = pd.read_parquet(
        input_path / "mmd_point_sample_manifest.parquet",
        columns=["marker", "scope", "axis", "sample_group", "dataset", "fov_name", "lineage_id"],
    )
    samples["lineage_uid"] = (
        samples["dataset"].astype(str)
        + "::"
        + samples["fov_name"].astype(str)
        + "::"
        + samples["lineage_id"].astype(str)
    )
    rows: list[dict[str, object]] = []
    for marker in markers:
        frame = pd.read_parquet(input_path / f"{marker.lower()}_cell_manifest.parquet")
        perturbed = frame.loc[
            frame["perturbation"].astype(str).eq(perturbation) & frame["pal_onset_hpi"].notna()
        ].copy()
        for scope, marker_datasets in scopes.items():
            scoped = perturbed.loc[perturbed["dataset"].isin(marker_datasets[marker])]
            candidate_hpi_ids = set(
                scoped.loc[np.isfinite(scoped["hours_post_perturbation"]), "lineage_uid"].astype(str)
            )
            candidate_tau_ids = set(scoped.loc[np.isfinite(scoped["tau"]), "lineage_uid"].astype(str))
            candidate_intersection = candidate_hpi_ids & candidate_tau_ids
            candidate_exact_match = candidate_hpi_ids == candidate_tau_ids
            if not candidate_exact_match:
                raise ValueError(f"HPI/tau cohort mismatch for {marker} {scope}")
            sampled = samples.loc[
                samples["marker"].eq(marker) & samples["scope"].eq(scope) & samples["sample_group"].eq(perturbation)
            ]
            sampled_hpi_ids = set(sampled.loc[sampled["axis"].eq("hpi"), "lineage_uid"].astype(str))
            sampled_tau_ids = set(sampled.loc[sampled["axis"].eq("tau"), "lineage_uid"].astype(str))
            sampled_intersection = sampled_hpi_ids & sampled_tau_ids
            rows.append(
                {
                    "marker": marker,
                    "scope": scope,
                    "n_candidate_hpi_lineages": len(candidate_hpi_ids),
                    "n_candidate_tau_lineages": len(candidate_tau_ids),
                    "n_candidate_intersection_lineages": len(candidate_intersection),
                    "candidate_exact_lineage_id_match": candidate_exact_match,
                    "n_sampled_hpi_lineages": len(sampled_hpi_ids),
                    "n_sampled_tau_lineages": len(sampled_tau_ids),
                    "n_sampled_intersection_lineages": len(sampled_intersection),
                    "sampled_exact_lineage_id_match": sampled_hpi_ids == sampled_tau_ids,
                    "n_datasets": int(scoped["dataset"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _model(n_knots: int, ridge_alpha: float):
    return make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=3, include_bias=False),
        Ridge(alpha=ridge_alpha),
    )


def _baseline_mask(x: np.ndarray, axis: str, args: argparse.Namespace) -> np.ndarray:
    if axis == "tau":
        return (x >= args.tau_baseline_start) & (x <= args.tau_baseline_end)
    return x <= x.min() + args.hpi_baseline_hours


def _persistent_onset(
    group: pd.DataFrame,
    baseline_mask: np.ndarray,
    persistent_bins: int,
) -> tuple[float, float]:
    baseline_upper = float(np.quantile(group.loc[baseline_mask, "subsample_q975"], 0.95))
    above = group["subsample_q025"].to_numpy(float) > baseline_upper
    x = group["axis_center"].to_numpy(float)
    width = group["bin_width"].to_numpy(float)
    onset = np.nan
    for index in range(len(group) - persistent_bins + 1):
        stop = index + persistent_bins
        consecutive = np.allclose(np.diff(x[index:stop]), width[index])
        if above[index:stop].all() and consecutive:
            onset = float(x[index])
            break
    return onset, baseline_upper


def _curve_metrics(group: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    group = group.sort_values("axis_center").reset_index(drop=True)
    x = group["axis_center"].to_numpy(float)
    y = group["mmd2_unbiased"].to_numpy(float)
    design = x[:, None]
    model = _model(args.spline_knots, args.ridge_alpha)
    cv_prediction = cross_val_predict(model, design, y, cv=LeaveOneOut())
    cv_r2 = float(r2_score(y, cv_prediction))
    model.fit(design, y)
    fitted = model.predict(design)
    dense_x = np.linspace(x.min(), x.max(), 1_000)
    dense_y = model.predict(dense_x[:, None])
    baseline_mask = _baseline_mask(x, str(group.iloc[0]["axis"]), args)
    baseline = float(np.median(fitted[baseline_mask]))
    amplitude = float(dense_y.max() - baseline)
    normalized = (dense_y - baseline) / amplitude if amplitude > 0 else np.full_like(dense_y, np.nan)
    slope = np.gradient(normalized, dense_x)
    hit10 = np.flatnonzero(normalized >= 0.1)
    t10 = float(dense_x[hit10[0]]) if len(hit10) else np.nan
    hit90 = np.flatnonzero((dense_x >= t10) & (normalized >= 0.9)) if np.isfinite(t10) else np.array([], dtype=int)
    t90 = float(dense_x[hit90[0]]) if len(hit90) else np.nan
    onset, baseline_upper = _persistent_onset(
        group,
        baseline_mask,
        args.persistent_bins,
    )
    robust_range = float(np.quantile(y, 0.95) - np.quantile(y, 0.05))
    residual = float(np.sqrt(np.mean((y - fitted) ** 2)) / (robust_range + 1e-12))
    return {
        "cv_r2": cv_r2,
        "normalized_residual_rmse": residual,
        "max_normalized_slope_per_hour": float(np.nanmax(slope)),
        "transition_t10": t10,
        "transition_t90": t90,
        "transition_width_hours": t90 - t10 if np.isfinite(t90) else np.nan,
        "persistent_baseline_onset": onset,
        "baseline_upper_z": baseline_upper,
        "baseline_raw_mmd2": baseline,
        "response_amplitude_raw_mmd2": amplitude,
    }


def _metric_tables(curves: pd.DataFrame, args: argparse.Namespace):
    rows = []
    for key, group in curves.groupby(["marker", "axis", "scope"], observed=True):
        rows.append(
            {
                "marker": key[0],
                "axis": key[1],
                "scope": key[2],
                **_curve_metrics(group, args),
            }
        )
    metrics = pd.DataFrame(rows)
    wide = metrics.pivot(index=["marker", "scope"], columns="axis")  # noqa: PD010
    gain = pd.DataFrame(index=wide.index)
    gain["r2_gain_tau_minus_hpi"] = wide["cv_r2"]["tau"] - wide["cv_r2"]["hpi"]
    gain["residual_reduction_fraction"] = (
        1 - wide["normalized_residual_rmse"]["tau"] / wide["normalized_residual_rmse"]["hpi"]
    )
    gain["width_reduction_hours"] = wide["transition_width_hours"]["hpi"] - wide["transition_width_hours"]["tau"]
    gain["slope_ratio_tau_over_hpi"] = (
        wide["max_normalized_slope_per_hour"]["tau"] / wide["max_normalized_slope_per_hour"]["hpi"]
    )
    return metrics, gain.reset_index()


def _plot_curves(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    scope: str,
    markers: tuple[str, ...],
    scope_label: str,
    output: Path,
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, axis in zip(axes, ("hpi", "tau"), strict=True):
        subset = curves.loc[curves["axis"].eq(axis) & curves["scope"].eq(scope)]
        for marker in markers:
            group = subset.loc[subset["marker"].eq(marker)].sort_values("axis_center")
            x = group["axis_center"].to_numpy(float)
            y = group["mmd2_unbiased"].to_numpy(float)
            model = _model(args.spline_knots, args.ridge_alpha).fit(x[:, None], y)
            dense_x = np.linspace(x.min(), x.max(), 800)
            ax.plot(x, y, color=COLORS[marker], alpha=0.35, lw=1, marker="o", ms=2)
            ax.plot(dense_x, model.predict(dense_x[:, None]), color=COLORS[marker], lw=2.4, label=marker)
            result = metrics.loc[
                metrics["marker"].eq(marker) & metrics["axis"].eq(axis) & metrics["scope"].eq(scope)
            ].iloc[0]
            onset = result["persistent_baseline_onset"]
            if np.isfinite(onset):
                ax.axvline(onset, color=COLORS[marker], ls=":", alpha=0.75)
        if axis == "tau":
            ax.axvline(0, color="black", ls="--", lw=1)
            ax.set_xlabel("Hours relative to lineage pAL onset (τ)")
            ax.set_title("Aligned by direct pAL")
        else:
            ax.set_xlabel("Hours post infection")
            ax.set_title("Unaligned physical time")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Raw MMD² vs exact-HPI matched controls")
    axes[1].legend(frameon=False)
    fig.suptitle(f"Native 30-minute comparison on the same onset-bearing lineages — {scope_label}")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_gain(
    gain: pd.DataFrame,
    output: Path,
    markers: tuple[str, ...],
    scope_labels: dict[str, str],
) -> None:
    columns = (
        ("r2_gain_tau_minus_hpi", "Cross-validated R² gain"),
        ("residual_reduction_fraction", "Residual reduction"),
        ("width_reduction_hours", "10–90% width reduction (h)"),
        ("slope_ratio_tau_over_hpi", "Maximum-slope ratio"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(markers))
    for ax, (column, title) in zip(axes.flat, columns, strict=True):
        n_scopes = len(scope_labels)
        width = 0.7 / n_scopes
        for offset, (scope, label) in enumerate(scope_labels.items()):
            values = gain.loc[gain["scope"].eq(scope)].set_index("marker").reindex(markers)[column].to_numpy(float)
            center = offset - (n_scopes - 1) / 2
            ax.bar(x + center * width, values, width=width, label=label)
        reference = 1 if column == "slope_ratio_tau_over_hpi" else 0
        ax.axhline(reference, color="black", lw=1)
        ax.set_xticks(x, markers)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Alignment gain: τ relative to physical HPI on identical lineages")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    curves = pd.read_csv(args.input / "mmd_curves.csv")
    if not np.allclose(curves["bin_width"], 0.5):
        raise ValueError("Alignment-gain analysis requires native 0.5-hour bins")
    if not curves["sampling_unit"].eq("lineage").all():
        raise ValueError("Alignment-gain analysis requires lineage sampling")
    run_info_path = args.input / "run_info.json"
    run_info = json.loads(run_info_path.read_text())
    if not run_info["arguments"].get("same_onset_cohort", False):
        raise ValueError("Input did not use the same onset-bearing cohort on both axes")
    markers = tuple(run_info.get("markers", sorted(curves["marker"].unique())))
    perturbation = str(run_info.get("perturbation", "ZIKV"))
    scopes = run_info.get("scopes")
    if scopes is None:
        scopes = {
            "common_07_01": {marker: ["07_01_SEC61B_TOMM20_G3BP1"] for marker in markers},
            "replicate_augmented": {
                marker: sorted(
                    pd.read_parquet(
                        args.input / f"{marker.lower()}_cell_manifest.parquet",
                        columns=["dataset"],
                    )["dataset"]
                    .astype(str)
                    .unique()
                )
                for marker in markers
            },
        }
    scope_labels = run_info.get("scope_titles", {scope: scope.replace("_", " ") for scope in scopes})
    cohort_audit = _cohort_audit(args.input, markers, perturbation, scopes)
    cohort_audit.to_csv(args.input / "same_cohort_audit.csv", index=False)
    metrics, gain = _metric_tables(curves, args)
    metrics.to_csv(args.input / "alignment_curve_metrics.csv", index=False)
    gain.to_csv(args.input / "alignment_gain_summary.csv", index=False)
    for scope in scopes:
        _plot_curves(
            curves,
            metrics,
            scope,
            markers,
            scope_labels[scope],
            args.input / f"native30_same_cohort_hpi_vs_tau__{scope}.png",
            args,
        )
    _plot_gain(
        gain,
        args.input / "alignment_gain_summary.png",
        markers,
        scope_labels,
    )
    run_info["alignment_gain_postprocess"] = {
        "spline_knots": args.spline_knots,
        "ridge_alpha": args.ridge_alpha,
        "cross_validation": "leave-one-timepoint-out",
        "metric_signal": "raw unbiased MMD squared within marker",
        "persistent_bins": args.persistent_bins,
        "tau_baseline": [args.tau_baseline_start, args.tau_baseline_end],
        "hpi_baseline_hours_from_first_point": args.hpi_baseline_hours,
        "inference_limit": "descriptive; lineage-bootstrap intervals remain pending",
        "same_cohort_audit": (
            "same_cohort_audit.csv; onset-eligible candidate IDs must match; "
            "actual per-bin sampled lineage unions are reported separately"
        ),
    }
    run_info_path.write_text(json.dumps(run_info, indent=2))
    print(gain.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--spline-knots", type=int, default=6)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--persistent-bins", type=int, default=3)
    parser.add_argument("--tau-baseline-start", type=float, default=-10.0)
    parser.add_argument("--tau-baseline-end", type=float, default=-6.0)
    parser.add_argument("--hpi-baseline-hours", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
