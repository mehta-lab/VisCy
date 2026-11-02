"""Paired parent-family bootstrap for direct-pAL alignment-gain metrics.

The bootstrap conditions on fixed PCA coordinates, MMD bandwidths, and
per-point matched samples. Complete parent-ID-linked track families are
resampled within dataset, well, and condition. Identical family
multiplicities are applied to HPI and tau, making every gain replicate paired.

Intervals quantify family-sampling uncertainty conditional on the observed
wells and datasets. They are not biological-replicate confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
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
AXES = ("hpi", "tau")
COLORS = {"TOMM20": "#009E73", "SEC61B": "#0072B2", "G3BP1": "#CC79A7"}
METRICS = {
    "r2_gain_tau_minus_hpi": ("Δ cross-validated R²", 0.0),
    "residual_reduction_fraction": ("Residual reduction fraction", 0.0),
    "width_reduction_hours": ("Transition-width reduction (h)", 0.0),
    "slope_ratio_tau_over_hpi": ("Maximum-slope ratio (τ/HPI)", 1.0),
}


def _well_from_fov(value: object) -> str:
    parts = str(value).split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer well from fov_name={value!r}")
    return "/".join(parts[:2])


def _annotate_families(samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.copy()
    samples["well_id"] = samples["fov_name"].map(_well_from_fov)
    samples["family_uid"] = (
        samples["dataset"].astype(str)
        + "::"
        + samples["fov_name"].astype(str)
        + "::"
        + samples["lineage_id"].astype(str)
    )
    duplicated = samples.duplicated(["marker", "scope", "axis", "axis_center", "sample_group", "family_uid"])
    if duplicated.any():
        raise ValueError("A family appears more than once in an MMD point")
    return samples


def _bootstrap_counts(
    samples: pd.DataFrame,
    *,
    group: str,
    n_replicates: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int], pd.DataFrame]:
    families = (
        samples.loc[samples["sample_group"].eq(group)][["dataset", "well_id", "family_uid"]]
        .drop_duplicates()
        .sort_values(["dataset", "well_id", "family_uid"])
        .reset_index(drop=True)
    )
    families["family_index"] = np.arange(len(families), dtype=int)
    counts = np.zeros((n_replicates, len(families)), dtype=np.float32)
    for _, stratum in families.groupby(["dataset", "well_id"], observed=True):
        indices = stratum["family_index"].to_numpy(dtype=int)
        probabilities = np.full(len(indices), 1.0 / len(indices))
        counts[:, indices] = rng.multinomial(len(indices), probabilities, size=n_replicates)
    index = dict(zip(families["family_uid"].astype(str), families["family_index"], strict=True))
    return torch.from_numpy(counts).to(device), index, families


def _rbf_kernel(left: torch.Tensor, right: torch.Tensor, bandwidth: float) -> torch.Tensor:
    left2 = (left * left).sum(1, keepdim=True)
    right2 = (right * right).sum(1, keepdim=True).T
    distance2 = (left2 + right2 - 2.0 * (left @ right.T)).clamp_min_(0.0)
    return torch.exp(-distance2 / (2.0 * bandwidth))


def _weighted_mmd(
    scores: np.ndarray,
    point: pd.DataFrame,
    zikv_counts: torch.Tensor,
    control_counts: torch.Tensor,
    zikv_index: dict[str, int],
    control_index: dict[str, int],
    bandwidth: float,
    device: torch.device,
    perturbed_group: str = "ZIKV",
) -> tuple[np.ndarray, float]:
    zikv = point.loc[point["sample_group"].eq(perturbed_group)]
    control = point.loc[point["sample_group"].eq("control")]
    zikv_columns = [zikv_index[value] for value in zikv["family_uid"].astype(str)]
    control_columns = [control_index[value] for value in control["family_uid"].astype(str)]
    wz = zikv_counts[:, zikv_columns]
    wc = control_counts[:, control_columns]
    z = torch.from_numpy(np.asarray(scores[zikv["pc_row"].to_numpy(dtype=int)], dtype=np.float32)).to(device)
    c = torch.from_numpy(np.asarray(scores[control["pc_row"].to_numpy(dtype=int)], dtype=np.float32)).to(device)
    kzz = _rbf_kernel(z, z, bandwidth)
    kcc = _rbf_kernel(c, c, bandwidth)
    kzc = _rbf_kernel(z, c, bandwidth)
    nz = wz.sum(1)
    nc = wc.sum(1)
    valid = (nz > 1) & (nc > 1)
    result = torch.full_like(nz, torch.nan)
    zz = (wz @ kzz * wz).sum(1) - (wz * torch.diagonal(kzz)[None, :]).sum(1)
    cc = (wc @ kcc * wc).sum(1) - (wc * torch.diagonal(kcc)[None, :]).sum(1)
    zc = (wz @ kzc * wc).sum(1)
    result[valid] = (
        zz[valid] / (nz[valid] * (nz[valid] - 1))
        + cc[valid] / (nc[valid] * (nc[valid] - 1))
        - 2.0 * zc[valid] / (nz[valid] * nc[valid])
    )
    one_z = torch.ones((1, len(zikv)), dtype=torch.float32, device=device)
    one_c = torch.ones((1, len(control)), dtype=torch.float32, device=device)
    base_zz = (one_z @ kzz * one_z).sum() - torch.diagonal(kzz).sum()
    base_cc = (one_c @ kcc * one_c).sum() - torch.diagonal(kcc).sum()
    base_zc = (one_z @ kzc * one_c).sum()
    base = (
        base_zz / (len(zikv) * (len(zikv) - 1))
        + base_cc / (len(control) * (len(control) - 1))
        - 2.0 * base_zc / (len(zikv) * len(control))
    )
    return result.detach().cpu().numpy(), float(base.item())


def _ridge_metric_batch(
    x: np.ndarray,
    values: np.ndarray,
    *,
    axis: str,
    n_knots: int,
    ridge_alpha: float,
    tau_baseline_start: float,
    tau_baseline_end: float,
    hpi_baseline_hours: float,
) -> dict[str, np.ndarray]:
    transformer = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False).fit(x[:, None])
    features = transformer.transform(x[:, None])
    design = np.column_stack([np.ones(len(x)), features])
    penalty = np.diag([0.0] + [ridge_alpha] * features.shape[1])
    operator = np.linalg.solve(design.T @ design + penalty, design.T)
    coefficients = operator @ values.T
    fitted = design @ coefficients
    # The production metric refits SplineTransformer inside every held-out
    # fold. Its boundary knots can therefore change when an endpoint is held
    # out, so the usual fixed-design hat-matrix shortcut is not equivalent.
    # Build the exact linear prediction operator once, then apply it to every
    # bootstrap curve.
    loo_operator = np.zeros((len(x), len(x)), dtype=float)
    for held_out in range(len(x)):
        train = np.arange(len(x)) != held_out
        fold_transformer = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False).fit(x[train, None])
        train_features = fold_transformer.transform(x[train, None])
        train_design = np.column_stack([np.ones(train.sum()), train_features])
        fold_penalty = np.diag([0.0] + [ridge_alpha] * train_features.shape[1])
        fold_operator = np.linalg.solve(
            train_design.T @ train_design + fold_penalty,
            train_design.T,
        )
        test_features = fold_transformer.transform(x[[held_out], None])
        test_design = np.column_stack([np.ones(1), test_features])
        loo_operator[held_out, train] = test_design @ fold_operator
    loo = loo_operator @ values.T
    total = np.sum((values - np.nanmean(values, axis=1, keepdims=True)) ** 2, axis=1)
    cv_r2 = 1.0 - np.sum((values - loo.T) ** 2, axis=1) / total
    robust_range = np.nanquantile(values, 0.95, axis=1) - np.nanquantile(values, 0.05, axis=1)
    residual = np.sqrt(np.nanmean((values - fitted.T) ** 2, axis=1)) / (robust_range + 1e-12)
    dense_x = np.linspace(x.min(), x.max(), 1_000)
    dense_features = transformer.transform(dense_x[:, None])
    dense_design = np.column_stack([np.ones(len(dense_x)), dense_features])
    dense = (dense_design @ coefficients).T
    if axis == "tau":
        baseline_mask = (x >= tau_baseline_start) & (x <= tau_baseline_end)
    else:
        baseline_mask = x <= x.min() + hpi_baseline_hours
    baseline = np.nanmedian(fitted[baseline_mask, :], axis=0)
    amplitude = np.nanmax(dense, axis=1) - baseline
    normalized = (dense - baseline[:, None]) / amplitude[:, None]
    normalized[amplitude <= 0] = np.nan
    slope = np.gradient(normalized, dense_x, axis=1)
    max_slope = np.nanmax(slope, axis=1)
    hit10 = normalized >= 0.1
    valid10 = hit10.any(axis=1)
    index10 = np.argmax(hit10, axis=1)
    after10 = np.arange(len(dense_x))[None, :] >= index10[:, None]
    hit90 = (normalized >= 0.9) & after10 & valid10[:, None]
    valid90 = hit90.any(axis=1)
    index90 = np.argmax(hit90, axis=1)
    width = np.full(len(values), np.nan)
    width[valid90] = dense_x[index90[valid90]] - dense_x[index10[valid90]]
    return {
        "cv_r2": cv_r2,
        "normalized_residual_rmse": residual,
        "transition_width_hours": width,
        "max_normalized_slope_per_hour": max_slope,
    }


def _gain(metrics: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        "r2_gain_tau_minus_hpi": (metrics["tau"]["cv_r2"] - metrics["hpi"]["cv_r2"]),
        "residual_reduction_fraction": (
            1.0 - metrics["tau"]["normalized_residual_rmse"] / metrics["hpi"]["normalized_residual_rmse"]
        ),
        "width_reduction_hours": (metrics["hpi"]["transition_width_hours"] - metrics["tau"]["transition_width_hours"]),
        "slope_ratio_tau_over_hpi": (
            metrics["tau"]["max_normalized_slope_per_hour"] / metrics["hpi"]["max_normalized_slope_per_hour"]
        ),
    }


def _summarize(
    replicates: pd.DataFrame,
    point_estimates: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    rows = []
    for (marker, scope), group in replicates.groupby(["marker", "scope"], observed=True):
        point = point_estimates.loc[point_estimates["marker"].eq(marker) & point_estimates["scope"].eq(scope)].iloc[0]
        for metric, (_, null) in METRICS.items():
            values = group[metric].dropna().to_numpy(float)
            lower, median, upper = np.quantile(values, [lower_quantile, 0.5, upper_quantile])
            rows.append(
                {
                    "marker": marker,
                    "scope": scope,
                    "metric": metric,
                    "point_estimate": float(point[metric]),
                    "bootstrap_median": float(median),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "null_value": null,
                    "probability_favors_tau": float(np.mean(values > null)),
                    "verdict": ("supports_tau" if lower > null else "supports_hpi" if upper < null else "inconclusive"),
                    "n_valid_replicates": len(values),
                }
            )
    return pd.DataFrame(rows)


def _plot(
    summary: pd.DataFrame,
    output: Path,
    markers: tuple[str, ...],
    scopes: tuple[str, ...],
    scope_titles: dict[str, str],
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 8.5),
        sharey=True,
        layout="constrained",
    )
    order = [(marker, scope) for marker in markers for scope in scopes]
    positions = {key: index for index, key in enumerate(order)}
    labels = list(markers) if len(scopes) == 1 else [f"{marker} · {scope.replace('_', ' ')}" for marker, scope in order]
    for panel_index, (ax, (metric, (title, null))) in enumerate(zip(axes.flat, METRICS.items(), strict=True)):
        ordered = summary.loc[summary["metric"].eq(metric)].copy()
        ordered["position"] = [
            positions[(marker, scope)] for marker, scope in zip(ordered["marker"], ordered["scope"], strict=True)
        ]
        ordered = ordered.sort_values("position")
        for row in ordered.itertuples(index=False):
            position = int(row.position)
            ax.errorbar(
                row.bootstrap_median,
                position,
                xerr=[
                    [row.bootstrap_median - row.ci_lower],
                    [row.ci_upper - row.bootstrap_median],
                ],
                fmt="o",
                color=COLORS[row.marker],
                capsize=3,
            )
            ax.scatter(
                row.point_estimate,
                position,
                marker="x",
                color="black",
                s=28,
                zorder=3,
            )
        ax.axvline(null, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(labels)))
        if panel_index % 2 == 0:
            ax.set_yticklabels(labels)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.margins(x=0.12, y=0.18)
    scope_note = scope_titles[scopes[0]] if len(scopes) == 1 else "multiple scopes"
    fig.suptitle(
        "Paired parent-family bootstrap of τ alignment gain\n"
        f"{scope_note}\n"
        "Colored dot/line: bootstrap median and 95% interval; ×: point estimate; "
        "dashed line: no gain",
        fontsize=15,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curves = pd.read_csv(args.input / "mmd_curves.csv")
    curve_metrics = pd.read_csv(args.input / "alignment_curve_metrics.csv")
    point_estimates = pd.read_csv(args.input / "alignment_gain_summary.csv")
    samples = _annotate_families(pd.read_parquet(args.input / "mmd_point_sample_manifest.parquet"))
    analysis_info = json.loads((args.input / "run_info.json").read_text())
    markers = tuple(analysis_info.get("markers", sorted(curves["marker"].unique())))
    scopes = tuple(analysis_info.get("scopes", {value: {} for value in curves["scope"].unique()}))
    scope_titles = analysis_info.get("scope_titles", {scope: scope.replace("_", " ") for scope in scopes})
    perturbed_group = str(analysis_info.get("perturbation", "ZIKV"))
    rng = np.random.default_rng(args.random_seed)
    replicate_parts = []
    validation_rows = []
    stratum_rows = []
    for marker in markers:
        scores = np.load(args.input / f"{marker.lower()}_pca80_scores.npy")
        marker_samples = samples.loc[samples["marker"].eq(marker)]
        for scope in scopes:
            print(f"Bootstrap {marker} {scope}", flush=True)
            scoped = marker_samples.loc[marker_samples["scope"].eq(scope)]
            zikv_counts, zikv_index, zikv_families = _bootstrap_counts(
                scoped,
                group=perturbed_group,
                n_replicates=args.bootstrap_replicates,
                rng=rng,
                device=device,
            )
            control_counts, control_index, control_families = _bootstrap_counts(
                scoped,
                group="control",
                n_replicates=args.bootstrap_replicates,
                rng=rng,
                device=device,
            )
            for group, families in (
                (perturbed_group, zikv_families),
                ("control", control_families),
            ):
                counts = (
                    families.groupby(["dataset", "well_id"], observed=True).size().rename("n_families").reset_index()
                )
                counts.insert(0, "sample_group", group)
                counts.insert(0, "scope", scope)
                counts.insert(0, "marker", marker)
                stratum_rows.append(counts)
            axis_metrics: dict[str, dict[str, np.ndarray]] = {}
            for axis in AXES:
                selected_curves = (
                    curves.loc[curves["marker"].eq(marker) & curves["scope"].eq(scope) & curves["axis"].eq(axis)]
                    .sort_values("axis_center")
                    .reset_index(drop=True)
                )
                point_groups = {
                    float(center): group
                    for center, group in scoped.loc[scoped["axis"].eq(axis)].groupby("axis_center", observed=True)
                }
                bootstrap_values = []
                reconstructed = []
                for row in selected_curves.itertuples(index=False):
                    point = point_groups[float(row.axis_center)]
                    values, base = _weighted_mmd(
                        scores,
                        point,
                        zikv_counts,
                        control_counts,
                        zikv_index,
                        control_index,
                        float(row.bandwidth),
                        device,
                        perturbed_group,
                    )
                    bootstrap_values.append(values)
                    reconstructed.append(base)
                    validation_rows.append(
                        {
                            "marker": marker,
                            "scope": scope,
                            "axis": axis,
                            "axis_center": float(row.axis_center),
                            "stored_mmd2": float(row.mmd2_unbiased),
                            "reconstructed_mmd2": base,
                            "absolute_error": abs(base - float(row.mmd2_unbiased)),
                        }
                    )
                bootstrap_matrix = np.stack(bootstrap_values, axis=1)
                invalid_fraction = float(np.isnan(bootstrap_matrix).mean())
                if invalid_fraction > args.max_invalid_fraction:
                    raise ValueError(f"Too many invalid replicates for {marker} {scope} {axis}")
                axis_metrics[axis] = _ridge_metric_batch(
                    selected_curves["axis_center"].to_numpy(float),
                    bootstrap_matrix,
                    axis=axis,
                    n_knots=args.spline_knots,
                    ridge_alpha=args.ridge_alpha,
                    tau_baseline_start=args.tau_baseline_start,
                    tau_baseline_end=args.tau_baseline_end,
                    hpi_baseline_hours=args.hpi_baseline_hours,
                )
                base_metrics = _ridge_metric_batch(
                    selected_curves["axis_center"].to_numpy(float),
                    np.asarray(reconstructed)[None, :],
                    axis=axis,
                    n_knots=args.spline_knots,
                    ridge_alpha=args.ridge_alpha,
                    tau_baseline_start=args.tau_baseline_start,
                    tau_baseline_end=args.tau_baseline_end,
                    hpi_baseline_hours=args.hpi_baseline_hours,
                )
                expected = curve_metrics.loc[
                    curve_metrics["marker"].eq(marker)
                    & curve_metrics["scope"].eq(scope)
                    & curve_metrics["axis"].eq(axis)
                ].iloc[0]
                for name in (
                    "cv_r2",
                    "normalized_residual_rmse",
                    "transition_width_hours",
                    "max_normalized_slope_per_hour",
                ):
                    if not np.isclose(base_metrics[name][0], float(expected[name]), atol=2e-5):
                        raise ValueError(f"Metric reconstruction mismatch: {marker} {scope} {axis} {name}")
            gains = _gain(axis_metrics)
            part = pd.DataFrame(gains)
            part.insert(0, "bootstrap_replicate", np.arange(len(part)))
            part.insert(0, "scope", scope)
            part.insert(0, "marker", marker)
            replicate_parts.append(part)
            del zikv_counts, control_counts
            if device.type == "cuda":
                torch.cuda.empty_cache()
    replicates = pd.concat(replicate_parts, ignore_index=True)
    summary = _summarize(
        replicates,
        point_estimates,
        args.lower_quantile,
        args.upper_quantile,
    )
    validation = pd.DataFrame(validation_rows)
    if validation["absolute_error"].max() > 2e-5:
        raise ValueError("Reconstructed point MMD does not match stored curves")
    strata = pd.concat(stratum_rows, ignore_index=True)
    replicates.to_parquet(args.input / "family_bootstrap_gain_replicates.parquet", index=False)
    summary.to_csv(args.input / "family_bootstrap_gain_summary.csv", index=False)
    validation.to_csv(args.input / "family_bootstrap_point_reconstruction.csv", index=False)
    strata.to_csv(args.input / "family_bootstrap_strata.csv", index=False)
    _plot(
        summary,
        args.input / "family_bootstrap_alignment_gain.png",
        markers,
        scopes,
        scope_titles,
    )
    run_info = {
        "bootstrap_unit": "parent-ID-linked track family",
        "strata": [
            "dataset",
            "well inferred from first two fov_name components",
            "condition",
        ],
        "pairing": "same family multiplicities applied to HPI and tau",
        "conditioned_on": [
            "observed datasets and wells",
            "fixed PCA coordinates",
            "fixed marker bandwidth",
            "fixed per-point matched family sample",
        ],
        "not_supported": "biological-replicate inference",
        "perturbation": perturbed_group,
        "markers": list(markers),
        "scopes": list(scopes),
        "device": str(device),
        "arguments": vars(args) | {"input": str(args.input)},
    }
    (args.input / "family_bootstrap_run_info.json").write_text(json.dumps(run_info, indent=2))
    analysis_info["alignment_gain_postprocess"]["inference_limit"] = (
        "paired parent-family bootstrap complete; conditional on observed datasets and wells"
    )
    analysis_info["family_bootstrap"] = {
        "status": "complete",
        "run_info": "family_bootstrap_run_info.json",
        "summary": "family_bootstrap_gain_summary.csv",
        "replicates": "family_bootstrap_gain_replicates.parquet",
        "figure": "family_bootstrap_alignment_gain.png",
        "n_replicates": args.bootstrap_replicates,
    }
    (args.input / "run_info.json").write_text(json.dumps(analysis_info, indent=2))
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--random-seed", type=int, default=20260810)
    parser.add_argument("--lower-quantile", type=float, default=0.025)
    parser.add_argument("--upper-quantile", type=float, default=0.975)
    parser.add_argument("--max-invalid-fraction", type=float, default=0.01)
    parser.add_argument("--spline-knots", type=int, default=6)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--tau-baseline-start", type=float, default=-10.0)
    parser.add_argument("--tau-baseline-end", type=float, default=-6.0)
    parser.add_argument("--hpi-baseline-hours", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
