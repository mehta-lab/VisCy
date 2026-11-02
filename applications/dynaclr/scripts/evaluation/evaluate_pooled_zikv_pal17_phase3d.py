"""Pooled ZIKV pAL17-witness teacher with cross-dataset classifiers.

Four fixed-encoder datasets are normalized against their own controls. A pooled
pAL17 PCA80 representation defines one control-vs-ZIKV witness and one all-cell
joint/tied GMM teacher. pAL17->pAL17 and pAL17->Phase3D classifiers are evaluated
with leave-one-dataset-out folds and then refit on all datasets for deployment.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from viscy_utils.evaluation.mmd import median_heuristic, witness_function
from viscy_utils.evaluation.witness_gmm import fit_gmm_labels

EMBEDDING_SUFFIX = Path(
    "2-phenotyping/predictions/DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/embeddings"
)
DEFAULT_OUTPUT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_zikv_pal17_phase3d_joint_tied"
)
KEYS = ["fov_name", "track_id", "t"]
SENSORS = ("pAL17", "Phase3D")
SENSOR_LABELS = {"pAL17": "pAL -> pAL", "Phase3D": "pAL -> Phase"}
SENSOR_COLORS = {"pAL17": "#009E73", "Phase3D": "#0072B2"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path

    @property
    def embeddings(self) -> Path:
        return self.root / EMBEDDING_SUFFIX


DATASETS = (
    DatasetSpec(
        "07_01_SEC61B_TOMM20_G3BP1",
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV"),
    ),
    DatasetSpec(
        "03_24_SEC61B",
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_03_24_A549_SEC61_ZIKV"),
    ),
    DatasetSpec(
        "04_08_G3BP1",
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_04_08_A549_G3BP1_ZIKV"),
    ),
    DatasetSpec(
        "04_10_TOMM20",
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_04_10_A549_TOMM20_ZIKV"),
    ),
)


def _load_pal(spec: DatasetSpec) -> tuple[np.ndarray, pd.DataFrame]:
    data = ad.read_zarr(spec.embeddings / "pAL17.zarr")
    obs = data.obs.reset_index(drop=True).copy()
    if obs.duplicated(KEYS).any():
        raise ValueError(f"{spec.name}: duplicate pAL17 cell keys")
    if "is_blank" not in obs:
        raise KeyError(f"{spec.name}: pAL17 obs is missing required is_blank QC")
    keep = ~obs["is_blank"].to_numpy(dtype=bool)
    x = np.asarray(data.X, dtype=np.float32)[keep]
    return x, obs.loc[keep].reset_index(drop=True)


def _load_phase_aligned(spec: DatasetSpec, reference_obs: pd.DataFrame) -> np.ndarray:
    data = ad.read_zarr(spec.embeddings / "Phase3D.zarr")
    obs = data.obs.reset_index(drop=True)
    if obs.duplicated(KEYS).any():
        raise ValueError(f"{spec.name}: duplicate Phase3D cell keys")
    lookup = obs[KEYS].copy()
    lookup["_phase_row"] = np.arange(len(lookup))
    aligned = reference_obs[KEYS].merge(lookup, on=KEYS, how="left", validate="one_to_one")
    if aligned["_phase_row"].isna().any():
        raise ValueError(f"{spec.name}: pAL17 cells missing Phase3D embeddings")
    rows = aligned["_phase_row"].to_numpy(dtype=int)
    if "is_blank" not in obs:
        raise KeyError(f"{spec.name}: Phase3D obs is missing required is_blank QC")
    if obs.iloc[rows]["is_blank"].to_numpy(dtype=bool).any():
        raise ValueError(f"{spec.name}: retained pAL17 rows disagree with Phase3D blank QC")
    return np.asarray(data.X, dtype=np.float32)[rows]


def _normal_consistent_mad(x: np.ndarray) -> np.ndarray:
    median = np.median(x, axis=0)
    return (1.4826 * np.median(np.abs(x - median), axis=0)).astype(np.float32)


def _control_normalization_contract(
    x: np.ndarray,
    obs: pd.DataFrame,
    *,
    sigma_timepoints: float,
    floor_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit and apply the frozen control center/MAD contract.

    Returns normalized embeddings, the exact-HPI grid, the smoothed control
    center at every grid value, and one control MAD vector pooled across HPI
    after centering.
    """
    x = np.asarray(x, dtype=np.float32)
    perturbation = obs["perturbation"].astype(str).to_numpy()
    control = perturbation == "uninfected"
    hpi = obs["hours_post_perturbation"].to_numpy(dtype=float)
    grid = np.unique(hpi[np.isfinite(hpi)])
    centers = np.full((len(grid), x.shape[1]), np.nan, dtype=np.float32)
    for index, time in enumerate(grid):
        rows = control & np.isclose(hpi, time)
        if rows.any():
            centers[index] = np.median(x[rows], axis=0)
    occupied = np.flatnonzero(np.isfinite(centers[:, 0]))
    if len(occupied) < 2:
        raise ValueError("Controls must cover at least two HPI values")
    missing = np.flatnonzero(~np.isfinite(centers[:, 0]))
    if len(missing):
        for dimension in range(x.shape[1]):
            centers[missing, dimension] = np.interp(grid[missing], grid[occupied], centers[occupied, dimension])
    centers = gaussian_filter1d(centers, sigma=sigma_timepoints, axis=0, mode="nearest").astype(np.float32)
    for index, time in enumerate(grid):
        x[np.isclose(hpi, time)] -= centers[index]
    scale = _normal_consistent_mad(x[control])
    positive = scale[scale > 0]
    if not len(positive):
        raise ValueError("Every control dimension has zero MAD")
    floor = float(np.quantile(positive, floor_quantile))
    scale = np.maximum(scale, floor).astype(np.float32)
    x /= scale
    return x, grid, centers, scale


def _normalize_in_place(
    x: np.ndarray,
    obs: pd.DataFrame,
    *,
    sigma_timepoints: float,
    floor_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the contract while preserving the historical two-value API."""
    normalized, _grid, _centers, scale = _control_normalization_contract(
        x,
        obs,
        sigma_timepoints=sigma_timepoints,
        floor_quantile=floor_quantile,
    )
    return normalized, scale


def _balanced_indices(obs: pd.DataFrame, max_per_condition: int, rng: np.random.Generator) -> np.ndarray:
    perturbation = obs["perturbation"].astype(str).to_numpy()
    selected = []
    for condition in ("uninfected", "ZIKV"):
        rows = np.flatnonzero(perturbation == condition)
        n = min(max_per_condition, len(rows))
        selected.append(rng.choice(rows, n, replace=False))
    return np.concatenate(selected)


def _fit_pooled_pcas(
    specs: tuple[DatasetSpec, ...], args: argparse.Namespace
) -> tuple[dict[str, PCA], dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    samples: dict[str, list[np.ndarray]] = {sensor: [] for sensor in SENSORS}
    scales: dict[str, dict[str, np.ndarray]] = {sensor: {} for sensor in SENSORS}
    sample_rows = []
    for dataset_index, spec in enumerate(specs):
        print(f"PCA sample: {spec.name}", flush=True)
        pal_x, obs = _load_pal(spec)
        pal_x, pal_scale = _normalize_in_place(
            pal_x,
            obs,
            sigma_timepoints=args.smooth_sigma_timepoints,
            floor_quantile=args.scale_floor_quantile,
        )
        phase_x = _load_phase_aligned(spec, obs)
        phase_x, phase_scale = _normalize_in_place(
            phase_x,
            obs,
            sigma_timepoints=args.smooth_sigma_timepoints,
            floor_quantile=args.scale_floor_quantile,
        )
        rng = np.random.default_rng(args.random_seed + dataset_index)
        indices = _balanced_indices(obs, args.pca_max_per_condition, rng)
        samples["pAL17"].append(pal_x[indices].copy())
        samples["Phase3D"].append(phase_x[indices].copy())
        scales["pAL17"][spec.name] = pal_scale
        scales["Phase3D"][spec.name] = phase_scale
        selected = obs.iloc[indices][KEYS + ["perturbation"]].copy()
        selected.insert(0, "dataset", spec.name)
        selected["_row"] = indices
        sample_rows.append(selected)
        del pal_x, phase_x, obs
        gc.collect()

    models: dict[str, PCA] = {}
    contracts = []
    for sensor_index, sensor in enumerate(SENSORS):
        fit_x = np.concatenate(samples[sensor], axis=0)
        print(f"Fit pooled {sensor} PCA80 on {len(fit_x):,} cells", flush=True)
        model = PCA(
            n_components=args.pca_variance_threshold,
            svd_solver="full",
            random_state=args.random_seed + 100 + sensor_index,
        )
        model.fit(fit_x)
        models[sensor] = model
        contracts.append(
            {
                "sensor": sensor,
                "n_fit_cells": len(fit_x),
                "n_input_features": fit_x.shape[1],
                "n_components": int(model.n_components_),
                "explained_variance": float(model.explained_variance_ratio_.sum()),
                "variance_threshold": args.pca_variance_threshold,
            }
        )
        del fit_x
        samples[sensor].clear()
        gc.collect()
    return models, scales, pd.DataFrame(contracts), pd.concat(sample_rows, ignore_index=True)


def _transform_all(
    specs: tuple[DatasetSpec, ...],
    pcas: dict[str, PCA],
    scales: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, slice]]:
    features: dict[str, list[np.ndarray]] = {sensor: [] for sensor in SENSORS}
    observations = []
    slices: dict[str, slice] = {}
    offset = 0
    for spec in specs:
        print(f"Transform all cells: {spec.name}", flush=True)
        pal_x, obs = _load_pal(spec)
        pal_x, pal_scale = _normalize_in_place(
            pal_x,
            obs,
            sigma_timepoints=args.smooth_sigma_timepoints,
            floor_quantile=args.scale_floor_quantile,
        )
        if not np.allclose(pal_scale, scales["pAL17"][spec.name]):
            raise ValueError(f"{spec.name}: pAL17 scale changed between passes")
        phase_x = _load_phase_aligned(spec, obs)
        phase_x, phase_scale = _normalize_in_place(
            phase_x,
            obs,
            sigma_timepoints=args.smooth_sigma_timepoints,
            floor_quantile=args.scale_floor_quantile,
        )
        if not np.allclose(phase_scale, scales["Phase3D"][spec.name]):
            raise ValueError(f"{spec.name}: Phase3D scale changed between passes")
        features["pAL17"].append(pcas["pAL17"].transform(pal_x).astype(np.float32))
        features["Phase3D"].append(pcas["Phase3D"].transform(phase_x).astype(np.float32))
        frame = obs.copy()
        frame.insert(0, "dataset", spec.name)
        frame["_dataset_row"] = np.arange(len(frame))
        frame["_global_row"] = np.arange(offset, offset + len(frame))
        observations.append(frame)
        slices[spec.name] = slice(offset, offset + len(frame))
        offset += len(frame)
        del pal_x, phase_x, obs
        gc.collect()
    return (
        {sensor: np.concatenate(parts, axis=0) for sensor, parts in features.items()},
        pd.concat(observations, ignore_index=True),
        slices,
    )


def _reference_indices(
    obs: pd.DataFrame,
    specs: tuple[DatasetSpec, ...],
    condition: str,
    max_per_dataset: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for spec in specs:
        rows = obs.index[(obs["dataset"] == spec.name) & (obs["perturbation"].astype(str) == condition)].to_numpy(
            dtype=int
        )
        n = min(max_per_dataset, len(rows))
        selected.append(rng.choice(rows, n, replace=False))
    return np.concatenate(selected)


def _posterior_boundary(result, threshold: float) -> float:
    means = result.gmm.means_.ravel()
    stds = result.component_stds
    grid = np.linspace(
        float(np.min(means - 8 * stds)),
        float(np.max(means + 8 * stds)),
        100_000,
    )
    posterior = result.gmm.predict_proba(grid[:, None])[:, result.remod_component]
    return float(grid[np.argmin(np.abs(posterior - threshold))])


def _pooled_teacher(
    pal_pc: np.ndarray,
    obs: pd.DataFrame,
    specs: tuple[DatasetSpec, ...],
    args: argparse.Namespace,
):
    control_ref_idx = _reference_indices(obs, specs, "uninfected", args.witness_reference_per_dataset, args.random_seed)
    zikv_ref_idx = _reference_indices(obs, specs, "ZIKV", args.witness_reference_per_dataset, args.random_seed + 1)
    x_ref = pal_pc[control_ref_idx]
    y_ref = pal_pc[zikv_ref_idx]
    bandwidth = median_heuristic(x_ref, y_ref)
    print(
        f"Pooled pAL witness: {len(obs):,} cells; {len(x_ref):,} control + {len(y_ref):,} ZIKV references",
        flush=True,
    )
    witness = witness_function(
        pal_pc,
        x_ref,
        y_ref,
        bandwidth=bandwidth,
        chunk_size=args.witness_chunk_size,
    )
    result = fit_gmm_labels(
        witness,
        pos_threshold=args.posterior_threshold,
        fit_scores_1d=witness,
        covariance_type="tied",
        random_state=args.random_seed,
    )
    if not result.converged or not result.separated:
        raise RuntimeError("Pooled all-cell joint/tied witness GMM is not separated")
    posterior = result.posterior
    perturbation = obs["perturbation"].astype(str).to_numpy()
    control = perturbation == "uninfected"
    zikv = perturbation == "ZIKV"
    state = np.full(len(obs), np.nan, dtype=float)
    state[control] = 0.0
    state[zikv & (posterior >= args.posterior_threshold)] = 1.0
    state[zikv & (posterior <= 1.0 - args.posterior_threshold)] = 0.0
    return witness, posterior, state, result, bandwidth, control_ref_idx, zikv_ref_idx


def _balanced_classifier_indices(
    obs: pd.DataFrame,
    state: np.ndarray,
    datasets: list[str],
    max_per_dataset_class: int,
    seed: int,
    population: str = "zikv_only",
) -> np.ndarray:
    perturbation = obs["perturbation"].astype(str).to_numpy()
    if population not in {"zikv_only", "all_gmm_gated"}:
        raise ValueError(f"Unknown classifier population: {population}")
    candidate_groups = []
    counts = []
    for dataset in datasets:
        for label in (0, 1):
            population_mask = perturbation == "ZIKV" if population == "zikv_only" else np.ones(len(obs), dtype=bool)
            rows = np.flatnonzero(
                (obs["dataset"].to_numpy() == dataset) & population_mask & np.isfinite(state) & (state == label)
            )
            candidate_groups.append(rows)
            counts.append(len(rows))
    n = min(max_per_dataset_class, *counts)
    if n < 2:
        raise ValueError(f"Insufficient confident classes: counts={counts}")
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(rows, n, replace=False) for rows in candidate_groups])


def _score_rows(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "n_negative": int((y == 0).sum()),
        "n_positive": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auroc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _evaluate_classifiers(
    features: dict[str, np.ndarray],
    obs: pd.DataFrame,
    state: np.ndarray,
    specs: tuple[DatasetSpec, ...],
    args: argparse.Namespace,
    model_dir: Path,
):
    oof = {sensor: np.full(len(obs), np.nan, dtype=float) for sensor in SENSORS}
    training_parts = []
    dataset_names = [spec.name for spec in specs]
    for fold_index, held_out in enumerate(dataset_names):
        training_datasets = [name for name in dataset_names if name != held_out]
        train_idx = _balanced_classifier_indices(
            obs,
            state,
            training_datasets,
            args.classifier_max_per_dataset_class,
            args.random_seed + 1000 + fold_index,
            getattr(args, "classifier_population", "zikv_only"),
        )
        test_idx = np.flatnonzero(obs["dataset"].to_numpy() == held_out)
        y_train = state[train_idx].astype(int)
        manifest = obs.iloc[train_idx][["dataset"] + KEYS + ["perturbation", "hours_post_perturbation"]].copy()
        manifest.insert(0, "held_out_dataset", held_out)
        manifest["teacher_state_binary"] = y_train
        training_parts.append(manifest)
        for sensor_index, sensor in enumerate(SENSORS):
            model = LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=args.random_seed + 2000 + 10 * fold_index + sensor_index,
            )
            model.fit(features[sensor][train_idx], y_train)
            oof[sensor][test_idx] = model.predict_proba(features[sensor][test_idx])[:, 1]
            joblib.dump(model, model_dir / f"lodo_{held_out}__{sensor}.joblib")
        print(
            f"LODO {held_out}: trained on {len(train_idx):,} balanced cells",
            flush=True,
        )

    metric_rows = []
    perturbation = obs["perturbation"].astype(str).to_numpy()
    for sensor in SENSORS:
        for dataset in dataset_names + ["pooled_oof"]:
            dataset_mask = (
                np.ones(len(obs), dtype=bool) if dataset == "pooled_oof" else obs["dataset"].to_numpy() == dataset
            )
            primary = dataset_mask & (perturbation == "ZIKV") & np.isfinite(state)
            row = {
                "sensor": sensor,
                "classifier": SENSOR_LABELS[sensor],
                "evaluation_dataset": dataset,
                "scope": "zikv_confident",
            }
            row.update(_score_rows(state[primary].astype(int), oof[sensor][primary]))
            control = dataset_mask & (perturbation == "uninfected")
            row["control_n"] = int(control.sum())
            row["control_predicted_positive_rate"] = float((oof[sensor][control] >= 0.5).mean())
            metric_rows.append(row)

    final_indices = _balanced_classifier_indices(
        obs,
        state,
        dataset_names,
        args.classifier_max_per_dataset_class,
        args.random_seed + 3000,
        getattr(args, "classifier_population", "zikv_only"),
    )
    y_final = state[final_indices].astype(int)
    for sensor_index, sensor in enumerate(SENSORS):
        model = LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=args.random_seed + 4000 + sensor_index,
        )
        model.fit(features[sensor][final_indices], y_final)
        joblib.dump(model, model_dir / f"final_all_datasets__{sensor}.joblib")
    final_manifest = obs.iloc[final_indices][["dataset"] + KEYS + ["perturbation", "hours_post_perturbation"]].copy()
    final_manifest.insert(0, "held_out_dataset", "none_final_all_datasets")
    final_manifest["teacher_state_binary"] = y_final
    training_parts.append(final_manifest)
    return pd.DataFrame(metric_rows), oof, pd.concat(training_parts, ignore_index=True)


def _teacher_summary(
    obs: pd.DataFrame,
    witness: np.ndarray,
    posterior: np.ndarray,
    state: np.ndarray,
    specs: tuple[DatasetSpec, ...],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    perturbation = obs["perturbation"].astype(str).to_numpy()
    for dataset in [spec.name for spec in specs] + ["pooled"]:
        mask = np.ones(len(obs), dtype=bool) if dataset == "pooled" else obs["dataset"].to_numpy() == dataset
        control = mask & (perturbation == "uninfected")
        zikv = mask & (perturbation == "ZIKV")
        rows.append(
            {
                "dataset": dataset,
                "n_total": int(mask.sum()),
                "n_control": int(control.sum()),
                "n_zikv": int(zikv.sum()),
                "control_witness_median": float(np.median(witness[control])),
                "zikv_witness_median": float(np.median(witness[zikv])),
                "control_gmm_false_positive_rate": float((posterior[control] >= threshold).mean()),
                "n_zikv_positive": int((state[zikv] == 1).sum()),
                "n_zikv_negative": int((state[zikv] == 0).sum()),
                "n_zikv_abstain": int(np.isnan(state[zikv]).sum()),
                "zikv_positive_fraction_confident": float(np.nanmean(state[zikv])),
            }
        )
    return pd.DataFrame(rows)


def _plot_pca_scree(pcas: dict[str, PCA], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for sensor in SENSORS:
        cumulative = np.cumsum(pcas[sensor].explained_variance_ratio_)
        ax.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            lw=2,
            color=SENSOR_COLORS[sensor],
            label=f"{sensor}: {pcas[sensor].n_components_} PCs",
        )
    ax.axhline(0.8, color="black", ls="--", lw=1)
    ax.set_xlabel("principal component")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0, 1.01)
    ax.set_title("Pooled four-dataset PCA80 contracts")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_witness_by_dataset(
    obs: pd.DataFrame,
    witness: np.ndarray,
    result,
    specs: tuple[DatasetSpec, ...],
    threshold: float,
    output: Path,
) -> None:
    lo, hi = np.quantile(witness, [0.001, 0.999])
    bins = np.linspace(lo, hi, 70)
    grid = np.linspace(lo, hi, 1_000)
    means = result.gmm.means_.ravel()
    stds = result.component_stds
    weights = result.gmm.weights_.ravel()
    b_pos = _posterior_boundary(result, threshold)
    b_neg = _posterior_boundary(result, 1.0 - threshold)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, spec in zip(axes.ravel(), specs, strict=True):
        frame_mask = obs["dataset"].to_numpy() == spec.name
        perturbation = obs["perturbation"].astype(str).to_numpy()
        control = frame_mask & (perturbation == "uninfected")
        zikv = frame_mask & (perturbation == "ZIKV")
        ax.hist(
            witness[zikv],
            bins=bins,
            density=True,
            color="0.65",
            alpha=0.7,
            label=f"ZIKV n={zikv.sum():,}",
        )
        ax.hist(
            witness[control],
            bins=bins,
            density=True,
            histtype="step",
            color="#1f4e79",
            lw=1.7,
            label=f"control n={control.sum():,}",
        )
        for component in range(2):
            density = weights[component] / (stds[component] * np.sqrt(2 * np.pi))
            density *= np.exp(-0.5 * ((grid - means[component]) / stds[component]) ** 2)
            ax.plot(
                grid,
                density,
                color="tab:red" if component == result.remod_component else "tab:cyan",
                lw=1.5,
            )
        ax.axvspan(b_pos, b_neg, color="#F2C94C", alpha=0.25)
        ax.axvline(b_pos, color="tab:red", ls="--", lw=1)
        ax.axvline(b_neg, color="tab:blue", ls="--", lw=1)
        ax.set_title(spec.name)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, frameon=False)
    fig.supxlabel("pooled pAL17 witness score")
    fig.supylabel("density")
    fig.suptitle(
        "One pooled all-cell joint/tied GMM applied to every dataset\nyellow = 0.2–0.8 posterior abstention interval"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    order = [spec.name for spec in DATASETS] + ["pooled_oof"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, metric in zip(axes, ("accuracy", "balanced_accuracy", "auroc"), strict=True):
        x = np.arange(len(order))
        width = 0.36
        for sensor_index, sensor in enumerate(SENSORS):
            subset = metrics.set_index(["sensor", "evaluation_dataset"])
            values = [float(subset.loc[(sensor, dataset), metric]) for dataset in order]
            ax.bar(
                x + (sensor_index - 0.5) * width,
                values,
                width,
                color=SENSOR_COLORS[sensor],
                label=SENSOR_LABELS[sensor],
            )
        ax.set_xticks(x, [name.replace("_", "\n") for name in order], rotation=0)
        ax.set_ylim(0, 1.02)
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("leave-one-dataset-out score")
    axes[0].legend(frameon=False)
    fig.suptitle("Pooled pAL witness teacher: held-out dataset classifier agreement")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    specs = DATASETS
    final_output = args.output_dir
    output = final_output.with_name(final_output.name + ".in_progress")
    if final_output.exists() or output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {final_output} or {output}")
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "models"
    model_dir.mkdir()

    pcas, scales, contracts, pca_manifest = _fit_pooled_pcas(specs, args)
    for sensor, model in pcas.items():
        joblib.dump(model, model_dir / f"pooled_{sensor}_pca80.joblib")
    scale_payload = {
        f"{dataset}__{sensor}": scale for sensor, by_dataset in scales.items() for dataset, scale in by_dataset.items()
    }
    np.savez_compressed(model_dir / "per_dataset_control_mad_scales.npz", **scale_payload)
    contracts.to_csv(output / "pca_contracts.csv", index=False)
    pca_manifest.to_csv(output / "pca_fit_cells.csv", index=False)
    _plot_pca_scree(pcas, output / "pooled_pca80_scree.png")

    features, obs, slices = _transform_all(specs, pcas, scales, args)
    witness, posterior, state, gmm, bandwidth, control_ref_idx, zikv_ref_idx = _pooled_teacher(
        features["pAL17"], obs, specs, args
    )
    joblib.dump(gmm.gmm, model_dir / "pooled_pal17_witness_joint_tied_gmm.joblib")
    references = pd.concat(
        [
            obs.iloc[control_ref_idx][["dataset"] + KEYS].assign(reference="control"),
            obs.iloc[zikv_ref_idx][["dataset"] + KEYS].assign(reference="ZIKV"),
        ],
        ignore_index=True,
    )
    references.to_csv(output / "witness_reference_cells.csv", index=False)

    metrics, oof, training_manifest = _evaluate_classifiers(features, obs, state, specs, args, model_dir)
    metrics.to_csv(output / "classification_metrics.csv", index=False)
    training_manifest.to_csv(output / "classifier_training_cells.csv", index=False)
    summary = _teacher_summary(obs, witness, posterior, state, specs, args.posterior_threshold)
    summary.to_csv(output / "teacher_label_summary.csv", index=False)

    predictions = obs[["dataset"] + KEYS + ["perturbation", "hours_post_perturbation"]].copy()
    predictions["witness_score"] = witness
    predictions["gmm_positive_posterior"] = posterior
    predictions["teacher_state_binary"] = state
    predictions["teacher_state"] = np.select([state == 1, state == 0], ["infected", "uninfected"], default="abstain")
    predictions["pal_to_pal_probability"] = oof["pAL17"]
    predictions["pal_to_phase_probability"] = oof["Phase3D"]
    predictions.to_parquet(output / "leave_one_dataset_out_predictions.parquet", index=False)

    _plot_witness_by_dataset(
        obs,
        witness,
        gmm,
        specs,
        args.posterior_threshold,
        output / "pooled_witness_joint_tied_by_dataset.png",
    )
    _plot_metrics(metrics, output / "classification_accuracy_by_dataset.png")

    means = gmm.gmm.means_.ravel()
    stds = gmm.component_stds
    weights = gmm.gmm.weights_.ravel()
    remod = gmm.remod_component
    b_pos = _posterior_boundary(gmm, args.posterior_threshold)
    b_mid = _posterior_boundary(gmm, 0.5)
    b_neg = _posterior_boundary(gmm, 1.0 - args.posterior_threshold)
    gmm_diagnostics = {
        "fit_population": "all control and ZIKV cells from all four datasets",
        "covariance_type": "tied",
        "n_fit": int(gmm.n_fit_samples),
        "bandwidth": float(bandwidth),
        "remodeled_mean": float(means[remod]),
        "control_like_mean": float(means[1 - remod]),
        "shared_std": float(stds[0]),
        "remodeled_weight": float(weights[remod]),
        "control_like_weight": float(weights[1 - remod]),
        "posterior_0p8_boundary": b_pos,
        "posterior_0p5_boundary": b_mid,
        "posterior_0p2_boundary": b_neg,
        "ambiguous_midpoint": float((b_pos + b_neg) / 2),
        "gmm_delta_bic": float(gmm.bic_1comp - gmm.bic),
    }
    (output / "gmm_diagnostics.json").write_text(json.dumps(gmm_diagnostics, indent=2) + "\n")
    run_info = {
        "datasets": [asdict(spec) | {"root": str(spec.root)} for spec in specs],
        "normalization": (
            "per-dataset, per-channel smooth control median by exact HPI; "
            "one normal-consistent control MAD per dimension pooled across HPI"
        ),
        "pca": (
            "one pooled PCA per sensor fit on equal control/ZIKV samples from "
            "each dataset; smallest dimension retaining 80% variance"
        ),
        "teacher": (
            "one pooled pAL17 RBF witness; all-cell joint control+ZIKV GMM with "
            "tied covariance and 0.8/0.2 posterior gates"
        ),
        "evaluation": (
            "leave-one-dataset-out classifier fitting; pooled PCA and pooled "
            "teacher are transductive, but no held-out dataset labels train its classifier"
        ),
        "primary_scope": "confident cells within ZIKV wells",
        "interpretation": ("agreement with a pooled pAL17 pseudo-teacher, not independent infection ground truth"),
        "sensors": {"pAL17": "pAL->pAL", "Phase3D": "pAL->Phase"},
        "parameters": {
            "pca_variance_threshold": args.pca_variance_threshold,
            "pca_max_per_condition": args.pca_max_per_condition,
            "witness_reference_per_dataset": args.witness_reference_per_dataset,
            "posterior_threshold": args.posterior_threshold,
            "classifier_max_per_dataset_class": args.classifier_max_per_dataset_class,
            "smooth_sigma_timepoints": args.smooth_sigma_timepoints,
            "scale_floor_quantile": args.scale_floor_quantile,
            "random_seed": args.random_seed,
        },
        "pca_contracts": contracts.to_dict(orient="records"),
        "gmm": gmm_diagnostics,
        "dataset_slices": {name: [sl.start, sl.stop] for name, sl in slices.items()},
    }
    (output / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(
        metrics[
            [
                "classifier",
                "evaluation_dataset",
                "n",
                "accuracy",
                "balanced_accuracy",
                "auroc",
                "control_predicted_positive_rate",
            ]
        ].to_string(index=False),
        flush=True,
    )
    output.rename(final_output)
    print(f"Wrote {final_output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pca-variance-threshold", type=float, default=0.80)
    parser.add_argument("--pca-max-per-condition", type=int, default=5_000)
    parser.add_argument("--witness-reference-per-dataset", type=int, default=2_500)
    parser.add_argument("--witness-chunk-size", type=int, default=500)
    parser.add_argument("--posterior-threshold", type=float, default=0.8)
    parser.add_argument("--classifier-max-per-dataset-class", type=int, default=10_000)
    parser.add_argument("--smooth-sigma-timepoints", type=float, default=2.0)
    parser.add_argument("--scale-floor-quantile", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
