"""Control-MAD and marker-PCA preprocessing for biological MMD analyses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from dynaclr.evaluation.mmd.config import MMDRepresentationConfig


@dataclass(frozen=True)
class ExplainedVariancePCA:
    """Unwhitened PCA basis selected by a cumulative-variance target."""

    marker: str
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    realized_variance: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) @ self.components.T


@dataclass(frozen=True)
class PreparedMMDRepresentation:
    """Row-aligned transformed embeddings and their fitted metadata."""

    features: np.ndarray
    label: str
    n_components_by_marker: dict[str, int]
    normalization_contracts: tuple[dict, ...]
    pca_fits: dict[str, ExplainedVariancePCA]
    normalization_summary: pd.DataFrame
    pca_summary: pd.DataFrame


def fit_pca_for_explained_variance(
    x: np.ndarray,
    variance_threshold: float,
    *,
    marker: str = "unknown",
) -> ExplainedVariancePCA:
    """Fit PCA and retain the fewest PCs reaching ``variance_threshold``.

    Component selection lives outside MMD so one frozen marker basis is fitted
    once, recorded, and then applied unchanged to every comparison.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("PCA needs a two-dimensional matrix with at least two rows")
    mean = x.mean(axis=0, dtype=np.float64)
    centered = x.astype(np.float64) - mean
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError(f"{marker}: PCA reference cohort has zero total variance")
    explained = eigenvalues / total
    cumulative = np.cumsum(explained)
    n_components = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    return ExplainedVariancePCA(
        marker=str(marker),
        mean=mean.astype(np.float32),
        components=eigenvectors[:, :n_components].T.astype(np.float32),
        explained_variance_ratio=explained,
        cumulative_explained_variance=cumulative,
        realized_variance=float(cumulative[n_components - 1]),
    )


def _control_mad_normalize(
    x: np.ndarray,
    obs: pd.DataFrame,
    config: MMDRepresentationConfig,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[dict, ...], pd.DataFrame]:
    required = {config.control_key, config.hpi_key, config.marker_key}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"control-MAD normalization missing obs columns: {sorted(missing)}")
    experiments = (
        obs[config.experiment_key].astype(str).to_numpy()
        if config.experiment_key in obs
        else np.repeat("unknown", len(obs))
    )
    markers = obs[config.marker_key].astype(str).to_numpy()
    labels = obs[config.control_key].astype(str).to_numpy()
    hpi = obs[config.hpi_key].to_numpy(dtype=float)
    if not np.isfinite(hpi).all():
        raise ValueError("control-MAD normalization requires finite HPI values")
    control = np.isin(labels, np.asarray(config.control_values, dtype=str))
    normalized = np.empty_like(np.asarray(x, dtype=np.float32))
    contracts: list[dict] = []
    summaries: list[dict] = []

    for experiment in sorted(np.unique(experiments)):
        exp_rows = experiments == experiment
        for marker in sorted(np.unique(markers[exp_rows])):
            group = exp_rows & (markers == marker)
            group_fit = fit_mask[group]
            group_control = control[group] & group_fit
            if not group_control.any():
                raise ValueError(
                    f"{experiment}/{marker}: no controls ({config.control_key} in {config.control_values!r})"
                )
            group_x = np.asarray(x[group], dtype=np.float32)
            group_hpi = hpi[group]
            grid = np.unique(group_hpi)
            centers = np.full((len(grid), x.shape[1]), np.nan, dtype=np.float32)
            for index, time in enumerate(grid):
                rows = group_control & np.isclose(group_hpi, time)
                if rows.any():
                    centers[index] = np.median(group_x[rows], axis=0)
            occupied = np.flatnonzero(np.isfinite(centers[:, 0]))
            if not len(occupied):
                raise ValueError(f"{experiment}/{marker}: no HPI has controls")
            missing_hpi = np.flatnonzero(~np.isfinite(centers[:, 0]))
            if len(missing_hpi):
                for dimension in range(x.shape[1]):
                    centers[missing_hpi, dimension] = np.interp(
                        grid[missing_hpi], grid[occupied], centers[occupied, dimension]
                    )
            if config.smooth_sigma_timepoints > 0 and len(grid) > 1:
                centers = gaussian_filter1d(
                    centers,
                    sigma=config.smooth_sigma_timepoints,
                    axis=0,
                    mode="nearest",
                ).astype(np.float32)
            centered = group_x - centers[np.searchsorted(grid, group_hpi)]
            control_centered = centered[group_control]
            residual_median = np.median(control_centered, axis=0)
            scale = 1.4826 * np.median(np.abs(control_centered - residual_median), axis=0)
            positive = scale[scale > 0]
            if not len(positive):
                raise ValueError(f"{experiment}/{marker}: every control dimension has zero MAD")
            floor = float(np.quantile(positive, config.mad_floor_quantile))
            scale = np.maximum(scale, floor).astype(np.float32)
            normalized[group] = centered / scale
            contracts.append(
                {
                    "experiment": experiment,
                    "marker": marker,
                    "hpi_grid": grid,
                    "control_centers": centers,
                    "control_mad": scale,
                }
            )
            summaries.append(
                {
                    "experiment": experiment,
                    "marker": marker,
                    "n_cells": int(group_fit.sum()),
                    "n_control_cells": int(group_control.sum()),
                    "n_hpi": int(len(grid)),
                    "mad_floor": floor,
                    "median_mad": float(np.median(scale)),
                }
            )
    return normalized, tuple(contracts), pd.DataFrame(summaries)


def _balanced_pca_rows(
    obs: pd.DataFrame,
    marker_mask: np.ndarray,
    config: MMDRepresentationConfig,
    marker: str,
    fit_mask: np.ndarray,
) -> np.ndarray:
    experiments = (
        obs[config.experiment_key].astype(str).to_numpy()
        if config.experiment_key in obs
        else np.repeat("unknown", len(obs))
    )
    labels = obs[config.control_key].astype(str).to_numpy()
    is_control = np.isin(labels, np.asarray(config.control_values, dtype=str))
    seed = config.random_seed + int.from_bytes(hashlib.sha256(marker.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed % (2**32))
    selected: list[np.ndarray] = []
    for experiment in sorted(np.unique(experiments[marker_mask])):
        dataset_rows = np.flatnonzero(marker_mask & fit_mask & (experiments == experiment))
        control_rows = dataset_rows[is_control[dataset_rows]]
        perturbed_rows = dataset_rows[~is_control[dataset_rows]]
        if len(control_rows) and len(perturbed_rows):
            n = min(
                config.pca_max_cells_per_dataset_class,
                len(control_rows),
                len(perturbed_rows),
            )
            selected.extend(
                [
                    rng.choice(control_rows, n, replace=False),
                    rng.choice(perturbed_rows, n, replace=False),
                ]
            )
        else:
            n = min(config.pca_max_cells_per_dataset_class, len(dataset_rows))
            selected.append(rng.choice(dataset_rows, n, replace=False))
    return np.concatenate(selected)


def prepare_mmd_representation(
    x: np.ndarray,
    obs: pd.DataFrame,
    config: MMDRepresentationConfig,
    *,
    artifact_dir: Path | None = None,
    fit_mask: np.ndarray | None = None,
) -> PreparedMMDRepresentation:
    """Fit control-MAD/PCA once and return row-aligned MMD coordinates."""
    x = np.asarray(x, dtype=np.float32)
    obs = obs.copy()
    if config.marker_key not in obs:
        raise KeyError(f"representation marker column {config.marker_key!r} not found")
    if fit_mask is None:
        fit_mask = np.ones(len(obs), dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != (len(obs),):
            raise ValueError(f"fit_mask must have shape ({len(obs)},), got {fit_mask.shape}")
    if not fit_mask.any():
        raise ValueError("fit_mask does not select any rows")
    if config.normalization == "control_mad":
        normalized, contracts, normalization_summary = _control_mad_normalize(x, obs, config, fit_mask)
    else:
        normalized, contracts, normalization_summary = x.copy(), (), pd.DataFrame()

    markers = obs[config.marker_key].astype(str).to_numpy()
    pca_fits: dict[str, ExplainedVariancePCA] = {}
    pca_rows: list[dict] = []
    if config.pca_variance is None:
        transformed = normalized
        n_components = {marker: int(x.shape[1]) for marker in np.unique(markers)}
    else:
        outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for marker in sorted(np.unique(markers)):
            marker_mask = markers == marker
            fit_rows = _balanced_pca_rows(obs, marker_mask, config, marker, fit_mask)
            fit = fit_pca_for_explained_variance(normalized[fit_rows], config.pca_variance, marker=marker)
            indices = np.flatnonzero(marker_mask)
            outputs[marker] = (indices, fit.transform(normalized[indices]).astype(np.float32))
            pca_fits[marker] = fit
            pca_rows.append(
                {
                    "marker": marker,
                    "embedding_dimensions": int(x.shape[1]),
                    "pca_fit_cells": int(len(fit_rows)),
                    "n_components": int(fit.components.shape[0]),
                    "target_variance": float(config.pca_variance),
                    "realized_variance": fit.realized_variance,
                }
            )
        max_components = max(values.shape[1] for _, values in outputs.values())
        transformed = np.zeros((len(x), max_components), dtype=np.float32)
        for indices, values in outputs.values():
            transformed[indices, : values.shape[1]] = values
        n_components = {marker: int(fit.components.shape[0]) for marker, fit in pca_fits.items()}

    label = config.normalization
    if config.pca_variance is not None:
        label += f"_pca{int(round(100 * config.pca_variance))}"
    prepared = PreparedMMDRepresentation(
        features=transformed,
        label=label,
        n_components_by_marker=n_components,
        normalization_contracts=contracts,
        pca_fits=pca_fits,
        normalization_summary=normalization_summary,
        pca_summary=pd.DataFrame(pca_rows),
    )
    if artifact_dir is not None:
        save_representation_artifacts(prepared, config, artifact_dir)
    return prepared


def save_representation_artifacts(
    prepared: PreparedMMDRepresentation,
    config: MMDRepresentationConfig,
    artifact_dir: Path,
) -> None:
    """Save enough normalization/PCA state to audit and reuse the fit."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not prepared.normalization_summary.empty:
        prepared.normalization_summary.to_csv(artifact_dir / "control_mad_summary.csv", index=False)
    contract_dir = artifact_dir / "control_mad_contracts"
    for contract in prepared.normalization_contracts:
        contract_dir.mkdir(parents=True, exist_ok=True)
        safe = f"{contract['experiment']}__{contract['marker']}".replace(" ", "_").replace("/", "-")
        np.savez_compressed(contract_dir / f"{safe}.npz", **contract)
    if not prepared.pca_summary.empty:
        prepared.pca_summary.to_csv(artifact_dir / "marker_pca_summary.csv", index=False)
    model_dir = artifact_dir / "pca_models"
    scree: list[dict] = []
    for marker, fit in prepared.pca_fits.items():
        model_dir.mkdir(parents=True, exist_ok=True)
        safe = marker.replace(" ", "_").replace("/", "-")
        checksum = hashlib.sha256(fit.mean.tobytes() + fit.components.tobytes()).hexdigest()
        np.savez_compressed(
            model_dir / f"{safe}.npz",
            marker=marker,
            mean=fit.mean,
            components=fit.components,
            explained_variance_ratio=fit.explained_variance_ratio,
            cumulative_explained_variance=fit.cumulative_explained_variance,
            checksum=checksum,
        )
        scree.extend(
            {
                "marker": marker,
                "pc": pc,
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance": float(cumulative),
            }
            for pc, (ratio, cumulative) in enumerate(
                zip(fit.explained_variance_ratio, fit.cumulative_explained_variance, strict=True),
                start=1,
            )
        )
    if scree:
        pd.DataFrame(scree).to_csv(artifact_dir / "marker_pca_scree.csv", index=False)
    run_info = {
        "representation": prepared.label,
        "normalization": config.normalization,
        "control_key": config.control_key,
        "control_values": config.control_values,
        "smooth_sigma_timepoints": config.smooth_sigma_timepoints,
        "mad_floor_quantile": config.mad_floor_quantile,
        "pca_variance": config.pca_variance,
        "pca_max_cells_per_dataset_class": config.pca_max_cells_per_dataset_class,
        "random_seed": config.random_seed,
        "n_components_by_marker": prepared.n_components_by_marker,
    }
    (artifact_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
