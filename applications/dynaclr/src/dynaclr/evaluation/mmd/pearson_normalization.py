"""Control-reference MAD normalization and marker-PCA80 for Pearson QC.

This module implements the promoted representation contract used by the
batch-correction exploration:

1. subtract a Gaussian-smoothed control median at each exact HPI;
2. divide by one normal-consistent control MAD per dimension, pooled over HPI;
3. fit one unwhitened PCA per marker on a balanced source-dataset cohort;
4. retain the smallest number of components reaching the configured variance.

The fitted marker basis is then applied unchanged to every dataset summarized
by the embedding-consistency Pearson matrix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from dynaclr.evaluation.mmd.compute_mmd import _extract_embeddings
from dynaclr.evaluation.mmd.config import EmbeddingConsistencyConfig


@dataclass(frozen=True)
class MarkerPCAFit:
    """One source-fitted, unwhitened PCA basis for a marker."""

    marker: str
    reference_dataset: str
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    n_fit_per_class: int
    realized_variance: float
    checksum: str

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project rows into the retained marker-PC basis."""
        return (np.asarray(x) - self.mean) @ self.components.T


def _safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "-")


def _dataset_from_path(path: str) -> str:
    parts = Path(path).parts
    if "2-phenotyping" in parts:
        return parts[parts.index("2-phenotyping") - 1]
    adata = ad.read_zarr(path)
    return str(adata.obs["experiment"].iloc[0])


def _smooth_control_center(
    x: np.ndarray,
    hpi: np.ndarray,
    control: np.ndarray,
    sigma_timepoints: float,
) -> np.ndarray:
    """Evaluate a smoothed control-median trajectory at every row's HPI."""
    grid = np.unique(hpi)
    medians = np.full((len(grid), x.shape[1]), np.nan, dtype=np.float32)
    for index, time in enumerate(grid):
        rows = control & np.isclose(hpi, time)
        if rows.any():
            medians[index] = np.median(x[rows], axis=0)

    occupied = np.flatnonzero(np.isfinite(medians[:, 0]))
    if len(occupied) < 2:
        raise ValueError("Controls must cover at least two HPI values")
    missing = np.flatnonzero(~np.isfinite(medians[:, 0]))
    if len(missing):
        for dimension in range(x.shape[1]):
            medians[missing, dimension] = np.interp(
                grid[missing],
                grid[occupied],
                medians[occupied, dimension],
            )
    if sigma_timepoints > 0:
        medians = gaussian_filter1d(
            medians,
            sigma=sigma_timepoints,
            axis=0,
            mode="nearest",
        )
    positions = np.searchsorted(grid, hpi)
    return medians[positions].astype(np.float32, copy=False)


def _control_values(config: EmbeddingConsistencyConfig) -> list[str]:
    values = config.obs_filter_aliases.get(
        "perturbation",
        [config.pearson_control_value],
    )
    return list(dict.fromkeys([config.pearson_control_value, *values]))


def _control_mad_normalize(
    x: np.ndarray,
    obs: pd.DataFrame,
    config: EmbeddingConsistencyConfig,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Apply the canonical time-matched center plus time-constant MAD scale."""
    required = {"perturbation", "hours_post_perturbation"}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"control-MAD normalization missing obs columns: {sorted(missing)}")

    labels = obs["perturbation"].astype(str).to_numpy()
    hpi = obs["hours_post_perturbation"].to_numpy(dtype=float)
    control_values = _control_values(config)
    control = np.isin(labels, control_values)
    if not control.any():
        raise ValueError(f"No control cells found (perturbation in {control_values!r})")

    center = _smooth_control_center(
        x,
        hpi,
        control,
        config.pearson_smooth_sigma_timepoints,
    )
    centered = x - center
    control_centered = centered[control]
    control_median = np.median(control_centered, axis=0)
    scale = 1.4826 * np.median(
        np.abs(control_centered - control_median),
        axis=0,
    )
    positive = scale[scale > 0]
    if not len(positive):
        raise ValueError("Every control embedding dimension has zero MAD")
    scale_floor = float(np.quantile(positive, config.pearson_mad_floor_quantile))
    scale = np.maximum(scale, scale_floor).astype(np.float32, copy=False)
    normalized = centered / scale
    return normalized.astype(np.float32, copy=False), {
        "n_cells": int(len(x)),
        "n_control_cells": int(control.sum()),
        "n_hpi": int(len(np.unique(hpi))),
        "mad_floor": scale_floor,
        "median_mad": float(np.median(scale)),
    }


def _marker_view(
    adata: ad.AnnData,
    marker: str,
    embedding_key: str | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    rows = adata.obs["marker"].astype(str) == marker
    sub = adata[rows]
    return (
        _extract_embeddings(sub, embedding_key).astype(np.float32),
        sub.obs.copy(),
    )


def _choose_reference_path(
    marker: str,
    candidates: list[str],
    config: EmbeddingConsistencyConfig,
) -> tuple[str, str]:
    """Resolve the configured source or choose the largest balanced cohort."""
    requested = config.pearson_pca_reference_datasets.get(marker)
    if requested is not None:
        matches = [p for p in candidates if _dataset_from_path(p) == requested]
        if len(matches) != 1:
            raise ValueError(f"{marker}: PCA reference {requested!r} matched {len(matches)} stores")
        return matches[0], requested

    scored: list[tuple[int, int, str, str]] = []
    for path in candidates:
        adata = ad.read_zarr(path)
        x, obs = _marker_view(adata, marker, config.embedding_key)
        del x
        labels = obs["perturbation"].astype(str).to_numpy()
        control = np.isin(labels, _control_values(config))
        n_control = int(control.sum())
        n_perturbed = int((~control).sum())
        dataset = str(obs["experiment"].iloc[0])
        scored.append((min(n_control, n_perturbed), n_control + n_perturbed, dataset, path))
    best = max(scored, key=lambda row: (row[0], row[1], row[2]))
    if best[0] < 2:
        raise ValueError(f"{marker}: no dataset has a usable balanced PCA cohort")
    return best[3], best[2]


def _stable_marker_seed(base_seed: int, marker: str) -> int:
    digest = hashlib.sha256(marker.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def _fit_pca80(
    x: np.ndarray,
    variance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit exact covariance PCA and retain the first components reaching target."""
    mean = x.mean(axis=0, dtype=np.float64)
    centered = x.astype(np.float64) - mean
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("PCA reference cohort has zero total variance")
    explained = eigenvalues / total
    cumulative = np.cumsum(explained)
    n_components = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    components = eigenvectors[:, :n_components].T
    return mean, components, explained, cumulative, n_components


def fit_marker_pca_models(
    input_paths: list[str],
    config: EmbeddingConsistencyConfig,
    output_dir: Path,
) -> dict[str, MarkerPCAFit]:
    """Fit and persist one source PCA80 model for every comparable marker."""
    paths_by_marker: dict[str, list[str]] = {}
    for path in input_paths:
        paths_by_marker.setdefault(Path(path).stem, []).append(path)
    paths_by_marker = {
        marker: paths
        for marker, paths in paths_by_marker.items()
        if len({_dataset_from_path(path) for path in paths}) >= 2
    }

    artifact_dir = output_dir / "pearson_mad_pca80"
    model_dir = artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fits: dict[str, MarkerPCAFit] = {}
    summary_rows: list[dict[str, int | float | str]] = []
    scree_rows: list[dict[str, int | float | str]] = []
    manifest_rows: list[dict[str, int | str]] = []

    for marker in sorted(paths_by_marker):
        source_path, source_dataset = _choose_reference_path(
            marker,
            paths_by_marker[marker],
            config,
        )
        adata = ad.read_zarr(source_path)
        x, obs = _marker_view(adata, marker, config.embedding_key)
        normalized, normalization = _control_mad_normalize(x, obs, config)
        labels = obs["perturbation"].astype(str).to_numpy()
        is_control = np.isin(labels, _control_values(config))
        control_rows = np.flatnonzero(is_control)
        perturbed_rows = np.flatnonzero(~is_control)
        n_per_class = min(
            config.pearson_pca_max_cells_per_class,
            len(control_rows),
            len(perturbed_rows),
        )
        if n_per_class < 2:
            raise ValueError(f"{marker}/{source_dataset}: PCA needs control and perturbed cells")
        rng = np.random.default_rng(_stable_marker_seed(config.pearson_random_seed, marker))
        selected = np.concatenate(
            [
                rng.choice(control_rows, n_per_class, replace=False),
                rng.choice(perturbed_rows, n_per_class, replace=False),
            ]
        )
        fit_x = normalized[selected]
        mean, components, explained, cumulative, n_components = _fit_pca80(
            fit_x,
            config.pearson_pca_variance,
        )
        checksum = hashlib.sha256(mean.tobytes() + components.tobytes()).hexdigest()
        fit = MarkerPCAFit(
            marker=marker,
            reference_dataset=source_dataset,
            mean=mean.astype(np.float32),
            components=components.astype(np.float32),
            explained_variance_ratio=explained.astype(np.float64),
            cumulative_explained_variance=cumulative.astype(np.float64),
            n_fit_per_class=int(n_per_class),
            realized_variance=float(cumulative[n_components - 1]),
            checksum=checksum,
        )
        fits[marker] = fit
        safe = _safe_name(marker)
        np.savez_compressed(
            model_dir / f"{safe}_pca80.npz",
            mean=fit.mean,
            components=fit.components,
            explained_variance_ratio=fit.explained_variance_ratio,
            cumulative_explained_variance=fit.cumulative_explained_variance,
            marker=marker,
            reference_dataset=source_dataset,
            checksum=checksum,
        )
        summary_rows.append(
            {
                "marker": marker,
                "reference_dataset": source_dataset,
                "embedding_dimensions": int(x.shape[1]),
                "pca_fit_cells": int(2 * n_per_class),
                "pca_fit_cells_per_class": int(n_per_class),
                "n_components": int(n_components),
                "target_variance": float(config.pearson_pca_variance),
                "realized_variance": fit.realized_variance,
                "mad_floor": normalization["mad_floor"],
                "pca_checksum": checksum,
            }
        )
        for pc, (ratio, cumulative_ratio) in enumerate(
            zip(explained, cumulative, strict=True),
            start=1,
        ):
            scree_rows.append(
                {
                    "marker": marker,
                    "pc": pc,
                    "explained_variance_ratio": float(ratio),
                    "cumulative_explained_variance": float(cumulative_ratio),
                }
            )
        selected_index = obs.index.to_numpy()[selected]
        selected_class = np.where(
            is_control[selected],
            "control",
            "perturbed",
        )
        manifest_rows.extend(
            {
                "marker": marker,
                "reference_dataset": source_dataset,
                "obs_index": str(obs_index),
                "pca_class": str(pca_class),
                "selection_order": int(order),
            }
            for order, (obs_index, pca_class) in enumerate(zip(selected_index, selected_class, strict=True))
        )

    pd.DataFrame(summary_rows).to_csv(
        artifact_dir / "marker_pca_80pct_summary.csv",
        index=False,
    )
    pd.DataFrame(scree_rows).to_csv(
        artifact_dir / "marker_pca_scree_curves.csv",
        index=False,
    )
    pd.DataFrame(manifest_rows).to_csv(
        artifact_dir / "pca_fit_manifest.csv",
        index=False,
    )
    run_info = {
        "normalization": (
            "per-dataset and marker Gaussian-smoothed exact-HPI control median; "
            "one normal-consistent control MAD per dimension pooled across HPI"
        ),
        "pca": "source-fit, balanced control/perturbed, unwhitened marker PCA",
        "pca_variance_threshold": config.pearson_pca_variance,
        "smooth_sigma_timepoints": config.pearson_smooth_sigma_timepoints,
        "mad_floor_quantile": config.pearson_mad_floor_quantile,
        "max_cells_per_class": config.pearson_pca_max_cells_per_class,
        "random_seed": config.pearson_random_seed,
        "reference_datasets": {marker: fit.reference_dataset for marker, fit in fits.items()},
    }
    (artifact_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n",
        encoding="utf-8",
    )
    return fits


def _summary_embedding(
    emb: np.ndarray,
    hpi: np.ndarray,
    hpi_bin_hours: float | None,
) -> np.ndarray:
    if hpi_bin_hours is None:
        return emb.mean(axis=0)
    bins = np.floor(hpi / hpi_bin_hours).astype(int)
    return np.stack([emb[bins == value].mean(axis=0) for value in np.unique(bins)]).mean(axis=0)


def _corr_matrices(
    marker_means: dict[str, dict[str, np.ndarray]],
) -> dict[str, pd.DataFrame]:
    matrices: dict[str, pd.DataFrame] = {}
    for marker, means in marker_means.items():
        datasets = sorted(means)
        if len(datasets) < 2:
            continue
        corr = np.corrcoef(np.stack([means[dataset] for dataset in datasets]))
        matrices[marker] = pd.DataFrame(
            corr,
            index=datasets,
            columns=datasets,
            dtype=float,
        )
    return matrices


def normalized_corr_matrix_per_marker(
    input_paths: list[str],
    config: EmbeddingConsistencyConfig,
    fits: dict[str, MarkerPCAFit],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute per-marker Pearson matrices after control-MAD and shared PCA80."""
    marker_means: dict[str, dict[str, np.ndarray]] = {}
    normalization_rows: list[dict[str, int | float | str]] = []
    for path in input_paths:
        marker_hint = Path(path).stem
        if marker_hint not in fits:
            continue
        adata = ad.read_zarr(path)
        experiment = str(adata.obs["experiment"].iloc[0])
        for marker in adata.obs["marker"].astype(str).unique():
            if marker not in fits:
                continue
            x, obs = _marker_view(adata, marker, config.embedding_key)
            normalized, metadata = _control_mad_normalize(x, obs, config)
            mask = np.ones(len(obs), dtype=bool)
            if config.obs_filter:
                for column, value in config.obs_filter.items():
                    if column not in obs.columns:
                        raise KeyError(f"obs_filter column {column!r} not found in {experiment}")
                    accepted = config.obs_filter_aliases.get(
                        column,
                        [str(value)],
                    )
                    mask &= np.isin(obs[column].astype(str).to_numpy(), accepted)
            if not mask.any():
                continue
            hpi = obs["hours_post_perturbation"].to_numpy(dtype=float)[mask]
            summary = _summary_embedding(
                normalized[mask],
                hpi,
                config.pearson_hpi_bin_hours,
            )
            marker_means.setdefault(marker, {})[experiment] = fits[marker].transform(summary[None, :])[0]
            normalization_rows.append(
                {
                    "marker": marker,
                    "experiment": experiment,
                    "reference_dataset": fits[marker].reference_dataset,
                    "n_components": int(fits[marker].components.shape[0]),
                    **metadata,
                }
            )
    return _corr_matrices(marker_means), pd.DataFrame(normalization_rows)


def correlation_summary(
    raw: dict[str, pd.DataFrame],
    normalized: dict[str, pd.DataFrame],
    fits: dict[str, MarkerPCAFit],
) -> pd.DataFrame:
    """Summarize off-diagonal Pearson correlation before and after normalization."""
    rows: list[dict[str, int | float | str]] = []
    for marker in sorted(set(raw) & set(normalized)):
        for representation, matrix in (
            ("raw", raw[marker]),
            ("control_mad_pca80", normalized[marker]),
        ):
            values = matrix.to_numpy()
            off_diagonal = values[~np.eye(len(values), dtype=bool)]
            rows.append(
                {
                    "marker": marker,
                    "representation": representation,
                    "n_datasets": int(len(matrix)),
                    "mean_off_diagonal_pearson": float(np.nanmean(off_diagonal)),
                    "median_off_diagonal_pearson": float(np.nanmedian(off_diagonal)),
                    "n_components": (
                        int(fits[marker].components.shape[0])
                        if representation == "control_mad_pca80"
                        else int(fits[marker].mean.size)
                    ),
                    "pca_reference_dataset": fits[marker].reference_dataset,
                }
            )
    return pd.DataFrame(rows)
