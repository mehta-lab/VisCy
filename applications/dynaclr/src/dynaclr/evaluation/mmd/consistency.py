"""Per-marker dataset-to-dataset embedding-consistency QC.

Reports two complementary per-marker dataset x dataset matrices on control
cells, so batch effects between acquisitions can be read at a glance:

- **MMD²** (primary, distributional): aggregates the long-form pairwise output
  of :func:`dynaclr.evaluation.mmd.compute_mmd.run_mmd_combined`. Sensitive to
  differences in mean, spread, and shape — the same statistic LOT validation
  uses. Low off-diagonal = comparable; large off-diagonal = batch effect.
- **Pearson correlation** (companion, cheap/readable): correlation between
  per-dataset mean control embeddings. Bounded ``[-1, 1]`` and easy to read, but
  blind to distributional (covariance) shifts — a fast first-pass sanity check.
- **Fréchet distance** (mean + covariance): 2-Wasserstein between Gaussian
  summaries of each dataset's control cells (the FID statistic). Uses the first
  *two* moments — richer than Pearson-of-means, cheaper than full-distribution
  MMD, and exactly the mean + covariance shift LOT correction removes.

Inputs are enumerated via :func:`dynaclr.evaluation.paths.iter_embeddings`; the
QC reuses ``run_mmd_combined`` verbatim and adds the matrix pivots, plotting, and
the glob-driven driver. It detects and reports — it does not correct (see
``lot_correction``).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from dynaclr.evaluation.mmd.compute_mmd import _extract_embeddings, run_mmd_combined
from dynaclr.evaluation.mmd.config import EmbeddingConsistencyConfig, MMDCombinedConfig
from dynaclr.evaluation.mmd.pearson_normalization import (
    MarkerPCAFit,
    correlation_summary,
    fit_marker_pca_models,
    normalized_corr_matrix_per_marker,
)
from dynaclr.evaluation.paths import DATASETS_ROOT, iter_embeddings
from viscy_utils.compose import load_composed_config


def mmd_matrix_per_marker(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate long-form pairwise MMD into a symmetric matrix per marker.

    Averages ``mmd2`` over conditions and temporal bins for each dataset pair,
    then fills a symmetric square matrix indexed by dataset. The diagonal is 0
    (a dataset compared to itself has no batch effect). Datasets that never
    appear as ``exp_a``/``exp_b`` for a marker are absent from that marker's
    matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form output of :func:`run_mmd_combined` with at least the columns
        ``marker``, ``exp_a``, ``exp_b``, ``mmd2``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``marker -> square DataFrame`` whose index and columns are the
        datasets and whose cells are mean ``mmd2`` (diagonal 0).
    """
    matrices: dict[str, pd.DataFrame] = {}
    for marker, sub in df.groupby("marker"):
        pair_mean = sub.groupby(["exp_a", "exp_b"])["mmd2"].mean()
        datasets = sorted(set(sub["exp_a"]) | set(sub["exp_b"]))
        matrix = pd.DataFrame(0.0, index=datasets, columns=datasets, dtype=float)
        for (exp_a, exp_b), value in pair_mean.items():
            matrix.loc[exp_a, exp_b] = value
            matrix.loc[exp_b, exp_a] = value
        matrices[str(marker)] = matrix
    return matrices


def plot_consistency_matrix(matrix: pd.DataFrame, marker: str, output_path: Path) -> None:
    """Plot one symmetric dataset x dataset MMD heatmap for a marker.

    Parameters
    ----------
    matrix : pd.DataFrame
        Square symmetric MMD matrix (datasets x datasets), diagonal 0.
    marker : str
        Marker name, used in the title.
    output_path : Path
        Output file path.
    """
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.3), max(5, n * 1.1)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "MMD²", "shrink": 0.7},
    )
    ax.set_title(
        f"Embedding consistency — {marker}\n"
        "control cells, dataset × dataset MMD²\n"
        "Read magnitude, not significance — permutation p saturates at large N\n"
        "(p, effect_size, n kept in consistency_mmd_results.csv)",
        fontsize=10,
        pad=12,
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dataset_mean_embedding(emb: np.ndarray, hpi: np.ndarray | None, hpi_bin_hours: float | None) -> np.ndarray:
    """Per-dataset summary vector for the Pearson matrix.

    ``hpi_bin_hours=None`` → the grand mean over all cells. Otherwise the mean
    embedding is computed per ``hours_post_perturbation`` bin (width
    ``hpi_bin_hours``, anchored at 0 h) and those bin-means are averaged
    ("mean of means"), so each occupied HPI bin contributes equally regardless
    of how many cells it holds — de-biasing uneven time sampling across
    acquisitions.

    Parameters
    ----------
    emb : np.ndarray
        ``(N, D)`` control embeddings for one (dataset, marker).
    hpi : np.ndarray or None
        ``(N,)`` hours_post_perturbation per cell; required when binning.
    hpi_bin_hours : float or None
        Bin width in hours, or None for the grand mean.

    Returns
    -------
    np.ndarray
        ``(D,)`` summary embedding.
    """
    if hpi_bin_hours is None:
        return emb.mean(axis=0)
    return _hpi_bin_means(emb, hpi, hpi_bin_hours).mean(axis=0)


def _hpi_bin_means(emb: np.ndarray, hpi: np.ndarray | None, hpi_bin_hours: float) -> np.ndarray:
    """Stack of per-``hours_post_perturbation``-bin mean embeddings for one (dataset, marker).

    Bins have width ``hpi_bin_hours`` anchored at 0 h; only occupied bins appear.
    The mean-of-means summary and the temporal-std diagnostic both derive from
    this, so the binning is defined once.

    Parameters
    ----------
    emb : np.ndarray
        ``(N, D)`` embeddings.
    hpi : np.ndarray or None
        ``(N,)`` hours_post_perturbation; required (raises if None).
    hpi_bin_hours : float
        Bin width in hours.

    Returns
    -------
    np.ndarray
        ``(n_occupied_bins, D)`` per-bin mean embeddings.
    """
    if hpi is None:
        raise KeyError("HPI binning requested but obs lacks 'hours_post_perturbation'")
    bins = np.floor(hpi / hpi_bin_hours).astype(int)
    return np.stack([emb[bins == b].mean(axis=0) for b in np.unique(bins)])


def temporal_std_per_marker(
    input_paths: list[str],
    obs_filter: dict[str, str] | None,
    embedding_key: str | None,
    hpi_bin_hours: float,
    obs_filter_aliases: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-(dataset, marker) temporal drift of the control embedding centroid.

    For each dataset and marker, bins control cells by ``hours_post_perturbation``
    (width ``hpi_bin_hours``), takes the mean embedding per bin, then reports the
    std **across those bin-means** (per embedding dimension, reduced to one scalar
    by averaging over dimensions). This measures how much the control centroid
    drifts over the timecourse — context for the consistency matrices, since a
    high-drift dataset is inherently a moving target to compare against. Not a
    pairwise matrix: one number per (dataset, marker).

    Parameters
    ----------
    input_paths : list[str]
        Per-dataset embedding zarr paths.
    obs_filter : dict[str, str] or None
        Control-cell filter (same as the matrices).
    embedding_key : str or None
        obsm key; None uses raw ``.X``.
    hpi_bin_hours : float
        HPI bin width in hours.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``marker -> DataFrame`` indexed by dataset with columns
        ``temporal_std`` (mean over dims of the across-bin std) and ``n_bins``.
    """
    per_marker: dict[str, dict[str, tuple[float, int]]] = {}
    for path in input_paths:
        adata = ad.read_zarr(path)
        experiment = str(adata.obs["experiment"].iloc[0])
        if obs_filter:
            mask = pd.Series(True, index=adata.obs.index)
            for col, val in obs_filter.items():
                if col not in adata.obs.columns:
                    raise KeyError(f"obs_filter column '{col}' not found in {experiment}")
                accepted = (obs_filter_aliases or {}).get(col, [val])
                mask &= adata.obs[col].astype(str).isin(map(str, accepted))
            adata = adata[mask]
        has_hpi = "hours_post_perturbation" in adata.obs.columns
        for marker in adata.obs["marker"].unique():
            sub = adata[adata.obs["marker"] == marker]
            emb = _extract_embeddings(sub, embedding_key).astype(np.float32)
            if len(emb) == 0:
                continue
            hpi = sub.obs["hours_post_perturbation"].to_numpy() if has_hpi else None
            bin_means = _hpi_bin_means(emb, hpi, hpi_bin_hours)
            # std across bins (ddof=0), per dim, then averaged to one scalar.
            std_scalar = float(bin_means.std(axis=0).mean()) if len(bin_means) > 1 else 0.0
            per_marker.setdefault(str(marker), {})[experiment] = (std_scalar, len(bin_means))

    tables: dict[str, pd.DataFrame] = {}
    for marker, rows in per_marker.items():
        datasets = sorted(rows)
        tables[marker] = pd.DataFrame(
            {
                "temporal_std": [rows[d][0] for d in datasets],
                "n_bins": [rows[d][1] for d in datasets],
            },
            index=datasets,
        )
    return tables


def corr_matrix_per_marker(
    input_paths: list[str],
    obs_filter: dict[str, str] | None,
    embedding_key: str | None,
    hpi_bin_hours: float | None = None,
    obs_filter_aliases: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Pearson-correlation matrix of per-dataset mean control embeddings, per marker.

    Loads each dataset zarr, applies the same control ``obs_filter`` as the MMD
    path, computes a summary embedding for every (dataset, marker), and Pearson-
    correlates those summary vectors across datasets. Cheap and bounded
    ``[-1, 1]``, but only sees the centroid — a readable companion to the
    distributional MMD matrix, not a replacement.

    Parameters
    ----------
    input_paths : list[str]
        Per-dataset embedding zarr paths (one dataset each).
    obs_filter : dict[str, str] or None
        ``obs[col] == val`` filter selecting control cells; None keeps all cells.
    embedding_key : str or None
        obsm key to correlate; None uses raw ``.X``.
    hpi_bin_hours : float or None
        None (default) uses one grand mean over all cells per dataset. When set,
        the summary is a mean-of-means over ``hours_post_perturbation`` bins of
        this width (see :func:`_dataset_mean_embedding`), so uneven time sampling
        cannot masquerade as a batch effect.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``marker -> square DataFrame`` (datasets x datasets) of Pearson
        correlation between summary embeddings (diagonal 1). Markers present in
        fewer than two datasets are omitted.
    """
    marker_means: dict[str, dict[str, np.ndarray]] = {}
    for path in input_paths:
        adata = ad.read_zarr(path)
        experiment = adata.obs["experiment"].iloc[0]
        if obs_filter:
            mask = pd.Series(True, index=adata.obs.index)
            for col, val in obs_filter.items():
                if col not in adata.obs.columns:
                    raise KeyError(f"obs_filter column '{col}' not found in {experiment}")
                accepted = (obs_filter_aliases or {}).get(col, [val])
                mask &= adata.obs[col].astype(str).isin(map(str, accepted))
            adata = adata[mask]
        has_hpi = "hours_post_perturbation" in adata.obs.columns
        for marker in adata.obs["marker"].unique():
            sub = adata[adata.obs["marker"] == marker]
            emb = _extract_embeddings(sub, embedding_key).astype(np.float32)
            if len(emb) == 0:
                continue
            hpi = sub.obs["hours_post_perturbation"].to_numpy() if has_hpi else None
            marker_means.setdefault(str(marker), {})[str(experiment)] = _dataset_mean_embedding(emb, hpi, hpi_bin_hours)

    matrices: dict[str, pd.DataFrame] = {}
    for marker, means in marker_means.items():
        datasets = sorted(means)
        if len(datasets) < 2:
            continue
        stacked = np.stack([means[d] for d in datasets])
        corr = np.corrcoef(stacked)
        matrices[marker] = pd.DataFrame(corr, index=datasets, columns=datasets, dtype=float)
    return matrices


def _frechet_distance(mu_a: np.ndarray, cov_a: np.ndarray, mu_b: np.ndarray, cov_b: np.ndarray) -> float:
    """Fréchet (2-Wasserstein between Gaussians) distance between two moment sets.

    ``FD² = ||mu_a - mu_b||² + Tr(cov_a + cov_b - 2 (cov_a cov_b)^½)`` — the same
    statistic as FID, and exactly the mean + covariance shift LOT corrects. The
    matrix square root of the (symmetric PSD) product ``cov_a cov_b`` is taken via
    an eigendecomposition of the symmetrized product with negatives clipped to 0
    (numerical guard), avoiding a scipy ``sqrtm`` dependency.

    Parameters
    ----------
    mu_a, mu_b : np.ndarray
        Per-dataset mean embedding vectors ``(D,)``.
    cov_a, cov_b : np.ndarray
        Per-dataset covariance matrices ``(D, D)``.

    Returns
    -------
    float
        The (non-negative) Fréchet distance ``FD`` (square root of ``FD²``).
    """
    diff = mu_a - mu_b
    prod = cov_a @ cov_b
    eigvals = np.linalg.eigvalsh((prod + prod.T) / 2.0)
    covmean_trace = float(np.sqrt(np.clip(eigvals, 0.0, None)).sum())
    fd2 = float(diff @ diff) + float(np.trace(cov_a + cov_b)) - 2.0 * covmean_trace
    return float(np.sqrt(max(fd2, 0.0)))


def frechet_matrix_per_marker(
    input_paths: list[str],
    obs_filter: dict[str, str] | None,
    embedding_key: str | None,
) -> dict[str, pd.DataFrame]:
    """Fréchet (mean + covariance) distance matrix of control embeddings, per marker.

    Loads each dataset zarr, applies the same control ``obs_filter`` as the MMD
    path, and for every (dataset, marker) computes the mean **and** covariance of
    the control embeddings. The per-marker matrix cell is the Fréchet distance
    (see :func:`_frechet_distance`) between the two datasets' Gaussian summaries.
    Richer than the Pearson-of-means companion (it uses the second moment) and
    cheaper than the full-distribution MMD (closed form, no permutation test) —
    and it is precisely the mean + covariance shift LOT correction removes.

    Parameters
    ----------
    input_paths : list[str]
        Per-dataset embedding zarr paths (one dataset each).
    obs_filter : dict[str, str] or None
        ``obs[col] == val`` filter selecting control cells; None keeps all cells.
    embedding_key : str or None
        obsm key to summarize; None uses raw ``.X``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``marker -> square DataFrame`` (datasets x datasets) of Fréchet
        distance (diagonal 0, symmetric). Markers present in fewer than two
        datasets are omitted. A dataset with only one control cell (no covariance)
        is skipped for that marker.
    """
    marker_moments: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for path in input_paths:
        adata = ad.read_zarr(path)
        experiment = adata.obs["experiment"].iloc[0]
        if obs_filter:
            mask = pd.Series(True, index=adata.obs.index)
            for col, val in obs_filter.items():
                if col not in adata.obs.columns:
                    raise KeyError(f"obs_filter column '{col}' not found in {experiment}")
                mask &= adata.obs[col] == val
            adata = adata[mask]
        for marker in adata.obs["marker"].unique():
            sub = adata[adata.obs["marker"] == marker]
            emb = _extract_embeddings(sub, embedding_key).astype(np.float64)
            if len(emb) < 2:  # need >=2 cells for a covariance
                continue
            mu = emb.mean(axis=0)
            cov = np.cov(emb, rowvar=False)
            marker_moments.setdefault(str(marker), {})[str(experiment)] = (mu, cov)

    matrices: dict[str, pd.DataFrame] = {}
    for marker, moments in marker_moments.items():
        datasets = sorted(moments)
        if len(datasets) < 2:
            continue
        matrix = pd.DataFrame(0.0, index=datasets, columns=datasets, dtype=float)
        for i, da in enumerate(datasets):
            for db in datasets[i + 1 :]:
                mu_a, cov_a = moments[da]
                mu_b, cov_b = moments[db]
                fd = _frechet_distance(mu_a, cov_a, mu_b, cov_b)
                matrix.loc[da, db] = fd
                matrix.loc[db, da] = fd
        matrices[marker] = matrix
    return matrices


def plot_frechet_matrix(matrix: pd.DataFrame, marker: str, output_path: Path) -> None:
    """Plot one symmetric dataset x dataset Fréchet-distance heatmap for a marker.

    Parameters
    ----------
    matrix : pd.DataFrame
        Square symmetric Fréchet-distance matrix (datasets x datasets), diagonal 0.
    marker : str
        Marker name, used in the title.
    output_path : Path
        Output file path.
    """
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.3), max(5, n * 1.1)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="magma",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Fréchet distance", "shrink": 0.7},
    )
    ax.set_title(
        f"Embedding consistency — {marker}\ncontrol cells, mean + covariance (Fréchet) distance",
        pad=12,
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corr_matrix(
    matrix: pd.DataFrame,
    marker: str,
    output_path: Path,
    vmin: float = 0.8,
    representation: str = "raw",
) -> None:
    """Plot one symmetric dataset x dataset Pearson-correlation heatmap for a marker.

    Parameters
    ----------
    matrix : pd.DataFrame
        Square symmetric correlation matrix (datasets x datasets), diagonal 1.
    marker : str
        Marker name, used in the title.
    output_path : Path
        Output file path.
    vmin : float
        Fixed lower bound of the colour scale, shared across all markers so the
        heatmaps are visually comparable (default 0.8). Correlations here cluster
        near 1, so a fixed [-1, 1] scale washes the structure out.
    """
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.3), max(5, n * 1.1)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Pearson r", "shrink": 0.7},
    )
    ax.set_title(
        f"Embedding consistency — {marker}\ncontrol-cell summary correlation "
        f"({representation}; Pearson r similarity; color scaled from {vmin:.2f})",
        pad=12,
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_corr_summary_pdf(
    raw_matrices: dict[str, pd.DataFrame],
    normalized_matrices: dict[str, pd.DataFrame],
    config: EmbeddingConsistencyConfig,
    output_path: Path,
    comparison: pd.DataFrame | None = None,
) -> None:
    """Write one referenceable, multi-page PDF for a model-run Pearson QC.

    The first page records model provenance and the aggregate off-diagonal
    correlations. Each following page contains one marker's raw and normalized
    dataset x dataset matrices side by side. When normalization is disabled,
    the marker pages contain the raw matrix only.
    """
    markers = sorted(set(raw_matrices) | set(normalized_matrices))
    if not markers:
        return

    pca_artifact_dir = Path(config.output_dir) / "pearson_mad_pca80"
    pca_summary_path = pca_artifact_dir / "marker_pca_80pct_summary.csv"
    scree_path = pca_artifact_dir / "marker_pca_scree_curves.csv"
    pca_summary = pd.read_csv(pca_summary_path) if pca_summary_path.exists() else pd.DataFrame()
    scree_curves = pd.read_csv(scree_path) if scree_path.exists() else pd.DataFrame()
    pca_by_marker = (
        pca_summary.assign(marker=pca_summary["marker"].astype(str)).set_index("marker")
        if not pca_summary.empty
        else pd.DataFrame()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        fig.suptitle("Embedding consistency summary", fontsize=20, y=0.95)
        normalization_label = config.pearson_normalization or "none"
        if config.pearson_compare_raw:
            normalization_label = f"precomputed {config.embedding_key}"
        provenance = (
            f"Model family: {config.model_family}\n"
            f"Run: {config.run}\n"
            f"Checkpoint: {config.ckpt_name}\n"
            f"Control filter: {config.obs_filter}\n"
            f"Normalization: {normalization_label}\n"
            f"Markers: {', '.join(markers)}"
        )
        ax.text(0.02, 0.88, provenance, transform=ax.transAxes, va="top", fontsize=11)

        if comparison is not None and not comparison.empty:
            means = comparison.pivot_table(
                index="marker",
                columns="representation",
                values="mean_off_diagonal_pearson",
            )
            medians = comparison.pivot_table(
                index="marker",
                columns="representation",
                values="median_off_diagonal_pearson",
            )
            rows = []
            for marker in sorted(means.index):
                raw_mean = means.loc[marker, "raw"]
                normalized_mean = means.loc[marker, "control_mad_pca80"]
                rows.append(
                    [
                        marker,
                        f"{raw_mean:.3f}",
                        f"{normalized_mean:.3f}",
                        f"{normalized_mean - raw_mean:+.3f}",
                        f"{medians.loc[marker, 'raw']:.3f}",
                        f"{medians.loc[marker, 'control_mad_pca80']:.3f}",
                    ]
                )
            table = ax.table(
                cellText=rows,
                colLabels=[
                    "Marker",
                    "Raw mean r",
                    "Normalized mean r",
                    "Mean delta",
                    "Raw median r",
                    "Normalized median r",
                ],
                cellLoc="center",
                colLoc="center",
                bbox=[0.02, 0.34, 0.96, 0.29],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.auto_set_column_width(col=list(range(6)))
            ax.text(
                0.02,
                0.66,
                "Off-diagonal dataset-pair summary",
                transform=ax.transAxes,
                fontsize=12,
                weight="bold",
            )

            if not pca_summary.empty:
                pca_rows = [
                    [
                        str(row.marker),
                        str(int(row.embedding_dimensions)),
                        str(int(row.n_components)),
                        f"{float(row.realized_variance):.1%}",
                        str(row.reference_dataset),
                    ]
                    for row in pca_summary.itertuples(index=False)
                    if str(row.marker) in markers
                ]
                pca_table = ax.table(
                    cellText=pca_rows,
                    colLabels=["Marker", "Input dims", "PCs used", "Variance", "PCA reference dataset"],
                    cellLoc="center",
                    colLoc="center",
                    bbox=[0.02, 0.04, 0.96, 0.22],
                )
                pca_table.auto_set_font_size(False)
                pca_table.set_fontsize(7.5)
                pca_table.auto_set_column_width(col=list(range(5)))
                ax.text(
                    0.02,
                    0.28,
                    "Marker-specific PCA80 selection",
                    transform=ax.transAxes,
                    fontsize=12,
                    weight="bold",
                )
        else:
            ax.text(
                0.02,
                0.66,
                "Raw Pearson matrices only; normalization comparison was not requested.",
                transform=ax.transAxes,
                fontsize=11,
            )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not pca_summary.empty and not scree_curves.empty:
            scree_markers = [marker for marker in markers if marker in pca_by_marker.index]
            ncols = 2
            nrows = int(np.ceil(len(scree_markers) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 8.27), squeeze=False)
            for ax, marker in zip(axes.flat, scree_markers, strict=False):
                curve = scree_curves[scree_curves["marker"].astype(str) == marker]
                info = pca_by_marker.loc[marker]
                selected = int(info["n_components"])
                target = float(info["target_variance"])
                realized = float(info["realized_variance"])
                max_pc = min(
                    int(curve["pc"].max()),
                    max(20, selected + 10, int(np.ceil(selected * 1.5))),
                )
                visible = curve[curve["pc"] <= max_pc]
                ax.plot(
                    visible["pc"],
                    visible["cumulative_explained_variance"],
                    color="tab:blue",
                    linewidth=1.8,
                )
                ax.axhline(target, color="tab:red", linestyle="--", linewidth=1, label=f"target {target:.0%}")
                ax.axvline(selected, color="black", linestyle=":", linewidth=1, label=f"selected {selected} PCs")
                ax.scatter([selected], [realized], color="black", s=18, zorder=3)
                ax.set_title(
                    f"{marker}: {selected}/{int(info['embedding_dimensions'])} PCs ({realized:.1%} variance)",
                    fontsize=10,
                )
                ax.set_xlabel("Principal component")
                ax.set_ylabel("Cumulative explained variance")
                ax.set_ylim(0, 1.02)
                ax.grid(alpha=0.2)
                ax.legend(loc="lower right", fontsize=7)
            for ax in axes.flat[len(scree_markers) :]:
                ax.axis("off")
            fig.suptitle(
                f"Marker-specific PCA80 scree curves\n{config.model_family} / {config.run} / {config.ckpt_name}",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for marker in markers:
            panels: list[tuple[str, pd.DataFrame]] = []
            if marker in raw_matrices:
                panels.append(("Raw embeddings", raw_matrices[marker]))
            if marker in normalized_matrices:
                normalized_title = "Control MAD + marker PCA80"
                if marker in pca_by_marker.index:
                    info = pca_by_marker.loc[marker]
                    normalized_title += (
                        f"\n{int(info['n_components'])}/{int(info['embedding_dimensions'])} PCs; "
                        f"{float(info['realized_variance']):.1%} variance"
                    )
                panels.append((normalized_title, normalized_matrices[marker]))

            max_datasets = max(len(matrix) for _, matrix in panels)
            fig_width = max(11.69, len(panels) * max(7.0, max_datasets * 1.1))
            fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, 8.27), squeeze=False)
            for ax, (title, matrix) in zip(axes[0], panels, strict=True):
                sns.heatmap(
                    matrix,
                    ax=ax,
                    cmap="RdBu_r",
                    vmin=0.8,
                    vmax=1.0,
                    square=True,
                    linewidths=0.5,
                    annot=True,
                    fmt=".3f",
                    annot_kws={"fontsize": 7},
                    cbar_kws={"label": "Pearson r", "shrink": 0.7},
                )
                ax.set_title(title, fontsize=12, pad=10)
                ax.set_xlabel("Dataset")
                ax.set_ylabel("Dataset")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
            fig.suptitle(
                f"{marker} — dataset x dataset embedding consistency\n"
                f"{config.model_family} / {config.run} / {config.ckpt_name}",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_existing_corr_summary_pdf(config: EmbeddingConsistencyConfig) -> Path:
    """Build the combined PDF from an already completed Pearson QC run.

    This is the non-destructive backfill path used by ``--summary-only``. It
    reads the matrix CSVs and comparison table but does not reopen embeddings or
    rewrite any existing QC artifact.
    """
    if config.split_by is not None:
        raise ValueError("--summary-only currently requires a non-split embedding-consistency config")

    output_dir = Path(config.output_dir)
    comparison_path = output_dir / "consistency_corr_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"completed Pearson comparison not found: {comparison_path}")

    comparison = pd.read_csv(comparison_path)
    markers = sorted(comparison["marker"].astype(str).unique())
    raw_matrices: dict[str, pd.DataFrame] = {}
    normalized_matrices: dict[str, pd.DataFrame] = {}
    for marker in markers:
        safe = marker.replace(" ", "_").replace("/", "-")
        raw_path = output_dir / f"consistency_corr_matrix_{safe}.csv"
        normalized_path = output_dir / f"consistency_corr_matrix_{safe}_mad_pca80.csv"
        if not raw_path.exists() or not normalized_path.exists():
            raise FileNotFoundError(
                f"completed Pearson matrices missing for {marker}: "
                f"raw={raw_path.exists()}, normalized={normalized_path.exists()}"
            )
        raw_matrices[marker] = pd.read_csv(raw_path, index_col=0)
        normalized_matrices[marker] = pd.read_csv(normalized_path, index_col=0)

    output_path = output_dir / "embedding_consistency_summary.pdf"
    write_corr_summary_pdf(
        raw_matrices,
        normalized_matrices,
        config,
        output_path,
        comparison,
    )
    return output_path


def _cross_group_pairs_only(df: pd.DataFrame, group_of: dict[str, str]) -> pd.DataFrame:
    """Keep only the long-form MMD rows whose two datasets are in different groups.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form pairwise MMD (needs ``exp_a``, ``exp_b``).
    group_of : dict[str, str]
        Dataset name -> group label.

    Returns
    -------
    pd.DataFrame
        Only the rows where ``group_of[exp_a] != group_of[exp_b]``.
    """
    keep = df.apply(lambda r: group_of.get(r["exp_a"]) != group_of.get(r["exp_b"]), axis=1)
    return df[keep]


def _stage_precomputed_pca_artifacts(
    input_paths: list[str],
    embedding_key: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Validate exact marker widths and stage pooled PCA audit tables for a QC."""
    artifact_dirs: set[Path] = set()
    markers: set[str] = set()
    fit_store_counts: set[int] = set()
    for input_path in input_paths:
        adata = ad.read_zarr(input_path)
        if embedding_key not in adata.obsm:
            raise KeyError(f"{input_path}: obsm[{embedding_key!r}] not found")
        metadata = adata.uns.get(embedding_key)
        if not isinstance(metadata, dict):
            raise KeyError(f"{input_path}: uns[{embedding_key!r}] provenance not found")
        component_counts = metadata.get("n_components_by_marker", {})
        local_markers = adata.obs["marker"].astype(str).unique()
        for marker in local_markers:
            if marker not in component_counts:
                raise KeyError(f"{input_path}: no PCA component count recorded for {marker}")
            selected = int(component_counts[marker])
            actual = int(adata.obsm[embedding_key].shape[1])
            if actual != selected:
                raise ValueError(
                    f"{input_path}: obsm[{embedding_key!r}] has {actual} columns but "
                    f"marker {marker!r} records {selected}; rerun normalization without padding"
                )
            markers.add(marker)
        artifact_dirs.add(Path(str(metadata["artifact_dir"])))
        fit_store_counts.add(len(metadata.get("fit_input_paths", [])))
    if len(artifact_dirs) != 1:
        raise ValueError(
            f"precomputed PCA provenance spans multiple artifact directories: {sorted(map(str, artifact_dirs))}"
        )

    artifact_dir = artifact_dirs.pop()
    source_summary = artifact_dir / "marker_pca_summary.csv"
    source_scree = artifact_dir / "marker_pca_scree.csv"
    if not source_summary.exists() or not source_scree.exists():
        raise FileNotFoundError(
            f"pooled PCA audit tables missing: summary={source_summary.exists()}, scree={source_scree.exists()} "
            f"under {artifact_dir}"
        )
    summary = pd.read_csv(source_summary)
    summary = summary[summary["marker"].astype(str).isin(markers)].copy()
    fit_count = next(iter(fit_store_counts)) if len(fit_store_counts) == 1 else 0
    summary["reference_dataset"] = f"pooled global fit ({fit_count} stores)"
    staged = output_dir / "pearson_mad_pca80"
    staged.mkdir(parents=True, exist_ok=True)
    summary.to_csv(staged / "marker_pca_80pct_summary.csv", index=False)
    scree = pd.read_csv(source_scree)
    scree[scree["marker"].astype(str).isin(markers)].to_csv(
        staged / "marker_pca_scree_curves.csv",
        index=False,
    )
    return summary


def _precomputed_correlation_summary(
    raw: dict[str, pd.DataFrame],
    normalized: dict[str, pd.DataFrame],
    pca_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize raw versus globally normalized Pearson matrices."""
    info = pca_summary.assign(marker=pca_summary["marker"].astype(str)).set_index("marker")
    rows: list[dict[str, int | float | str]] = []
    for marker in sorted(set(raw) & set(normalized)):
        if marker not in info.index:
            raise KeyError(f"pooled PCA summary does not contain marker {marker!r}")
        marker_info = info.loc[marker]
        for representation, matrix, dimensions in (
            ("raw", raw[marker], int(marker_info["embedding_dimensions"])),
            ("control_mad_pca80", normalized[marker], int(marker_info["n_components"])),
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
                    "n_components": dimensions,
                    "pca_reference_dataset": str(marker_info["reference_dataset"]),
                }
            )
    return pd.DataFrame(rows)


def _write_matrices(
    df: pd.DataFrame | None,
    input_paths: list[str],
    config: EmbeddingConsistencyConfig,
    output_dir: Path,
    suffix: str = "",
    pca_fits: dict[str, MarkerPCAFit] | None = None,
) -> None:
    """Pivot + write the requested (``config.metrics``) matrices for one block.

    Parameters
    ----------
    df : pd.DataFrame or None
        Long-form pairwise MMD restricted to this block's dataset pairs, or None
        when ``"mmd"`` is not in ``config.metrics`` (the MMD pass was skipped).
    input_paths : list[str]
        The block's per-dataset zarrs (for the Pearson / Fréchet matrices).
    config : EmbeddingConsistencyConfig
        QC config (metrics / obs_filter / embedding_key / save_plots).
    output_dir : Path
        Directory to write into (created by the caller).
    suffix : str
        Appended to each filename stem (e.g. ``"_mantis_v2"``) to keep blocks
        from colliding when several are written under one dir.
    """
    if "mmd" in config.metrics and df is not None:
        df.to_csv(output_dir / f"consistency_mmd_results{suffix}.csv", index=False)
        for marker, matrix in mmd_matrix_per_marker(df).items():
            safe = marker.replace(" ", "_").replace("/", "-")
            matrix.to_csv(output_dir / f"consistency_mmd_matrix_{safe}{suffix}.csv")
            if config.save_plots:
                for fmt in ("pdf", "png"):
                    plot_consistency_matrix(matrix, marker, output_dir / f"consistency_mmd_matrix_{safe}{suffix}.{fmt}")

    if "pearson" in config.metrics:
        selected_matrices = corr_matrix_per_marker(
            input_paths,
            config.obs_filter,
            config.embedding_key,
            config.pearson_hpi_bin_hours,
            config.obs_filter_aliases,
        )
        corr_matrices = selected_matrices
        normalized_matrices: dict[str, pd.DataFrame] = {}
        comparison: pd.DataFrame | None = None
        precomputed_summary: pd.DataFrame | None = None
        if config.pearson_compare_raw:
            if config.embedding_key is None:
                raise ValueError("pearson_compare_raw requires a precomputed embedding_key")
            if pca_fits is not None:
                raise ValueError("pearson_compare_raw cannot be combined with pearson_normalization")
            corr_matrices = corr_matrix_per_marker(
                input_paths,
                config.obs_filter,
                None,
                config.pearson_hpi_bin_hours,
                config.obs_filter_aliases,
            )
            normalized_matrices = selected_matrices
            precomputed_summary = _stage_precomputed_pca_artifacts(
                input_paths,
                config.embedding_key,
                output_dir,
            )

        for marker, matrix in corr_matrices.items():
            safe = marker.replace(" ", "_").replace("/", "-")
            matrix.to_csv(output_dir / f"consistency_corr_matrix_{safe}{suffix}.csv")
            if config.save_plots:
                for fmt in ("pdf", "png"):
                    plot_corr_matrix(matrix, marker, output_dir / f"consistency_corr_matrix_{safe}{suffix}.{fmt}")

        if pca_fits is not None:
            normalized_matrices, normalization_table = normalized_corr_matrix_per_marker(input_paths, config, pca_fits)
            normalization_table.to_csv(
                output_dir / f"consistency_control_mad_summary{suffix}.csv",
                index=False,
            )
            comparison = correlation_summary(corr_matrices, normalized_matrices, pca_fits)
        elif precomputed_summary is not None:
            comparison = _precomputed_correlation_summary(
                corr_matrices,
                normalized_matrices,
                precomputed_summary,
            )

        for marker, matrix in normalized_matrices.items():
            safe = marker.replace(" ", "_").replace("/", "-")
            stem = f"consistency_corr_matrix_{safe}_mad_pca80{suffix}"
            matrix.to_csv(output_dir / f"{stem}.csv")
            if config.save_plots:
                for fmt in ("pdf", "png"):
                    plot_corr_matrix(
                        matrix,
                        marker,
                        output_dir / f"{stem}.{fmt}",
                        representation="control MAD + marker PCA80",
                    )
        if comparison is not None:
            comparison.to_csv(
                output_dir / f"consistency_corr_comparison{suffix}.csv",
                index=False,
            )

        if config.save_plots:
            write_corr_summary_pdf(
                corr_matrices,
                normalized_matrices,
                config,
                output_dir / f"embedding_consistency_summary{suffix}.pdf",
                comparison,
            )

        # Temporal drift diagnostic (needs HPI bins) — std of the control centroid
        # across HPI bins, per (dataset, marker). Context for the matrices above.
        if config.pearson_hpi_bin_hours is not None:
            std_tables = temporal_std_per_marker(
                input_paths,
                config.obs_filter,
                config.embedding_key,
                config.pearson_hpi_bin_hours,
                config.obs_filter_aliases,
            )
            for marker, table in std_tables.items():
                safe = marker.replace(" ", "_").replace("/", "-")
                table.to_csv(output_dir / f"consistency_temporal_std_{safe}{suffix}.csv")

    if "frechet" not in config.metrics:
        return
    for marker, matrix in frechet_matrix_per_marker(input_paths, config.obs_filter, config.embedding_key).items():
        safe = marker.replace(" ", "_").replace("/", "-")
        matrix.to_csv(output_dir / f"consistency_frechet_matrix_{safe}{suffix}.csv")
        if config.save_plots:
            for fmt in ("pdf", "png"):
                plot_frechet_matrix(matrix, marker, output_dir / f"consistency_frechet_matrix_{safe}{suffix}.{fmt}")


def _run_split_qc(
    config: EmbeddingConsistencyConfig,
    input_paths: list[str],
    output_dir: Path,
    pca_fits: dict[str, MarkerPCAFit] | None = None,
) -> pd.DataFrame | None:
    """Partition datasets by ``config.split_by`` and emit within + cross blocks.

    For each group value, writes a within-group block (needs >=2 datasets); for
    each pair of groups, writes a cross-group block containing only the
    across-group dataset pairs (needs >=1 dataset on each side). Degenerate
    blocks are skipped with a log line. Blocks land in per-block subdirectories
    of ``output_dir`` (``within_{group}/`` and ``cross_{a}__{b}/``).

    Returns the full long-form MMD over all datasets (also written at the top
    level) so callers still get the complete pairwise record.
    """
    # One obs read per zarr: dataset name + its split-group label.
    name_of: dict[str, str] = {}
    group_of: dict[str, str] = {}
    paths_of_group: dict[str, list[str]] = {}
    for p in input_paths:
        obs = ad.read_zarr(p).obs
        name = str(obs["experiment"].iloc[0])
        if config.split_by not in obs.columns:
            raise KeyError(f"split_by column '{config.split_by}' not found in obs of {p}")
        values = obs[config.split_by].astype(str).unique()
        if len(values) != 1:
            raise ValueError(f"split_by column '{config.split_by}' is not constant in {p}: found {sorted(values)}")
        name_of[p] = name
        group_of[name] = str(values[0])
        paths_of_group.setdefault(str(values[0]), []).append(p)
    groups = sorted(paths_of_group)
    click.echo(f"split_by={config.split_by!r}: {[(g, len(paths_of_group[g])) for g in groups]}")

    # Full pairwise MMD once (all datasets); blocks filter this + re-pivot.
    # Skipped entirely when "mmd" is not requested (Pearson/Fréchet need no df).
    df: pd.DataFrame | None = None
    if "mmd" in config.metrics:
        df = run_mmd_combined(_combined_config(config, input_paths))
        df.to_csv(output_dir / "consistency_mmd_results.csv", index=False)

    # Within-group blocks. Degeneracy is measured in DISTINCT DATASETS, not zarr
    # paths (one dataset contributes several per-marker zarrs).
    for g in groups:
        g_paths = paths_of_group[g]
        n_datasets = len({name_of[p] for p in g_paths})
        safe_g = str(g).replace(" ", "_").replace("/", "-")
        if n_datasets < 2:
            click.echo(f"  [skip] within '{g}': only {n_datasets} dataset(s), need >=2")
            continue
        names = {name_of[p] for p in g_paths}
        sub = None if df is None else df[df["exp_a"].isin(names) & df["exp_b"].isin(names)]
        block_dir = output_dir / f"within_{safe_g}"
        block_dir.mkdir(parents=True, exist_ok=True)
        _write_matrices(
            sub,
            g_paths,
            config,
            block_dir,
            suffix=f"_{safe_g}",
            pca_fits=pca_fits,
        )
        click.echo(f"  [write] within '{g}': {n_datasets} datasets -> {block_dir}")

    # Cross-group blocks (one per unordered pair of groups).
    for i, ga in enumerate(groups):
        for gb in groups[i + 1 :]:
            a_paths, b_paths = paths_of_group[ga], paths_of_group[gb]
            if not a_paths or not b_paths:
                click.echo(f"  [skip] cross '{ga}' x '{gb}': empty side")
                continue
            safe_a = str(ga).replace(" ", "_").replace("/", "-")
            safe_b = str(gb).replace(" ", "_").replace("/", "-")
            pair_paths = a_paths + b_paths
            names = {name_of[p] for p in pair_paths}
            sub = (
                None
                if df is None
                else _cross_group_pairs_only(df[df["exp_a"].isin(names) & df["exp_b"].isin(names)], group_of)
            )
            block_dir = output_dir / f"cross_{safe_a}__{safe_b}"
            block_dir.mkdir(parents=True, exist_ok=True)
            _write_matrices(
                sub,
                pair_paths,
                config,
                block_dir,
                suffix=f"_{safe_a}__{safe_b}",
                pca_fits=pca_fits,
            )
            click.echo(f"  [write] cross '{ga}' x '{gb}' -> {block_dir}")

    return df


def _combined_config(config: EmbeddingConsistencyConfig, input_paths: list[str]) -> MMDCombinedConfig:
    """Build the ``MMDCombinedConfig`` for a set of input paths from the QC config."""
    return MMDCombinedConfig(
        input_paths=input_paths,
        output_dir=config.output_dir,
        group_by=config.group_by,
        obs_filter=config.obs_filter,
        embedding_key=config.embedding_key,
        mmd=config.mmd,
        map_settings=config.map_settings,
        temporal_bin_size=config.temporal_bin_size,
        temporal_bins=config.temporal_bins,
        save_plots=config.save_plots,
        center_per_experiment=config.center_per_experiment,
    )


def run_consistency_qc(
    config: EmbeddingConsistencyConfig,
    input_paths: list[str] | None = None,
) -> pd.DataFrame | None:
    """Run the per-marker embedding-consistency QC end to end.

    Enumerates the embedding zarrs for the configured model/run/checkpoint
    across datasets, runs pairwise cross-dataset MMD on control cells, writes the
    long-form CSV plus one square-matrix CSV and heatmap per marker, and returns
    the long-form DataFrame.

    Parameters
    ----------
    config : EmbeddingConsistencyConfig
        QC configuration (provenance tuple, control filter, MMD settings).

    Returns
    -------
    pd.DataFrame
        The long-form pairwise MMD results (same schema as ``run_mmd_combined``).

    Notes
    -----
    The two matrices do **not** always summarize the same population:

    - **MMD²** partitions each dataset pair by ``group_by`` condition and, when
      ``temporal_bin_size``/``temporal_bins`` is set, by temporal bin. It runs
      one MMD² per (condition, bin) and the matrix cell is the **mean** of those
      values (see :func:`mmd_matrix_per_marker`). With the default (no temporal
      bins, ``obs_filter`` collapsing ``group_by`` to a single value) this is a
      single MMD² over all control cells at all timepoints pooled. The per-bin
      values survive in ``consistency_mmd_results.csv`` even though the matrix
      shows only their average.
    - **Pearson correlation** ignores ``group_by`` and time entirely: it pools
      every cell passing ``obs_filter`` into one mean embedding per dataset
      (see :func:`corr_matrix_per_marker`).

    Consequence: with no temporal bins the two matrices describe the same
    population, but **turning on temporal bins makes the MMD matrix a
    mean-over-bins statistic while the Pearson matrix stays all-time pooled** —
    they then measure different things. Per-bin-then-average (MMD) controls for
    differences in *time sampling* between acquisitions, which is usually the
    right batch-effect choice, but the matrix itself is not time-resolved.

    The heatmaps display **MMD² magnitude only, not the permutation p-value.**
    At the cell counts typical here (thousands per group), the permutation test
    saturates: every dataset pair hits the ``1/(n_permutations+1)`` p-value floor,
    so p cannot distinguish a trivial offset from a real batch effect — only the
    magnitude does. The per-pair ``p_value`` (and ``effect_size``, ``n_a``,
    ``n_b``) stay in ``consistency_mmd_results.csv`` for the record.
    """
    datasets_root = config.datasets_root if config.datasets_root is not None else DATASETS_ROOT
    if input_paths is None:
        input_paths = [
            str(p)
            for p in iter_embeddings(
                config.model_family,
                config.run,
                config.ckpt_name,
                datasets_root=datasets_root,
            )
        ]
    if len(input_paths) < 2:
        raise ValueError(
            f"embedding-consistency QC needs >=2 dataset zarrs for "
            f"{config.model_family}/{config.run}/{config.ckpt_name} under {datasets_root}, "
            f"found {len(input_paths)}: {input_paths}"
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_fits = None
    if "pearson" in config.metrics and config.pearson_normalization == "control_mad_pca80":
        pca_fits = fit_marker_pca_models(input_paths, config, output_dir)

    if config.split_by is not None:
        return _run_split_qc(config, input_paths, output_dir, pca_fits)

    df = run_mmd_combined(_combined_config(config, input_paths)) if "mmd" in config.metrics else None
    _write_matrices(df, input_paths, config, output_dir, pca_fits=pca_fits)
    return df


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to embedding-consistency QC YAML config",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Build the combined PDF from existing matrix CSVs without rerunning QC",
)
def main(config: Path, summary_only: bool) -> None:
    """Compute the per-marker dataset x dataset embedding-consistency MMD matrix."""
    raw = load_composed_config(config)
    cfg = EmbeddingConsistencyConfig(**raw)
    if summary_only:
        output_path = write_existing_corr_summary_pdf(cfg)
        click.echo(f"Saved embedding-consistency summary to: {output_path}")
        return
    run_consistency_qc(cfg)
    click.echo(f"Saved consistency QC (metrics={cfg.metrics}) to: {cfg.output_dir}")
