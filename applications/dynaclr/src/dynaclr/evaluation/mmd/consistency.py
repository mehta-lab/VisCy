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

from dynaclr.evaluation.mmd.compute_mmd import _extract_embeddings, run_mmd_combined
from dynaclr.evaluation.mmd.config import EmbeddingConsistencyConfig, MMDCombinedConfig
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
                mask &= adata.obs[col] == val
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
                mask &= adata.obs[col] == val
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


def plot_corr_matrix(matrix: pd.DataFrame, marker: str, output_path: Path) -> None:
    """Plot one symmetric dataset x dataset Pearson-correlation heatmap for a marker.

    Parameters
    ----------
    matrix : pd.DataFrame
        Square symmetric correlation matrix (datasets x datasets), diagonal 1.
    marker : str
        Marker name, used in the title.
    output_path : Path
        Output file path.
    """
    n = len(matrix)
    # Correlations here cluster near 1; a fixed [-1, 1] scale washes the structure
    # out. Anchor vmin at the off-diagonal minimum (floored a touch) so real
    # differences are visible, and annotate the cells with the r values.
    off = matrix.to_numpy()[~np.eye(n, dtype=bool)]
    vmin = float(np.floor(off.min() * 20) / 20) if off.size else -1.0  # nearest 0.05 below
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
        f"Embedding consistency — {marker}\ncontrol cells, mean-embedding correlation "
        f"(similarity; color scaled from {vmin:.2f})",
        pad=12,
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def _write_matrices(
    df: pd.DataFrame | None,
    input_paths: list[str],
    config: EmbeddingConsistencyConfig,
    output_dir: Path,
    suffix: str = "",
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
        corr_matrices = corr_matrix_per_marker(
            input_paths, config.obs_filter, config.embedding_key, config.pearson_hpi_bin_hours
        )
        for marker, matrix in corr_matrices.items():
            safe = marker.replace(" ", "_").replace("/", "-")
            matrix.to_csv(output_dir / f"consistency_corr_matrix_{safe}{suffix}.csv")
            if config.save_plots:
                for fmt in ("pdf", "png"):
                    plot_corr_matrix(matrix, marker, output_dir / f"consistency_corr_matrix_{safe}{suffix}.{fmt}")

        # Temporal drift diagnostic (needs HPI bins) — std of the control centroid
        # across HPI bins, per (dataset, marker). Context for the matrices above.
        if config.pearson_hpi_bin_hours is not None:
            std_tables = temporal_std_per_marker(
                input_paths, config.obs_filter, config.embedding_key, config.pearson_hpi_bin_hours
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
        _write_matrices(sub, g_paths, config, block_dir, suffix=f"_{safe_g}")
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
            _write_matrices(sub, pair_paths, config, block_dir, suffix=f"_{safe_a}__{safe_b}")
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


def run_consistency_qc(config: EmbeddingConsistencyConfig) -> pd.DataFrame | None:
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
    input_paths = [
        str(p) for p in iter_embeddings(config.model_family, config.run, config.ckpt_name, datasets_root=datasets_root)
    ]
    if len(input_paths) < 2:
        raise ValueError(
            f"embedding-consistency QC needs >=2 dataset zarrs for "
            f"{config.model_family}/{config.run}/{config.ckpt_name} under {datasets_root}, "
            f"found {len(input_paths)}: {input_paths}"
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.split_by is not None:
        return _run_split_qc(config, input_paths, output_dir)

    df = run_mmd_combined(_combined_config(config, input_paths)) if "mmd" in config.metrics else None
    _write_matrices(df, input_paths, config, output_dir)
    return df


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to embedding-consistency QC YAML config",
)
def main(config: Path) -> None:
    """Compute the per-marker dataset x dataset embedding-consistency MMD matrix."""
    raw = load_composed_config(config)
    cfg = EmbeddingConsistencyConfig(**raw)
    run_consistency_qc(cfg)
    click.echo(f"Saved consistency QC (metrics={cfg.metrics}) to: {cfg.output_dir}")
