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
    ax.set_title(f"Embedding consistency — {marker}\ncontrol cells, dataset × dataset MMD²", pad=12)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def corr_matrix_per_marker(
    input_paths: list[str],
    obs_filter: dict[str, str] | None,
    embedding_key: str | None,
) -> dict[str, pd.DataFrame]:
    """Pearson-correlation matrix of per-dataset mean control embeddings, per marker.

    Loads each dataset zarr, applies the same control ``obs_filter`` as the MMD
    path, computes the mean embedding for every (dataset, marker), and Pearson-
    correlates those mean vectors across datasets. Cheap and bounded ``[-1, 1]``,
    but only sees the centroid — a readable companion to the distributional MMD
    matrix, not a replacement.

    Parameters
    ----------
    input_paths : list[str]
        Per-dataset embedding zarr paths (one dataset each).
    obs_filter : dict[str, str] or None
        ``obs[col] == val`` filter selecting control cells; None keeps all cells.
    embedding_key : str or None
        obsm key to correlate; None uses raw ``.X``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``marker -> square DataFrame`` (datasets x datasets) of Pearson
        correlation between mean embeddings (diagonal 1). Markers present in
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
        for marker in adata.obs["marker"].unique():
            sub = adata[adata.obs["marker"] == marker]
            emb = _extract_embeddings(sub, embedding_key).astype(np.float32)
            if len(emb) == 0:
                continue
            marker_means.setdefault(str(marker), {})[str(experiment)] = emb.mean(axis=0)

    matrices: dict[str, pd.DataFrame] = {}
    for marker, means in marker_means.items():
        datasets = sorted(means)
        if len(datasets) < 2:
            continue
        stacked = np.stack([means[d] for d in datasets])
        corr = np.corrcoef(stacked)
        matrices[marker] = pd.DataFrame(corr, index=datasets, columns=datasets, dtype=float)
    return matrices


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
    fig, ax = plt.subplots(figsize=(max(6, n * 1.3), max(5, n * 1.1)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r", "shrink": 0.7},
    )
    ax.set_title(f"Embedding consistency — {marker}\ncontrol cells, mean-embedding correlation", pad=12)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_consistency_qc(config: EmbeddingConsistencyConfig) -> pd.DataFrame:
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

    combined = MMDCombinedConfig(
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
    df = run_mmd_combined(combined)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "consistency_mmd_results.csv", index=False)

    mmd_matrices = mmd_matrix_per_marker(df)
    for marker, matrix in mmd_matrices.items():
        safe = marker.replace(" ", "_").replace("/", "-")
        matrix.to_csv(output_dir / f"consistency_mmd_matrix_{safe}.csv")
        if config.save_plots:
            for fmt in ("pdf", "png"):
                plot_consistency_matrix(matrix, marker, output_dir / f"consistency_mmd_matrix_{safe}.{fmt}")

    corr_matrices = corr_matrix_per_marker(input_paths, config.obs_filter, config.embedding_key)
    for marker, matrix in corr_matrices.items():
        safe = marker.replace(" ", "_").replace("/", "-")
        matrix.to_csv(output_dir / f"consistency_corr_matrix_{safe}.csv")
        if config.save_plots:
            for fmt in ("pdf", "png"):
                plot_corr_matrix(matrix, marker, output_dir / f"consistency_corr_matrix_{safe}.{fmt}")
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
    df = run_consistency_qc(cfg)
    n_markers = df["marker"].nunique() if len(df) else 0
    click.echo(f"Saved consistency QC ({n_markers} marker matrices) to: {cfg.output_dir}")
