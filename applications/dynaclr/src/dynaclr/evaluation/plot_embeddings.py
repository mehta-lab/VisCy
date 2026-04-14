"""CLI tool for generating scatter plots from AnnData embedding stores.

For high-dimensional embeddings (PCA): generates a seaborn pairplot of the
first N components, one figure per color variable.
For low-dimensional embeddings (PHATE, UMAP): generates a simple scatter
colored by each metadata column.

Usage
-----
dynaclr plot-embeddings -c plot_config.yaml
"""

from pathlib import Path
from typing import Optional

import anndata as ad
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from viscy_utils.cli_utils import load_config


class PlotEmbeddingsConfig(BaseModel):
    """Configuration for plot-embeddings command.

    Parameters
    ----------
    input_path : str, optional
        Path to a single AnnData zarr store. Mutually exclusive with input_paths.
    input_paths : list[str], optional
        Paths to multiple AnnData zarr stores. All are concatenated before plotting.
        Use for combined embeddings (X_pca_combined, X_phate_combined) to get one
        figure across all experiments. Mutually exclusive with input_path.
    output_dir : str
        Directory to save plots.
    embedding_keys : list[str]
        obsm keys to plot (e.g. X_phate, X_pca).
    color_by : list[str]
        obs columns to use as hue in pairplots / color in scatter plots.
    pairplot_components : int
        Number of leading components to include in pairplots. Default: 10.
    point_size : float
        Scatter plot point size (passed as ``s`` to matplotlib and
        ``plot_kws`` to seaborn). Default: 1.0.
    format : str
        Output format: "pdf", "png", or "both". Default: "pdf".
    low_dim_threshold : int
        Embeddings with <= this many components use the simple scatter path
        instead of pairplot. Default: 4.
    """

    input_path: Optional[str] = None
    input_paths: Optional[list[str]] = None
    output_dir: str = Field(...)
    embedding_keys: list[str] = ["X_pca_combined", "X_phate_combined"]
    color_by: list[str] = ["perturbation", "hours_post_perturbation", "experiment", "marker"]
    pairplot_components: int = 10
    point_size: float = 1.0
    format: str = "pdf"
    low_dim_threshold: int = 4

    @model_validator(mode="after")
    def validate_input(self):
        if self.input_path is None and self.input_paths is None:
            raise ValueError("Either input_path or input_paths must be provided")
        if self.input_path is not None and self.input_paths is not None:
            raise ValueError("Provide either input_path or input_paths, not both")
        return self


_PALETTE = [
    "#1b69a1",
    "#d9534f",
    "#5cb85c",
    "#f0ad4e",
    "#9b59b6",
    "#1abc9c",
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#e67e22",
]


def _save_fig(fig: plt.Figure, output_dir: Path, stem: str, fmt: str) -> None:
    if fmt in ("pdf", "both"):
        fig.savefig(output_dir / f"{stem}.pdf", dpi=150, bbox_inches="tight")
    if fmt in ("png", "both"):
        fig.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"  Saved {stem}.{fmt}")


def _pairplot(
    emb: np.ndarray,
    obs: pd.DataFrame,
    color_col: str,
    n_components: int,
    point_size: float,
    emb_key: str,
) -> plt.Figure:
    """Build a seaborn pairplot of the first n_components."""
    import seaborn as sns

    n = min(n_components, emb.shape[1])
    cols = [f"{emb_key}_{i}" for i in range(n)]
    df = pd.DataFrame(emb[:, :n], columns=cols)

    values = obs[color_col].to_numpy()
    is_categorical = values.dtype.kind in ("U", "O", "S") or hasattr(values, "cat")

    if is_categorical:
        cats = sorted(str(v) for v in np.unique(values))
        palette = {cat: _PALETTE[i % len(_PALETTE)] for i, cat in enumerate(cats)}
        df[color_col] = [str(v) for v in values]
        pg = sns.pairplot(
            df,
            hue=color_col,
            palette=palette,
            plot_kws={"s": point_size, "alpha": 0.4, "rasterized": True, "zorder": 0},
            diag_kind="hist",
            corner=True,
        )
        pg.legend.set(title=color_col)
        for lh in pg.legend.legend_handles:
            lh.set_alpha(1.0)
            if hasattr(lh, "set_sizes"):
                lh.set_sizes([40])
            else:
                lh.set_markersize(8)
        for ax_row in pg.axes:
            for ax in ax_row:
                if ax is not None:
                    ax.set_rasterization_zorder(1)
    else:
        # Continuous: no hue support in pairplot — use a custom scatter matrix
        df[color_col] = values.astype(float)
        pg = sns.pairplot(
            df,
            plot_kws={"s": point_size, "alpha": 0.4, "rasterized": True, "color": "#888888", "zorder": 0},
            diag_kind="hist",
            corner=True,
        )
        # Overlay color on lower-triangle axes
        norm = plt.Normalize(df[color_col].min(), df[color_col].max())
        cmap = plt.cm.viridis
        for i in range(1, n):
            for j in range(i):
                ax = pg.axes[i][j]
                if ax is None:
                    continue
                ax.collections[0].set_visible(False)
                sc = ax.scatter(
                    df.iloc[:, j],
                    df.iloc[:, i],
                    c=df[color_col],
                    cmap=cmap,
                    norm=norm,
                    s=point_size,
                    alpha=0.4,
                    rasterized=True,
                    zorder=0,
                )
        pg.figure.colorbar(sc, ax=pg.axes[-1][-1], label=color_col)
        for ax_row in pg.axes:
            for ax in ax_row:
                if ax is not None:
                    ax.set_rasterization_zorder(1)

    pg.figure.suptitle(f"{emb_key} — {color_col}", y=1.01, fontsize=11, fontweight="bold")
    return pg.figure


def _scatter_2d(
    emb: np.ndarray,
    obs: pd.DataFrame,
    color_cols: list[str],
    point_size: float,
    emb_key: str,
) -> plt.Figure:
    """Simple scatter for low-dimensional embeddings (PHATE, UMAP)."""
    ncols = min(4, len(color_cols))
    nrows = (len(color_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    rng = np.random.default_rng(42)
    shuffle = rng.permutation(len(emb))
    x, y = emb[shuffle, 0], emb[shuffle, 1]

    for ax_idx, col in enumerate(color_cols):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        values = obs[col].to_numpy()[shuffle]
        is_categorical = values.dtype.kind in ("U", "O", "S") or hasattr(values, "cat")

        if is_categorical:
            cats = sorted(str(v) for v in np.unique(values))
            for i, cat in enumerate(cats):
                mask = np.array([str(v) == cat for v in values])
                ax.scatter(
                    x[mask], y[mask], s=point_size, c=_PALETTE[i % len(_PALETTE)], label=cat, alpha=0.5, rasterized=True
                )
            ax.legend(
                markerscale=6, fontsize=10, loc="best", framealpha=1.0, edgecolor="black", ncol=max(1, len(cats) // 8)
            )
        else:
            sc = ax.scatter(x, y, s=point_size, c=values.astype(float), cmap="viridis", alpha=0.5, rasterized=True)
            plt.colorbar(sc, ax=ax, shrink=0.8)

        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel(f"{emb_key} 0")
        ax.set_ylabel(f"{emb_key} 1")

    for ax_idx in range(len(color_cols), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.suptitle(f"Embeddings: {emb_key}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path) -> None:
    """Generate pairplots (PCA) and scatter plots (PHATE/UMAP) from an AnnData store."""
    matplotlib.use("Agg")

    raw = load_config(config)
    cfg = PlotEmbeddingsConfig(**raw)

    if cfg.input_paths is not None:
        click.echo(f"Concatenating {len(cfg.input_paths)} zarr stores...")
        adata = ad.concat([ad.read_zarr(p) for p in cfg.input_paths], join="outer")
    else:
        adata = ad.read_zarr(cfg.input_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_color_cols = [c for c in cfg.color_by if c in adata.obs.columns]
    missing = set(cfg.color_by) - set(valid_color_cols)
    if missing:
        click.echo(f"Warning: obs columns not found, skipping: {sorted(missing)}", err=True)
    if not valid_color_cols:
        click.echo("No valid color columns found, nothing to plot.", err=True)
        return

    for emb_key in cfg.embedding_keys:
        if emb_key not in adata.obsm:
            click.echo(f"Warning: {emb_key} not in obsm, skipping", err=True)
            continue

        emb = np.asarray(adata.obsm[emb_key])
        click.echo(f"Plotting {emb_key} ({emb.shape[1]} components)...")

        if emb.shape[1] <= cfg.low_dim_threshold:
            # Simple scatter (PHATE, UMAP)
            fig = _scatter_2d(emb, adata.obs, valid_color_cols, cfg.point_size, emb_key)
            _save_fig(fig, output_dir, f"scatter_{emb_key}", cfg.format)
        else:
            # Pairplot per color variable (PCA)
            for col in valid_color_cols:
                try:
                    fig = _pairplot(emb, adata.obs, col, cfg.pairplot_components, cfg.point_size, emb_key)
                    _save_fig(fig, output_dir, f"pairplot_{emb_key}_{col}", cfg.format)
                except Exception as e:
                    click.echo(f"  Warning: pairplot {emb_key}/{col} failed: {e}", err=True)


if __name__ == "__main__":
    main()
