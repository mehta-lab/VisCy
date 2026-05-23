"""Print summary information about an AnnData zarr store."""

import warnings
from pathlib import Path

import click
import numpy as np


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def main(path: Path):
    """Print summary of an AnnData zarr store."""
    import anndata as ad
    import scipy.sparse as sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = ad.read_zarr(path)

    click.echo(f"Path:  {path}")
    click.echo(f"Shape: {adata.n_obs:,} obs × {adata.n_vars:,} vars")
    X = adata.X
    if sp.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X
    sparse = sp.issparse(adata.X)
    xmin, xmax = np.nanmin(X_dense), np.nanmax(X_dense)
    click.echo(f"X:     dtype={X_dense.dtype}, sparse={sparse}, range=[{xmin:.4f}, {xmax:.4f}]")

    if len(adata.obs.columns):
        click.echo("\nobs columns:")
        for col in adata.obs.columns:
            s = adata.obs[col]
            nuniq = s.nunique()
            if nuniq <= 10:
                vals = ", ".join(str(v) for v in sorted(s.dropna().unique()[:10]))
                click.echo(f"  {col}: {s.dtype}, {nuniq} unique — [{vals}]")
            else:
                click.echo(f"  {col}: {s.dtype}, {nuniq} unique")

    if adata.obsm:
        click.echo("\nobsm:")
        for k, v in adata.obsm.items():
            click.echo(f"  {k}: {v.shape}, dtype={v.dtype}, range=[{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]")

    if adata.uns:
        click.echo("\nuns:")
        for k, v in adata.uns.items():
            click.echo(f"  {k}: {v}")

    if adata.layers:
        click.echo("\nlayers:")
        for k, v in adata.layers.items():
            click.echo(f"  {k}: {v.shape}, dtype={v.dtype}")
