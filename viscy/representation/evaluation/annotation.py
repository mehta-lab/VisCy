from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import xarray as xr
from natsort import natsorted

from viscy.representation.embedding_writer import get_available_index_columns


def convert(
    embeddings_ds: xr.Dataset | Path,
    output_path: Path,
    overwrite: bool = False,
    return_anndata: bool = False,
) -> ad.AnnData | None:
    """
    Convert an Xarray embeddings dataset to an AnnData object.

    Parameters
    ----------
    embeddings_ds : xr.Dataset | Path
        The Xarray embeddings dataset to convert or the path to the embeddings dataset.
    output_path : Path
        Path to the zarr store to write the AnnData object to.
    overwrite : bool, optional
        Whether to overwrite existing zarr store, by default False.
    return_anndata : bool, optional
        Whether to return the AnnData object, by default False.

    Returns
    -------
    ad.AnnData | None
        The AnnData object if return_anndata is True, otherwise None.

    Raises
    ------
    FileExistsError
        If output_path exists and overwrite is False.

    Examples
    --------
    >>> embeddings_ds = xr.open_zarr(embeddings_path)
    >>> adata = convert_xarray_annotation_to_anndata(embeddings_ds, output_path, overwrite=True, return_anndata=True)
    >>> adata
    AnnData object with n_obs × n_vars = 18861 × 768
        obs: 'id', 'fov_name', 'track_id', 'parent_track_id', 'parent_id', 't', 'y', 'x'
        obsm: 'X_projections', 'X_PCA', 'X_UMAP', 'X_PHATE'
    """
    # Check if output_path exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output path {output_path} already exists.")

    # Tracking
    if isinstance(embeddings_ds, Path):
        embeddings_ds = xr.open_zarr(embeddings_ds)

    available_cols = get_available_index_columns(embeddings_ds)
    tracking_df = pd.DataFrame(
        {
            col: embeddings_ds.coords[col].data
            if col != "fov_name"
            else embeddings_ds.coords[col].to_pandas().str.strip("/")
            for col in available_cols
        }
    )

    obsm = {}
    # Projections
    if "projections" in embeddings_ds.coords:
        obsm["X_projections"] = embeddings_ds.coords["projections"].data

    # Embeddings
    for embedding in ["PCA", "UMAP", "PHATE"]:
        embedding_coords = natsorted(
            [coord for coord in embeddings_ds.coords if embedding in coord]
        )
        if embedding_coords:
            obsm[f"X_{embedding.lower()}"] = np.column_stack(
                [embeddings_ds.coords[coord] for coord in embedding_coords]
            )

    # X, "expression" matrix (NN embedding features)
    X = embeddings_ds["features"].data

    adata = ad.AnnData(X=X, obs=tracking_df, obsm=obsm)

    adata.write_zarr(output_path)
    if return_anndata:
        return adata
