from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from viscy_utils.callbacks.embedding_writer import get_available_index_columns

if TYPE_CHECKING:
    import anndata as ad


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
    ImportError
        If anndata or natsort are not installed.

    Examples
    --------
    >>> embeddings_ds = xr.open_zarr(embeddings_path)
    >>> adata = convert(embeddings_ds, output_path, overwrite=True, return_anndata=True)
    >>> adata
    AnnData object with n_obs × n_vars = 18861 × 768
        obs: 'id', 'fov_name', 'track_id', 'parent_track_id', 'parent_id', 't', 'y', 'x'
        obsm: 'X_projections', 'X_pca', 'X_umap', 'X_phate'
    """
    try:
        import anndata as ad
        from natsort import natsorted
    except ImportError as e:
        raise ImportError(
            "anndata and natsort are required for embedding conversion. "
            "Install with: pip install 'viscy-utils[anndata]'"
        ) from e

    # Check if output_path exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output path {output_path} already exists.")

    # Tracking
    if isinstance(embeddings_ds, Path):
        embeddings_ds = xr.open_zarr(embeddings_ds)

    available_cols = get_available_index_columns(embeddings_ds)
    tracking_df = pd.DataFrame(
        {
            col: (
                embeddings_ds.coords[col].data
                if col != "fov_name"
                else embeddings_ds.coords[col].to_pandas().str.strip("/")
            )
            for col in available_cols
        }
    )

    obsm = {}
    # Projections
    if "projections" in embeddings_ds.coords:
        obsm["X_projections"] = embeddings_ds.coords["projections"].data

    # Embeddings
    for embedding in ["PCA", "UMAP", "PHATE"]:
        embedding_coords = natsorted([coord for coord in embeddings_ds.coords if embedding in coord])
        if embedding_coords:
            obsm[f"X_{embedding.lower()}"] = np.column_stack(
                [embeddings_ds.coords[coord].data for coord in embedding_coords]
            )

    # X, "expression" matrix (NN embedding features)
    X = embeddings_ds["features"].data

    adata = ad.AnnData(X=X, obs=tracking_df, obsm=obsm)

    adata.write_zarr(output_path)
    if return_anndata:
        return adata


def load_annotation_anndata(
    adata: ad.AnnData,
    path: str,
    name: str,
    categories: dict | None = None,
    spatial_tolerance: float = 4.0,
):
    """
    Load annotations from a CSV file and map them to the AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to map the annotations to.
    path : str
        Path to the CSV file containing annotations.
    name : str
        The column name in the CSV file to be used as annotations.
    categories : dict, optional
        A dictionary to rename categories in the annotation column. Default is None.
    spatial_tolerance : float, optional
        Pixel tolerance (Chebyshev / box kernel half-width) used to disambiguate
        duplicate annotation rows at division frames. At a mitosis split the
        segmenter typically produces two daughter cells that briefly share the
        parent's ``track_id`` for one frame, yielding two annotation rows with
        the same ``(fov_name, t, track_id)`` but distinct ``(y, x)``. We resolve
        the duplicate by picking the annotation row whose ``(y, x)`` falls
        within ``spatial_tolerance`` pixels of the embedding's centroid; if no
        candidate is within range the label stays NaN. Default 4 pixels.

    Returns
    -------
    anndata.AnnData
        The AnnData object with annotations added to adata.obs[name].
    """
    annotation = pd.read_csv(path)
    annotation["fov_name"] = annotation["fov_name"].str.strip("/")

    # Normalize obs fov_name: strip leading/trailing slashes so both sides match.
    obs_fov = adata.obs["fov_name"].astype(object).str.strip("/")

    if "id" in adata.obs.columns and "id" in annotation.columns:
        key_cols = ["fov_name", "id"]
        mi = pd.MultiIndex.from_arrays([obs_fov, adata.obs["id"]], names=key_cols)
    elif all(c in adata.obs.columns for c in ("fov_name", "t", "track_id")) and all(
        c in annotation.columns for c in ("fov_name", "t", "track_id")
    ):
        key_cols = ["fov_name", "t", "track_id"]
        mi = pd.MultiIndex.from_arrays(
            [obs_fov, adata.obs["t"], adata.obs["track_id"]],
            names=key_cols,
        )
    else:
        raise KeyError(
            "Cannot join annotations: embeddings have neither (fov_name, id) nor (fov_name, t, track_id) columns."
        )

    annotation_indexed = annotation.set_index(key_cols)

    if annotation_indexed.index.is_unique:
        # Fast path: unique annotation index, plain reindex.
        selected = annotation_indexed.reindex(mi)[name]
    else:
        # Slow path: annotation has duplicate keys (typically mitotic daughters
        # briefly sharing a parent track_id at the division frame). Resolve by
        # picking the duplicate whose (y, x) is closest to the embedding's
        # (y, x), provided it falls within ``spatial_tolerance`` pixels.
        spatial_cols = ("y", "x")
        if not all(c in annotation.columns for c in spatial_cols) or not all(
            c in adata.obs.columns for c in spatial_cols
        ):
            raise ValueError(
                f"Annotation index {key_cols} has duplicate keys (typical of mitosis "
                f"split frames) but cannot disambiguate: both annotation and embedding "
                f"obs must carry (y, x) columns for spatial nearest-neighbor matching."
            )

        selected = _spatial_nearest_select(
            annotation_indexed,
            mi=mi,
            embedding_y=np.asarray(adata.obs["y"], dtype=float),
            embedding_x=np.asarray(adata.obs["x"], dtype=float),
            value_col=name,
            tolerance=spatial_tolerance,
        )

    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)

    selected.index = adata.obs.index
    adata.obs[name] = selected

    return adata


def _spatial_nearest_select(
    annotation_indexed: pd.DataFrame,
    *,
    mi: pd.MultiIndex,
    embedding_y: np.ndarray,
    embedding_x: np.ndarray,
    value_col: str,
    tolerance: float,
) -> pd.Series:
    """Resolve duplicate annotation keys via spatial nearest-neighbor.

    For each row in ``mi`` (one per embedding observation), find rows in
    ``annotation_indexed`` whose key matches, and among them pick the one
    whose ``(y, x)`` is closest to the embedding's ``(y, x)`` — but only if
    the Chebyshev (box) distance is within ``tolerance`` pixels. Returns the
    ``value_col`` of the chosen row, or NaN if no match passes the threshold.

    The Chebyshev metric (``max(|dy|, |dx|)``) matches a square kernel; use
    Euclidean if a circular kernel is preferred.
    """
    # Strategy: pair each annotation row with every embedding row that shares
    # the same key (via pandas merge), compute Chebyshev distance, and keep
    # the closest annotation per embedding row whose distance ≤ tolerance.
    n = len(mi)

    emb_df = pd.DataFrame(
        {"_emb_idx": np.arange(n), "_emb_y": embedding_y, "_emb_x": embedding_x},
        index=mi,
    ).reset_index()

    ann_df = annotation_indexed[["y", "x", value_col]].reset_index()
    ann_df = ann_df.rename(columns={"y": "_ann_y", "x": "_ann_x"})

    key_cols = list(mi.names)
    paired = emb_df.merge(ann_df, on=key_cols, how="left")

    dy = (paired["_ann_y"] - paired["_emb_y"]).abs()
    dx = (paired["_ann_x"] - paired["_emb_x"]).abs()
    paired["_dist"] = np.maximum(dy, dx)
    paired.loc[paired["_dist"] > tolerance, "_dist"] = np.nan

    # Pick the row with minimum distance per embedding row; idxmin skips NaN,
    # so embedding rows with no candidate within tolerance silently drop out.
    best = paired.dropna(subset=["_dist"]).loc[lambda d: d.groupby("_emb_idx")["_dist"].idxmin()]

    out = pd.Series(pd.NA, index=np.arange(n), name=value_col, dtype="object")
    out.loc[best["_emb_idx"].to_numpy()] = best[value_col].to_numpy()
    return out
