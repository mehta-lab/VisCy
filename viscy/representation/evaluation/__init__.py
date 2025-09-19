"""
This module enables evaluation of learned representations using annotations, such as
* cell division labels,
* infection state labels,
* labels predicted using supervised classifiers,
* computed image features.

Following evaluation methods are implemented:
* Linear classifier accuracy when labels are provided.
* Clustering evaluation using normalized mutual information (NMI) and adjusted rand index (ARI).
* Correlation between embeddings and computed features using rank correlation.

TODO: consider time- and condition-dependent clustering and UMAP visualization of patches developed earlier:
https://github.com/mehta-lab/dynacontrast/blob/master/analysis/gmm.py
"""

from pathlib import Path

import anndata as ad
import natsort as ns
import numpy as np
import pandas as pd
import xarray as xr

from viscy.data.triplet import TripletDataModule


def load_annotation(da, path, name, categories: dict | None = None):
    """
    Load annotations from a CSV file and map them to the dataset.

    Parameters
    ----------
    da : xarray.DataArray
        The dataset array containing 'fov_name' and 'id' coordinates.
    path : str
        Path to the CSV file containing annotations.
    name : str
        The column name in the CSV file to be used as annotations.
    categories : dict, optional
        A dictionary to rename categories in the annotation column. Default is None.

    Returns
    -------
    pd.Series
        A pandas Series containing the selected annotations mapped to the dataset.
    """
    # Read the annotation CSV file
    annotation = pd.read_csv(path)

    # Set the index of the annotation DataFrame to ['fov_name', 'id']
    annotation = annotation.set_index(["fov_name", "id"])

    # Create a MultiIndex from the dataset array's 'fov_name' and 'id' values
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )

    # Select the annotations corresponding to the MultiIndex
    selected = annotation.loc[mi][name]

    # If categories are provided, rename the categories in the selected annotations
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)

    return selected


def load_annotation_anndata(
    adata: ad.AnnData, path: str, name: str, categories: dict | None = None
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
    """
    annotation = pd.read_csv(path)
    annotation["fov_name"] = annotation["fov_name"].str.strip("/")

    annotation = annotation.set_index(["fov_name", "id"])
    print(annotation.head())

    mi = pd.MultiIndex.from_arrays(
        [adata.obs["fov_name"], adata.obs["id"]], names=["fov_name", "id"]
    )
    selected = annotation.loc[mi][name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected


def convert_xarray_annotation_to_anndata(
    embeddings_ds: xr.Dataset,
    output_path: Path,
    overwrite: bool = False,
    return_anndata: bool = False,
) -> ad.AnnData | None:
    """
    Convert an Xarray embeddings dataset to an AnnData object.

    Parameters
    ----------
    embeddings_ds : xr.Dataset
        The Xarray embeddings dataset to convert.
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
    tracking_df = pd.DataFrame(
        data={
            "id": embeddings_ds.coords["id"].data,
            "fov_name": embeddings_ds.coords["fov_name"].to_pandas().str.strip("/"),
            "track_id": embeddings_ds.coords["track_id"].data,
            "parent_track_id": embeddings_ds.coords["parent_track_id"].data,
            "parent_id": embeddings_ds.coords["parent_id"].data,
            "t": embeddings_ds.coords["t"].data,
            "y": embeddings_ds.coords["y"].data,
            "x": embeddings_ds.coords["x"].data,
        },
    )

    obsm = {}
    # Projections
    if "projections" in embeddings_ds.coords:
        obsm["X_projections"] = embeddings_ds.coords["projections"].data

    # Embeddings
    for embedding in ["PCA", "UMAP", "PHATE"]:
        embedding_coords = ns.natsorted(
            [coord for coord in embeddings_ds.coords if embedding in coord]
        )
        if embedding_coords:
            obsm[f"X_{embedding.lower()}"] = np.column_stack(
                [embeddings_ds.coords[coord] for coord in embedding_coords]
            )

    # X, "expression" matrix
    X = embeddings_ds["features"].data

    adata = ad.AnnData(X=X, obs=tracking_df, obsm=obsm)

    adata.write_zarr(output_path)
    if return_anndata:
        return adata


def dataset_of_tracks(
    data_path,
    tracks_path,
    fov_list,
    track_id_list,
    source_channel=["Phase3D", "RFP"],
    z_range=(28, 43),
    initial_yx_patch_size=(128, 128),
    final_yx_patch_size=(128, 128),
):
    data_module = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_path,
        include_fov_names=fov_list,
        include_track_ids=track_id_list,
        source_channel=source_channel,
        z_range=z_range,
        initial_yx_patch_size=initial_yx_patch_size,
        final_yx_patch_size=final_yx_patch_size,
        batch_size=1,
        num_workers=16,
        normalizations=None,
        predict_cells=True,
    )
    # for train and val
    data_module.setup("predict")
    prediction_dataset = data_module.predict_dataset
    return prediction_dataset
