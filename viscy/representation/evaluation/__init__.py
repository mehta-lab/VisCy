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

"""

import pandas as pd
import numpy as np

from viscy.data.triplet import TripletDataModule
import xarray as xr


def load_annotation(
    da: xr.DataArray | xr.Dataset,
    path: str,
    name: str,
    categories: dict | None = None,
    fov_column: str = "fov_name",
    id_column: str = "id",
) -> pd.Series:
    """
    Load annotations from a CSV file and map them to the dataset.

    This function supports both simple datasets (with fov_name and id coordinates)
    and complex datasets with additional coordinates like time (t) and track_id.

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        The dataset containing coordinate information. For simple datasets,
        should have 'fov_name' and 'id' coordinates. For complex datasets,
        can have additional coordinates like 't' and 'track_id'.
    path : str
        Path to the CSV file containing annotations.
    name : str
        The column name in the CSV file to be used as annotations.
    categories : dict, optional
        A dictionary to rename categories in the annotation column.
        If -1 is present in the data but not in categories, it will be mapped to np.nan.
        Default is None.
    fov_column : str, optional
        The column name in the CSV file that contains FOV information.
        Default is "fov_name".
    id_column : str, optional
        The column name in the CSV file that contains ID information.
        Default is "id".

    Returns
    -------
    pd.Series
        A pandas Series containing the selected annotations mapped to the dataset.

    Notes
    -----
    The function automatically detects the dataset structure and handles:
    - Simple datasets: Uses fov_name and id coordinates
    - Complex datasets: Uses fov_name, id, t, and track_id coordinates if available

    For complex datasets, the CSV should contain columns: fov_column, id_column, 't', 'track_id'
    """
    # Read the annotation CSV file
    annotation = pd.read_csv(path)

    # Determine if this is a simple or complex dataset
    has_time_coord = hasattr(da, "t") and "t" in da.coords
    has_track_id_coord = hasattr(da, "track_id") and "track_id" in da.coords

    if has_time_coord and has_track_id_coord:
        # Complex dataset with time and track_id coordinates
        # Handle FOV column mapping
        if fov_column == "fov_name" and "fov ID" in annotation.columns:
            annotation["fov_name"] = "/" + annotation["fov ID"]
        elif fov_column != "fov_name":
            annotation["fov_name"] = "/" + annotation[fov_column]
        else:
            annotation["fov_name"] = "/" + annotation["fov_name"]

        # Create MultiIndex for complex dataset
        embedding_index = pd.MultiIndex.from_arrays(
            [
                da["fov_name"].values,
                da["id"].values,
                da["t"].values,
                da["track_id"].values,
            ],
            names=["fov_name", "id", "t", "track_id"],
        )

        # Set index and reindex
        annotation = annotation.set_index(["fov_name", "id", "t", "track_id"])
        selected = annotation.reindex(embedding_index)[name]

    else:
        # Simple dataset with just fov_name and id
        # Handle FOV column mapping
        if fov_column == "fov_name" and "fov_name" in annotation.columns:
            annotation["fov_name"] = "/" + annotation["fov_name"]
        elif fov_column != "fov_name":
            annotation["fov_name"] = "/" + annotation[fov_column]
        else:
            annotation["fov_name"] = "/" + annotation["fov_name"]

        # Set the index of the annotation DataFrame
        annotation = annotation.set_index(["fov_name", id_column])

        # Create a MultiIndex from the dataset array's coordinates
        mi = pd.MultiIndex.from_arrays(
            [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
        )

        # Select the annotations corresponding to the MultiIndex
        selected = annotation.loc[mi][name]

    # Ensure we have a Series
    if not isinstance(selected, pd.Series):
        selected = pd.Series(selected)

    # Handle category mapping
    if categories:
        # Check if -1 is present in data but not in categories
        if -1 in selected.values and -1 not in categories:
            categories = categories.copy()
            categories[-1] = np.nan

        # Use replace for category renaming (more appropriate than map for dict)
        selected = selected.replace(categories)

    return selected


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
        normalizations=[],
        predict_cells=True,
    )
    # for train and val
    data_module.setup("predict")
    prediction_dataset = data_module.predict_dataset
    return prediction_dataset
