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

import pandas as pd

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

    # Add a leading slash to 'fov name' column and set it as 'fov_name'
    annotation["fov_name"] = "/" + annotation["fov_name"]

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
