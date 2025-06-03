"""PCA and UMAP dimensionality reduction."""

import pandas as pd
import phate
import umap
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xarray import Dataset


def compute_phate(
    embedding_dataset,
    n_components: int = 2,
    knn: int = 5,
    decay: int = 40,
    update_dataset: bool = False,
    **phate_kwargs,
) -> tuple[phate.PHATE, NDArray]:
    """
    Compute PHATE embeddings for features and optionally update dataset.

    Parameters
    ----------
    embedding_dataset : xarray.Dataset or NDArray
        The dataset containing embeddings, timepoints, fov_name, and track_id,
        or a numpy array of embeddings.
    n_components : int, optional
        Number of dimensions in the PHATE embedding, by default None
    knn : int, optional
        Number of nearest neighbors to use in the KNN graph, by default 5
    decay : int, optional
        Decay parameter for the Markov operator, by default 40
    update_dataset : bool, optional
        Whether to update the PHATE coordinates in the dataset, by default False
    phate_kwargs : dict, optional
        Additional keyword arguments for PHATE, by default None

    Returns
    -------
    phate.PHATE, NDArray
        PHATE model and PHATE embeddings
    """
    import phate

    # Get embeddings from dataset if needed
    embeddings = (
        embedding_dataset["features"].values
        if isinstance(embedding_dataset, Dataset)
        else embedding_dataset
    )

    # Compute PHATE embeddings
    phate_model = phate.PHATE(
        n_components=n_components, knn=knn, decay=decay, **phate_kwargs
    )
    phate_embedding = phate_model.fit_transform(embeddings)

    # Update dataset if requested
    if update_dataset and isinstance(embedding_dataset, Dataset):
        for i in range(
            min(2, phate_embedding.shape[1])
        ):  # Only update PHATE1 and PHATE2
            embedding_dataset[f"PHATE{i+1}"].values = phate_embedding[:, i]

    return phate_model, phate_embedding


def compute_pca(embedding_dataset, n_components=None, normalize_features=True):
    features = embedding_dataset["features"]
    projections = embedding_dataset["projections"]

    if normalize_features:
        scaled_projections = StandardScaler().fit_transform(projections.values)
        scaled_features = StandardScaler().fit_transform(features.values)
    else:
        scaled_projections = projections.values
        scaled_features = features.values

    # Compute PCA with specified number of components
    PCA_features = PCA(n_components=n_components, random_state=42)
    PCA_projection = PCA(n_components=n_components, random_state=42)
    pc_features = PCA_features.fit_transform(scaled_features)
    pc_projection = PCA_projection.fit_transform(scaled_projections)

    # Prepare DataFrame with id and PCA coordinates
    pca_df = pd.DataFrame(
        {
            "id": embedding_dataset["id"].values,
            "fov_name": embedding_dataset["fov_name"].values,
            "PCA1": pc_features[:, 0],
            "PCA2": pc_features[:, 1],
            "PCA3": pc_features[:, 2],
            "PCA4": pc_features[:, 3],
            "PCA5": pc_features[:, 4],
            "PCA6": pc_features[:, 5],
            "PCA1_proj": pc_projection[:, 0],
            "PCA2_proj": pc_projection[:, 1],
            "PCA3_proj": pc_projection[:, 2],
            "PCA4_proj": pc_projection[:, 3],
            "PCA5_proj": pc_projection[:, 4],
            "PCA6_proj": pc_projection[:, 5],
        }
    )

    return PCA_features, PCA_projection, pca_df


def _fit_transform_umap(
    embeddings: NDArray, n_components: int = 2, normalize: bool = True
) -> tuple[umap.UMAP, NDArray]:
    """Fit UMAP model and transform embeddings."""
    if normalize:
        embeddings = StandardScaler().fit_transform(embeddings)
    umap_model = umap.UMAP(n_components=n_components, random_state=42)
    umap_embedding = umap_model.fit_transform(embeddings)
    return umap_model, umap_embedding


def compute_umap(
    embedding_dataset: Dataset, normalize_features: bool = True
) -> tuple[umap.UMAP, umap.UMAP, pd.DataFrame]:
    """Compute UMAP embeddings for features and projections.

    Parameters
    ----------
    embedding_dataset : Dataset
        Xarray dataset with features and projections.
    normalize_features : bool, optional
        Scale the input to zero mean and unit variance before fitting UMAP,
        by default True

    Returns
    -------
    tuple[umap.UMAP, umap.UMAP, pd.DataFrame]
        UMAP models for features and projections,
        and DataFrame with UMAP embeddings
    """
    features = embedding_dataset["features"].values
    projections = embedding_dataset["projections"].values

    umap_features, umap_features_embedding = _fit_transform_umap(
        features, n_components=2, normalize=normalize_features
    )
    umap_projection, umap_projection_embedding = _fit_transform_umap(
        projections, n_components=2, normalize=normalize_features
    )

    # Prepare DataFrame with id and UMAP coordinates
    umap_df = pd.DataFrame(
        {
            "id": embedding_dataset["id"].values,
            "track_id": embedding_dataset["track_id"].values,
            "t": embedding_dataset["t"].values,
            "fov_name": embedding_dataset["fov_name"].values,
            "UMAP1": umap_features_embedding[:, 0],
            "UMAP2": umap_features_embedding[:, 1],
            "UMAP1_proj": umap_projection_embedding[:, 0],
            "UMAP2_proj": umap_projection_embedding[:, 1],
        }
    )

    return umap_features, umap_projection, umap_df
