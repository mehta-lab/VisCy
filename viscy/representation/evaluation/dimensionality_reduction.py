"""PCA and UMAP dimensionality reduction."""

import pandas as pd
import umap
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xarray import Dataset


def compute_phate(
    embedding_dataset,
    scale_embeddings: bool = False,
    n_components: int = 2,
    knn: int = 5,
    decay: int = 40,
    update_dataset: bool = False,
    **phate_kwargs,
) -> tuple[object, NDArray]:
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
    tuple[object, NDArray]
        PHATE model and PHATE embeddings

    Raises
    ------
    ImportError
        If PHATE is not installed. Install with: pip install viscy[phate]
    """
    try:
        import phate
    except ImportError:
        raise ImportError(
            "PHATE is not available. Install with: pip install viscy[phate]"
        )

    # Get embeddings from dataset if needed
    embeddings = (
        embedding_dataset["features"].values
        if isinstance(embedding_dataset, Dataset)
        else embedding_dataset
    )

    if scale_embeddings:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
    else:
        embeddings_scaled = embeddings

    # Compute PHATE embeddings
    phate_model = phate.PHATE(
        n_components=n_components, knn=knn, decay=decay, random_state=42, **phate_kwargs
    )

    phate_embedding = phate_model.fit_transform(embeddings_scaled)

    # Update dataset if requested
    if update_dataset and isinstance(embedding_dataset, Dataset):
        for i in range(
            min(2, phate_embedding.shape[1])
        ):  # Only update PHATE1 and PHATE2
            embedding_dataset[f"PHATE{i + 1}"].values = phate_embedding[:, i]

    return phate_model, phate_embedding


def compute_pca(embedding_dataset, n_components=None, normalize_features=True):
    """Compute PCA embeddings for features and optionally update dataset.

    Parameters
    ----------
    embedding_dataset : xarray.Dataset or NDArray
        The dataset containing embeddings, timepoints, fov_name, and track_id,
        or a numpy array of embeddings.
    n_components : int, optional
        Number of components to keep in the PCA, by default None
    normalize_features : bool, optional
        Whether to normalize the features, by default True

    Returns
    -------
    tuple[NDArray, pd.DataFrame]
        PCA embeddings and PCA DataFrame
    """

    embeddings = (
        embedding_dataset["features"].values
        if isinstance(embedding_dataset, Dataset)
        else embedding_dataset
    )

    if normalize_features:
        scaled_features = StandardScaler().fit_transform(embeddings)
    else:
        scaled_features = embeddings

    # Compute PCA with specified number of components
    PCA_features = PCA(n_components=n_components, random_state=42)
    pc_features = PCA_features.fit_transform(scaled_features)

    # Create base dictionary with id and fov_name
    if isinstance(embedding_dataset, Dataset):
        pca_dict = {
            "id": embedding_dataset["id"].values,
            "fov_name": embedding_dataset["fov_name"].values,
            "t": embedding_dataset["t"].values,
            "track_id": embedding_dataset["track_id"].values,
        }
    else:
        pca_dict = {}

    # Add PCA components for features
    for i in range(pc_features.shape[1]):
        pca_dict[f"PC{i + 1}"] = pc_features[:, i]

    # Create DataFrame with all components
    pca_df = pd.DataFrame(pca_dict)

    return pc_features, pca_df


def _fit_transform_umap(
    embeddings: NDArray,
    n_components: int = 2,
    n_neighbors: int = 15,
    normalize: bool = True,
) -> tuple[umap.UMAP, NDArray]:
    """Fit UMAP model and transform embeddings."""
    if normalize:
        embeddings = StandardScaler().fit_transform(embeddings)
    umap_model = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, random_state=42
    )
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
