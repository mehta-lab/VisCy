"""PCA and UMAP dimensionality reduction."""

import pandas as pd
import umap
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xarray import Dataset


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
