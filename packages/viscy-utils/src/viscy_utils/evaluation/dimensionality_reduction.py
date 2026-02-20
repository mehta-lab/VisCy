"""PCA, UMAP, and PHATE dimensionality reduction."""

import pandas as pd
from numpy.typing import NDArray
from xarray import Dataset


def compute_phate(
    embedding_dataset,
    scale_embeddings: bool = False,
    n_components: int = 2,
    knn: int = 5,
    decay: int = 40,
    update_dataset: bool = False,
    random_state: int = 42,
    **phate_kwargs,
) -> tuple[object, NDArray]:
    """Compute PHATE embeddings.

    Parameters
    ----------
    embedding_dataset : xarray.Dataset or NDArray
        Dataset containing embeddings or a numpy array.
    scale_embeddings : bool, optional
        Whether to scale embeddings, by default False.
    n_components : int, optional
        Number of PHATE dimensions, by default 2.
    knn : int, optional
        Number of nearest neighbors, by default 5.
    decay : int, optional
        Decay parameter for the Markov operator, by default 40.
    update_dataset : bool, optional
        Whether to update the dataset, by default False.
    random_state : int, optional
        Random state, by default 42.

    Returns
    -------
    tuple[object, NDArray]
        PHATE model and embeddings.
    """
    try:
        import phate
    except ImportError:
        raise ImportError("PHATE is not available. Install with: pip install viscy-utils[eval]")

    embeddings = (
        embedding_dataset["features"].to_numpy() if isinstance(embedding_dataset, Dataset) else embedding_dataset
    )

    from sklearn.preprocessing import StandardScaler

    if scale_embeddings:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
    else:
        embeddings_scaled = embeddings

    phate_model = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        random_state=random_state,
        **phate_kwargs,
    )

    phate_embedding = phate_model.fit_transform(embeddings_scaled)

    if update_dataset and isinstance(embedding_dataset, Dataset):
        for i in range(min(2, phate_embedding.shape[1])):
            embedding_dataset[f"PHATE{i + 1}"].values = phate_embedding[:, i]  # noqa: PD011

    return phate_model, phate_embedding


def compute_pca(embedding_dataset, n_components=None, normalize_features=True):
    """Compute PCA embeddings.

    Parameters
    ----------
    embedding_dataset : xarray.Dataset or NDArray
        Dataset containing embeddings or a numpy array.
    n_components : int, optional
        Number of PCA components.
    normalize_features : bool, optional
        Whether to normalize features, by default True.

    Returns
    -------
    tuple[NDArray, pd.DataFrame]
        PCA embeddings and PCA DataFrame.
    """
    embeddings = (
        embedding_dataset["features"].to_numpy() if isinstance(embedding_dataset, Dataset) else embedding_dataset
    )

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if normalize_features:
        scaled_features = StandardScaler().fit_transform(embeddings)
    else:
        scaled_features = embeddings

    PCA_features = PCA(n_components=n_components, random_state=42)
    pc_features = PCA_features.fit_transform(scaled_features)

    if isinstance(embedding_dataset, Dataset):
        pca_dict = {
            "id": embedding_dataset["id"].to_numpy(),
            "fov_name": embedding_dataset["fov_name"].to_numpy(),
            "t": embedding_dataset["t"].to_numpy(),
            "track_id": embedding_dataset["track_id"].to_numpy(),
        }
    else:
        pca_dict = {}

    for i in range(pc_features.shape[1]):
        pca_dict[f"PC{i + 1}"] = pc_features[:, i]

    pca_df = pd.DataFrame(pca_dict)

    return pc_features, pca_df


def _fit_transform_umap(
    embeddings: NDArray,
    n_components: int = 2,
    n_neighbors: int = 15,
    normalize: bool = True,
):
    """Fit UMAP model and transform embeddings."""
    import umap
    from sklearn.preprocessing import StandardScaler

    if normalize:
        embeddings = StandardScaler().fit_transform(embeddings)
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    umap_embedding = umap_model.fit_transform(embeddings)
    return umap_model, umap_embedding


def compute_umap(embedding_dataset: Dataset, normalize_features: bool = True):
    """Compute UMAP embeddings for features and projections.

    Parameters
    ----------
    embedding_dataset : Dataset
        Xarray dataset with features and projections.
    normalize_features : bool, optional
        Whether to scale inputs before UMAP, by default True.

    Returns
    -------
    tuple[umap.UMAP, umap.UMAP, pd.DataFrame]
        UMAP models for features and projections, and DataFrame.
    """
    features = embedding_dataset["features"].to_numpy()
    projections = embedding_dataset["projections"].to_numpy()

    umap_features, umap_features_embedding = _fit_transform_umap(features, n_components=2, normalize=normalize_features)
    umap_projection, umap_projection_embedding = _fit_transform_umap(
        projections, n_components=2, normalize=normalize_features
    )

    umap_df = pd.DataFrame(
        {
            "id": embedding_dataset["id"].to_numpy(),
            "track_id": embedding_dataset["track_id"].to_numpy(),
            "t": embedding_dataset["t"].to_numpy(),
            "fov_name": embedding_dataset["fov_name"].to_numpy(),
            "UMAP1": umap_features_embedding[:, 0],
            "UMAP2": umap_features_embedding[:, 1],
            "UMAP1_proj": umap_projection_embedding[:, 0],
            "UMAP2_proj": umap_projection_embedding[:, 1],
        }
    )

    return umap_features, umap_projection, umap_df
