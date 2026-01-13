"""PCA and UMAP dimensionality reduction."""

import anndata as ad
import pandas as pd
import umap
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_phate(
    embedding_dataset: ad.AnnData | NDArray,
    scale_embeddings: bool = False,
    n_components: int = 2,
    knn: int = 5,
    decay: int = 40,
    random_state: int = 42,
    **phate_kwargs,
) -> tuple[object, NDArray]:
    """
    Compute PHATE embeddings from AnnData or array.

    Parameters
    ----------
    embedding_dataset : ad.AnnData | NDArray
        AnnData object with features in .X,
        or a numpy array of embeddings.
    scale_embeddings : bool, optional
        Whether to scale embeddings before computing PHATE, by default False
    n_components : int, optional
        Number of dimensions in the PHATE embedding, by default 2
    knn : int, optional
        Number of nearest neighbors to use in the KNN graph, by default 5
    decay : int, optional
        Decay parameter for the Markov operator, by default 40
    random_state : int, optional
        Random state for reproducibility, by default 42
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

    # Extract features (support both AnnData and NDArray)
    if isinstance(embedding_dataset, ad.AnnData):
        embeddings = embedding_dataset.X
    else:
        embeddings = embedding_dataset

    # Scale if requested
    if scale_embeddings:
        embeddings_scaled = StandardScaler().fit_transform(embeddings)
    else:
        embeddings_scaled = embeddings

    # Compute PHATE embeddings
    phate_model = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        random_state=random_state,
        **phate_kwargs,
    )

    phate_embedding = phate_model.fit_transform(embeddings_scaled)

    return phate_model, phate_embedding


def compute_pca(
    embedding_dataset: ad.AnnData | NDArray,
    n_components: int | None = None,
    normalize_features: bool = True,
) -> tuple[NDArray, pd.DataFrame]:
    """Compute PCA embeddings from AnnData or array.

    Parameters
    ----------
    embedding_dataset : ad.AnnData | NDArray
        AnnData object with features in .X and metadata in .obs,
        or a numpy array of embeddings.
    n_components : int, optional
        Number of components to keep in the PCA, by default None
    normalize_features : bool, optional
        Whether to normalize the features, by default True

    Returns
    -------
    tuple[NDArray, pd.DataFrame]
        PCA embeddings array and DataFrame with metadata + PCA components
    """

    # Extract features and metadata
    if isinstance(embedding_dataset, ad.AnnData):
        embeddings = embedding_dataset.X
        metadata = embedding_dataset.obs.copy()
    else:
        embeddings = embedding_dataset
        metadata = pd.DataFrame()

    # Normalize if requested
    if normalize_features:
        scaled_features = StandardScaler().fit_transform(embeddings)
    else:
        scaled_features = embeddings

    # Compute PCA with specified number of components
    pca_model = PCA(n_components=n_components, random_state=42)
    pc_features = pca_model.fit_transform(scaled_features)

    # Create result DataFrame with metadata + PCA components
    result_dict = metadata.to_dict("list") if not metadata.empty else {}
    for i in range(pc_features.shape[1]):
        result_dict[f"PC{i + 1}"] = pc_features[:, i]

    pca_df = pd.DataFrame(result_dict)

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
    embedding_dataset: ad.AnnData,
    n_components: int = 2,
    n_neighbors: int = 15,
    normalize_features: bool = True,
) -> tuple[umap.UMAP, NDArray, pd.DataFrame]:
    """Compute UMAP embeddings from AnnData.

    Parameters
    ----------
    embedding_dataset : ad.AnnData
        AnnData object with features in .X and metadata in .obs
    n_components : int, optional
        Number of UMAP dimensions, by default 2
    n_neighbors : int, optional
        Number of neighbors for UMAP, by default 15
    normalize_features : bool, optional
        Scale the input to zero mean and unit variance before fitting UMAP,
        by default True

    Returns
    -------
    tuple[umap.UMAP, NDArray, pd.DataFrame]
        UMAP model, embeddings array, and DataFrame with metadata + UMAP coordinates
    """
    features = embedding_dataset.X

    # Fit UMAP
    umap_model, umap_embedding = _fit_transform_umap(
        features,
        n_components=n_components,
        n_neighbors=n_neighbors,
        normalize=normalize_features,
    )

    # Create result DataFrame with metadata + UMAP coordinates
    umap_df = embedding_dataset.obs.copy()
    for i in range(umap_embedding.shape[1]):
        umap_df[f"UMAP{i + 1}"] = umap_embedding[:, i]

    return umap_model, umap_embedding, umap_df
