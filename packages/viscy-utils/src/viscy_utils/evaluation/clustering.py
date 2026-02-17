"""Methods for evaluating clustering performance."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import KNeighborsClassifier


def knn_accuracy(embeddings, annotations, k=5):
    """
    Evaluate the k-NN classification accuracy.

    Parameters
    ----------
    k : int, optional
        Number of neighbors to use for k-NN. Default is 5.

    Returns
    -------
    float
        Accuracy of the k-NN classifier.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, annotations)
    predictions = knn.predict(embeddings)
    accuracy = accuracy_score(annotations, predictions)
    return accuracy


def pairwise_distance_matrix(
    features: ArrayLike, metric: str = "cosine", device: str = "auto"
) -> NDArray:
    """Compute pairwise distances between all samples in the feature matrix.

    Uses PyTorch with GPU acceleration when available for significant speedup.
    Falls back to scipy for unsupported metrics or when PyTorch is unavailable.

    Parameters
    ----------
    features : ArrayLike
        Feature matrix (n_samples, n_features)
    metric : str, optional
        Distance metric to use, by default "cosine"
        Supports "cosine" and "euclidean" with PyTorch acceleration.
        Other scipy metrics will use scipy fallback.
    device : str, optional
        Device to use for computation, by default "auto"
        - "auto": automatically use GPU if available, otherwise CPU
        - "cuda" or "gpu": force GPU usage
        - "cpu": force CPU usage
        - None or "scipy": force scipy fallback

    Returns
    -------
    NDArray
        Distance matrix of shape (n_samples, n_samples)
    """
    if device in (None, "scipy") or metric not in ("cosine", "euclidean"):
        return cdist(features, features, metric=metric)

    try:
        import torch

        if device == "auto":
            device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            device_torch = torch.device("cuda")
        elif device == "cpu":
            device_torch = torch.device("cpu")
        else:
            raise ValueError(
                f"Invalid device: {device}. Use 'auto', 'cuda', 'cpu', or 'scipy'"
            )
        features_array = np.asarray(features)
        if features_array.dtype == np.float32:
            features_tensor = torch.from_numpy(features_array).double().to(device_torch)
        else:
            features_tensor = torch.from_numpy(features_array).to(device_torch)
            if features_tensor.dtype not in (torch.float32, torch.float64):
                features_tensor = features_tensor.double()

        if metric == "cosine":
            features_norm = torch.nn.functional.normalize(features_tensor, p=2, dim=1)
            similarity = features_norm @ features_norm.T
            distances = 1 - similarity
        elif metric == "euclidean":
            distances = torch.cdist(features_tensor, features_tensor, p=2)
        return distances.cpu().numpy()

    except ImportError:
        return cdist(features, features, metric=metric)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return cdist(features, features, metric=metric)


def rank_nearest_neighbors(
    cross_dissimilarity: NDArray, normalize: bool = True
) -> NDArray:
    """Rank each sample by (dis)similarity to all other samples.

    Parameters
    ----------
    cross_dissimilarity : NDArray
        Dissimilarity square matrix (n_samples, n_samples)
    normalize : bool, optional
        Normalize the rank matrix by sample size, by default True
        If normalized, self (diagonal) will be at fraction 0,
        and the farthest sample will be at fraction 1.

    Returns
    -------
    NDArray
        Rank matrix (n_samples, n_samples)
        Ranking is done on axis=1
    """
    rankings = np.argsort(np.argsort(cross_dissimilarity, axis=1), axis=1)
    if normalize:
        rankings = rankings.astype(np.float64) / (rankings.shape[1] - 1)
    return rankings


def select_block(distances: NDArray, index: NDArray) -> NDArray:
    """Select with the same indexes along both dimensions for a square matrix."""
    return distances[index][:, index]


def compare_time_offset(
    single_track_distances: NDArray, time_offset: int = 1
) -> NDArray:
    """Extract the nearest neighbor distances/rankings
    of the next sample compared to each sample.

    Parameters
    ----------
    single_track_distances : NDArray
        Distances or rankings of a single track (n_samples, n_samples)
        If the matrix is not symmetric (e.g. is rankings),
        it should measured along dimension 1
    sample_offset : int, optional
        Offset from the diagonal, by default 1 (the next sample in time)

    Returns
    -------
    NDArray
        Distances/rankings vector (n_samples - time_offset,)
    """
    return single_track_distances.diagonal(offset=-time_offset)


def dbscan_clustering(embeddings, eps=0.5, min_samples=5):
    """
    Apply DBSCAN clustering to the embeddings.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered as in the same neighborhood. Default is 0.5.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point. Default is 5.

    Returns
    -------
    np.ndarray
        Clustering labels assigned by DBSCAN.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(embeddings)
    return clusters


def clustering_evaluation(embeddings, annotations, method="nmi"):
    """
    Evaluate the clustering of the embeddings compared to the ground truth labels.

    Parameters
    ----------
    method : str, optional
        Metric to use for evaluation ('nmi' or 'ari'). Default is 'nmi'.

    Returns
    -------
    float
        NMI or ARI score depending on the method chosen.
    """
    clusters = dbscan_clustering(embeddings)

    if method == "nmi":
        score = normalized_mutual_info_score(annotations, clusters)
    elif method == "ari":
        score = adjusted_rand_score(annotations, clusters)
    else:
        raise ValueError("Invalid method. Choose 'nmi' or 'ari'.")

    return score
