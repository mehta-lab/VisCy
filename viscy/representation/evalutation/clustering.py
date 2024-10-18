"""Methods for evaluating clustering performance."""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier


class GMMClustering:
    def __init__(self, features_data, n_clusters_range=np.arange(2, 10)):
        self.features_data = features_data
        self.n_clusters_range = n_clusters_range
        self.best_n_clusters = None
        self.best_gmm = None
        self.aic_scores = None
        self.bic_scores = None

    def find_best_n_clusters(self):
        """Find the best number of clusters using AIC/BIC scores."""
        aic_scores = []
        bic_scores = []
        for n in self.n_clusters_range:
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(self.features_data)
            aic_scores.append(gmm.aic(self.features_data))
            bic_scores.append(gmm.bic(self.features_data))

        self.aic_scores = aic_scores
        self.bic_scores = bic_scores

        return aic_scores, bic_scores

    def fit_best_model(self, criterion="bic", n_clusters=None):
        """
        Fit the best GMM model based on AIC or BIC scores, or a user-specified number of clusters.

        Parameters:
        - criterion: 'aic' or 'bic' to select the best model based on the chosen criterion.
        - n_clusters: Specify a fixed number of clusters (overrides the 'best' search).
        """
        # Case 1: If the user provides n_clusters, use it directly
        if n_clusters is not None:
            self.best_n_clusters = n_clusters

        # Case 2: If no n_clusters is provided but find_best_n_clusters was run, use stored AIC/BIC results
        elif self.aic_scores is not None and self.bic_scores is not None:
            if criterion == "bic":
                self.best_n_clusters = self.n_clusters_range[np.argmin(self.bic_scores)]
            else:
                self.best_n_clusters = self.n_clusters_range[np.argmin(self.aic_scores)]

        # Case 3: If find_best_n_clusters hasn't been run, compute AIC/BIC scores now
        else:
            aic_scores, bic_scores = self.find_best_n_clusters()
            if criterion == "bic":
                self.best_n_clusters = self.n_clusters_range[np.argmin(bic_scores)]
            else:
                self.best_n_clusters = self.n_clusters_range[np.argmin(aic_scores)]

        self.best_gmm = GaussianMixture(
            n_components=self.best_n_clusters, random_state=42
        )
        self.best_gmm.fit(self.features_data)

        return self.best_gmm

    def predict_clusters(self):
        """Run prediction on the fitted best GMM model."""
        if self.best_gmm is None:
            raise Exception(
                "No GMM model is fitted yet. Please run fit_best_model() first."
            )
        cluster_labels = self.best_gmm.predict(self.features_data)
        return cluster_labels


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
