import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    normalized_mutual_information_score,
    adjusted_rand_score,
    silhouette_score,
)
import torch.nn as nn
import torch.optim as optim


class RepresentationEvaluator:
    def __init__(self, embeddings: np.ndarray, annotations: np.ndarray):
        """
        Initialize the evaluator with embeddings and annotations.

        Parameters
        ----------
        embeddings : np.ndarray
            The learned representations. Shape: (n_samples, n_features).
        annotations : np.ndarray
            The ground truth labels in one-hot encoding. Shape: (n_samples, p_class).
        """
        self.embeddings = embeddings
        self.annotations = np.argmax(
            annotations, axis=1
        )  # Convert one-hot encoding to class labels

    def knn_accuracy(self, k=5):
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
        knn.fit(self.embeddings, self.annotations)
        predictions = knn.predict(self.embeddings)
        accuracy = accuracy_score(self.annotations, predictions)
        return accuracy

    def dbscan_clustering(self, eps=0.5, min_samples=5):
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
        clusters = dbscan.fit_predict(self.embeddings)
        return clusters

    def silhouette_score(self, clusters):
        """
        Compute the silhouette score for the DBSCAN clustering results.

        Parameters
        ----------
        clusters : np.ndarray
            Clustering labels assigned by DBSCAN.

        Returns
        -------
        float
            Silhouette score for the clustering.
        """
        score = silhouette_score(self.embeddings, clusters)
        return score

    def clustering_evaluation(self, method="nmi"):
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
        clusters = self.dbscan_clustering()

        if method == "nmi":
            score = normalized_mutual_information_score(self.annotations, clusters)
        elif method == "ari":
            score = adjusted_rand_score(self.annotations, clusters)
        else:
            raise ValueError("Invalid method. Choose 'nmi' or 'ari'.")

        return score

    def linear_classifier_accuracy(self, batch_size=32, learning_rate=0.01, epochs=10):
        """
        Evaluate the accuracy of a single-layer neural network trained on the embeddings.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for training. Default is 32.
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 0.01.
        epochs : int, optional
            Number of training epochs. Default is 10.

        Returns
        -------
        float
            Accuracy of the neural network classifier.
        """

        class SingleLayerNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(SingleLayerNN, self).__init__()
                self.fc = nn.Linear(input_dim, output_dim)

            def forward(self, x):
                return self.fc(x)

        # Convert numpy arrays to PyTorch tensors
        inputs = torch.tensor(self.embeddings, dtype=torch.float32)
        labels = torch.tensor(self.annotations, dtype=torch.long)

        # Create a dataset and data loader
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the neural network, loss function, and optimizer
        input_dim = self.embeddings.shape[1]
        output_dim = len(np.unique(self.annotations))
        model = SingleLayerNN(input_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(epochs):
            for batch_inputs, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            accuracy = accuracy_score(labels.numpy(), predictions.numpy())

        return accuracy


# Example usage:
# embeddings = np.random.rand(100, 128)  # Example embeddings
# annotations = np.eye(10)[np.random.choice(10, 100)]  # Example one-hot encoded labels
# evaluator = ContrastiveLearningEvaluator(embeddings, annotations)
# print("k-NN Accuracy:", evaluator.knn_accuracy(k=5))
# dbscan_clusters = evaluator.dbscan_clustering(eps=0.3, min_samples=10)
# print("Silhouette Score:", evaluator.silhouette_score(dbscan_clusters))
# print("NMI Score:", evaluator.clustering_evaluation(method='nmi'))
# print("ARI Score:", evaluator.clustering_evaluation(method='ari'))
# print("Linear Classifier Accuracy:", evaluator.linear_classifier_accuracy(batch_size=32, learning_rate=0.01, epochs=10))
