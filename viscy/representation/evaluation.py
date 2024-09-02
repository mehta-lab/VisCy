import numpy as np
from numpy import fft
from skimage import color
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu
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
        criterion = (
            nn.CrossEntropyLoss()
        )  # Works with logits, so no softmax in the last layer

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


class FeatureExtractor:

    def __init__(self):
        pass

    def compute_fourier_descriptors(image):

        # Convert contour to complex numbers
        contour_complex = image[:, 0] + 1j * image[:, 1]

        # Compute Fourier descriptors
        descriptors = np.fft.fft(contour_complex)

        return descriptors

    def analyze_symmetry(descriptors):
        # Normalize descriptors
        descriptors = np.abs(descriptors) / np.max(np.abs(descriptors))
        # Check symmetry (for a perfect circle, descriptors should be quite uniform)
        return np.std(descriptors)  # Lower standard deviation indicates higher symmetry

    def compute_area(input_image, sigma=0.6):
        """Create a binary mask using morphological operations
        :param np.array input_image: generate masks from this 3D image
        :param float sigma: Gaussian blur standard deviation, increase in value increases blur
        :return: volume mask of input_image, 3D np.array
        """

        input_image_blur = gaussian(input_image, sigma=sigma)

        thresh = threshold_otsu(input_image_blur)
        mask = input_image >= thresh

        # Apply sensor mask to the image
        masked_image = input_image * mask

        # Compute the mean intensity inside the sensor area
        masked_intensity = np.mean(masked_image)

        return masked_intensity, np.sum(mask)

    def compute_spectral_entropy(image):
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # Compute the 2D Fourier Transform
        f_transform = fft.fft2(image)

        # Compute the power spectrum
        power_spectrum = np.abs(f_transform) ** 2

        # Compute the probability distribution
        power_spectrum += 1e-10  # Avoid log(0) issues
        prob_distribution = power_spectrum / np.sum(power_spectrum)

        # Compute the spectral entropy
        entropy = -np.sum(prob_distribution * np.log(prob_distribution))

        return entropy

    def compute_glcm_features(image):

        # Normalize the input image from 0 to 255
        image = (image - np.min(image)) * (255 / (np.max(image) - np.min(image)))
        image = image.astype(np.uint8)

        # Compute the GLCM
        distances = [1]  # Distance between pixels
        angles = [0]  # Angle in radians

        glcm = graycomatrix(image, distances, angles, symmetric=True, normed=True)

        # Compute GLCM properties
        contrast = graycoprops(glcm, "contrast")[0, 0]
        dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
        homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

        return contrast, dissimilarity, homogeneity

    # def detect_edges(image):

    #     # Apply Canny edge detection
    #     edges = cv2.Canny(image, 100, 200)

    #     return edges

    def compute_iqr(image):

        # Compute the interquartile range of pixel intensities
        iqr = np.percentile(image, 75) - np.percentile(image, 25)

        return iqr

    def compute_mean_intensity(image):

        # Compute the mean pixel intensity
        mean_intensity = np.mean(image)

        return mean_intensity

    def compute_std_dev(image):

        # Compute the standard deviation of pixel intensities
        std_dev = np.std(image)

        return std_dev


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
