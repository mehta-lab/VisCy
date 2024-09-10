import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import fft
from skimage import color
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
import umap
from torch.utils.data import DataLoader, TensorDataset

from viscy.data.triplet import TripletDataModule

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


"""
Utilities for loading datasets.
"""


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
    initial_yx_patch_size=(256, 256),
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


""" Methods for evaluating clustering performance.
"""


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


def silhouette_score(embeddings, clusters):
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
    score = silhouette_score(embeddings, clusters)
    return score


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


def compute_umap(embedding_dataset, normalize_features=True):
    features = embedding_dataset["features"]
    projections = embedding_dataset["projections"]

    if normalize_features:
        scaled_projections = StandardScaler().fit_transform(projections.values)
        scaled_features = StandardScaler().fit_transform(features.values)
    else:
        scaled_projections = projections.values
        scaled_features = features.values

    # Compute UMAP for features and projections
    # Computing 3 components to enable 3D visualization.
    umap_features = umap.UMAP(random_state=42, n_components=3)
    umap_projection = umap.UMAP(random_state=42, n_components=3)
    umap_features_embedding = umap_features.fit_transform(scaled_features)
    umap_projection_embedding = umap_projection.fit_transform(scaled_projections)

    # Prepare DataFrame with id and UMAP coordinates
    umap_df = pd.DataFrame(
        {
            "id": embedding_dataset["id"].values,
            "fov_name": embedding_dataset["fov_name"].values,
            "UMAP1": umap_features_embedding[:, 0],
            "UMAP2": umap_features_embedding[:, 1],
            "UMAP3": umap_features_embedding[:, 2],
            "UMAP1_proj": umap_projection_embedding[:, 0],
            "UMAP2_proj": umap_projection_embedding[:, 1],
            "UMAP3_proj": umap_projection_embedding[:, 2],
        }
    )

    return umap_features, umap_projection, umap_df


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
