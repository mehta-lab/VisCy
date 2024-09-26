from collections import defaultdict

import numpy as np
import pandas as pd
import umap
from numpy import fft
from skimage import color
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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

__all__ = [
    # re-exporting from sklearn
    "silhouette_score",
    "load_annotation",
    "dataset_of_tracks",
    "knn_accuracy",
    "clustering_evaluation",
    "compute_pca",
    "compute_umap",
    "FeatureExtractor",
]


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


"""Methods for evaluating clustering performance."""


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


# Function to extract embeddings and calculate cosine similarities for a specific cell
def calculate_cosine_similarity_cell(embedding_dataset, fov_name, track_id):
    # Filter the dataset for the specific infected cell
    filtered_data = embedding_dataset.where(
        (embedding_dataset["fov_name"] == fov_name)
        & (embedding_dataset["track_id"] == track_id),
        drop=True,
    )

    # Extract the feature embeddings and time points
    features = filtered_data["features"].values  # (sample, features)
    time_points = filtered_data["t"].values  # (sample,)

    # Get the first time point's embedding
    first_time_point_embedding = features[0].reshape(1, -1)

    # Calculate cosine similarity between each time point and the first time point
    cosine_similarities = []
    for i in range(len(time_points)):
        similarity = cosine_similarity(
            first_time_point_embedding, features[i].reshape(1, -1)
        )
        cosine_similarities.append(similarity[0][0])

    return time_points, cosine_similarities


# Function to compute the norm of differences between embeddings at t and t + tau
def compute_displacement_mean_std(
    embedding_dataset, max_tau=10, use_cosine=False, use_dissimilarity=False
):
    # Get the arrays of (fov_name, track_id, t, and embeddings)
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values
    embeddings = embedding_dataset["features"].values

    # Dictionary to store displacements for each tau
    displacement_per_tau = defaultdict(list)

    # Iterate over all entries in the dataset
    for i in range(len(fov_names)):
        fov_name = fov_names[i]
        track_id = track_ids[i]
        current_time = timepoints[i]
        current_embedding = embeddings[i]

        # For each time point t, compute displacements for t + tau
        for tau in range(1, max_tau + 1):
            future_time = current_time + tau

            # Find if future_time exists for the same (fov_name, track_id)
            matching_indices = np.where(
                (fov_names == fov_name)
                & (track_ids == track_id)
                & (timepoints == future_time)
            )[0]

            if len(matching_indices) == 1:
                # Get the embedding at t + tau
                future_embedding = embeddings[matching_indices[0]]

                if use_cosine:
                    # Compute cosine similarity
                    similarity = cosine_similarity(
                        current_embedding.reshape(1, -1),
                        future_embedding.reshape(1, -1),
                    )[0][0]
                    # Choose whether to use similarity or dissimilarity
                    if use_dissimilarity:
                        displacement = 1 - similarity  # Cosine dissimilarity
                    else:
                        displacement = similarity  # Cosine similarity
                else:
                    # Compute the Euclidean distance, elementwise square on difference
                    displacement = np.sum((current_embedding - future_embedding) ** 2)

                # Store the displacement for the given tau
                displacement_per_tau[tau].append(displacement)

    # Compute mean and std displacement for each tau by averaging the displacements
    mean_displacement_per_tau = {
        tau: np.mean(displacements)
        for tau, displacements in displacement_per_tau.items()
    }
    std_displacement_per_tau = {
        tau: np.std(displacements)
        for tau, displacements in displacement_per_tau.items()
    }

    return mean_displacement_per_tau, std_displacement_per_tau


# Function to compute the norm of differences between embeddings at t and t + tau
def compute_displacement(
    embedding_dataset,
    max_tau=10,
    use_cosine=False,
    use_dissimilarity=False,
    use_umap=False,
):
    # Get the arrays of (fov_name, track_id, t, and embeddings)
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values

    if use_umap:
        umap1 = embedding_dataset["UMAP1"].values
        umap2 = embedding_dataset["UMAP2"].values
        embeddings = np.vstack((umap1, umap2)).T
    else:
        embeddings = embedding_dataset["features"].values

    # Dictionary to store displacements for each tau
    displacement_per_tau = defaultdict(list)

    # Iterate over all entries in the dataset
    for i in range(len(fov_names)):
        fov_name = fov_names[i]
        track_id = track_ids[i]
        current_time = timepoints[i]
        current_embedding = embeddings[i]

        # For each time point t, compute displacements for t + tau
        for tau in range(1, max_tau + 1):
            future_time = current_time + tau

            # Find if future_time exists for the same (fov_name, track_id)
            matching_indices = np.where(
                (fov_names == fov_name)
                & (track_ids == track_id)
                & (timepoints == future_time)
            )[0]

            if len(matching_indices) == 1:
                # Get the embedding at t + tau
                future_embedding = embeddings[matching_indices[0]]

                if use_cosine:
                    # Compute cosine similarity
                    similarity = cosine_similarity(
                        current_embedding.reshape(1, -1),
                        future_embedding.reshape(1, -1),
                    )[0][0]
                    # Choose whether to use similarity or dissimilarity
                    if use_dissimilarity:
                        displacement = 1 - similarity  # Cosine dissimilarity
                    else:
                        displacement = similarity  # Cosine similarity
                else:
                    # Compute the Euclidean distance, elementwise square on difference
                    displacement = np.sum((current_embedding - future_embedding) ** 2)

                # Store the displacement for the given tau
                displacement_per_tau[tau].append(displacement)

    return displacement_per_tau

def calculate_normalized_euclidean_distance_cell(embedding_dataset, fov_name, track_id):
    filtered_data = embedding_dataset.where(
        (embedding_dataset["fov_name"] == fov_name)
        & (embedding_dataset["track_id"] == track_id),
        drop=True,
    )

    features = filtered_data["features"].values  # (sample, features)
    time_points = filtered_data["t"].values  # (sample,)

    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Get the first time point's normalized embedding
    first_time_point_embedding = normalized_features[0].reshape(1, -1)

    euclidean_distances = []
    for i in range(len(time_points)):
        distance = np.linalg.norm(first_time_point_embedding - normalized_features[i].reshape(1, -1))
        euclidean_distances.append(distance)

    return time_points, euclidean_distances

def compute_displacement_mean_std_full(
    embedding_dataset, max_tau=10
):
    fov_names = embedding_dataset["fov_name"].values
    track_ids = embedding_dataset["track_id"].values
    timepoints = embedding_dataset["t"].values
    embeddings = embedding_dataset["features"].values

    cell_identifiers = np.array(list(zip(fov_names, track_ids)), dtype=[('fov_name', 'O'), ('track_id', 'int64')])

    unique_cells = np.unique(cell_identifiers)

    displacement_per_tau = defaultdict(list)

    for cell in unique_cells:
        fov_name = cell['fov_name']
        track_id = cell['track_id']

        indices = np.where((fov_names == fov_name) & (track_ids == track_id))[0]
        
        cell_timepoints = timepoints[indices]
        cell_embeddings = embeddings[indices]

        sorted_indices = np.argsort(cell_timepoints)
        cell_timepoints = cell_timepoints[sorted_indices]
        cell_embeddings = cell_embeddings[sorted_indices]

        for i in range(len(cell_timepoints)):
            current_time = cell_timepoints[i]
            current_embedding = cell_embeddings[i]

            current_embedding = current_embedding / np.linalg.norm(current_embedding)

            for tau in range(0, max_tau + 1):
                future_time = current_time + tau

                future_index = np.where(cell_timepoints == future_time)[0]

                if len(future_index) >= 1:
                    future_embedding = cell_embeddings[future_index[0]]
                    future_embedding = future_embedding / np.linalg.norm(future_embedding)

                    distance = np.linalg.norm(current_embedding - future_embedding)
                    
                    displacement_per_tau[tau].append(distance)

    mean_displacement_per_tau = {
        tau: np.mean(displacements)
        for tau, displacements in displacement_per_tau.items()
    }
    std_displacement_per_tau = {
        tau: np.std(displacements)
        for tau, displacements in displacement_per_tau.items()
    }

    return mean_displacement_per_tau, std_displacement_per_tau

