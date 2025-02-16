""" Script to compute the correlation between PCA and UMAP features and computed features
* finds the computed features best representing the PCA and UMAP components
* outputs a heatmap of the correlation between PCA and UMAP features and computed features
"""

# %%
import os
import sys
from pathlib import Path

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
from sklearn.decomposition import PCA
import mahotas as mh
import pandas as pd
import scipy.stats
from numpy import fft
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks

# %%
features_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)

# %%

source_channel = ["Phase3D", "raw GFP EX488 EM525-45"]
seg_channel = ["nuclei_prediction_labels_labels"]
z_range = (16, 21)
normalizations = None
fov_name = "/B/1/000000"
track_id = 74

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

# load all unprojected features:
features = embedding_dataset["features"]

# %% PCA analysis of the features

pca = PCA(n_components=8)
pca_features = pca.fit_transform(features.values)
features = (
    features.assign_coords(PCA1=("sample", pca_features[:, 0]))
    .assign_coords(PCA2=("sample", pca_features[:, 1]))
    .assign_coords(PCA3=("sample", pca_features[:, 2]))
    .assign_coords(PCA4=("sample", pca_features[:, 3]))
    .assign_coords(PCA5=("sample", pca_features[:, 4]))
    .assign_coords(PCA6=("sample", pca_features[:, 5]))
    .assign_coords(PCA7=("sample", pca_features[:, 6]))
    .assign_coords(PCA8=("sample", pca_features[:, 7]))
    .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8"], append=True)
)


# %% convert the xarray to dataframe structure and add columns for computed features
features_df = features.to_dataframe()
features_df = features_df.drop(columns=["features"])
df = features_df.drop_duplicates()
features = df.reset_index(drop=True)

features = features[features["fov_name"].str.startswith(("/C/2/000000", "/B/3/000000", "/C/1/000000", "/B/2/000000"))]

# features computed to extract the value and texture features from image patches
# mean intensity
features["Phase Mean Intensity"] = np.nan
features["Fluor Mean Intensity"] = np.nan
# standard deviation
features["Phase Standard Deviation"] = np.nan
features["Fluor Standard Deviation"] = np.nan
# kurtosis
features["Fluor kurtosis"] = np.nan
features["Phase kurtosis"] = np.nan
# skewness
features["Fluor skewness"] = np.nan
features["Phase skewness"] = np.nan
# entropy
features["Entropy Phase"] = np.nan
features["Entropy Fluor"] = np.nan
# iqr
features["Phase IQR"] = np.nan
features["Fluor IQR"] = np.nan
# contrast
features["Contrast Phase"] = np.nan
features["Contrast Fluor"] = np.nan
# texture
features["Phase texture"] = np.nan
features["Fluor texture"] = np.nan

# organelle segmentation features
features["Fluor Area"] = np.nan
features["Masked fluor Intensity"] = np.nan
features["Perimeter"] = np.nan

# nuclear segmentation based features
features["Fluor weighted intensity gradient"] = np.nan
features["Phase weighted intensity gradient"] = np.nan

features["Perimeter"] = np.nan
features["Nuclear area"] = np.nan
features["Perimeter area ratio"] = np.nan
features["Nucleus eccentricity"] = np.nan

features["Instantaneous velocity"] = np.nan
features["Fluor localization"] = np.nan

# Zernike moment 0
features["Zernike moment std"] = np.nan
features["Zernike moment mean"] = np.nan

# %% iterate over new features and compute them

def normalize_image(image):
    """Normalize the image to 0-255 range."""
    return (image - image.min()) / (image.max() - image.min()) * 255

# weighted intensity gradient
def compute_weighted_intensity_gradient(image):
    """Compute the radial gradient profile and its slope.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        float: Slope of the azimuthally averaged radial gradient profile
    """
    # Get image dimensions
    h, w = image.shape
    center_y, center_x = h // 2, w // 2
    
    # Create meshgrid of coordinates
    y, x = np.ogrid[:h, :w]
    
    # Calculate radial distances from center
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate gradients in x and y directions
    gy, gx = np.gradient(image)
    
    # Calculate magnitude of gradient
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    # Weight gradient by intensity
    weighted_gradient = gradient_magnitude * image
    
    # Calculate maximum radius (to edge of image)
    max_radius = int(min(h//2, w//2))
    
    # Initialize arrays for radial profile
    radial_profile = np.zeros(max_radius)
    counts = np.zeros(max_radius)
    
    # Bin pixels by radius
    for i in range(h):
        for j in range(w):
            radius = int(r[i,j])
            if radius < max_radius:
                radial_profile[radius] += weighted_gradient[i,j]
                counts[radius] += 1
    
    # Average by counts (avoiding division by zero)
    valid_mask = counts > 0
    radial_profile[valid_mask] /= counts[valid_mask]
    
    # Calculate slope using linear regression
    x = np.arange(max_radius)[valid_mask]
    y = radial_profile[valid_mask]
    slope = np.polyfit(x, y, 1)[0]
    
    return slope

# perimeter of nuclear segmentations found inside the patch
def compute_perimeter_area_ratio(image):
    """Compute the perimeter of the nuclear segmentations found inside the patch.
    
    Args:
        image (np.ndarray): Input image with nuclear segmentation labels
        
    Returns:
        float: Total perimeter of the nuclear segmentations found inside the patch
    """
    total_perimeter = 0
    total_area = 0
    
    # Get the binary mask of each nuclear segmentation labels
    for label in np.unique(image):
        if label != 0:  # Skip background
            continue
            
        # Create binary mask for current label
        mask = (image == label)
        
        # Convert to proper format for OpenCV
        mask = mask.astype(np.uint8)
        
        # Ensure we have a 2D array
        if mask.ndim > 2:
            # Take the first channel if multi-channel
            mask = mask[:, :, 0] if mask.shape[-1] > 1 else mask.squeeze()
        
        # Ensure we have values 0 and 1 only
        mask = (mask > 0).astype(np.uint8) * 255
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add perimeter of all contours for this label
        for contour in contours:
            total_perimeter += cv2.arcLength(contour, closed=True)
        
        # Add area of all contours for this label
        for contour in contours:
            total_area += cv2.contourArea(contour)
    average_area = total_area / (len(np.unique(image))-1)
    average_perimeter = total_perimeter / (len(np.unique(image))-1)

    return average_perimeter, average_area, total_perimeter / total_area

def compute_texture_features(image):
    """Compute the texture features of the image.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        float: Mean of the texture features
    """
    # rescale image to 0 to 255 and convert to uint8
    image_rescaled = (image - image.min()) / (image.max() - image.min()) * 255
    texture_features = mh.features.haralick(image_rescaled.astype('uint8')).ptp(0)
    return np.mean(texture_features)

def nucleus_eccentricity(image):
    """Compute the eccentricity of the nucleus.
    
    Args:
        image (np.ndarray): Input image with nuclear segmentation labels
        
    Returns:
        float: Eccentricity of the nucleus
    """
    # convert the label image to a binary image and ensure single channel
    binary_image = (image > 0).astype(np.uint8)
    if binary_image.ndim > 2:
        binary_image = binary_image[:,:,0]  # Take first channel if multi-channel
    binary_image = cv2.convertScaleAbs(binary_image * 255)  # Ensure proper OpenCV format
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    eccentricities = []
    
    for contour in contours:
        # Fit an ellipse to the contour (works well for ellipse-like shapes)
        if len(contour) >= 5:  # At least 5 points are required to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            
            # Extract the lengths of the semi-major and semi-minor axes
            major_axis, minor_axis = max(axes), min(axes)
            
            # Calculate the eccentricity using the formula: e = sqrt(1 - (b^2 / a^2))
            eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2))
            eccentricities.append(eccentricity)
    
    return np.mean(eccentricities)

def Eucledian_distance_transform(image):
    """Compute the Euclidean distance transform of a binary mask.
    
    Args:
        image (np.ndarray): Binary mask (0s and 1s)
        
    Returns:
        np.ndarray: Distance transform where each pixel value represents 
                   the Euclidean distance to the nearest non-zero pixel
    """
    # Ensure the image is binary
    binary_mask = (image > 0).astype(np.uint8)
    
    # Compute the distance transform
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    return dist_transform

def fluor_localization(image, mask):
    """ Compute localization of fluor using Eucledian distance transformation and fluor intensity"""
    # compute EDT of mask
    edt = Eucledian_distance_transform(mask)
    # compute the intensity weighted center of the fluor
    intensity_weighted_center = np.sum(image * edt) / np.sum(edt)
    return intensity_weighted_center

def compute_instantaneous_velocity(track_info, row_idx):
    """Compute the instantaneous velocity of the cell.

    Args:
        track_info (pd.DataFrame): DataFrame containing track information
        row_idx (int): Current row index in the track_info DataFrame
        
    Returns:
        float: Instantaneous velocity of the cell
    """
    # Check if previous timepoint exists
    has_prev = row_idx > 0
    # Check if next timepoint exists
    has_next = row_idx < len(track_info) - 1
    
    if has_prev:
        # Use previous timepoint
        prev_row = track_info.iloc[row_idx - 1]
        curr_row = track_info.iloc[row_idx]
        distance = np.sqrt((curr_row["x"] - prev_row["x"])**2 + 
                         (curr_row["y"] - prev_row["y"])**2)
        time_diff = curr_row["t"] - prev_row["t"]
    elif has_next:
        # Use next timepoint if previous doesn't exist
        next_row = track_info.iloc[row_idx + 1]
        curr_row = track_info.iloc[row_idx]
        distance = np.sqrt((next_row["x"] - curr_row["x"])**2 + 
                         (next_row["y"] - curr_row["y"])**2)
        time_diff = next_row["t"] - curr_row["t"]
    else:
        # No neighboring timepoints exist
        return 0.0
        
    # Compute velocity (avoid division by zero)
    velocity = distance / max(time_diff, 1e-6)
    return velocity

def compute_kurtosis(image):
    """Compute the kurtosis of the image."""
    normalized_image = normalize_image(image)
    kurtosis = scipy.stats.kurtosis(normalized_image, fisher=True, axis=None)
    return kurtosis

def compute_skewness(image):
    """Compute the skewness of the image."""
    normalized_image = normalize_image(image)
    skewness = scipy.stats.skew(normalized_image, axis=None)
    return skewness

def compute_spectral_entropy(image):
    """
    Compute the spectral entropy of the image
    High frequency components are observed to increase in phase and reduce in sensor when cell is infected
    :param np.array image: input image
    :return: spectral entropy
    """

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

def compute_iqr(image):
    """
    Compute the interquartile range of pixel intensities
    Observed to increase when cell is infected
    :param np.array image: input image
    :return: interquartile range of pixel intensities
    """

    # Compute the interquartile range of pixel intensities
    iqr = np.percentile(image, 75) - np.percentile(image, 25)

    return iqr

def compute_mean_intensity(image):
    """
    Compute the mean pixel intensity
    Expected to vary when cell morphology changes due to infection, divison or death
    :param np.array image: input image
    :return: mean pixel intensity
    """

    # Compute the mean pixel intensity
    mean_intensity = np.mean(image)

    return mean_intensity

def compute_std_dev(image):
    """
    Compute the standard deviation of pixel intensities
    Expected to vary when cell morphology changes due to infection, divison or death
    :param np.array image: input image
    :return: standard deviation of pixel intensities
    """
    # Compute the standard deviation of pixel intensities
    std_dev = np.std(image)

    return std_dev

def compute_glcm_features(image):
    """
    Compute the contrast, dissimilarity and homogeneity of the image
    Both sensor and phase texture changes when infected, smooth in sensor, and rough in phase
    :param np.array image: input image
    :return: contrast, dissimilarity, homogeneity
    """

    # Normalize the input image from 0 to 255
    image = (image - np.min(image)) * (255 / (np.max(image) - np.min(image)))
    image = image.astype(np.uint8)

    # Compute the GLCM
    distances = [1]  # Distance between pixels
    angles = [45]  # Angle in radians

    glcm = graycomatrix(image, distances, angles, symmetric=True, normed=True)

    # Compute GLCM properties - average across all angles
    contrast = graycoprops(glcm, "contrast")[0, 0]
    dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    return contrast, dissimilarity, homogeneity

def compute_area(input_image, sigma=0.6):
    """Create a binary mask using morphological operations
    Sensor area will increase when infected due to expression in nucleus
    :param np.array input_image: generate masks from this 3D image
    :param float sigma: Gaussian blur standard deviation, increase in value increases blur
    :return: area of the sensor mask & mean intensity inside the sensor area
    """

    input_image_blur = gaussian(input_image, sigma=sigma)

    thresh = threshold_otsu(input_image_blur)
    mask = input_image >= thresh

    # Apply sensor mask to the image
    masked_image = input_image * mask

    # Compute the mean intensity inside the sensor area
    masked_intensity = np.mean(masked_image)

    return masked_intensity, np.sum(mask)

def compute_radial_intensity_gradient(image):
    """
    Compute the radial intensity gradient of the image
    The sensor relocalizes inside the nucleus, which is center of the image when cells are infected
    Expected negative gradient when infected and zero to positive gradient when not infected
    :param np.array image: input image
    :return: radial intensity gradient
    """
    # normalize the image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # compute the intensity gradient from center to periphery
    y, x = np.indices(image.shape)
    center = np.array(image.shape) / 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    radial_intensity_values = tbin / nr

    # get the slope radial_intensity_values
    from scipy.stats import linregress

    radial_intensity_gradient = linregress(
        range(len(radial_intensity_values)), radial_intensity_values
    )

    return radial_intensity_gradient[0]

def compute_zernike_momemnts(image):
    """ 
    Compute the Zernike moments of the image
    :param np.array image: input image
    :return: Zernike moments
    """
    # compute the Zernike moments
    zernike_moments = mh.features.zernike_moments(image, 32)
    # return standard deviation (for level of variation in texture) and mean (for overall level of texture) of the zernike moments
    return np.std(zernike_moments), np.mean(zernike_moments)

# %% compute the computed features and add them to the dataset

fov_names_list = features["fov_name"].unique()
unique_fov_names = sorted(list(set(fov_names_list)))
max_iterations = 50

for fov_name in unique_fov_names:
    csv_files = list((Path(str(tracks_path) + str(fov_name))).glob("*.csv"))
    tracks_df = pd.read_csv(str(csv_files[0]))

    unique_track_ids = features[features["fov_name"] == fov_name]["track_id"].unique()
    unique_track_ids = list(set(unique_track_ids))

    iteration_count = 0

    for track_id in unique_track_ids:
        if iteration_count >= max_iterations:
            break
        track_subdf = tracks_df[tracks_df["track_id"] == track_id]
            
        prediction_dataset = dataset_of_tracks(
            data_path,
            tracks_path,
            [fov_name],
            [track_id],
            z_range=z_range,
            source_channel=source_channel,
        )
        track_channel = dataset_of_tracks(
            tracks_path,
            tracks_path,
            [fov_name],
            [track_id],
            z_range=(0,1),
            source_channel=seg_channel,
        )

        whole = np.stack([p["anchor"] for p in prediction_dataset])
        seg_mask = np.stack([p["anchor"] for p in track_channel])
        phase = whole[:, 0, 2]
        # Normalize phase image to 0-255 range
        # phase = ((phase - phase.min()) / (phase.max() - phase.min()) * 255).astype(np.uint8)
        # Normalize fluorescence image to 0-255 range
        fluor = np.max(whole[:, 1], axis=1)
        # fluor = ((fluor - fluor.min()) / (fluor.max() - fluor.min()) * 255).astype(np.uint8)
        nucl_mask = seg_mask[:, 0, 0]

        for t in range(phase.shape[0]):

            # compute mean intensity of the fluor
            fluor_mean_intensity = compute_mean_intensity(fluor[t])
            #compute mean intensity of the phase
            phase_mean_intensity = compute_mean_intensity(phase[t])

            # compute standard deviation of the fluor
            fluor_std_dev = compute_std_dev(fluor[t])
            # compute standard deviation of the phase
            phase_std_dev = compute_std_dev(phase[t])

            # compute kurtosis of the fluor
            fluor_kurtosis = compute_kurtosis(fluor[t])
            # compute kurtosis of the phase
            phase_kurtosis = compute_kurtosis(phase[t])

            # compute skewness of the fluor
            fluor_skewness = compute_skewness(fluor[t])
            # compute skewness of the phase
            phase_skewness = compute_skewness(phase[t])

            # Compute area of fluor
            masked_intensity, area = compute_area(fluor[t])

            # Compute higher frequency features using spectral entropy
            entropy_phase = compute_spectral_entropy(phase[t])

            # Compute texture analysis using GLCM
            contrast_phase, dissimilarity_phase, homogeneity_phase = (
                compute_glcm_features(phase[t])
            )
            contrast_fluor, dissimilarity_fluor, homogeneity_fluor = (
                compute_glcm_features(fluor[t])
            )

            # Compute interqualtile range of pixel intensities
            iqr = compute_iqr(phase[t])

            # Compute mean pixel intensity
            fluor_mean_intensity = compute_mean_intensity(fluor[t])

            # Compute standard deviation of pixel intensities
            phase_std_dev = compute_std_dev(phase[t])
            fluor_std_dev = compute_std_dev(fluor[t])

            # Compute gradient for localization in organelle channel
            fluor_weighted_gradient = compute_weighted_intensity_gradient(fluor[t])
            # Compute gradient for localization in phase channel
            phase_weighted_gradient = compute_weighted_intensity_gradient(phase[t])

            # compute the texture features using haralick
            phase_texture = compute_texture_features(phase[t])
            fluor_texture = compute_texture_features(fluor[t])

            # compute the perimeter of the nuclear segmentations found inside the patch
            perimeter, nucl_area, perimeter_area_ratio = compute_perimeter_area_ratio(nucl_mask[t])

            # compute the eccentricity of the nucleus
            seg_eccentricity = nucleus_eccentricity(nucl_mask[t])

            # compute instantaneous velocity of cell 
            inst_velocity = compute_instantaneous_velocity(track_subdf, t)

            # compute the localization of the fluor
            fluor_location = fluor_localization(fluor[t], nucl_mask[t])

            # compute the zernike moments
            zernike_std_fluor, zernike_mean_fluor = compute_zernike_momemnts(fluor[t])
            zernike_std_phase, zernike_mean_phase = compute_zernike_momemnts(phase[t])

            # Create dictionary mapping feature names to their computed values
            feature_values = {
                "Phase Mean Intensity": phase_mean_intensity,
                "Phase Kurtosis": phase_kurtosis,
                "Phase Skewness": phase_skewness,
                "Fluor Mean Intensity": fluor_mean_intensity,
                "Fluor Kurtosis": fluor_kurtosis,
                "Fluor Skewness": fluor_skewness,
                "Fluor Area": area,
                "Masked fluor Intensity": masked_intensity,
                "Entropy Phase": entropy_phase,
                "Contrast Phase": contrast_phase,
                "Contrast Fluor": contrast_fluor,
                "Phase IQR": iqr,
                "Fluor Mean Intensity": fluor_mean_intensity,
                "Phase Standard Deviation": phase_std_dev,
                "Fluor Standard Deviation": fluor_std_dev,
                "Fluor weighted intensity gradient": fluor_weighted_gradient,
                "Phase weighted intensity gradient": phase_weighted_gradient,
                "Phase texture": phase_texture,
                "Fluor texture": fluor_texture,
                "Perimeter": perimeter,
                "Nuclear area": nucl_area,
                "Perimeter area ratio": perimeter_area_ratio,
                "Nucleus eccentricity": seg_eccentricity,
                "Instantaneous velocity": inst_velocity,
                "Fluor localization": fluor_location,
                "Zernike moment std": zernike_std_fluor,
                "Zernike moment mean": zernike_mean_fluor,
                "Zernike moment std phase": zernike_std_phase,
                "Zernike moment mean phase": zernike_mean_phase,
            }

            # Update features dataframe using the dictionary
            for feature_name, value in feature_values.items():
                features.loc[
                    (features["fov_name"] == fov_name)
                    & (features["track_id"] == track_id)
                    & (features["t"] == t),
                    feature_name
                ] = value

        iteration_count += 1
        print(f"Processed {iteration_count}")

# %%

# Save the features dataframe to a CSV file
features.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan_organelle_multiwell.csv",
    index=False,
)

# # read the features dataframe from the CSV file
# features = pd.read_csv(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan_organelle.csv"
# )
# remove the column "Perimeter"
# features = features.drop(columns=["Perimeter"])

# remove the rows with missing values
features = features.dropna()

# sub_features = features[features["Time"] == 20]
feature_df_removed = features.drop(
    columns=["fov_name", "track_id", "t", "id", "parent_track_id", "parent_id", "PHATE1", "PHATE2", "UMAP1", "UMAP2"]
)

# Compute correlation between PCA features and computed features
correlation = feature_df_removed.corr(method="spearman")

# %% display PCA correlation as a heatmap

plt.figure(figsize=(20, 8))
sns.heatmap(
    correlation.drop(columns=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8"]).loc[
        "PCA1":"PCA8", :
    ],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={'size': 12}  # Increase annotation text size
)
plt.title("Correlation between PCA features and computed features", fontsize=12)
plt.xlabel("Computed Features", fontsize=12)
plt.ylabel("PCA Features", fontsize=12)
plt.xticks(fontsize=12)  # Increase x-axis tick labels
plt.yticks(fontsize=12)  # Increase y-axis tick labels
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_organelle_multiwell.png",
    dpi=300
)


# %% plot PCA vs set of computed features

# set_features = [
#     "Phase Standard Deviation",
#     "Entropy Fluor",
#     "Fluor Standard Deviation",
#     "Dissimilarity Fluor",
#     "Fluor Area",
#     "Fluor Symmetry Score",
#     "Entropy Fluor",
#     "Masked fluor Intensity",
# ]

# plt.figure(figsize=(8, 10))
# sns.heatmap(
#     correlation.loc[set_features, "PCA1":"PCA8"],
#     annot=True,
#     cmap="coolwarm",
#     fmt=".2f",
#     vmin=-1,
#     vmax=1,
# )

# plt.savefig(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_setfeatures.svg"
# )
