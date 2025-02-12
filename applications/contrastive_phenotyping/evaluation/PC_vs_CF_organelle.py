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
import mahotas.features
import pandas as pd

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks
from viscy.representation.evaluation.feature import (
    FeatureExtractor as FE,
)

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
z_range = (30, 35)
normalizations = None
# fov_name = "/0/6/000000"
# track_id = 21

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

features = features[features["fov_name"].str.startswith("/C/2/000000")]

features["Phase Symmetry Score"] = np.nan
features["Fluor Symmetry Score"] = np.nan
features["Fluor Area"] = np.nan
features["Masked fluor Intensity"] = np.nan
features["Entropy Phase"] = np.nan
features["Contrast Phase"] = np.nan
features["Dissimilarity Phase"] = np.nan
features["Homogeneity Phase"] = np.nan
features["Contrast Fluor"] = np.nan
features["Dissimilarity Fluor"] = np.nan
features["Homogeneity Fluor"] = np.nan
features["Phase IQR"] = np.nan
features["Fluor Mean Intensity"] = np.nan
features["Phase Standard Deviation"] = np.nan
features["Fluor Standard Deviation"] = np.nan
features["Fluor weighted intensity gradient"] = np.nan
features["Fluor texture"] = np.nan
features["Phase texture"] = np.nan
features["Perimeter area ratio"] = np.nan
features["Nucleus eccentricity"] = np.nan
features["Instantaneous velocity"] = np.nan
features["Fluor localization"] = np.nan

# %% iterate over new features and compute them

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

    return total_perimeter / total_area

def texture_features(image):
    """Compute the texture features of the image.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        float: Mean of the texture features
    """
    # rescale image to 0 to 255 and convert to uint8
    image_rescaled = (image - image.min()) / (image.max() - image.min()) * 255
    texture_features = mahotas.features.haralick(image_rescaled.astype('uint8')).ptp(0)
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

# %% compute the computed features and add them to the dataset

fov_names_list = features["fov_name"].unique()
unique_fov_names = sorted(list(set(fov_names_list)))

iteration_count = 0
max_iterations = 100

for fov_name in unique_fov_names:
    csv_files = list((Path(str(tracks_path) + str(fov_name))).glob("*.csv"))
    tracks_df = pd.read_csv(str(csv_files[0]))

    unique_track_ids = features[features["fov_name"] == fov_name]["track_id"].unique()
    unique_track_ids = list(set(unique_track_ids))

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
        fluor = np.max(whole[:, 1], axis=1)
        nucl_mask = seg_mask[:, 0, 0]

        for t in range(phase.shape[0]):

            # Compute Fourier descriptors for phase image
            phase_descriptors = FE.compute_fourier_descriptors(phase[t])
            # Analyze symmetry of phase image
            phase_symmetry_score = FE.analyze_symmetry(phase_descriptors)

            # Compute Fourier descriptors for fluor image
            fluor_descriptors = FE.compute_fourier_descriptors(fluor[t])
            # Analyze symmetry of fluor image
            fluor_symmetry_score = FE.analyze_symmetry(fluor_descriptors)

            # Compute area of fluor
            masked_intensity, area = FE.compute_area(fluor[t])

            # Compute higher frequency features using spectral entropy
            entropy_phase = FE.compute_spectral_entropy(phase[t])

            # Compute texture analysis using GLCM
            contrast_phase, dissimilarity_phase, homogeneity_phase = (
                FE.compute_glcm_features(phase[t])
            )
            contrast_fluor, dissimilarity_fluor, homogeneity_fluor = (
                FE.compute_glcm_features(fluor[t])
            )

            # Compute interqualtile range of pixel intensities
            iqr = FE.compute_iqr(phase[t])

            # Compute mean pixel intensity
            fluor_mean_intensity = FE.compute_mean_intensity(fluor[t])

            # Compute standard deviation of pixel intensities
            phase_std_dev = FE.compute_std_dev(phase[t])
            fluor_std_dev = FE.compute_std_dev(fluor[t])

            # Compute gradient for localization in organelle channel
            fluor_weighted_gradient = compute_weighted_intensity_gradient(fluor[t])

            # compute the texture features using haralick
            phase_texture = texture_features(phase[t])
            fluor_texture = texture_features(fluor[t])

            # compute the perimeter of the nuclear segmentations found inside the patch
            perimeter_area_ratio = compute_perimeter_area_ratio(nucl_mask[t])

            # compute the eccentricity of the nucleus
            seg_eccentricity = nucleus_eccentricity(nucl_mask[t])

            # compute instantaneous velocity of cell 
            inst_velocity = compute_instantaneous_velocity(track_subdf, t)

            # compute the localization of the fluor
            fluor_location = fluor_localization(fluor[t], nucl_mask[t])

            # Create dictionary mapping feature names to their computed values
            feature_values = {
                "Fluor Symmetry Score": fluor_symmetry_score,
                "Phase Symmetry Score": phase_symmetry_score,
                "Fluor Area": area,
                "Masked fluor Intensity": masked_intensity,
                "Entropy Phase": entropy_phase,
                "Contrast Phase": contrast_phase,
                "Dissimilarity Phase": dissimilarity_phase,
                "Homogeneity Phase": homogeneity_phase,
                "Contrast Fluor": contrast_fluor,
                "Dissimilarity Fluor": dissimilarity_fluor,
                "Homogeneity Fluor": homogeneity_fluor,
                "Phase IQR": iqr,
                "Fluor Mean Intensity": fluor_mean_intensity,
                "Phase Standard Deviation": phase_std_dev,
                "Fluor Standard Deviation": fluor_std_dev,
                "Fluor weighted intensity gradient": fluor_weighted_gradient,
                "Phase texture": phase_texture,
                "Fluor texture": fluor_texture,
                "Perimeter area ratio": perimeter_area_ratio,
                "Nucleus eccentricity": seg_eccentricity,
                "Instantaneous velocity": inst_velocity,
                "Fluor localization": fluor_location,
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
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan_organelle.csv",
    index=False,
)

# # read the features dataframe from the CSV file
# features = pd.read_csv(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan.csv"
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
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_organelle.svg"
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
