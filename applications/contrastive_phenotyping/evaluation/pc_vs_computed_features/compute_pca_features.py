from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks
from viscy.representation.evaluation.feature import CellFeatures


## function to read the embedding dataset and return the features
def compute_PCA(features_path: Path):
    """Compute PCA components from embedding features and combine with original features.

    This function reads an embedding dataset, standardizes the features, and computes
    8 principal components. The PCA components are then combined with the original
    features in an xarray dataset structure.

    Parameters
    ----------
    features_path : Path
        Path to the embedding dataset containing the feature vectors.

    Returns
    -------
    features: xarray dataset with PCA components as new coordinates

    """
    embedding_dataset = read_embedding_dataset(features_path)
    embedding_dataset

    # load all unprojected features:
    features = embedding_dataset["features"]
    scaled_features = StandardScaler().fit_transform(features.values)
    # PCA analysis of the features

    pca = PCA(n_components=8)
    pca_features = pca.fit_transform(scaled_features)
    features = (
        features.assign_coords(PCA1=("sample", pca_features[:, 0]))
        .assign_coords(PCA2=("sample", pca_features[:, 1]))
        .assign_coords(PCA3=("sample", pca_features[:, 2]))
        .assign_coords(PCA4=("sample", pca_features[:, 3]))
        .assign_coords(PCA5=("sample", pca_features[:, 4]))
        .assign_coords(PCA6=("sample", pca_features[:, 5]))
        .assign_coords(PCA7=("sample", pca_features[:, 6]))
        .assign_coords(PCA8=("sample", pca_features[:, 7]))
        .set_index(
            sample=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8"],
            append=True,
        )
    )

    return features


def compute_features(
    features_path: Path,
    data_path: Path,
    tracks_path: Path,
    source_channel: list,
    seg_channel: list,
    z_range: tuple,
    fov_list: list,
):
    """Compute various cell features and combine them with PCA features.

    This function processes cell tracking data to compute various morphological and
    intensity-based features for both phase and fluorescence channels, and combines
    them with PCA features from an embedding dataset.

    Parameters
    ----------
    features_path : Path
        Path to the embedding dataset containing PCA features.
    data_path : Path
        Path to the raw data directory containing image data.
    tracks_path : Path
        Path to the directory containing tracking data in CSV format.
    source_channel : list
        List of source channels to process from the data.
    seg_channel : list
        List of segmentation channels to process from the data.
    z_range : tuple
        Tuple specifying the z-range to process (min_z, max_z).
    fov_list : list
        List of field of view names to process.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all computed features including:
        - Basic features (mean intensity, std dev, kurtosis, etc.) for both Phase and Fluor channels
        - Organelle features (area, masked intensity)
        - Nuclear features (area, perimeter, eccentricity)
        - PCA components (PCA1-PCA8)
        - Original tracking information (fov_name, track_id, time points)
    """

    embedding_dataset = compute_PCA(features_path)
    features_npy = embedding_dataset["features"].values

    # convert the xarray to dataframe structure and add columns for computed features
    embedding_df = embedding_dataset["sample"].to_dataframe().reset_index(drop=True)
    feature_columns = pd.DataFrame(
        features_npy, columns=[f"feature_{i + 1}" for i in range(768)]
    )

    embedding_df = pd.concat([embedding_df, feature_columns], axis=1)
    embedding_df = embedding_df.drop(columns=["sample", "UMAP1", "UMAP2"])

    # Filter features based on FOV names that start with any of the items in fov_list
    embedding_df = embedding_df[
        embedding_df["fov_name"].apply(
            lambda x: any(x.startswith(fov) for fov in fov_list)
        )
    ]

    # Define feature categories and their corresponding column names
    feature_columns = {
        "basic_features": [
            ("Mean Intensity", ["Phase", "Fluor"]),
            ("Std Dev", ["Phase", "Fluor"]),
            ("Kurtosis", ["Phase", "Fluor"]),
            ("Skewness", ["Phase", "Fluor"]),
            ("Entropy", ["Phase", "Fluor"]),
            ("Interquartile Range", ["Phase", "Fluor"]),
            ("Dissimilarity", ["Phase", "Fluor"]),
            ("Contrast", ["Phase", "Fluor"]),
            ("Texture", ["Phase", "Fluor"]),
            ("Weighted Intensity Gradient", ["Phase", "Fluor"]),
            ("Radial Intensity Gradient", ["Phase", "Fluor"]),
            ("Zernike Moment Std", ["Phase", "Fluor"]),
            ("Zernike Moment Mean", ["Phase", "Fluor"]),
            ("Intensity Localization", ["Phase", "Fluor"]),
        ],
        "organelle_features": [
            "Fluor Area",
            "Fluor Masked Intensity",
        ],
        "nuclear_features": [
            "Nuclear area",
            "Perimeter",
            "Perimeter area ratio",
            "Nucleus eccentricity",
        ],
    }

    # Initialize all feature columns
    for category, feature_list in feature_columns.items():
        if isinstance(feature_list[0], tuple):  # Handle features with multiple channels
            for feature, channels in feature_list:
                for channel in channels:
                    col_name = f"{channel} {feature}"
                    embedding_df[col_name] = np.nan
        else:  # Handle single features
            for feature in feature_list:
                embedding_df[feature] = np.nan

    # compute the computed features and add them to the dataset

    fov_names_list = embedding_df["fov_name"].unique()
    unique_fov_names = sorted(list(set(fov_names_list)))

    for fov_name in unique_fov_names:
        unique_track_ids = embedding_df[embedding_df["fov_name"] == fov_name][
            "track_id"
        ].unique()
        unique_track_ids = list(set(unique_track_ids))

        # iteration_count = 0

        for track_id in unique_track_ids:
            if not embedding_df[
                (embedding_df["fov_name"] == fov_name)
                & (embedding_df["track_id"] == track_id)
            ].empty:
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
                    z_range=(0, 1),
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

                for i, t in enumerate(
                    embedding_df[
                        (embedding_df["fov_name"] == fov_name)
                        & (embedding_df["track_id"] == track_id)
                    ]["t"]
                ):
                    # Basic statistical features for both channels
                    phase_features = CellFeatures(phase[i], nucl_mask[i])
                    PF = phase_features.compute_all_features()

                    # Get all basic statistical measures at once
                    phase_stats = {
                        "Mean Intensity": PF["mean_intensity"],
                        "Std Dev": PF["std_dev"],
                        "Kurtosis": PF["kurtosis"],
                        "Skewness": PF["skewness"],
                        "Interquartile Range": PF["iqr"],
                        "Entropy": PF["spectral_entropy"],
                        "Dissimilarity": PF["dissimilarity"],
                        "Contrast": PF["contrast"],
                        "Texture": PF["texture"],
                        "Zernike Moment Std": PF["zernike_std"],
                        "Zernike Moment Mean": PF["zernike_mean"],
                        "Radial Intensity Gradient": PF["radial_intensity_gradient"],
                        "Weighted Intensity Gradient": PF[
                            "weighted_intensity_gradient"
                        ],
                        "Intensity Localization": PF["intensity_localization"],
                    }

                    fluor_cell_features = CellFeatures(fluor[i], nucl_mask[i])

                    FF = fluor_cell_features.compute_all_features()

                    fluor_stats = {
                        "Mean Intensity": FF["mean_intensity"],
                        "Std Dev": FF["std_dev"],
                        "Kurtosis": FF["kurtosis"],
                        "Skewness": FF["skewness"],
                        "Interquartile Range": FF["iqr"],
                        "Entropy": FF["spectral_entropy"],
                        "Contrast": FF["contrast"],
                        "Dissimilarity": FF["dissimilarity"],
                        "Texture": FF["texture"],
                        "Masked Area": FF["masked_area"],
                        "Masked Intensity": FF["masked_intensity"],
                        "Weighted Intensity Gradient": FF[
                            "weighted_intensity_gradient"
                        ],
                        "Radial Intensity Gradient": FF["radial_intensity_gradient"],
                        "Zernike Moment Std": FF["zernike_std"],
                        "Zernike Moment Mean": FF["zernike_mean"],
                        "Intensity Localization": FF["intensity_localization"],
                        "Area": FF["area"],
                    }

                    mask_features = CellFeatures(nucl_mask[i], nucl_mask[i])
                    MF = mask_features.compute_all_features()

                    mask_stats = {
                        "perimeter": MF["perimeter"],
                        "area": MF["area"],
                        "eccentricity": MF["eccentricity"],
                        "perimeter_area_ratio": MF["perimeter_area_ratio"],
                    }

                    # Create dictionaries for each feature category
                    phase_feature_mapping = {
                        f"Phase {k.replace('_', ' ').title()}": v
                        for k, v in phase_stats.items()
                    }

                    fluor_feature_mapping = {
                        f"Fluor {k.replace('_', ' ').title()}": v
                        for k, v in fluor_stats.items()
                    }

                    mask_feature_mapping = {
                        "Nuclear area": mask_stats["area"],
                        "Perimeter": mask_stats["perimeter"],
                        "Perimeter area ratio": mask_stats["perimeter_area_ratio"],
                        "Nucleus eccentricity": mask_stats["eccentricity"],
                    }

                    # Combine all feature dictionaries
                    feature_values = {
                        **phase_feature_mapping,
                        **fluor_feature_mapping,
                        **mask_feature_mapping,
                    }

                    # update the features dataframe
                    for feature_name, value in feature_values.items():
                        embedding_df.loc[
                            (embedding_df["fov_name"] == fov_name)
                            & (embedding_df["track_id"] == track_id)
                            & (embedding_df["t"] == t),
                            feature_name,
                        ] = value[0]

                # iteration_count += 1
                print(f"Processed {fov_name}+{track_id}")

    return embedding_df


## save all feature dataframe to png file
def compute_correlation_and_save_png(features: pd.DataFrame, filename: str):
    """Compute correlation between PCA features and computed features, and save as heatmap.

    This function calculates the Spearman correlation between PCA components and all
    computed features, then visualizes the results as a heatmap. The heatmap focuses
    on the correlation between PCA components (PCA1-PCA8) and all other computed features.

    Parameters
    ----------
    features : pandas.DataFrame
        DataFrame containing all features including:
        - PCA components (PCA1-PCA8)
        - Computed features (morphological, intensity-based, etc.)
        - Tracking metadata (fov_name, track_id, t, etc.)
    filename : str
        Path where the correlation heatmap will be saved as a PNG or SVG file.

    Returns
    -------
    pandas.DataFrame
        The correlation matrix between all features.
    """
    # remove the rows with missing values
    features = features.dropna()

    # sub_features = features[features["Time"] == 20]
    feature_df_removed = features.drop(
        columns=["fov_name", "track_id", "t", "id", "parent_track_id", "parent_id"]
    )

    # Compute correlation between PCA features and computed features
    correlation = feature_df_removed.corr(method="spearman")

    # display PCA correlation as a heatmap

    plt.figure(figsize=(30, 10))
    sns.heatmap(
        correlation.drop(
            columns=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8"]
        ).loc["PCA1":"PCA8", :],
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 18},
        cbar=False,
    )
    plt.title("Correlation between PCA features and computed features", fontsize=12)
    plt.xlabel("Computed Features", fontsize=18)
    plt.ylabel("PCA Features", fontsize=18)
    plt.xticks(fontsize=18, rotation=45, ha="right")  # Rotate labels and align them
    plt.yticks(fontsize=18)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.5,  # Add padding around the figure
    )
    plt.close()

    return correlation
