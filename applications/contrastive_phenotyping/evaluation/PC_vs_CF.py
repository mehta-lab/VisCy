""" Script to compute the correlation between PCA and UMAP features and computed features
* finds the computed features best representing the PCA and UMAP components
* outputs a heatmap of the correlation between PCA and UMAP features and computed features
"""

# %%
import sys
from pathlib import Path

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks
from viscy.representation.evaluation.feature import CellFeatures

# %% function to read the embedding dataset and return the features
def compute_PCA(features_path: Path):
    """
    Read the embedding dataset and return the features and 8 PCA components
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
        .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8"], append=True)
    )

    return features

def compute_features(features_path: Path, data_path: Path, tracks_path: Path, source_channel: list, seg_channel: list, z_range: tuple, fov_list: list):

    embedding_dataset = compute_PCA(features_path)
    features_npy = embedding_dataset["features"].values

    # convert the xarray to dataframe structure and add columns for computed features
    embedding_df = embedding_dataset["sample"].to_dataframe().reset_index(drop=True)
    feature_columns = pd.DataFrame(features_npy, columns=[f"feature_{i+1}" for i in range(768)])
    
    embedding_df = pd.concat([embedding_df, feature_columns], axis=1)
    embedding_df = embedding_df.drop(columns=["sample", "UMAP1", "UMAP2"])

    # Filter features based on FOV names that start with any of the items in fov_list
    embedding_df = embedding_df[embedding_df["fov_name"].apply(lambda x: any(x.startswith(fov) for fov in fov_list))]

    # Define feature categories and their corresponding column names
    feature_columns = {
        'basic_features': [
            ('Mean Intensity', ['Phase', 'Fluor']),
            ('Std Dev', ['Phase', 'Fluor']),
            ('Kurtosis', ['Phase', 'Fluor']),
            ('Skewness', ['Phase', 'Fluor']),
            ('Entropy', ['Phase', 'Fluor']),
            ('Interquartile Range', ['Phase', 'Fluor']),
            ('Dissimilarity', ['Phase', 'Fluor']),
            ('Contrast', ['Phase', 'Fluor']),
            ('Texture', ['Phase', 'Fluor']),
            ('Weighted Intensity Gradient', ['Phase', 'Fluor']),
            ('Radial Intensity Gradient', ['Phase', 'Fluor']),
            ('Zernike Moment Std', ['Phase', 'Fluor']),
            ('Zernike Moment Mean', ['Phase', 'Fluor']),
            ('Intensity Localization', ['Phase','Fluor']),
        ],
        'organelle_features': [
            'Fluor Area',
            'Fluor Masked Intensity',
        ],
        'nuclear_features': [
            'Nuclear Area',
            'Perimeter',
            'Perimeter area ratio',
            'Nucleus eccentricity',
        ],
        'dynamic_features': [
            'Instantaneous velocity',
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
    # max_iterations = 50

    for fov_name in unique_fov_names:
        csv_files = list((Path(str(tracks_path) + str(fov_name))).glob("*.csv"))
        tracks_df = pd.read_csv(str(csv_files[0]))

        unique_track_ids = embedding_df[embedding_df["fov_name"] == fov_name]["track_id"].unique()
        unique_track_ids = list(set(unique_track_ids))

        # iteration_count = 0

        for track_id in unique_track_ids:
            if not embedding_df[(embedding_df["fov_name"] == fov_name) & (embedding_df["track_id"] == track_id)].empty:
                
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

                # find the minimum time point
                t_min_track = np.min(track_subdf["t"])
                for i, t in enumerate(embedding_df[(embedding_df["fov_name"] == fov_name) & (embedding_df["track_id"] == track_id)]["t"]):

                    # Basic statistical features for both channels
                    phase_features = CellFeatures(phase[i], nucl_mask[i])
                    PF = phase_features.compute_all_features()

                    # Get all basic statistical measures at once
                    phase_stats = {
                        'Mean Intensity': PF['mean_intensity'],
                        'Std Dev': PF['std_dev'],
                        'Kurtosis': PF['kurtosis'],
                        'Skewness': PF['skewness'],
                        'Interquartile Range': PF['iqr'],
                        'Entropy': PF['spectral_entropy'],
                        'Dissimilarity': PF['dissimilarity'],
                        'Contrast': PF['contrast'],
                        'Texture': PF['texture'],
                        'Zernike Moment Std': PF['zernike_std'],
                        'Zernike Moment Mean': PF['zernike_mean'],
                        'Radial Intensity Gradient': PF['radial_intensity_gradient'],
                        'Weighted Intensity Gradient': PF['weighted_intensity_gradient'],
                        'Intensity Localization': PF['intensity_localization'],
                    }

                    fluor_cell_features = CellFeatures(fluor[i], nucl_mask[i])
        
                    FF = fluor_cell_features.compute_all_features()

                    fluor_stats = {
                        'Mean Intensity': FF['mean_intensity'],
                        'Std Dev': FF['std_dev'],
                        'Kurtosis': FF['kurtosis'],
                        'Skewness': FF['skewness'],
                        'Interquartile Range': FF['iqr'],
                        'Entropy': FF['spectral_entropy'],
                        'Contrast': FF['contrast'],
                        'Dissimilarity': FF['dissimilarity'],
                        'Texture': FF['texture'],
                        'Masked Area': FF['masked_area'],
                        'Masked Intensity': FF['masked_intensity'],
                        'Weighted Intensity Gradient': FF['weighted_intensity_gradient'],
                        'Radial Intensity Gradient': FF['radial_intensity_gradient'],
                        'Zernike Moment Std': FF['zernike_std'],
                        'Zernike Moment Mean': FF['zernike_mean'],
                        'Intensity Localization': FF['intensity_localization'],
                        'Masked Intensity': FF['masked_intensity'],
                        'Area': FF['area'],
                    }

                    mask_features = CellFeatures(nucl_mask[i], nucl_mask[i])
                    MF = mask_features.compute_all_features()

                    mask_stats = {
                        'perimeter': MF['perimeter'],
                        'area': MF['area'],
                        'eccentricity': MF['eccentricity'],
                        'perimeter_area_ratio': MF['perimeter_area_ratio'],
                    }

                    # dynamic_features = DynamicFeatures(tracks_df)
                    # DF = dynamic_features.compute_all_features()
                    # dynamic_stats = {
                    #     'instantaneous_velocity': DF['instantaneous_velocity'],
                    # }

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
                        "Nuclear area": mask_stats['area'],
                        "Perimeter": mask_stats['perimeter'],
                        "Perimeter area ratio": mask_stats['perimeter_area_ratio'],
                        "Nucleus eccentricity": mask_stats['eccentricity']
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
                            (embedding_df["fov_name"] == fov_name) & 
                            (embedding_df["track_id"] == track_id) & 
                            (embedding_df["t"] == t),
                            feature_name
                        ] = value[0]

                # iteration_count += 1
                print(f"Processed {fov_name}+{track_id}")

    return embedding_df

# %% save all feature dataframe to png file
def compute_correlation_and_save_png(features: pd.DataFrame, filename: str):
    # remove the rows with missing values
    features = features.dropna()

    # sub_features = features[features["Time"] == 20]
    feature_df_removed = features.drop(
        columns=["fov_name", "track_id", "t", "id", "parent_track_id", "parent_id", "UMAP1", "UMAP2"]
    )

    # Compute correlation between PCA features and computed features
    correlation = feature_df_removed.corr(method="spearman")

    # display PCA correlation as a heatmap

    plt.figure(figsize=(30, 10))
    sns.heatmap(
        correlation.drop(columns=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8"]).loc[
            "PCA1":"PCA8", :
        ],
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={'size': 18},
        cbar=False
    )
    plt.title("Correlation between PCA features and computed features", fontsize=12)
    plt.xlabel("Computed Features", fontsize=18)
    plt.ylabel("PCA Features", fontsize=18)
    plt.xticks(fontsize=18, rotation=45, ha='right')  # Rotate labels and align them
    plt.yticks(fontsize=18)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.5  # Add padding around the figure
    )
    plt.close()

    return correlation

# %% for organelle features

features_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/Soorya/timeAware_2chan_ntxent_192patch_91ckpt_rev7_GT.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)

source_channel = ["Phase3D", "raw GFP EX488 EM525-45"]
seg_channel = ["nuclei_prediction_labels_labels"]
z_range = (16, 21)
normalizations = None
fov_list = ["/B/2/000000", "/B/3/000000", "/C/2/000000"]
# fov_name = "/B/2/000000"
# track_id = 24

features_organelle = compute_features(features_path, data_path, tracks_path, source_channel, seg_channel, z_range, fov_list)

# Save the features dataframe to a CSV file
features_organelle.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan_organelle_multiwell.csv",
    index=False,
)

correlation_organelle = compute_correlation_and_save_png(features_organelle, "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_organelle_multiwell.png")

# %% for sensor features

features_path = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_94ckpt_rev6_2.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
)

source_channel = ["Phase3D", "RFP"]
seg_channel = ["Nuclei_prediction_labels"]
z_range = (28, 43)
fov_list = ["/A/3","/B/3","/B/4"]
# fov_name = "/B/4/5"
# track_id = 11

features_sensor = compute_features(features_path, data_path, tracks_path, source_channel, seg_channel, z_range, fov_list)

features_sensor.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_allset_sensor.csv",
    index=False,
)
correlation_sensor = compute_correlation_and_save_png(features_sensor, "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_sensor.png")

# %% plot PCA vs set of computed features for sensor features

set_features = [
    "Fluor Radial Intensity Gradient",
    "Fluor Kurtosis",
    "Phase Entropy",
    "Phase Std Dev",
    "Perimeter area ratio",
    "Phase Interquartile Range",
    "Phase Skewness",
    "Fluor Interquartile Range",
    "Nuclear area",
    "Fluor Area",
    "Perimeter",
    "Fluor Texture",
]

plt.figure(figsize=(8, 10))
sns.heatmap(
    correlation_sensor.loc[set_features, "PCA1":"PCA6"],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={'size': 18},
    vmin=-1,
    vmax=1,
)
plt.xlabel("Computed Features", fontsize=18)
plt.ylabel("PCA Features", fontsize=18)
plt.xticks(fontsize=18)  # Increase x-axis tick labels
plt.yticks(fontsize=18)  # Increase y-axis tick labels

plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_setfeatures.svg"
)

# plot the PCA1 vs PCA2 map for sensor features

plt.figure(figsize=(10, 10))
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    data=features_sensor,
)

# %% plot PCA vs set of computed features for organelle features

set_features = [
    "Fluor Radial Intensity Gradient",
    "Phase Interquartile Range",
    "Perimeter area ratio",  
    "Fluor Zernike Moment Mean",
    "Fluor Mean Intensity",
    "Phase Entropy",
    "Fluor Interquartile Range",
    "Fluor Masked Area",
    "Fluor Skewness",
    "Phase Dissimilarity",
]

plt.figure(figsize=(10, 10))
sns.heatmap(
    correlation_organelle.loc[set_features, "PCA1":"PCA8"],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={'size': 18},
    vmin=-1,
    vmax=1,
)
plt.xlabel("Computed Features", fontsize=18)
plt.ylabel("PCA Features", fontsize=18)
plt.xticks(fontsize=18)  # Increase x-axis tick labels
plt.yticks(fontsize=18)  # Increase y-axis tick labels
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_setfeatures_organelle_refinedPCA.svg"
)

# %%