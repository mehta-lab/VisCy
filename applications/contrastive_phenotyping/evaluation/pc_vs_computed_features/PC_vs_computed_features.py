"""Script to compute the correlation between PCA and UMAP features and computed features
* finds the computed features best representing the PCA and UMAP components
* outputs a heatmap of the correlation between PCA and UMAP features and computed features
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from compute_pca_features import compute_correlation_and_save_png, compute_features

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
fov_list = ["/A/3", "/B/3", "/B/4"]

features_sensor = compute_features(
    features_path,
    data_path,
    tracks_path,
    source_channel,
    seg_channel,
    z_range,
    fov_list,
)

features_sensor.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_allset_sensor.csv",
    index=False,
)

# features_sensor = pd.read_csv("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_allset_sensor.csv")

# take a subset without the 768 features
feature_columns = [f"feature_{i + 1}" for i in range(768)]
features_subset_sensor = features_sensor.drop(columns=feature_columns)
correlation_sensor = compute_correlation_and_save_png(
    features_subset_sensor,
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_sensor_allset.svg",
)

# %% plot PCA vs set of computed features for sensor features

set_features = [
    "Fluor Radial Intensity Gradient",
    "Phase Interquartile Range",
    "Perimeter area ratio",
    "Fluor Interquartile Range",
    "Phase Entropy",
    "Fluor Zernike Moment Mean",
]

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_sensor.loc[set_features, "PCA1":"PCA6"],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={"size": 24},
    vmin=-1,
    vmax=1,
)
plt.xlabel("Computed Features", fontsize=24)
plt.ylabel("PCA Features", fontsize=24)
plt.xticks(fontsize=24)  # Increase x-axis tick labels
plt.yticks(fontsize=24)  # Increase y-axis tick labels

plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_allset_sensor_6features.svg"
)

# plot the PCA1 vs PCA2 map for sensor features

plt.figure(figsize=(10, 10))
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    data=features_sensor,
)


#   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.
#  / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \
# '-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'-'   '-'


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

features_organelle = compute_features(
    features_path,
    data_path,
    tracks_path,
    source_channel,
    seg_channel,
    z_range,
    fov_list,
)

# Save the features dataframe to a CSV file
features_organelle.to_csv(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan_organelle_multiwell.csv",
    index=False,
)

correlation_organelle = compute_correlation_and_save_png(
    features_organelle,
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_organelle_multiwell.svg",
)

# features_organelle = pd.read_csv("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan_organelle_multiwell_refinedPCA.csv")

# %% plot PCA vs set of computed features for organelle features

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_organelle.loc[set_features, "PCA1":"PCA6"],
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={"size": 24},
    vmin=-1,
    vmax=1,
)
plt.xlabel("Computed Features", fontsize=24)
plt.ylabel("PCA Features", fontsize=24)
plt.xticks(fontsize=24)  # Increase x-axis tick labels
plt.yticks(fontsize=24)  # Increase y-axis tick labels
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/PC_vs_CF_2chan_pca_setfeatures_organelle_6features.svg"
)

# %%
