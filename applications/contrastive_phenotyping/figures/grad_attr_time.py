# %%
import logging
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from cmap import Colormap
from lightning.pytorch import seed_everything
from skimage.exposure import rescale_intensity
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.engine import ContrastiveEncoder, ContrastiveModule
from viscy.representation.evaluation import load_annotation
from viscy.representation.evaluation.lca import (
    AssembledClassifier,
    fit_logistic_regression,
    linear_from_binary_logistic_regression,
)
from viscy.transforms import NormalizeSampled, ScaleIntensityRangePercentilesd

seed_everything(42, workers=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# %%
# Dataset for display and occlusion analysis
data_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
tracks_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
annotation_occlusion_infection_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv"
annotation_occlusion_division_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/9-lineage-cell-division/lineages_gt/cell_division_state_test_set.csv"
fov = "/B/4/8"
track = [44, 46]

# %%
dm = TripletDataModule(
    data_path=data_path,
    tracks_path=tracks_path,
    source_channel=["Phase3D", "RFP"],
    z_range=[25, 40],
    batch_size=48,
    num_workers=0,
    initial_yx_patch_size=(128, 128),
    final_yx_patch_size=(128, 128),
    normalizations=[
        NormalizeSampled(
            keys=["Phase3D"], level="fov_statistics", subtrahend="mean", divisor="std"
        ),
        ScaleIntensityRangePercentilesd(
            keys=["RFP"], lower=50, upper=99, b_min=0.0, b_max=1.0
        ),
    ],
    predict_cells=True,
    include_fov_names=[fov] * len(track),
    include_track_ids=track,
)
dm.setup("predict")
len(dm.predict_dataset)

# %%
# load model
model = ContrastiveModule.load_from_checkpoint(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/epoch=178-step=16826.ckpt",
    encoder=ContrastiveEncoder(
        backbone="convnext_tiny",
        in_channels=2,
        in_stack_depth=15,
        stem_kernel_size=(5, 4, 4),
        stem_stride=(5, 4, 4),
        embedding_dim=768,
        projection_dim=32,
    ),
).eval()

# %%
# TODO add the patsh to the combination of sec61 and tomm20
# train linear classifier
# INFECTION
##  Embedding and Annotations

path_infection_embedding_1 = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)

path_annotations_infection_1 = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv"
)
# TOMM20
path_infection_embedding_2 = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/sensor_160patch_99ckpt_max.zarr"
)
path_annotations_infection_2 = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_infection_annotation.csv"
)

# SEC61
path_infection_embedding_3 = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/prediction_infection/2chan_192patch_100ckpt_timeAware_ntxent_rerun.zarr"
)

path_annotations_infection_3 = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/0-annotation/track_infection_annotation.csv"
)

# CELL DIVISION
path_annotations_division_1 = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/9-lineage-cell-division/lineages_gt/cell_division_state_test_set.csv"
)
path_division_embedding_1 = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178_gt_tracks.zarr"
)
# TOMM20
path_annotations_division_2 = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_cell_state_annotation.csv"
)
# SEC61
path_annotations_division_3 = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/0-annotation/track_cell_state_annotation.csv"
)
# %%
#########
# Make tuple of tuples of embedding and annotations

# Train FOVs - use a broader set since we have multiple datasets

infection_classifier_pairs = (
    (
        path_infection_embedding_1,
        path_annotations_infection_1,
        ["/A/3/7", "/A/3/8", "/A/3/9", "/B/4/6", "/B/4/7"],
    ),
    (path_infection_embedding_2, path_annotations_infection_2, "all"),
    (path_infection_embedding_3, path_annotations_infection_3, "all"),
)

# NOTE: embedding 1 and annotations 1 are not used. They are not wll annotated for division
division_classifier_pairs = (
    # (
    #     path_division_embedding_1,
    #     path_annotations_division_1,
    #     ["/A/3/7", "/A/3/8", "/A/3/9", "/B/4/6", "/B/4/7"],
    # ),
    # (path_infection_embedding_2, path_annotations_division_2, "all"),
    (path_infection_embedding_3, path_annotations_division_3, "all"),
)


def load_and_combine_datasets(
    datasets,
    target_type="infection",
    standardization_mapping=None,
):
    """Load and combine multiple embedding datasets with their annotations.

    Parameters
    ----------
    datasets : list of tuple
        List of (embedding_path, annotation_path, train_fovs) tuples containing
        paths to embedding files, annotation CSV files, and training FOVs.
    target_type : str, default='infection'
        Type of classification target. Either 'infection' or 'division' - determines
        which column to look for in the annotation files.
    standardization_mapping : dict, optional
        Dictionary to standardize different annotation formats across datasets.
        Maps original values to standardized values.
        Example: {'infected': 2, 'uninfected': 1, 'background': 0,
                  2.0: 2, 1.0: 1, 0.0: 0, 'mitosis': 2, 'interphase': 1, 'unknown': 0}

    Returns
    -------
    combined_features : xarray.DataArray
        Combined feature embeddings from all successfully loaded datasets.
    combined_annotations : pandas.Series
        Combined and standardized annotations from all datasets.

    Raises
    ------
    ValueError
        If no datasets were successfully loaded.
    """

    all_features = []
    all_annotations = []

    # Default standardization mappings
    if standardization_mapping is None:
        if target_type == "infection":
            standardization_mapping = {
                # String formats
                "infected": 2,
                "uninfected": 1,
                "background": 0,
                "unknown": 0,
                # Numeric formats
                2.0: 2,
                1.0: 1,
                0.0: 0,
                2: 2,
                1: 1,
                0: 0,
            }
        elif target_type == "division":
            standardization_mapping = {
                # String formats
                "mitosis": 2,
                "interphase": 1,
                "unknown": 0,
                # Numeric formats
                2.0: 2,
                1.0: 1,
                0.0: 0,
                2: 2,
                1: 1,
                0: 0,
            }

    for emb_path, ann_path, train_fovs in datasets:
        try:
            logger.debug(f"Loading dataset: {emb_path}")
            dataset = read_embedding_dataset(emb_path)

            # Read annotation CSV to detect column names
            logger.debug(f"Reading annotation CSV: {ann_path}")
            ann_df = pd.read_csv(ann_path)
            # make sure the ann_fov_names start with '/' otherwise add it, and strip whitespace
            ann_df["fov_name"] = ann_df["fov_name"].apply(
                lambda x: (
                    "/" + x.strip() if not x.strip().startswith("/") else x.strip()
                )
            )

            if train_fovs == "all":
                train_fovs = np.unique(dataset["fov_name"])

            # Auto-detect annotation column based on target_type
            annotation_key = None
            if target_type == "infection":
                for col in [
                    "infection_state",
                    "infection",
                    "infection_status",
                ]:
                    if col in ann_df.columns:
                        annotation_key = col
                        break

            elif target_type == "division":
                for col in ["division", "cell_division", "cell_state"]:
                    if col in ann_df.columns:
                        annotation_key = col
                        break

            if annotation_key is None:
                print(f"  No {target_type} column found, skipping...")
                continue

            # Filter the dataset to only include the FOVs in the annotation
            # Use xarray's native filtering methods
            ann_fov_names = set(ann_df["fov_name"].unique())
            train_fovs = set(train_fovs)

            logger.debug(f"Dataset FOVs: {dataset['fov_name'].values}")
            logger.debug(f"Annotation FOV names: {ann_fov_names}")
            logger.debug(f"Train FOVs: {train_fovs}")
            logger.debug(f"Dataset samples before filtering: {len(dataset.sample)}")

            # Filter and get only the intersection of train_fovs and ann_fov_names
            common_fovs = train_fovs.intersection(ann_fov_names)
            # missed out fovs in the dataset
            missed_fovs = train_fovs - common_fovs
            # missed out fovs in the annotations
            missed_fovs_ann = ann_fov_names - common_fovs

            if len(common_fovs) == 0:
                raise ValueError(
                    f"No common FOVs found between dataset and annotations: {train_fovs} not in {ann_fov_names}"
                )
            elif len(missed_fovs) > 0:
                warnings.warn(
                    f"No matching found for FOVs in the train dataset: {missed_fovs}"
                )
            elif len(missed_fovs_ann) > 0:
                warnings.warn(
                    f"No matching found for FOVs in the annotations: {missed_fovs_ann}"
                )

            logger.debug(f"Intersection of train_fovs and ann_fov_names: {common_fovs}")

            # Filter the dataset to only include the intersection of train_fovs and ann_fov_names
            dataset = dataset.where(
                dataset["fov_name"].isin(list(common_fovs)), drop=True
            )

            logger.debug(f"Dataset samples after filtering: {len(dataset.sample)}")

            # Load annotations without class mapping first
            annotations = load_annotation(dataset, ann_path, annotation_key)

            # Check unique values before standardization
            unique_vals = annotations.unique()
            logger.debug(f"Original unique values: {unique_vals}")

            # Apply standardization mapping
            standardized_annotations = annotations.copy()
            if standardization_mapping:
                for original_val, standard_val in standardization_mapping.items():
                    mask = annotations == original_val
                    if mask.any():
                        standardized_annotations[mask] = standard_val
                        logger.debug(
                            f"Mapped {original_val} -> {standard_val} ({mask.sum()} instances)"
                        )

            # Check standardized values
            std_unique_vals = standardized_annotations.unique()
            logger.debug(f"Standardized unique values: {std_unique_vals}")

            # Convert to categorical for consistency
            standardized_annotations = standardized_annotations.astype("category")

            # Keep features as xarray DataArray for compatibility with fit_logistic_regression
            all_features.append(dataset["features"])
            all_annotations.append(standardized_annotations)

            logger.debug(f"Features shape: {dataset['features'].shape}")
            logger.debug(f"Annotations shape: {standardized_annotations.shape}")
        except Exception as e:
            raise ValueError(f"Error loading dataset {emb_path}: {e}")

    # Combine all datasets
    if all_features:
        # Extract features and coordinates from each dataset
        all_features_arrays = []
        all_coords = []

        for dataset in all_features:
            # Extract the features array
            features_array = dataset["features"].values
            all_features_arrays.append(features_array)

            # Extract coordinates
            coords_dict = {}
            for coord_name in dataset.coords:
                if coord_name != "sample":  # skip sample coordinate
                    coords_dict[coord_name] = dataset.coords[coord_name].values
            all_coords.append(coords_dict)

        # Combine feature arrays
        combined_features_array = np.concatenate(all_features_arrays, axis=0)

        # Combine coordinates (excluding 'features' from coordinates)
        combined_coords = {}
        for coord_name in all_coords[0].keys():
            if coord_name != "features":  # Don't include 'features' in coordinates
                coord_values = []
                for coords_dict in all_coords:
                    coord_values.extend(coords_dict[coord_name])
                combined_coords[coord_name] = coord_values

        # Create new combined dataset in the correct format
        coords_dict = {
            "sample": range(len(combined_features_array)),
        }

        # Add each coordinate as a 1D coordinate along the sample dimension
        for coord_name, coord_values in combined_coords.items():
            coords_dict[coord_name] = ("sample", coord_values)

        combined_dataset = xr.Dataset(
            {
                "features": (("sample", "features"), combined_features_array),
            },
            coords=coords_dict,
        )

        # Set the index properly like the original datasets
        if "fov_name" in combined_coords:
            available_coords = [
                coord
                for coord in combined_coords.keys()
                if coord in ["fov_name", "track_id", "t"]
            ]
            combined_dataset = combined_dataset.set_index(sample=available_coords)

        combined_annotations = pd.concat(all_annotations, ignore_index=True)

        logger.debug(f"Combined features shape: {combined_dataset['features'].shape}")
        logger.debug(f"Combined annotations shape: {combined_annotations.shape}")

        # Final check of combined annotations
        final_unique = combined_annotations.unique()
        logger.debug(f"Final combined unique values: {final_unique}")

    return combined_dataset["features"], combined_annotations


# %%

# Load and combine infection datasets
logger.info("Loading infection classification datasets...")
infection_features, infection_labels = load_and_combine_datasets(
    infection_classifier_pairs,
    target_type="infection",
)
# %%
# Load and combine division datasets
logger.info("Loading division classification datasets...")
division_features, division_labels = load_and_combine_datasets(
    division_classifier_pairs,
    target_type="division",
)


# %%

logistic_regression_infection, _ = fit_logistic_regression(
    features=infection_features.copy(),
    annotations=infection_labels.copy(),
    train_ratio=0.8,
    remove_background_class=True,
    scale_features=True,
    class_weight="balanced",
    solver="liblinear",
    random_state=42,
)
# %%

logistic_regression_division, _ = fit_logistic_regression(
    division_features.copy(),
    division_labels.copy(),
    train_ratio=0.8,
    remove_background_class=True,
    scale_features=True,
    class_weight="balanced",
    solver="liblinear",
    random_state=42,
)

# %%
linear_classifier_infection = linear_from_binary_logistic_regression(
    logistic_regression_infection
)
assembled_classifier_infection = (
    AssembledClassifier(model.model, linear_classifier_infection)
    .eval()
    .to(model.device)
)

# %%
linear_classifier_division = linear_from_binary_logistic_regression(
    logistic_regression_division
)
assembled_classifier_division = (
    AssembledClassifier(model.model, linear_classifier_division).eval().to(model.device)
)


# %%
# Loading the lineage images
img = []
for sample in dm.predict_dataloader():
    img.append(sample["anchor"].numpy())
img = np.concatenate(img, axis=0)
print(img.shape)

# %%
img_tensor = torch.from_numpy(img).to(model.device)

with torch.inference_mode():
    infection_probs = assembled_classifier_infection(img_tensor).sigmoid()
    division_probs = assembled_classifier_division(img_tensor).sigmoid()

# %%
attr_kwargs = dict(
    img=img_tensor,
    sliding_window_shapes=(1, 15, 12, 12),
    strides=(1, 15, 4, 4),
    show_progress=True,
)


infection_attribution = (
    assembled_classifier_infection.attribute_occlusion(**attr_kwargs).cpu().numpy()
)
division_attribution = (
    assembled_classifier_division.attribute_occlusion(**attr_kwargs).cpu().numpy()
)


# %%
def clip_rescale(img, low, high):
    return rescale_intensity(img.clip(low, high), out_range=(0, 1))


def clim_percentile(heatmap, low=1, high=99):
    lo, hi = np.percentile(heatmap, (low, high))
    return clip_rescale(heatmap, lo, hi)


g_lim = 1
z_slice = 5
phase = clim_percentile(img[:, 0, z_slice])
rfp = clim_percentile(img[:, 1, z_slice])
img_render = np.concatenate([phase, rfp], axis=2)
phase_heatmap_inf = infection_attribution[:, 0, z_slice]
rfp_heatmap_inf = infection_attribution[:, 1, z_slice]
inf_render = clip_rescale(
    np.concatenate([phase_heatmap_inf, rfp_heatmap_inf], axis=2), -g_lim, g_lim
)
phase_heatmap_div = division_attribution[:, 0, z_slice]
rfp_heatmap_div = division_attribution[:, 1, z_slice]
div_render = clip_rescale(
    np.concatenate([phase_heatmap_div, rfp_heatmap_div], axis=2), -g_lim, g_lim
)


# %%
# Filter the dataframe to only include the fovs and track_id of the current fov
infection = pd.read_csv(annotation_occlusion_infection_path)
infection = infection[infection["fov_name"] == fov[1:]]
infection = infection[infection["track_id"].isin(track)]
track_classes_infection = infection["infection_state"]

# load division annotations
division = pd.read_csv(annotation_occlusion_division_path)
division = division[division["fov_name"] == fov[1:]]
division = division[division["track_id"].isin(track)]

division["division"] = 1  # default: not dividing
division.loc[division["t"].between(16, 22, inclusive="both"), "division"] = (
    2  # dividing for t in 16-20
)

track_classes_division = division["division"]


# %%
plt.style.use("./figure.mplstyle")

all_time_points = list(range(len(img_render)))
selected_time_points = all_time_points[
    :: max(1, len(all_time_points) // 8)
]  # Show up to 8 time points


sps = len(selected_time_points)

icefire = Colormap("icefire").to_mpl()

f, ax = plt.subplots(3, sps, figsize=(2 * sps, 3), layout="compressed")
for i, time in enumerate(selected_time_points):
    hpi = 3 + 0.5 * time
    prob = infection_probs[time].item()
    inf_binary = str(bool(track_classes_infection.iloc[time] - 1)).lower()
    div_binary = str(bool(track_classes_division.iloc[time] - 1)).lower()
    ax[0, i].imshow(img_render[time], cmap="gray")
    ax[0, i].set_title(f"{hpi} HPI")
    ax[1, i].imshow(inf_render[time], cmap=icefire, vmin=0, vmax=1)
    ax[1, i].set_title(
        f"infected: {prob:.3f}\n" f"label: {inf_binary}",
    )
    ax[2, i].imshow(div_render[time], cmap=icefire, vmin=0, vmax=1)
    ax[2, i].set_title(
        f"dividing: {division_probs[time].item():.3f}\n" f"label: {div_binary}",
    )
for a in ax.ravel():
    a.axis("off")
norm = mpl.colors.Normalize(vmin=-g_lim, vmax=g_lim)
cbar = f.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=icefire),
    orientation="vertical",
    ax=ax[1:].ravel().tolist(),
    format=mpl.ticker.StrMethodFormatter("{x:.1f}"),
)
cbar.set_label("occlusion attribution")

# %%
# f.savefig(
#     Path.home()
#     / "mydata"
#     / "gdrive/publications/dynaCLR/2025_dynaCLR_paper/fig_manuscript_svg/figure_occlusion_analysis/figure_parts/fig_explanation_patch12_stride4.pdf",
#     dpi=300,
# )

# %%
# Create video animation of occlusion analysis
icefire = Colormap("icefire").to_mpl()
plt.style.use("./figure.mplstyle")

fig, ax = plt.subplots(3, 1, figsize=(6, 8), layout="compressed")

# Initialize plots
im1 = ax[0].imshow(img_render[0], cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

im2 = ax[1].imshow(inf_render[0], cmap=icefire, vmin=0, vmax=1)
ax[1].set_title("Infection Occlusion Attribution")
ax[1].axis("off")

im3 = ax[2].imshow(div_render[0], cmap=icefire, vmin=0, vmax=1)
ax[2].set_title("Division Occlusion Attribution")
ax[2].axis("off")

# Store initial border colors
for a in ax:
    for spine in a.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")

# Add colorbar
norm = mpl.colors.Normalize(vmin=-g_lim, vmax=g_lim)
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=icefire),
    ax=ax[1:],
    orientation="horizontal",
    shrink=0.8,
    pad=0.1,
)
cbar.set_label("Occlusion Attribution")


# Animation function
def animate(frame):
    time = frame
    hpi = 3 + 0.5 * time

    # Update images
    im1.set_array(img_render[time])
    im2.set_array(inf_render[time])
    im3.set_array(div_render[time])

    # Update titles with probabilities
    inf_prob = infection_probs[time].item()
    div_prob = division_probs[time].item()
    inf_binary = bool(track_classes_infection.iloc[time] - 1)
    div_binary = bool(track_classes_division.iloc[time] - 1)

    # Color code labels - red for true, green for false
    inf_color = "darkorange" if inf_binary else "blue"
    div_color = "darkorange" if div_binary else "blue"

    # Make label text bold when true
    inf_weight = "bold" if inf_binary else "normal"
    div_weight = "bold" if div_binary else "normal"

    # Update border colors to highlight true labels
    for spine in ax[1].spines.values():
        spine.set_color(inf_color)
        spine.set_linewidth(4 if inf_binary else 2)

    for spine in ax[2].spines.values():
        spine.set_color(div_color)
        spine.set_linewidth(4 if div_binary else 2)

    ax[0].set_title(f"Original Image - {hpi:.1f} HPI", fontsize=12, fontweight="bold")
    ax[1].set_title(
        f"Infection Attribution - Prob: {inf_prob:.3f} (Label: {str(inf_binary).lower()})",
        fontsize=12,
        fontweight=inf_weight,
        color=inf_color,
    )
    ax[2].set_title(
        f"Division Attribution - Prob: {div_prob:.3f} (Label: {str(div_binary).lower()})",
        fontsize=12,
        fontweight=div_weight,
        color=div_color,
    )

    return [im1, im2, im3]


# Create animation
anim = animation.FuncAnimation(
    fig, animate, frames=len(img_render), interval=200, blit=True, repeat=True
)

# %%
# Save as video
video_path = (
    Path.home()
    / "mydata"
    / "gdrive/2025_dynaCLR_paper/fig_manuscript_svg/figure_occlusion_analysis/figure_parts/occlusion_analysis_video.mp4"
)
video_path.parent.mkdir(parents=True, exist_ok=True)

# Save as MP4
Writer = animation.writers["ffmpeg"]
writer = Writer(fps=5, metadata=dict(artist="VisCy"), bitrate=1800)
anim.save(str(video_path), writer=writer)

print(f"Video saved to: {video_path}")


# %%
# Performance metrics over time
def calculate_metrics_over_time(y_true, y_pred_probs, threshold=0.5):
    """Calculate accuracy, F1, and AUC for each time point"""
    y_pred = (y_pred_probs > threshold).astype(int)

    metrics = {"accuracy": [], "f1": [], "auc": []}

    for i in range(len(y_true)):
        # Get predictions up to current time point
        true_up_to_i = y_true[: i + 1]
        pred_up_to_i = y_pred[: i + 1]
        prob_up_to_i = y_pred_probs[: i + 1]

        # Skip if we don't have both classes
        if len(np.unique(true_up_to_i)) < 2:
            metrics["accuracy"].append(np.nan)
            metrics["f1"].append(np.nan)
            metrics["auc"].append(np.nan)
            continue

        # Calculate metrics
        acc = accuracy_score(true_up_to_i, pred_up_to_i)
        f1 = f1_score(true_up_to_i, pred_up_to_i, average="binary")
        try:
            auc_score = roc_auc_score(true_up_to_i, prob_up_to_i)
        except:
            auc_score = np.nan

        metrics["accuracy"].append(acc)
        metrics["f1"].append(f1)
        metrics["auc"].append(auc_score)

    return metrics


# Ensure we have matching lengths - use the minimum length
min_length = min(
    len(track_classes_infection), len(track_classes_division), len(infection_probs)
)

# Convert labels to binary for metrics calculation - truncate to min_length
inf_true = (track_classes_infection.values[:min_length] - 1).astype(bool).astype(int)
div_true = track_classes_division.values[:min_length].astype(bool).astype(int)

inf_probs = infection_probs[:min_length].cpu().numpy()
div_probs = division_probs[:min_length].cpu().numpy()

print(f"Using {min_length} time points for metrics calculation")
print(f"Infection labels shape: {inf_true.shape}")
print(f"Division labels shape: {div_true.shape}")
print(f"Infection probs shape: {inf_probs.shape}")
print(f"Division probs shape: {div_probs.shape}")

# Calculate metrics
inf_metrics = calculate_metrics_over_time(inf_true, inf_probs)
div_metrics = calculate_metrics_over_time(div_true, div_probs)

# Time points
time_points = np.arange(len(inf_true))
hpi_values = 3 + 0.5 * time_points

# Create metrics plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="compressed")

# Infection metrics
axes[0, 0].plot(
    hpi_values, inf_metrics["accuracy"], "b-", linewidth=2, label="Accuracy"
)
axes[0, 0].set_title("Infection Classification Accuracy Over Time")
axes[0, 0].set_xlabel("Hours Post Infection (HPI)")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, 1)

axes[0, 1].plot(hpi_values, inf_metrics["f1"], "g-", linewidth=2, label="F1 Score")
axes[0, 1].set_title("Infection Classification F1 Score Over Time")
axes[0, 1].set_xlabel("Hours Post Infection (HPI)")
axes[0, 1].set_ylabel("F1 Score")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 1)

axes[0, 2].plot(hpi_values, inf_metrics["auc"], "r-", linewidth=2, label="AUC")
axes[0, 2].set_title("Infection Classification AUC Over Time")
axes[0, 2].set_xlabel("Hours Post Infection (HPI)")
axes[0, 2].set_ylabel("AUC")
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_ylim(0, 1)

# Division metrics
axes[1, 0].plot(
    hpi_values, div_metrics["accuracy"], "b-", linewidth=2, label="Accuracy"
)
axes[1, 0].set_title("Division Classification Accuracy Over Time")
axes[1, 0].set_xlabel("Hours Post Infection (HPI)")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0, 1)

axes[1, 1].plot(hpi_values, div_metrics["f1"], "g-", linewidth=2, label="F1 Score")
axes[1, 1].set_title("Division Classification F1 Score Over Time")
axes[1, 1].set_xlabel("Hours Post Infection (HPI)")
axes[1, 1].set_ylabel("F1 Score")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 1)

axes[1, 2].plot(hpi_values, div_metrics["auc"], "r-", linewidth=2, label="AUC")
axes[1, 2].set_title("Division Classification AUC Over Time")
axes[1, 2].set_xlabel("Hours Post Infection (HPI)")
axes[1, 2].set_ylabel("AUC")
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_ylim(0, 1)

plt.tight_layout()

# %%
# Save metrics plot
metrics_path = (
    Path.home()
    / "mydata"
    / "gdrive/2025_dynaCLR_paper/fig_manuscript_svg/figure_occlusion_analysis/figure_parts/performance_metrics_over_time.pdf"
)
fig.savefig(str(metrics_path), dpi=300, bbox_inches="tight")
print(f"Metrics plot saved to: {metrics_path}")
