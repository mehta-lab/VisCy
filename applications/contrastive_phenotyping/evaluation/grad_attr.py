# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cmap import Colormap
from lightning.pytorch import seed_everything
from skimage.exposure import rescale_intensity

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

# %%
seed_everything(42, workers=True)

fov = "/B/4/8"
track = 44

# %%
dm = TripletDataModule(
    data_path="/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr",
    tracks_path="/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr",
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
    include_fov_names=[fov],
    include_track_ids=[track],
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
# train linear classifier
path_infection_embedding = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)
path_division_embedding = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178_gt_tracks.zarr"
)
path_annotations_infection = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv"
)
path_annotations_division = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/9-lineage-cell-division/lineages_gt/cell_division_state_test_set.csv"
)

infection_dataset = read_embedding_dataset(path_infection_embedding)
infection_features = infection_dataset["features"]
infection = load_annotation(
    infection_dataset,
    path_annotations_infection,
    "infection_state",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)

division_dataset = read_embedding_dataset(path_division_embedding)
division_features = division_dataset["features"]
division = load_annotation(division_dataset, path_annotations_division, "division")
# move the unknown class to the 0 label
division[division == 1] = -2
division += 2
division /= 2
division = division.astype("category")

# %%
train_fovs = ["/A/3/7", "/A/3/8", "/A/3/9", "/B/4/6", "/B/4/7"]

# %%
logistic_regression_infection, _ = fit_logistic_regression(
    infection_features.copy(),
    infection.copy(),
    train_fovs,
    remove_background_class=True,
    scale_features=False,
    class_weight="balanced",
    solver="liblinear",
)
# %%
logistic_regression_division, _ = fit_logistic_regression(
    division_features.copy(),
    division.copy(),
    train_fovs,
    remove_background_class=True,
    scale_features=False,
    class_weight="balanced",
    solver="liblinear",
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
# load infection annotations
infection = pd.read_csv(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv",
)
track_classes_infection = infection[infection["fov_name"] == fov[1:]]
track_classes_infection = track_classes_infection[
    track_classes_infection["track_id"] == track
]["infection_state"]

# %%
# load division annotations
division = pd.read_csv(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/9-lineage-cell-division/lineages_gt/cell_division_state_test_set.csv",
)
track_classes_division = division[division["fov_name"] == fov[1:]]
track_classes_division = track_classes_division[
    track_classes_division["track_id"] == track
]["division"]


# %%
for sample in dm.predict_dataloader():
    img = sample["anchor"].numpy()

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
plt.style.use("./figure.mplstyle")

selected_time_points = [3, 6, 15, 16]
selected_div_states = [False] * 3 + [True]

sps = len(selected_time_points)

icefire = Colormap("icefire").to_mpl()

f, ax = plt.subplots(3, sps, figsize=(5.5, 3), layout="compressed")
for i, time in enumerate(selected_time_points):
    hpi = 3 + 0.5 * time
    prob = infection_probs[time].item()
    inf_binary = str(bool(track_classes_infection.iloc[time] - 1)).lower()
    div_binary = str(selected_div_states[i]).lower()
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
f.savefig(
    Path.home()
    / "gdrive/publications/learning_impacts_of_infection/fig_manuscript/fig_explanation/fig_explanation_patch12_stride4.pdf",
    dpi=300,
)

# %%
