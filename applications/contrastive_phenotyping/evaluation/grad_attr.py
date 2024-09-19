# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from cmap import Colormap
from lightning.pytorch import seed_everything
from skimage.exposure import rescale_intensity

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.engine import ContrastiveEncoder, ContrastiveModule
from viscy.representation.evaluation import load_annotation
from viscy.representation.lca import (
    AssembledClassifier,
    fit_logistic_regression,
    linear_from_binary_logistic_regression,
)
from viscy.transforms import NormalizeSampled, ScaleIntensityRangePercentilesd

# %%
seed_everything(42, workers=True)

fov = "/B/4/6"
track = 4

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
path_embedding = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)
path_annotations_infection = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv"
)

dataset = read_embedding_dataset(path_embedding)
features = dataset["features"]
infection = load_annotation(
    dataset,
    path_annotations_infection,
    "infection_state",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)

# %%
train_fovs = ["/A/3/7", "/A/3/8", "/A/3/9", "/B/4/7", "/B/4/8"]

# %%
logistic_regression, data_split = fit_logistic_regression(
    features.copy(),
    infection.copy(),
    train_fovs,
    remove_background_class=True,
    scale_features=False,
    class_weight="balanced",
    solver="liblinear",
)

# %%
linear_classifier = linear_from_binary_logistic_regression(logistic_regression)
assembled_classifier = AssembledClassifier(model.model, linear_classifier).eval().cpu()

# %%
# load infection annotations
infection = pd.read_csv(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv",
)
track_classes = infection[infection["fov_name"] == fov[1:]]
track_classes = track_classes[track_classes["track_id"] == track]["infection_state"]


# %%
def attribute_sample(img, assembled_classifier):
    ig = IntegratedGradients(assembled_classifier, multiply_by_inputs=True)
    assembled_classifier.zero_grad()
    attribution = ig.attribute(torch.from_numpy(img)).numpy()
    return img, attribution


def color_and_clim(heatmap, cmap, low=1, high=99):
    lo, hi = np.percentile(heatmap, (low, high))
    rescaled = rescale_intensity(heatmap.clip(lo, hi), out_range=(0, 1))
    return Colormap(cmap)(rescaled)


# %%
for sample in dm.predict_dataloader():
    img = sample["anchor"].numpy()

# %%
with torch.inference_mode():
    probs = assembled_classifier(torch.from_numpy(img)).sigmoid()
img, attribution = attribute_sample(img, assembled_classifier)

# %%
z_slice = 5
phase = color_and_clim(img[:, 0, z_slice], cmap="gray")
rfp = color_and_clim(img[:, 1, z_slice], cmap="gray")
phase_heatmap = color_and_clim(attribution[:, 0, z_slice], cmap="icefire")
rfp_heatmap = color_and_clim(attribution[:, 1, z_slice], cmap="icefire")
grid = np.concatenate(
    [
        np.concatenate([phase, phase_heatmap], axis=1),
        np.concatenate([rfp, rfp_heatmap], axis=1),
    ],
    axis=2,
)
print(grid.shape)

# %%
selected_time_points = [0, 4, 8, 34]
class_text = {0: "none", 1: "uninfected", 2: "infected"}

sps = len(selected_time_points)
f, ax = plt.subplots(1, sps, figsize=(4 * sps, 4))
for time, a in zip(selected_time_points, ax.flatten()):
    rendered = grid[time]
    prob = probs[time].item()
    a.imshow(rendered)
    hpi = 3 + 0.5 * time
    text_label = class_text[track_classes.iloc[time]]
    a.set_title(
        f"{hpi} HPI,\npredicted infection probability: {prob:.2f},\nannotation: {text_label}"
    )
    a.axis("off")
f.tight_layout()

# %%
