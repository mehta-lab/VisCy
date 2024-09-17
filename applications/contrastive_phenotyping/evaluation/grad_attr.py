# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from cmap import Colormap
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from skimage.exposure import rescale_intensity

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.engine import ContrastiveEncoder, ContrastiveModule
from viscy.representation.evaluation import load_annotation
from viscy.representation.lca import LinearClassifier, LinearProbingDataModule
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
ds = dm.predict_dataset
len(ds)

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
linear_data = LinearProbingDataModule(
    embeddings=torch.from_numpy(features.values).float(),
    labels=torch.from_numpy(infection.cat.codes.values).long(),
    split_ratio=(0.4, 0.2, 0.4),
    batch_size=2**14,
)
linear_data.setup("fit")

linear_classifier = LinearClassifier(
    in_features=features.shape[1], out_features=3, lr=0.001
)

log_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/test"
)

trainer = Trainer(
    max_epochs=60,
    logger=CSVLogger(log_path),
    log_every_n_steps=1,
)

trainer.fit(linear_classifier, linear_data)
losses = pd.read_csv(
    log_path / "lightning_logs" / "version_0" / "metrics.csv", index_col="epoch"
)
losses = pd.merge(
    losses["loss/train"].dropna(), losses["loss/val"].dropna(), on="epoch"
)
losses.plot()

# %%
linear_classifier = linear_classifier.eval()


# %%
class AssembledClassifier(torch.nn.Module):
    def __init__(self, model, classifier):
        super().__init__()
        self.model = model
        self.classifier = classifier

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.encoder(x)
        x = self.classifier(x)
        return x


assembled_classifier = AssembledClassifier(model.model, linear_classifier).eval().cpu()

# %%
# load infection annotations
infection = pd.read_csv(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv",
)
track_classes = infection[infection["fov_name"] == fov[1:]]
track_classes = track_classes[track_classes["track_id"] == track]["infection_state"]
track_classes

# %%
ig = IntegratedGradients(assembled_classifier, multiply_by_inputs=True)

# %%
sample_idx = 7

sample = dm.predict_dataset[sample_idx]
img = sample["anchor"].numpy()[None]
target = int(track_classes.values[sample_idx])
sample["index"], target

# %%
assembled_classifier.zero_grad()
attribution = ig.attribute(torch.from_numpy(img), target=target).numpy()

# %%
with torch.inference_mode():
    probs = assembled_classifier(torch.from_numpy(img)).softmax(dim=1)


# %%
def clim(heatmap):
    lo, hi = np.percentile(heatmap, (1, 99))
    return rescale_intensity(heatmap.clip(lo, hi), out_range=(0, 1))


# %%
phase = Colormap("gray")(clim(img[0, 0]))
rfp = Colormap("gray")(clim(img[0, 1]))
phase_heatmap = Colormap("icefire")(clim(attribution[0, 0]))
rfp_heatmap = Colormap("icefire")(clim(attribution[0, 1]))
grid = np.concatenate(
    [
        np.concatenate([phase, phase_heatmap], axis=1),
        np.concatenate([rfp, rfp_heatmap], axis=1),
    ],
    axis=2,
)

f, ax = plt.subplots(3, 5, figsize=(10, 6))
for i, (z_slice, a) in enumerate(zip(grid, ax.flatten())):
    a.imshow(z_slice)
    a.set_title(f"z={i}")
    a.axis("off")
f.suptitle(f"t={sample_idx}, prediction={probs.numpy().tolist()}")
f.tight_layout()

# %%
