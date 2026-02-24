# %% [markdown]
"""
# Infected cell segmentation

Interactive script to demonstrate PyTorch Lightning training
with a semantic segmentation task.
"""

# %%
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from skimage.color import label2rgb
from torchview import draw_graph

from viscy.data.hcs import HCSDataModule
from viscy.scripts.infection_phenotyping.classify_infection_2D import SemanticSegUNet2D
from viscy.transforms import NormalizeSampled, RandWeightedCropd

# use tf32 for matmul
torch.set_float32_matmul_precision("high")

# %% [markdown]
"""
## Dataset

In this dataset, we have images of A549 cells infected with Dengue virus
in two channels:

- The cells are engineered to express a fluorescent protein (viral sensor)
that translocates from the cytoplasm to the nucleus upon infection.
- Quantitative phase images are reconstructed from brightfield images.

## Task
The goal is to identify infected and uninfected cells from these images.
For the training target, cell nuclei were segmented from virtual staining,
and manually labelled as infected (1) or uninfected (2),
while background was labelled as 0.
We will train a U-Net to predict these labels from the images.
Is is a semantic segmentation task,
where assign a label (class) to each pixel in the image.
"""


# %%
# setup datamodule
data_module = HCSDataModule(
    data_path="/hpc/mydata/ziwen.liu/demo/Exp_2024_02_13_DENV_3infMarked_trainVal.zarr",
    source_channel=["Sensor", "Phase"],
    target_channel=["Inf_mask"],
    yx_patch_size=(128, 128),
    split_ratio=0.5,
    z_window_size=1,
    architecture="2D",
    num_workers=8,
    batch_size=128,
    normalizations=[
        NormalizeSampled(
            keys=["Sensor", "Phase"],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
    augmentations=[
        RandWeightedCropd(
            num_samples=8,
            spatial_size=[-1, 128, 128],
            keys=["Sensor", "Phase", "Inf_mask"],
            w_key="Inf_mask",
        )
    ],
)

data_module.prepare_data()
data_module.setup(stage="fit")

# %%
# sample from training data
num_samples = 8

for batch in data_module.train_dataloader():
    image = batch["source"][:num_samples].numpy()
    label = batch["target"][:num_samples].numpy().astype("uint8")
    break

# %%
# visualize the samples
fig, ax = plt.subplots(num_samples, 3, figsize=(3, 8))

for i in range(num_samples):
    ax[i, 0].imshow(image[i, 0, 0], cmap="gray")
    ax[i, 1].imshow(image[i, 1, 0], cmap="gray")
    ax[i, 2].imshow(label2rgb(label[i, 0, 0], bg_label=0))

for a in ax.ravel():
    a.axis("off")

fig.tight_layout()

# %%
model = SemanticSegUNet2D(
    in_channels=2,
    out_channels=3,
    loss_function=torch.nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.25, 0.7])),
)

# %%
model_graph = draw_graph(
    model=model,
    input_data=torch.rand(1, 2, 1, 128, 128),
    graph_name="2D UNet",
    roll=True,
    depth=2,
    device="cpu",
)

model_graph.visual_graph

# %%
trainer = Trainer(
    accelerator="gpu",
    precision=32,
    devices=1,
    num_nodes=1,
    fast_dev_run=True,
    max_epochs=100,
    logger=TensorBoardLogger(
        save_dir="/hpc/mydata/ziwen.liu/demo/logs",
        version="interactive_demo",
        log_graph=True,
    ),
    log_every_n_steps=10,
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="loss/validate", save_top_k=5, every_n_epochs=1, save_last=True
        ),
    ],
)


# %%
trainer.fit(model, data_module)

# %%
