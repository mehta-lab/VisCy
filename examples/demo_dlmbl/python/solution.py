# %% [markdown]
"""
# Image translation
---

Written by Ziwen Liu and Shalin Mehta, CZ Biohub San Francisco.


In this exercise, we will solve an image translation task to predict fluorescence images of nuclei and membrane markers from quantitative phase images of cells. In other words, we will _virtually stain_ the nuclei and membrane visible in the phase image. 

Here, the source domain is label-free microscopy (material density) and the target domain is fluorescence microscopy (fluorophore density). The goal is to learn a mapping from the source domain to the target domain. We will use a deep convolutional neural network (CNN), specifically, a U-Net model with residual connections to learn the mapping. The preprocessing, training, prediction, evaluation, and deployment steps are unified in a computer vision pipeline for single-cell analysis that we call [VisCy](https://github.com/mehta-lab/VisCy).

VisCy evolved from our previous work on virtual staining of cellular components from their density and anisotropy.
![](https://iiif.elifesciences.org/lax/55502%2Felife-55502-fig1-v2.tif/full/1500,/0/default.jpg)

[Guo et al. (2020) Revealing architectural order with quantitative label-free imaging and deep learning
. eLife](https://elifesciences.org/articles/55502).

VisCy exploits recent advances in the data and metadata formats ([OME-zarr](https://www.nature.com/articles/s41592-021-01326-w)) and DL frameworks, [PyTorch Lightning](https://lightning.ai/) and [MONAI](https://monai.io/). Our previous pipeline, [microDL](https://github.com/mehta-lab/microDL), is deprecated and is now a public archive.

"""

# %% [markdown]
"""
Today, we will train a 2D image translation model using a 2D U-Net with residual connections. We will use a dataset of 301 fields of view (FOVs) of Human Embryonic Kidney (HEK) cells, each FOV has 3 channels (phase, membrane, and nuclei). The cells were labeled with CRISPR editing. Intrestingly, not all cells during this experiment were labeled due to the stochastic nature of CRISPR editing. In such situations, virtual staining rescues missing labels.
![HEK](https://github.com/mehta-lab/VisCy/blob/dlmbl2023/docs/figures/phase_to_nuclei_membrane.svg?raw=true)

The exercise is organized in 3 parts.

* [Part 1](#1_phase2fluor) - Explore the data and model using tensorboard. Launch the training before lunch.
* Lunch break - The model will continue training during lunch.
* [Part 2](#2_fluor2phase) - Evaluate the training with tensorboard. Train a model to predict phase from fluorescence.
* [Part 3, bonus](#3_tuning) - Tune the capacity of networks and other hyperparameters to improve the performance.

<div class="alert alert-info">
Each part should take ~1.5 hours, depending on your practice with python.
During parts 2 and 3, please summarize the models you train and their performance in [this google doc](https://docs.google.com/document/d/1hZWSVRvt9KJEdYu7ib-vFBqAVQRYL8cWaP_vFznu7D8/edit#heading=h.n5u485pmzv2z).
At checkpoints 2 and 3, we will discuss your results!
</div>

Before you start,

<div class="alert alert-danger">
Set your python kernel to <span style="color:black;">04-image-translation</span>
</div>
"""
# %% [markdown] <a id='1_phase2fluor'></a>
"""
# Part 1: Visualize data using tensorboard, start training a model.

Learning goals:

- Load the and visualize the images from OME-Zarr
- Configure and understand the data loader
- Log the data to tensorboard.
- Initialize a 2D U-Net model for virtual staining
- Start training the model to predict nuclei and membrane from phase.

"""

# %% Imports and paths

import matplotlib.pyplot as plt
import torch
from iohub import open_ome_zarr
from torchview import draw_graph
import os
from pathlib import Path
import numpy as np


from viscy.light.data import HCSDataModule
from viscy.light.engine import VSTrainer, VSUNet
from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard
from tensorboard import notebook  # for viewing tensorboard in notebook

# Paths to data and log directory
data_path = Path(
    "~/data/04_image_translation/HEK_nuclei_membrane_pyramid.zarr/"
).expanduser()
log_dir = Path("~/data/04_image_translation/logs/").expanduser()

# %% [markdown]
"""
## Load Dataset.

<div class="alert alert-info">
Task 1.1
Use <a href=https://czbiohub-sf.github.io/iohub/main/api/ngff.html#open-ome-zarr>
<code>iohub.open_ome_zarr</code></a> to read the dataset.
</div>

There should be 301 FOVs in the dataset (12 GB compressed).

Each FOV consists of 3 channels of 2048x2048 images,
saved in the <a href="https://ngff.openmicroscopy.org/latest/#hcs-layout">
High-Content Screening (HCS) layout</a>
specified by the Open Microscopy Environment Next Generation File Format
(OME-NGFF).

"""

# %%
# set dataset path here


dataset = open_ome_zarr(data_path)

print(len(list(dataset.positions())))

# %% [markdown]
"""
View images with matplotlib.

The layout on the disk is: row/col/field/pyramid_level/timepoint/channel/z/y/x.


Note that labelling is not perfect,
as some cells are not expressing the fluorophore.
"""

# %%

# Change the parameters below to visualize data.
row = "0"
col = "0"
field = "42"
# This dataset contains images at 3 resolutions.
# '0' is the highest resolution
# '1' is down-scaled 2x2,
# '2' is down-scaled 4x4.
# Such datasets are called image pyramids.
pyaramid_level = "2"

# `channel_names` is the metadata that is stored with data accoring to the OME-NGFF spec.
n_channels = len(dataset.channel_names)

image = dataset[f"{row}/{col}/{field}/{pyaramid_level}"].numpy()
print(f"data shape: {image.shape}, FOV: {field}, pyramid level: {pyaramid_level}")

figure, axes = plt.subplots(1, n_channels, figsize=(9, 3))

for i in range(n_channels):
    for i in range(n_channels):
        channel_image = image[0, i, 0]
        # Adjust contrast to 0.5th and 99.5th percentile of pixel values.
        p1, p99 = np.percentile(channel_image, (0.5, 99.5))
        channel_image = np.clip(channel_image, p1, p99)
        axes[i].imshow(channel_image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(dataset.channel_names[i])
plt.tight_layout()

# %% [markdown]
"""
## Initialize the data loaders.
"""

# %%

BATCH_SIZE = 32

data_module = HCSDataModule(
    data_path,
    source_channel="Phase",
    target_channel=["Nuclei", "Membrane"],
    z_window_size=1,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    architecture="2D",
    yx_patch_size=(256, 256),
)

data_module.setup("fit")

print(len(data_module.train_dataset), len(data_module.val_dataset))

# %% [markdown]
"""
## Initialize tensorboard.

<div class="alert alert-info">
Task 1.2
Log the batches drawn with data loader to tensorboard.
"""

# %% [markdown]
"""
Launch TensorBoard :

```
%load_ext tensorboard
%tensorboard --logdir model_log_dir
```
"""

# %%
notebook.list()

# %%
notebook.display(port=6006, height=800)


# %% [markdown]
"""
<div class="alert alert-info">
Task 1.2

Understand the data loader.
</div>
"""

# %%
train_dataloader = data_module.train_dataloader()

for i, batch in enumerate(train_dataloader):
    # The batch is a dictionary consisting of three keys: 'index', 'source', 'target'.
    # index is the tuple consisting of (image name, time, and z-slice)
    # source is the tensor of size 1x1x256x256
    # target is the tensor of size 2x1x256x256
    ...
    # plot one image from each of the batch and break
    break

# %% Load the data with tensorboard

train_dataloader = data_module.train_dataloader()

# create a SummaryWriter object to write to the tensorboard log
writer = SummaryWriter()

for i, batch in enumerate(train_dataloader):
    # The batch is a dictionary consisting of three keys: 'index', 'source', 'target'.
    # index is the tuple consisting of (image name, time, and z-slice)
    # source is the tensor of size 1x1x256x256
    # target is the tensor of size 2x1x256x256

    if i >= 5:
        break
    FOV = batch["index"][0][0]
    input_tensor = batch["source"][0, 0, :, :].squeeze()
    target_nuclei_tensor = batch["target"][0, 0, :, :].squeeze()
    target_membrane_tensor = batch["target"][0, 1, :, :].squeeze()

    # add the images to the tensorboard log
    writer.add_images(f"input/{FOV}", input_tensor.unsqueeze(0), global_step=i)
    writer.add_images(
        f"target_nuclei/{FOV}", target_nuclei_tensor.unsqueeze(0), global_step=i
    )
    writer.add_images(
        f"target_membrane/{FOV}", target_membrane_tensor.unsqueeze(0), global_step=i
    )

# close the SummaryWriter object
writer.close()

# %% tags=["solution"]
train_dataloader = data_module.train_dataloader()


fig, axs = plt.subplots(3, 8, figsize=(20, 8))

# Draw 8 batches, each with 32 images. Show the first image in each batch.

for i, batch in enumerate(train_dataloader):
    # The batch is a dictionary consisting of three keys: 'index', 'source', 'target'.
    # index is the tuple consisting of (image name, time, and z-slice)
    # source is the tensor of size 1x1x256x256
    # target is the tensor of size 2x1x256x256

    if i >= 8:
        break
    FOV = batch["index"][0][0]
    input_tensor = batch["source"][0, 0, :, :].squeeze()
    target_nuclei_tensor = batch["target"][0, 0, :, :].squeeze()
    target_membrane_tensor = batch["target"][0, 1, :, :].squeeze()

    axs[0, i].imshow(input_tensor, cmap="gray")
    axs[1, i].imshow(target_nuclei_tensor, cmap="gray")
    axs[2, i].imshow(target_membrane_tensor, cmap="gray")
    axs[0, i].set_title(f"input@{FOV}")
    axs[1, i].set_title("target-nuclei")
    axs[2, i].set_title("target-membrane")
    axs[0, i].axis("off")
    axs[1, i].axis("off")
    axs[2, i].axis("off")

plt.tight_layout()
plt.show()


# %% [markdown]
"""
Construct a 2D U-Net for image translation.

See ``viscy.unet.networks.Unet2D.Unet2d`` for configuration details.
Increase the ``depth`` in ``draw_graph`` to zoom in.
"""

# %%


model_config = {
    "architecture": "2D",
    "in_channels": 1,
    "out_channels": 2,
    "residual": True,
    "dropout": 0.1,
    "task": "reg",
}

model = VSUNet(
    model_config=model_config.copy(),
    batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.mse_loss,
    schedule="WarmupCosine",
    log_num_samples=10,
)

# visualize graph
model_graph = draw_graph(model, model.example_input_array, depth=2, device="cpu")
graph = model_graph.visual_graph
graph

# %% [markdown]
"""
Configure trainer class.
Here we use the ``fast_dev_run`` flag to run a sanity check first.
"""

# %%
GPU_ID = 0
trainer = VSTrainer(accelerator="gpu", devices=[GPU_ID], fast_dev_run=True)

trainer.fit(model, datamodule=data_module)

# %% [markdown]
"""
<div class="alert alert-info">
Task 1.3

Modify the trainer to train the model for 20 epochs.
</div>
"""

# %% [markdown]
"""
Tips:

- See ``VSTrainer?`` for all the available parameters.
- Set ``default_root_dir`` to store the logs and checkpoints
in a specific directory.
"""

# %% [markdown]
"""
Bonus:

- Tweak model hyperparameters
- Adjust batch size to fully utilize the VRAM
"""

# %% tags=["solution"]
wider_config = model_config | {"num_filters": [24, 48, 96, 192, 384]}

model = model = VSUNet(
    model_config=wider_config.copy(),
    batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.mse_loss,
    schedule="WarmupCosine",
    log_num_samples=10,
)


trainer = VSTrainer(
    accelerator="gpu",
    max_epochs=20,
    log_every_n_steps=8,
    default_root_dir=model_log_dir,
)

trainer.fit(model, datamodule=data_module)


# %% [markdown]
"""
<div class="alert alert-success">
Checkpoint 1

Now the training has started,
we can come back after a while and evaluate the performance!
</div>
"""

# %% [markdown] <a id='1_fluor2phase'></a>
"""
# Part 2: Visualize the previous model and training with tensorboard. Train fluorescence to phase contrast translation model.
--------------------------------------------------

Learning goals:
- Visualize the previous model and training with tensorboard
- Train fluorescence to phase contrast translation model
- Compare the performance of the two models.

"""

# %% [markdown]
"""
<div class="alert alert-success">
Checkpoint 2
Please summarize hyperparameters and performance of your models in [this google doc](https://docs.google.com/document/d/1hZWSVRvt9KJEdYu7ib-vFBqAVQRYL8cWaP_vFznu7D8/edit#heading=h.n5u485pmzv2z)

Now that you have trained two models, let's think about the following questions:
- What is the information content of each channel in the dataset?
- How would you use image translation models?
- What can you try to improve the performance of each model?


</div>
"""

# %% [markdown] <a id='3_tuning'></a>
"""
# Part 3: Tune the capacity of networks and other hyperparameters to improve the performance.
--------------------------------------------------

Learning goals:
- Visualize the previous model and training with tensorboard
- Train fluorescence to phase contrast translation model
- Compare the performance of the two models.

"""

# %% [markdown]
"""
<div class="alert alert-success">
Checkpoint 3

Congratulations! You have trained several image translation models now!
Please summarize hyperparameters and performance of your models in [this google doc](https://docs.google.com/document/d/1hZWSVRvt9KJEdYu7ib-vFBqAVQRYL8cWaP_vFznu7D8/edit#heading=h.n5u485pmzv2z)
</div>
"""
