# %% [markdown]
"""
# Image translation (Virtual Staining - Part 1

Written by Eduardo Hirata-Miyasaki, Ziwen Liu, and Shalin Mehta, CZ Biohub San Francisco.

## Overview

In this exercise, we will predict fluorescence images of
nuclei and plasma membrane markers from quantitative phase images of cells,
i.e., we will _virtually stain_ the nuclei and plasma membrane
visible in the phase image.
This is an example of an image translation task.
We will apply spatial and intensity augmentations to train robust models
and evaluate their performance using a regression approach.

[![HEK293T](https://raw.githubusercontent.com/mehta-lab/VisCy/main/docs/figures/svideo_1.png)](https://github.com/mehta-lab/VisCy/assets/67518483/d53a81eb-eb37-44f3-b522-8bd7bddc7755)
(Click on image to play video)
"""

# %% [markdown]
"""
### Goals

#### Part 1: Learn to use iohub (I/O library), VisCy dataloaders, and TensorBoard.

  - Use a OME-Zarr dataset of 34 FOVs of adenocarcinomic human alveolar basal epithelial cells (A549),
  each FOV has 3 channels (phase, nuclei, and cell membrane).
  The nuclei were stained with DAPI and the cell membrane with Cellmask.
  - Explore OME-Zarr using [iohub](https://czbiohub-sf.github.io/iohub/main/index.html)
  and the high-content-screen (HCS) format.
  - Use [MONAI](https://monai.io/) to implement data augmentations.

#### Part 2: Train and evaluate the model to translate phase into fluorescence.
  - Train a 2D UNeXt2 model to predict nuclei and membrane from phase images.
  - Compare the performance of the trained model and a pre-trained model.
  - Evaluate the model using pixel-level and instance-level metrics.


Checkout [VisCy](https://github.com/mehta-lab/VisCy/tree/main/examples/demos),
our deep learning pipeline for training and deploying computer vision models
for image-based phenotyping including the robust virtual staining of landmark organelles.
VisCy exploits recent advances in data and metadata formats
([OME-zarr](https://www.nature.com/articles/s41592-021-01326-w)) and DL frameworks,
[PyTorch Lightning](https://lightning.ai/) and [MONAI](https://monai.io/).

### References

- [Liu, Z. and Hirata-Miyasaki, E. et al. (2024) Robust Virtual Staining of Cellular Landmarks](https://www.biorxiv.org/content/10.1101/2024.05.31.596901v2.full.pdf)
- [Guo et al. (2020) Revealing architectural order with quantitative label-free imaging and deep learning. eLife](https://elifesciences.org/articles/55502)
"""


# %% [markdown]
"""
📖 As you work through parts 2, please share the layouts of your models (output of torchview)
and their performance with everyone via
[this Google Doc](https://docs.google.com/document/d/1Mq-yV8FTG02xE46Mii2vzPJVYSRNdeOXkeU-EKu-irE/edit?usp=sharing). 📖
"""
# %% [markdown]
"""
<div class="alert alert-warning">
The exercise is organized in 2 parts 

<ul>
<li><b>Part 1</b> - Learn to use iohub (I/O library), VisCy dataloaders, and tensorboard.</li>
<li><b>Part 2</b> - Train and evaluate the model to translate phase into fluorescence.</li>
</ul>

</div>
"""

# %% [markdown]
"""
<div class="alert alert-danger">
Set your python kernel to <span style="color:black;">06_image_translation</span>
</div>
"""
# %% [markdown]
"""
## Part 1: Log training data to tensorboard, start training a model.
---------
Learning goals:

- Load the OME-zarr dataset and examine the channels (A549).
- Configure and understand the data loader.
- Log some patches to tensorboard.
- Initialize a 2D UNeXt2 model for virtual staining of nuclei and membrane from phase.
- Start training the model to predict nuclei and membrane from phase.
"""

# %% Imports
import os
from glob import glob
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchview
import torchvision
from cellpose import models
from iohub import open_ome_zarr
from iohub.reader import print_info
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from natsort import natsorted
from numpy.typing import ArrayLike
from skimage import metrics  # for metrics.
# pytorch lightning wrapper for Tensorboard.
from skimage.color import label2rgb
from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard
from torchmetrics.functional import accuracy, dice, jaccard_index
from tqdm import tqdm
# HCSDataModule makes it easy to load data during training.
from viscy.data.hcs import HCSDataModule
from viscy.evaluation.evaluation_metrics import mean_average_precision
# Trainer class and UNet.
from viscy.light.engine import MixedLoss, VSUNet
from viscy.light.trainer import VSTrainer
# training augmentations
from viscy.transforms import (NormalizeSampled, RandAdjustContrastd,
                              RandAffined, RandGaussianNoised,
                              RandGaussianSmoothd, RandScaleIntensityd,
                              RandWeightedCropd)

# %%
# seed random number generators for reproducibility.
seed_everything(42, workers=True)

# Paths to data and log directory
top_dir = Path(
    f"/hpc/mydata/{os.environ['USER']}/data/"
)  # TODO: Change this to point to your data directory.

data_path = (
    top_dir / "06_image_translation/training/a549_hoechst_cellmask_train_val.zarr"
)
log_dir = top_dir / "06_image_translation/logs/"

if not data_path.exists():
    raise FileNotFoundError(
        f"Data not found at {data_path}. Please check the top_dir and data_path variables."
    )

# %%
# Create log directory if needed, and launch tensorboard
log_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
"""
The next cell starts tensorboard.

<div class="alert alert-warning">
If you launched jupyter lab from ssh terminal, add <code>--host &lt;your-server-name&gt;</code> to the tensorboard command below. <code>&lt;your-server-name&gt;</code> is the address of your compute node that ends in amazonaws.com.

</div>

<div class="alert alert-warning">
If you are using VSCode and a remote server, you will need to forward the port to view the tensorboard. <br>
Take note of the port number was assigned in the previous cell.(i.e <code> http://localhost:{port_number_assigned}</code>) <br>

Locate the your VSCode terminal and select the <code>Ports</code> tab <br>
<ul>
<li>Add a new port with the <code>port_number_assigned</code>
</ul>
Click on the link to view the tensorboard and it should open in your browser.
</div>
"""

# %% Imports and paths tags=[]


# Function to find an available port
def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Launch TensorBoard on the browser
def launch_tensorboard(log_dir):
    import subprocess

    port = find_free_port()
    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port}"
    process = subprocess.Popen(tensorboard_cmd, shell=True)
    print(
        f"TensorBoard started at http://localhost:{port}. \n"
        "If you are using VSCode remote session, forward the port using the PORTS tab next to TERMINAL."
    )
    return process


# Launch tensorboard and click on the link to view the logs.
tensorboard_process = launch_tensorboard(log_dir)

# %% [markdown]
"""
## Load OME-Zarr Dataset

There should be 34 FOVs in the dataset.

Each FOV consists of 3 channels of 2048x2048 images,
saved in the [High-Content Screening (HCS) layout](https://ngff.openmicroscopy.org/latest/#hcs-layout)
specified by the Open Microscopy Environment Next Generation File Format
(OME-NGFF).

- The layout on the disk is: `row/col/field/pyramid_level/timepoint/channel/z/y/x.`
- These datasets only have 1 level in the pyramid (highest resolution) which is '0'.
"""

# %% [markdown]
"""
<div class="alert alert-warning">
You can inspect the tree structure by using your terminal:
<code> iohub info -v "path-to-ome-zarr" </code>

<br>
More info on the CLI:
<code>iohub info --help </code> to see the help menu.
</div>
"""
# %%
# This is the python function called by `iohub info` CLI command
print_info(data_path, verbose=True)

# Open and inspect the dataset.
dataset = open_ome_zarr(data_path)

# %%
# Use the field and pyramid_level below to visualize data.
row = 0
col = 0
field = 9  # TODO: Change this to explore data.


# NOTE: this dataset only has one level
pyaramid_level = 0

# `channel_names` is the metadata that is stored with data according to the OME-NGFF spec.
n_channels = len(dataset.channel_names)

image = dataset[f"{row}/{col}/{field}/{pyaramid_level}"].numpy()
print(f"data shape: {image.shape}, FOV: {field}, pyramid level: {pyaramid_level}")

figure, axes = plt.subplots(1, n_channels, figsize=(9, 3))

for i in range(n_channels):
    for i in range(n_channels):
        channel_image = image[0, i, 0]
        # Adjust contrast to 0.5th and 99.5th percentile of pixel values.
        p_low, p_high = np.percentile(channel_image, (0.5, 99.5))
        channel_image = np.clip(channel_image, p_low, p_high)
        axes[i].imshow(channel_image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(dataset.channel_names[i])
plt.tight_layout()

# %% [markdown]
# <div class="alert alert-info">
#
# ### Task 1.1
#
# Look at a couple different fields of view by changing the value in the cell above.
# Check the cell density, the cell morphologies, and fluorescence signal.
# </div>

# %% [markdown]
"""
## Explore the effects of augmentation on batch.

VisCy builds on top of PyTorch Lightning. PyTorch Lightning is a thin wrapper around PyTorch that allows rapid experimentation. It provides a [DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) to handle loading and processing of data during training. VisCy provides a child class, `HCSDataModule` to make it intuitve to access data stored in the HCS layout.
  
The dataloader in `HCSDataModule` returns a batch of samples. A `batch` is a list of dictionaries. The length of the list is equal to the batch size. Each dictionary consists of following key-value pairs.
- `source`: the input image, a tensor of size 1*1*Y*X
- `target`: the target image, a tensor of size 2*1*Y*X
- `index` : the tuple of (location of field in HCS layout, time, and z-slice) of the sample.
"""

# %% [markdown]
# <div class="alert alert-info">
#
# ### Task 1.2
#
# Setup the data loader and log several batches to tensorboard.
#
# Based on the tensorboard images, what are the two channels in the target image?
#
# Note: If tensorboard is not showing images, try refreshing and using the "Images" tab.
# </div>

# %%
# Define a function to write a batch to tensorboard log.


def log_batch_tensorboard(batch, batchno, writer, card_name):
    """
    Logs a batch of images to TensorBoard.

    Args:
        batch (dict): A dictionary containing the batch of images to be logged.
        writer (SummaryWriter): A TensorBoard SummaryWriter object.
        card_name (str): The name of the card to be displayed in TensorBoard.

    Returns:
        None
    """
    batch_phase = batch["source"][:, :, 0, :, :]  # batch_size x z_size x Y x X tensor.
    batch_membrane = batch["target"][:, 1, 0, :, :].unsqueeze(
        1
    )  # batch_size x 1 x Y x X tensor.
    batch_nuclei = batch["target"][:, 0, 0, :, :].unsqueeze(
        1
    )  # batch_size x 1 x Y x X tensor.

    p1, p99 = np.percentile(batch_membrane, (0.1, 99.9))
    batch_membrane = np.clip((batch_membrane - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_nuclei, (0.1, 99.9))
    batch_nuclei = np.clip((batch_nuclei - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_phase, (0.1, 99.9))
    batch_phase = np.clip((batch_phase - p1) / (p99 - p1), 0, 1)

    [N, C, H, W] = batch_phase.shape
    interleaved_images = torch.zeros((3 * N, C, H, W), dtype=batch_phase.dtype)
    interleaved_images[0::3, :] = batch_phase
    interleaved_images[1::3, :] = batch_nuclei
    interleaved_images[2::3, :] = batch_membrane

    grid = torchvision.utils.make_grid(interleaved_images, nrow=3)

    # add the grid to tensorboard
    writer.add_image(card_name, grid, batchno)


# %%
# Define a function to visualize a batch on jupyter, in case tensorboard is finicky
def log_batch_jupyter(batch):
    """
    Logs a batch of images on jupyter using ipywidget.

    Args:
        batch (dict): A dictionary containing the batch of images to be logged.

    Returns:
        None
    """
    batch_phase = batch["source"][:, :, 0, :, :]  # batch_size x z_size x Y x X tensor.
    batch_size = batch_phase.shape[0]
    batch_membrane = batch["target"][:, 1, 0, :, :].unsqueeze(
        1
    )  # batch_size x 1 x Y x X tensor.
    batch_nuclei = batch["target"][:, 0, 0, :, :].unsqueeze(
        1
    )  # batch_size x 1 x Y x X tensor.

    p1, p99 = np.percentile(batch_membrane, (0.1, 99.9))
    batch_membrane = np.clip((batch_membrane - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_nuclei, (0.1, 99.9))
    batch_nuclei = np.clip((batch_nuclei - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_phase, (0.1, 99.9))
    batch_phase = np.clip((batch_phase - p1) / (p99 - p1), 0, 1)

    plt.figure()
    fig, axes = plt.subplots(
        batch_size, n_channels, figsize=(n_channels * 2, batch_size * 2)
    )
    [N, C, H, W] = batch_phase.shape
    for sample_id in range(batch_size):
        axes[sample_id, 0].imshow(batch_phase[sample_id, 0])
        axes[sample_id, 1].imshow(batch_nuclei[sample_id, 0])
        axes[sample_id, 2].imshow(batch_membrane[sample_id, 0])

        for i in range(n_channels):
            axes[sample_id, i].axis("off")
            axes[sample_id, i].set_title(dataset.channel_names[i])
    plt.tight_layout()
    plt.show()


# %%
# Initialize the data module.

BATCH_SIZE = 4

# 5 is a perfectly reasonable batch size
# (batch size does not have to be a power of 2)
# See: https://sebastianraschka.com/blog/2022/batch-size-2.html

data_module = HCSDataModule(
    data_path,
    z_window_size=1,
    architecture="UNeXt2_2D",
    source_channel=["Phase3D"],
    target_channel=["Nucl", "Mem"],
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    yx_patch_size=(256, 256),  # larger patch size makes it easy to see augmentations.
    augmentations=[],  # Turn off augmentation for now.
    normalizations=[],  # Turn off normalization for now.
)
data_module.setup("fit")

print(
    f"Samples in training set: {len(data_module.train_dataset)}, "
    f"samples in validation set:{len(data_module.val_dataset)}"
)
train_dataloader = data_module.train_dataloader()

# Instantiate the tensorboard SummaryWriter, logs the first batch and then iterates through all the batches and logs them to tensorboard.

writer = SummaryWriter(log_dir=f"{log_dir}/view_batch")
# Draw a batch and write to tensorboard.
batch = next(iter(train_dataloader))
log_batch_tensorboard(batch, 0, writer, "augmentation/none")
writer.close()


# %% [markdown]
# If your tensorboard is causing issues, you can visualize directly on Jupyter /VSCode

# %%
log_batch_jupyter(batch)

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 1.3
# Add augmentations to the datamodule and rerun the setup.
#
# What kind of augmentations do you think are important for this task?
#
# How do they make the model more robust?
#
# Add augmentations to rotate about $\pi$ around z-axis, 30% scale in y,x,
# shearing of 10% and no padding with zeros with a probablity of 80%.
#
# Add a Gaussian noise with a mean of 0.0 and standard deviation of 0.3 with a probability of 50%.
#
# HINT: `RandAffined()` and `RandGaussianNoised()` are from
# `viscy.transforms` [here](https://github.com/mehta-lab/VisCy/blob/main/viscy/transforms.py).
# *Note these are MONAI transforms that have been redefined for VisCy.*
# Can you tell what augmentation were applied from looking at the augmented images in Tensorboard?
#
# HINT:
# [Compare your choice of augmentations by dowloading the pretrained models and config files](https://github.com/mehta-lab/VisCy/releases/download/v0.1.0/VisCy-0.1.0-VS-models.zip).
# </div>
# %%
# Here we turn on data augmentation and rerun setup
source_channel = ["Phase3D"]
target_channel = ["Nucl", "Mem"]

augmentations = [
    RandWeightedCropd(
        keys=source_channel + target_channel,
        spatial_size=(1, 256, 256),
        num_samples=2,
        w_key=target_channel[0],
    ),
    RandAdjustContrastd(keys=source_channel, prob=0.5, gamma=(0.8, 1.2)),
    RandScaleIntensityd(keys=source_channel, factors=0.5, prob=0.5),
    RandGaussianSmoothd(
        keys=source_channel,
        sigma_x=(0.25, 0.75),
        sigma_y=(0.25, 0.75),
        sigma_z=(0.0, 0.0),
        prob=0.5,
    ),
    # #######################
    # ##### TODO  ########
    # #######################
    ##TODO: Add Random Affine Transorms
    ## Write code below
    ## TODO: Add Random Gaussian Noise
    ## Write code below
]

normalizations = [
    NormalizeSampled(
        keys=source_channel + target_channel,
        level="fov_statistics",
        subtrahend="mean",
        divisor="std",
    )
]

data_module.augmentations = augmentations
data_module.setup("fit")

# get the new data loader with augmentation turned on
augmented_train_dataloader = data_module.train_dataloader()

# Draw batches and write to tensorboard
writer = SummaryWriter(log_dir=f"{log_dir}/view_batch")
augmented_batch = next(iter(augmented_train_dataloader))
log_batch_tensorboard(augmented_batch, 0, writer, "augmentation/some")
writer.close()

# %% tags=["solution"]
# #######################
# ##### SOLUTION ########
# #######################
source_channel = ["Phase3D"]
target_channel = ["Nucl", "Mem"]

augmentations = [
    RandWeightedCropd(
        keys=source_channel + target_channel,
        spatial_size=(1, 384, 384),
        num_samples=2,
        w_key=target_channel[0],
    ),
    RandAffined(
        keys=source_channel + target_channel,
        rotate_range=[3.14, 0.0, 0.0],
        scale_range=[0.0, 0.3, 0.3],
        prob=0.8,
        padding_mode="zeros",
        shear_range=[0.0, 0.01, 0.01],
    ),
    RandAdjustContrastd(keys=source_channel, prob=0.5, gamma=(0.8, 1.2)),
    RandScaleIntensityd(keys=source_channel, factors=0.5, prob=0.5),
    RandGaussianNoised(keys=source_channel, prob=0.5, mean=0.0, std=0.3),
    RandGaussianSmoothd(
        keys=source_channel,
        sigma_x=(0.25, 0.75),
        sigma_y=(0.25, 0.75),
        sigma_z=(0.0, 0.0),
        prob=0.5,
    ),
]

normalizations = [
    NormalizeSampled(
        keys=source_channel + target_channel,
        level="fov_statistics",
        subtrahend="mean",
        divisor="std",
    )
]

data_module.augmentations = augmentations
data_module.setup("fit")

# get the new data loader with augmentation turned on
augmented_train_dataloader = data_module.train_dataloader()

# Draw batches and write to tensorboard
writer = SummaryWriter(log_dir=f"{log_dir}/view_batch")
augmented_batch = next(iter(augmented_train_dataloader))
log_batch_tensorboard(augmented_batch, 0, writer, "augmentation/some")
writer.close()


# %% [markdown]
# Visualize directly on Jupyter

# %%
log_batch_jupyter(augmented_batch)

# %% [markdown]
"""
## Train a 2D U-Net model to predict nuclei and membrane from phase.

### Construct a 2D UNeXt2 using VisCy
See ``viscy.unet.networks.Unet2D.Unet2d`` ([source code](https://github.com/mehta-lab/VisCy/blob/7c5e4c1d68e70163cf514d22c475da8ea7dc3a88/viscy/unet/networks/Unet2D.py#L7)) for configuration details.
"""
# %%
# Create a 2D UNet.
GPU_ID = 0

BATCH_SIZE = 12
YX_PATCH_SIZE = (256, 256)

# Dictionary that specifies key parameters of the model.

phase2fluor_config = dict(
    in_channels=1,
    out_channels=2,
    encoder_blocks=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    decoder_conv_blocks=2,
    stem_kernel_size=(1, 2, 2),
    in_stack_depth=1,
    pretraining=False,
)

phase2fluor_model = VSUNet(
    architecture="UNeXt2_2D",  # 2D UNeXt2 architecture
    model_config=phase2fluor_config.copy(),
    loss_function=MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5),
    schedule="WarmupCosine",
    lr=2e-4,
    log_batches_per_epoch=5,  # Number of samples from each batch to log to tensorboard.
    freeze_encoder=False,
)

# %% [markdown]
"""
### Instantiate data module and trainer, test that we are setup to launch training.
"""
# %%
source_channel = ["Phase3D"]
target_channel = ["Nucl", "Mem"]
# Setup the data module.
phase2fluor_2D_data = HCSDataModule(
    data_path,
    architecture="UNeXt2_2D",
    source_channel=source_channel,
    target_channel=target_channel,
    z_window_size=1,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    yx_patch_size=YX_PATCH_SIZE,
    augmentations=augmentations,
    normalizations=normalizations,
)
phase2fluor_2D_data.setup("fit")
# fast_dev_run runs a single batch of data through the model to check for errors.
trainer = VSTrainer(accelerator="gpu", devices=[GPU_ID], fast_dev_run=True)

# trainer class takes the model and the data module as inputs.
trainer.fit(phase2fluor_model, datamodule=phase2fluor_2D_data)


# %% [markdown]
# ## View model graph.
#
# PyTorch uses dynamic graphs under the hood.
# The graphs are constructed on the fly.
# This is in contrast to TensorFlow,
# where the graph is constructed before the training loop and remains static.
# In other words, the graph of the network can change with every forward pass.
# Therefore, we need to supply an input tensor to construct the graph.
# The input tensor can be a random tensor of the correct shape and type.
# We can also supply a real image from the dataset.
# The latter is more useful for debugging.

# %% [markdown]
# <div class="alert alert-info">
#
# ### Task 1.4
# Run the next cell to generate a graph representation of the model architecture.
# Can you recognize the UNet structure and skip connections in this graph visualization?
# </div>

# %%
# visualize graph of phase2fluor model as image.
model_graph_phase2fluor = torchview.draw_graph(
    phase2fluor_model,
    phase2fluor_2D_data.train_dataset[0]["source"][0].unsqueeze(dim=0),
    roll=True,
    depth=3,  # adjust depth to zoom in.
    device="cpu",
    # expand_nested=True,
)
# Print the image of the model.
model_graph_phase2fluor.visual_graph


# %% [markdown]
"""
<div class="alert alert-info">

<h3> Task 1.5 </h3>
Start training by running the following cell. Check the new logs on the tensorboard.
</div>
"""

# %%
# Check if GPU is available
# You can check by typing `nvidia-smi`
GPU_ID = 0

n_samples = len(phase2fluor_2D_data.train_dataset)
steps_per_epoch = n_samples // BATCH_SIZE  # steps per epoch.
n_epochs = 50  # Set this to 50 or the number of epochs you want to train for.

trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    max_epochs=n_epochs,
    log_every_n_steps=steps_per_epoch // 2,
    # log losses and image samples 2 times per epoch.
    logger=TensorBoardLogger(
        save_dir=log_dir,
        # lightning trainer transparently saves logs and model checkpoints in this directory.
        name="phase2fluor",
        log_graph=True,
    ),
)
# Launch training and check that loss and images are being logged on tensorboard.
trainer.fit(phase2fluor_model, datamodule=phase2fluor_2D_data)

# %% [markdown]
"""
<div class="alert alert-success">

<h2> Checkpoint 1 </h2>

While your model is training, let's think about the following questions:<br>
<ul>
<li>What is the information content of each channel in the dataset?</li>
<li>How would you use image translation models?</li>
<li>What can you try to improve the performance of each model?</li>
</ul>

Now the training has started,
we can come back after a while and evaluate the performance!

</div>
"""

# %% [markdown]
"""
## Part 2: Assess your trained model

Now we will look at some metrics of performance of previous model.
We typically evaluate the model performance on a held out test data.
We will use the following metrics to evaluate the accuracy of regression of the model:

- [Person Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
- [Structural similarity](https://en.wikipedia.org/wiki/Structural_similarity) (SSIM).

You should also look at the validation samples on tensorboard
(hint: the experimental data in nuclei channel is imperfect.)
"""

# %% [markdown]
"""
<div class="alert alert-info">

<h3> Task 2.1 Define metrics </h3>

For each of the above metrics, write a brief definition of what they are and what they mean
for this image translation task. Use your favorite search engine and/or resources.

</div>
"""

# %% [markdown]
# ```
# #######################
# ##### Todo ############
# #######################
#
# ```
#
# - Pearson Correlation:
#
# - Structural similarity:

# %% [markdown]
"""
Let's compute metrics directly and plot below.
"""
# %%
# Setup the test data module.
test_data_path = top_dir / "06_image_translation/test/a549_hoechst_cellmask_test.zarr"
source_channel = ["Phase3D"]
target_channel = ["Nucl", "Mem"]

test_data = HCSDataModule(
    test_data_path,
    source_channel=source_channel,
    target_channel=target_channel,
    z_window_size=1,
    batch_size=1,
    num_workers=8,
    architecture="UNeXt2",
)
test_data.setup("test")

test_metrics = pd.DataFrame(
    columns=["pearson_nuc", "SSIM_nuc", "pearson_mem", "SSIM_mem"]
)

# %% Compute metrics directly and plot here.
def min_max_scale(input):
    return (input - np.min(input)) / (np.max(input) - np.min(input))


for i, sample in enumerate(test_data.test_dataloader()):
    phase_image = sample["source"]
    with torch.inference_mode():  # turn off gradient computation.
        predicted_image = phase2fluor_model(phase_image)

    target_image = (
        sample["target"].cpu().numpy().squeeze(0)
    )  # Squeezing batch dimension.
    predicted_image = predicted_image.cpu().numpy().squeeze(0)
    phase_image = phase_image.cpu().numpy().squeeze(0)
    target_mem = min_max_scale(target_image[1, 0, :, :])
    target_nuc = min_max_scale(target_image[0, 0, :, :])
    # slicing channel dimension, squeezing z-dimension.
    predicted_mem = min_max_scale(predicted_image[1, :, :, :].squeeze(0))
    predicted_nuc = min_max_scale(predicted_image[0, :, :, :].squeeze(0))

    # Compute SSIM and pearson correlation.
    ssim_nuc = metrics.structural_similarity(target_nuc, predicted_nuc, data_range=1)
    ssim_mem = metrics.structural_similarity(target_mem, predicted_mem, data_range=1)
    pearson_nuc = np.corrcoef(target_nuc.flatten(), predicted_nuc.flatten())[0, 1]
    pearson_mem = np.corrcoef(target_mem.flatten(), predicted_mem.flatten())[0, 1]

    test_metrics.loc[i] = {
        "pearson_nuc": pearson_nuc,
        "SSIM_nuc": ssim_nuc,
        "pearson_mem": pearson_mem,
        "SSIM_mem": ssim_mem,
    }

test_metrics.boxplot(
    column=["pearson_nuc", "SSIM_nuc", "pearson_mem", "SSIM_mem"],
    rot=30,
)
# %%
# Plot the predicted image
channel_titles = ["Phase", "Nuclei", "Membrane"]
fig, axes = plt.subplots(2, 3, figsize=(30, 20))

for i, sample in enumerate(test_data.test_dataloader()):
    # Plot the phase image
    phase_image = sample["source"]
    channel_image = phase_image[0, 0, 0]
    p_low, p_high = np.percentile(channel_image, (0.5, 99.5))
    channel_image = np.clip(channel_image, p_low, p_high)
    axes[0, 0].imshow(channel_image, cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title(channel_titles[0])

    with torch.inference_mode():  # turn off gradient computation.
        predicted_image = (
            phase2fluor_model(phase_image.to(phase2fluor_model.device))
            .cpu()
            .numpy()
            .squeeze(0)
        )

    target_image = sample["target"].cpu().numpy().squeeze(0)
    # Plot the predicted images
    for i in range(predicted_image.shape[-4]):
        channel_image = predicted_image[i, 0]
        p_low, p_high = np.percentile(channel_image, (0.1, 99.5))
        channel_image = np.clip(channel_image, p_low, p_high)
        axes[0, i + 1].imshow(channel_image, cmap="gray")
        axes[0, i + 1].axis("off")
        axes[0, i + 1].set_title(f"VS {channel_titles[i + 1]}")

    # Plot the target images
    for i in range(target_image.shape[-4]):
        channel_image = target_image[i, 0]
        p_low, p_high = np.percentile(channel_image, (0.5, 99.5))
        channel_image = np.clip(channel_image, p_low, p_high)
        axes[1, i].imshow(channel_image, cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Target {dataset.channel_names[i+1]}")

    # Remove any unused subplots
    for j in range(i + 1, 3):
        fig.delaxes(axes[1, j])

    plt.tight_layout()
    plt.show()
    break

# %% [markdown] tags=[]
"""
<div class="alert alert-info">

<h3> Task 2.2 Compute the metrics with respect to the pretrained model VSCyto2D </h3>
Here we will compare your model with the VSCyto2D pretrained model by computing the pixel-based metrics and segmentation-based metrics.
 
<ul>
<li>When you ran the `setup.sh` you also downloaded the models in `/06_image_translation/pretrained_models/VSCyto2D/*.ckpt`</li>
<li>Load the <b>VSCyto2 model</b> model checkpoint and the configuration file</li>
<li>Compute the pixel-based metrics and segmentation-based metrics between the model you trained and the pretrained model</li>
</ul>
<br>

We will evaluate the performance of your trained model with a pre-trained model using pixel based metrics as above and
segmantation based metrics including (mAP@0.5, dice, accuracy and jaccard index).
</div>

"""

# %% tags=[]
#################
##### TODO ######
#################
# Let's load the pretrained model checkpoint  
pretrained_model_ckpt = top_dir/...## Add the path to the "VSCyto2D/epoch=399-step=23200.ckpt"

# TODO: Load the phase2fluor_config just like the model you trained
phase2fluor_config = dict() ##

# TODO: Load the checkpoint. Write the architecture name. HINT: look at the previous config.
pretrained_phase2fluor = VSUNet.load_from_checkpoint(
    pretrained_model_ckpt,
    architecture=....,
    module_config=phase2fluor_config,
)

# %% tags=["solution"]
# #######################
# ##### SOLUTION ########
# #######################

pretrained_model_ckpt = (
    top_dir / "06_image_translation/pretrained_models/VSCyto2D/epoch=399-step=23200.ckpt"
)

phase2fluor_config = dict(
    in_channels=1,
    out_channels=2,
    encoder_blocks=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    decoder_conv_blocks=2,
    stem_kernel_size=(1, 2, 2),
    in_stack_depth=1,
    pretraining=False,
)
# Load the model checkpoint
pretrained_phase2fluor = VSUNet.load_from_checkpoint(
    pretrained_model_ckpt,
    architecture="UNeXt2_2D",
    model_config = phase2fluor_config,
)
pretrained_phase2fluor.eval()

### Re-load your trained model 
#NOTE: assuming the latest checkpoint it your latest training and model
#TODO: modify above is not the case
phase2fluor_model_ckpt = natsorted(glob(
    str(top_dir / "06_image_translation/logs/phase2fluor/version*/checkpoints/*.ckpt")
))[-1]

phase2fluor_config = dict(
    in_channels=1,
    out_channels=2,
    encoder_blocks=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    decoder_conv_blocks=2,
    stem_kernel_size=(1, 2, 2),
    in_stack_depth=1,
    pretraining=False,
)
# Load the model checkpoint
phase2fluor_model = VSUNet.load_from_checkpoint(
    phase2fluor_model_ckpt,
    architecture="UNeXt2_2D",
    model_config = phase2fluor_config,
)
phase2fluor_model.eval()

#%%[markdown]
"""
### Let's compute the metrics for the test dataset
Before you run the following code, make sure you have the pretrained model loaded and the test data is ready.

The following code will compute the following:
- the pixel-based metrics  (pearson correlation, SSIM)
- segmentation-based metrics (mAP@0.5, dice, accuracy, jaccard index)

#### Note:
- The segmentation-based metrics are computed using the cellpose stock `nuclei` model
- The metrics will be store in the `test_pixel_metrics` and `test_segmentation_metrics` dataframes
- The segmentations will be stored in the `segmentation_store` zarr file
- Analyze the code while it runs.
"""
#%%
# Define the function to compute the cellpose segmentation and the normalization
def min_max_scale(input):
    return (input - np.min(input)) / (np.max(input) - np.min(input))

def cellpose_segmentation(prediction:ArrayLike,target:ArrayLike)->Tuple[torch.ShortTensor]:
    #NOTE these are hardcoded for this notebook and A549 dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp_nuc_kwargs = {
        "diameter": 65,
        "channels": [0, 0],
        "cellprob_threshold": 0.0,
    }
    cellpose_model = models.CellposeModel(
            gpu=True, model_type='nuclei', device=torch.device(device)
    )
    pred_label, _, _ = cellpose_model.eval(prediction, **cp_nuc_kwargs)
    target_label, _, _ = cellpose_model.eval(target, **cp_nuc_kwargs)

    pred_label = pred_label.astype(np.int32)
    target_label = target_label.astype(np.int32)
    pred_label = torch.ShortTensor(pred_label)
    target_label = torch.ShortTensor(target_label)

    return (pred_label,target_label)

#%% 
# Setting the paths for the test data and the output segmentation
test_data_path = top_dir / "06_image_translation/test/a549_hoechst_cellmask_test.zarr"
output_segmentation_path=top_dir /"06_image_translation/pretrained_model_segmentations.zarr"

# Creating the dataframes to store the pixel and segmentation metrics
test_pixel_metrics = pd.DataFrame(
    columns=["model", "fov","pearson_nuc", "SSIM_nuc", "pearson_mem", "SSIM_mem"]
)
test_segmentation_metrics= pd.DataFrame(
    columns=["model", "fov","masks_per_fov","accuracy","dice","jaccard","mAP","mAP_50","mAP_75","mAR_100"]
)
# Opening the test dataset
test_dataset = open_ome_zarr(test_data_path)

# Creating an output store for the predictions and segmentations
segmentation_store = open_ome_zarr(output_segmentation_path,channel_names=['nuc_pred','mem_pred','nuc_labels'],mode='w',layout='hcs')

# Looking at the test dataset
print('Test dataset:')
test_dataset.print_tree()
channel_names = test_dataset.channel_names
print(f'Channel names: {channel_names}')

# Finding the channel indeces for the corresponding channel names
phase_cidx = channel_names.index("Phase3D")
nuc_cidx = channel_names.index("Nucl")
mem_cidx =  channel_names.index("Mem")
nuc_label_cidx =  channel_names.index("nuclei_segmentation")

#%%
# Iterating through the test dataset positions to:
positions = list(test_dataset.positions())
total_positions = len(positions)

# Initializing the progress bar with the total number of positions
with tqdm(total=total_positions, desc="Processing FOVs") as pbar:
    # Iterating through the test dataset positions
    for fov, pos in positions:
        T,C,Z,Y,X = pos.data.shape
        Z_slice = slice(Z//2,Z//2+1)
        # Getting the arrays and the center slices
        phase_image = pos.data[:,phase_cidx:phase_cidx+1,Z_slice]
        target_nucleus =  pos.data[0,nuc_cidx:nuc_cidx+1,Z_slice]
        target_membrane =  pos.data[0,mem_cidx:mem_cidx+1,Z_slice]
        target_nuc_label = pos.data[0,nuc_label_cidx:nuc_label_cidx+1,Z_slice]
        
        # Running the prediction for both models
        phase_image = torch.from_numpy(phase_image).type(torch.float32)
        phase_image = phase_image.to(phase2fluor_model.device)
        with torch.inference_mode():  # turn off gradient computation.
            predicted_image_phase2fluor = phase2fluor_model(phase_image)
            predicted_image_pretrained = pretrained_phase2fluor(phase_image)

        # Loading and Normalizing the target and predictions for both models 
        predicted_image_phase2fluor = predicted_image_phase2fluor.cpu().numpy().squeeze(0)
        predicted_image_pretrained = predicted_image_pretrained.cpu().numpy().squeeze(0)
        phase_image = phase_image.cpu().numpy().squeeze(0)

        target_mem = min_max_scale(target_membrane[0,0])
        target_nuc = min_max_scale(target_nucleus[0,0])

        # Noramalize the dataset using min-max scaling
        predicted_mem_phase2fluor = min_max_scale(
            predicted_image_phase2fluor[1, :, :, :].squeeze(0)
        )
        predicted_nuc_phase2fluor = min_max_scale(
            predicted_image_phase2fluor[0, :, :, :].squeeze(0)
        )

        predicted_mem_pretrained = min_max_scale(
            predicted_image_pretrained[1, :, :, :].squeeze(0)
        )
        predicted_nuc_pretrained = min_max_scale(
            predicted_image_pretrained[0, :, :, :].squeeze(0)
        )

        #######  Pixel-based Metrics ############
        # Compute SSIM and Pearson correlation for phase2fluor_model
        print('Computing Pixel Metrics')
        ssim_nuc_phase2fluor = metrics.structural_similarity(
            target_nuc, predicted_nuc_phase2fluor, data_range=1
        )
        ssim_mem_phase2fluor = metrics.structural_similarity(
            target_mem, predicted_mem_phase2fluor, data_range=1
        )
        pearson_nuc_phase2fluor = np.corrcoef(
            target_nuc.flatten(), predicted_nuc_phase2fluor.flatten()
        )[0, 1]
        pearson_mem_phase2fluor = np.corrcoef(
            target_mem.flatten(), predicted_mem_phase2fluor.flatten()
        )[0, 1]

        test_pixel_metrics.loc[len(test_pixel_metrics)] = {
            "model": "phase2fluor",
            "fov":fov,
            "pearson_nuc": pearson_nuc_phase2fluor,
            "SSIM_nuc": ssim_nuc_phase2fluor,
            "pearson_mem": pearson_mem_phase2fluor,
            "SSIM_mem": ssim_mem_phase2fluor,
        }
        # Compute SSIM and Pearson correlation for pretrained_model
        ssim_nuc_pretrained = metrics.structural_similarity(
            target_nuc, predicted_nuc_pretrained, data_range=1
        )
        ssim_mem_pretrained = metrics.structural_similarity(
            target_mem, predicted_mem_pretrained, data_range=1
        )
        pearson_nuc_pretrained = np.corrcoef(
            target_nuc.flatten(), predicted_nuc_pretrained.flatten()
        )[0, 1]
        pearson_mem_pretrained = np.corrcoef(
            target_mem.flatten(), predicted_mem_pretrained.flatten()
        )[0, 1]

        test_pixel_metrics.loc[len(test_pixel_metrics)] = {
            "model": "pretrained_phase2fluor",
            "fov":fov,
            "pearson_nuc": pearson_nuc_pretrained,
            "SSIM_nuc": ssim_nuc_pretrained,
            "pearson_mem": pearson_mem_pretrained,
            "SSIM_mem": ssim_mem_pretrained,
        }

        ###### Segmentation based metrics #########
        # Load the manually curated nuclei target label
        print('Computing Segmentation Metrics')
        pred_label,target_label= cellpose_segmentation(predicted_nuc_phase2fluor,target_nucleus)
        # Binary labels
        pred_label_binary = pred_label > 0
        target_label_binary = target_label > 0

        # Use Coco metrics to get mean average precision
        coco_metrics = mean_average_precision(pred_label, target_label)
        # Find unique number of labels
        num_masks_fov = len(np.unique(pred_label))

        test_segmentation_metrics.loc[len(test_segmentation_metrics)] = {
            "model": "phase2fluor",
            "fov":fov,
            "masks_per_fov": num_masks_fov,
            "accuracy": accuracy(pred_label_binary, target_label_binary, task="binary").item(),
            "dice":  dice(pred_label_binary, target_label_binary).item(),
            "jaccard": jaccard_index(pred_label_binary, target_label_binary, task="binary").item(),
            "mAP":coco_metrics["map"].item(),
            "mAP_50":coco_metrics["map_50"].item(),
            "mAP_75":coco_metrics["map_75"].item(),
            "mAR_100":coco_metrics["mar_100"].item()
        }

        pred_label,target_label= cellpose_segmentation(predicted_nuc_pretrained,target_nucleus)
        # Binary labels
        pred_label_binary = pred_label > 0
        target_label_binary = target_label > 0

        # Use Coco metrics to get mean average precision
        coco_metrics = mean_average_precision(pred_label, target_label)
        # Find unique number of labels
        num_masks_fov = len(np.unique(pred_label))

        test_segmentation_metrics.loc[len(test_segmentation_metrics)] = {
            "model": "phase2fluor_pretrained",
            "fov":fov,
            "masks_per_fov": num_masks_fov,
            "accuracy": accuracy(pred_label_binary, target_label_binary, task="binary").item(),
            "dice":  dice(pred_label_binary, target_label_binary).item(),
            "jaccard": jaccard_index(pred_label_binary, target_label_binary, task="binary").item(),
            "mAP":coco_metrics["map"].item(),
            "mAP_50":coco_metrics["map_50"].item(),
            "mAP_75":coco_metrics["map_75"].item(),
            "mAR_100":coco_metrics["mar_100"].item()
        }
        
        #Save the predictions and segmentations
        position = segmentation_store.create_position(*Path(fov).parts[-3:])
        output_array = np.zeros((T,3,1,Y,X),dtype=np.float32)
        output_array[0,0,0]=predicted_nuc_pretrained
        output_array[0,1,0]=predicted_mem_pretrained
        output_array[0,2,0]=np.array(pred_label)
        position.create_image("0",output_array)
        
        # Update the progress bar
        pbar.update(1)

# Close the OME-Zarr files
test_dataset.close()
segmentation_store.close()
#%%
#Save the test metrics into a dataframe
pixel_metrics_path = top_dir/"06_image_translation/VS_metrics_pixel_part_1.csv"
segmentation_metrics_path = top_dir/"06_image_translation/VS_metrics_segments_part_1.csv"
test_pixel_metrics.to_csv(pixel_metrics_path)
test_segmentation_metrics.to_csv(segmentation_metrics_path)

# %% [markdown] tags=[]
"""
<div class="alert alert-info">

<h3> Task 2.3 Compare the model's metrics </h3>
In the previous section, we computed the pixel-based metrics and segmentation-based metrics.
Now we will compare the performance of the model you trained with the pretrained model by plotting the boxplots.

After you plot the metrics answer the following:
<ul>
<li>What do these metrics tells us about the performance of the model?</li>
<li>How do you interpret the differences in the metrics between the models?</li>
<li>How is your model compared to the pretrained model? How can you improve it?</li>
</ul>
</div>

"""
#%%
# Show boxplot of the metrics
# Boxplot of the metrics
test_pixel_metrics.boxplot(
    by="model",
    column=["pearson_nuc", "SSIM_nuc", "pearson_mem", "SSIM_mem"],
    rot=30,
    figsize=(8, 8),
)
plt.suptitle("Model Pixel Metrics")
plt.show()
# Show boxplot of the metrics
# Boxplot of the metrics
test_segmentation_metrics.boxplot(
    by="model",
    column=["jaccard", "accuracy", "mAP_75","mAP_50"],
    rot=30,
    figsize=(8, 8),
)
plt.suptitle("Model Segmentation Metrics")
plt.show()

#%%[markdown]
"""
########## TODO ##############
- What do these metrics tells us about the performance of the model?
- How do you interpret the differences in the metrics between the models?
- How is your model compared to the pretrained model? How can you improve it?

"""

#%%[markdown]
"""
## Plotting the predictions and segmentations
Here we will plot the predictions and segmentations side by side for the pretrained and trained models.

Please modify the crop size and Y,X slicing to view different areas of the FOV.
"""
#%%
######## TODO ##########
# Modify the crop size and Y,X slicing to view different areas of the FOV
crop = 256
y_slice=slice(Y//2-crop//2,Y//2+crop//2)
x_slice=slice(X//2-crop//2,X//2+crop//2)
#######################
# Plotting side by side comparisons
fig, axs = plt.subplots(4, 3, figsize=(15, 20))

# First row: phase_image, target_nuc, target_mem
axs[0, 0].imshow(phase_image[0,0,y_slice,x_slice], cmap='gray')
axs[0, 0].set_title("Phase Image")
axs[0, 1].imshow(target_nuc[y_slice,x_slice], cmap='gray')
axs[0, 1].set_title("Target Nucleus")
axs[0, 2].imshow(target_mem[y_slice,x_slice], cmap='gray')
axs[0, 2].set_title("Target Membrane")

# Second row: target_nuc, pred_nuc_phase2fluor, pred_nuc_pretrained
axs[1, 0].imshow(target_nuc[y_slice,x_slice], cmap='gray')
axs[1, 0].set_title("Target Nucleus")
axs[1, 1].imshow(predicted_nuc_phase2fluor[y_slice,x_slice], cmap='gray')
axs[1, 1].set_title("Pred Nucleus Phase2Fluor")
axs[1, 2].imshow(predicted_nuc_pretrained[y_slice,x_slice], cmap='gray')
axs[1, 2].set_title("Pred Nucleus Pretrained")

# Third row: target_mem, pred_mem_phase2fluor, pred_mem_pretrained
axs[2, 0].imshow(target_mem[y_slice,x_slice], cmap='gray')
axs[2, 0].set_title("Target Membrane")
axs[2, 1].imshow(predicted_mem_phase2fluor[y_slice,x_slice], cmap='gray')
axs[2, 1].set_title("Pred Membrane Phase2Fluor")
axs[2, 2].imshow(predicted_mem_pretrained[y_slice,x_slice], cmap='gray')
axs[2, 2].set_title("Pred Membrane Pretrained")

# Fourth row: target_nuc, segment_nuc, segment_nuc2
axs[3, 0].imshow(target_nuc[y_slice,x_slice], cmap='gray')
axs[3, 0].set_title("Target Nucleus")
axs[3, 1].imshow(label2rgb(np.array(target_label[y_slice,x_slice],dtype='int')), cmap='gray')
axs[3, 1].set_title("Segmented Nucleus (Target)")
axs[3, 2].imshow(label2rgb(np.array(pred_label[y_slice,x_slice],dtype='int')), cmap='gray')
axs[3, 2].set_title("Segmented Nucleus")

# Hide axes ticks
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# %% [markdown] tags=[]
"""
<div class="alert alert-success">

<h2> 
🎉 The end of the notebook 🎉 
Continue to Part 2: Image translation with generative models.
</h2>

Congratulations! You have trained several image translation models now!
<br>
Please remember to document the hyperparameters,
snapshots of predictions on validation set,
and loss curves for your models and add the final performance in
<a href="https://docs.google.com/document/d/1Mq-yV8FTG02xE46Mii2vzPJVYSRNdeOXkeU-EKu-irE/edit?usp=sharing">
this google doc
</a>.
We'll discuss our combined results as a group.
</div>
"""

# %%
