# %% [markdown]
"""
# Image translation (Virtual Staining)

Written by Eduardo Hirata-Miyasaki, Ziwen Liu, and Shalin Mehta, CZ Biohub San Francisco.

## Overview

In this exercise, we will predict fluorescence images of
nuclei and plasma membrane markers from quantitative phase images of cells,
i.e., we will _virtually stain_ the nuclei and plasma membrane
visible in the phase image.
This is an example of an image translation task.
We will apply spatial and intensity augmentations to train robust models
and evaluate their performance.
Finally, we will explore the opposite process of predicting a phase image
from a fluorescence membrane label.

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
  
#### Part 2: Train a model that predicts fluorescence from phase, and vice versa, using the UNeXt2 architecture.

  - Create a model for image translation mapping from source domain to target domain
  where the source domain is label-free microscopy (material density)
  and the target domain is fluorescence microscopy (fluorophore density).
  - Use the UNeXt2 architecture, a _purely convolutional architecture_
  that draws on the design principles of transformer models to complete this task.
  Here we will use a *UNeXt2*, an efficient image translation architecture inspired by ConvNeXt v2 and SparK.
  - We will perform the preprocessing, training, prediction, evaluation, and deployment steps
  that borrow from our computer vision pipeline for single-cell analysis in
  our pipeline called [VisCy](https://github.com/mehta-lab/VisCy).
  - Reuse the same architecture as above and create a similar model doing the inverse task (fluorescence to phase).
  - Evaluate the model.

#### (Extra) Play with the hyperparameters to improve the models or train a 3D UNeXt2

Our guesstimate is that each of the three parts will take ~1-1.5 hours.
A reasonable 2D UNet can be trained in ~30 min on a typical AWS node.
The focus of the exercise is on understanding the information content of the data,
how to train and evaluate 2D image translation models, and exploring some hyperparameters of the model.
If you complete this exercise and have time to spare, try the bonus exercise on 3D image translation.

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
📖 As you work through parts 2 and 3, please share the layouts of your models (output of torchview)
and their performance with everyone via
[this Google Doc](https://docs.google.com/document/d/1Mq-yV8FTG02xE46Mii2vzPJVYSRNdeOXkeU-EKu-irE/edit?usp=sharing). 📖
"""
# %% [markdown]
"""
<div class="alert alert-warning">
The exercise is organized in 3 parts + Extra part.

<ul>
<li><b>Part 1</b> - Learn to use iohub (I/O library), VisCy dataloaders, and tensorboard.</li>
<li><b>Part 2</b> - Train and evaluate the model to translate phase into fluorescence, and vice versa.</li>
<li><b>Extra task</b> - Tune the models to improve performance.</li>
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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchview
import torchvision
from iohub import open_ome_zarr
from iohub.reader import print_info
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from skimage import metrics  # for metrics.

# pytorch lightning wrapper for Tensorboard.
from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard

# HCSDataModule makes it easy to load data during training.
from viscy.data.hcs import HCSDataModule

# Trainer class and UNet.
from viscy.light.engine import MixedLoss, VSUNet
from viscy.light.trainer import VSTrainer

# training augmentations
from viscy.transforms import (
    NormalizeSampled,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
)

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

# This dataset contains images at 3 resolutions.
# '0' is the highest resolution
# '1' is down-scaled 2x2,
# '2' is down-scaled 4x4.
# Such datasets are called image pyramids.
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

BATCH_SIZE = 5

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
    ##TODO: Add rotation agumentations
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

Now the training has started,
we can come back after a while and evaluate the performance!

</div>
"""

# %% [markdown]
"""
## Part 2: Assess previous model, train fluorescence to phase contrast translation model.

We now look at some metrics of performance of previous model.
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

<h3>Task 2.2 Train fluorescence to phase contrast translation model</h3>

Instantiate a data module, model, and trainer for fluorescence to phase contrast translation. Copy over the code from previous cells and update the parameters. Give the variables and paths a different name/suffix (fluor2phase) to avoid overwriting objects used to train phase2fluor models.
</div>
"""
# %% tags=[]
##########################
######## TODO ########
##########################

fluor2phase_data = HCSDataModule(
    # Your code here (copy from above and modify as needed)
)
fluor2phase_data.setup("fit")

# Dictionary that specifies key parameters of the model.
fluor2phase_config = {
    # Your config here
}

fluor2phase_model = VSUNet(
    # Your code here (copy from above and modify as needed)
)

# Visualize the graph of fluor2phase model as image.
model_graph_fluor2phase = torchview.draw_graph(
    fluor2phase_model,
    fluor2phase_data.train_dataset[0]["source"],
    depth=2,  # adjust depth to zoom in.
    device="cpu",
)
model_graph_fluor2phase.visual_graph

# %% tags=["solution"]

##########################
######## Solution ########
##########################

# The entire training loop is contained in this cell.
source_channel = ["Mem"]  # or 'Nuc' depending on choice
target_channel = ["Phase3D"]
YX_PATCH_SIZE = (256, 256)
BATCH_SIZE = 12
n_epochs = 50

# Setup the new augmentations
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

# Setup the dataloader
fluor2phase_data = HCSDataModule(
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
fluor2phase_data.setup("fit")

n_samples = len(fluor2phase_data.train_dataset)

steps_per_epoch = n_samples // BATCH_SIZE  # steps per epoch.

# Dictionary that specifies key parameters of the model.
fluor2phase_config = dict(
    in_channels=1,
    out_channels=1,
    encoder_blocks=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    decoder_conv_blocks=2,
    stem_kernel_size=(1, 2, 2),
    in_stack_depth=1,
    pretraining=False,
)

fluor2phase_model = VSUNet(
    architecture="UNeXt2_2D",
    model_config=fluor2phase_config.copy(),
    loss_function=MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5),
    schedule="WarmupCosine",
    lr=2e-4,
    log_batches_per_epoch=5,  # Number of samples from each batch to log to tensorboard.
    freeze_encoder=False,
)

# Visualize the graph of fluor2phase model as image.
model_graph_fluor2phase = torchview.draw_graph(
    fluor2phase_model,
    next(iter(fluor2phase_data.train_dataloader()))["source"],
    depth=3,  # adjust depth to zoom in.
    device="cpu",
)
model_graph_fluor2phase.visual_graph

# %% tags=[]
##########################
######## TODO ########
##########################

trainer = VSTrainer(
    # Your code here (copy from above and modify as needed)
)
trainer.fit(fluor2phase_model, datamodule=fluor2phase_data)


# %%  tags=["solution"]
trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    max_epochs=n_epochs,
    log_every_n_steps=steps_per_epoch // 2,
    logger=TensorBoardLogger(
        save_dir=log_dir,
        # lightning trainer transparently saves logs and model checkpoints in this directory.
        name="fluor2phase",
        log_graph=True,
    ),
)
trainer.fit(fluor2phase_model, datamodule=fluor2phase_data)


# %% [markdown] tags=[]
"""
<div class="alert alert-info">

<h3>Task 2.3 </h3>

While your model is training, let's think about the following questions:
- What is the information content of each channel in the dataset?
- How would you use image translation models?
- What can you try to improve the performance of each model?
</div>
"""
# %%
test_data_path = Path(
    "~/data/06_image_translation/test/a549_hoechst_cellmask_test.zarr"
).expanduser()

test_data = HCSDataModule(
    test_data_path,
    source_channel="Mem",  # or Nuc, depending on your choice of source
    target_channel="Phase3D",
    z_window_size=1,
    batch_size=1,
    num_workers=8,
    architecture="UNeXt2",
)
test_data.setup("test")

test_metrics = pd.DataFrame(columns=["pearson_phase", "SSIM_phase"])


# %%
for i, sample in enumerate(test_data.test_dataloader()):
    source_image = sample["source"]
    with torch.inference_mode():  # turn off gradient computation.
        predicted_image = fluor2phase_model(source_image.to(fluor2phase_model.device))

    target_image = (
        sample["target"].cpu().numpy().squeeze(0)
    )  # Squeezing batch dimension.
    predicted_image = predicted_image.cpu().numpy().squeeze(0)
    source_image = source_image.cpu().numpy().squeeze(0)
    target_phase = min_max_scale(target_image[0, 0, :, :])
    # slicing channel dimension, squeezing z-dimension.
    predicted_phase = min_max_scale(predicted_image[0, :, :, :].squeeze(0))

    # Compute SSIM and pearson correlation.
    ssim_phase = metrics.structural_similarity(
        target_phase, predicted_phase, data_range=1
    )
    pearson_phase = np.corrcoef(target_phase.flatten(), predicted_phase.flatten())[0, 1]

    test_metrics.loc[i] = {
        "pearson_phase": pearson_phase,
        "SSIM_phase": ssim_phase,
    }

test_metrics.boxplot(
    column=["pearson_phase", "SSIM_phase"],
    rot=30,
)
# %%
# Plot the predicted image
channel_titles = [
    "Membrane",
    "Target Phase",
    "Predicted_Phase",
]
fig, axes = plt.subplots(1, 3, figsize=(30, 20))

for i, sample in enumerate(test_data.test_dataloader()):
    # Plot the phase image
    mem_image = sample["source"]
    channel_image = mem_image[0, 0, 0]
    p_low, p_high = np.percentile(channel_image, (0.5, 99.5))
    channel_image = np.clip(channel_image, p_low, p_high)
    axes[0].imshow(channel_image, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title(channel_titles[0])

    with torch.inference_mode():  # turn off gradient computation.
        predicted_image = (
            phase2fluor_model(phase_image.to(phase2fluor_model.device))
            .cpu()
            .numpy()
            .squeeze(0)
        )

    target_image = sample["target"].cpu().numpy().squeeze(0)
    # Plot the predicted images
    channel_image = target_image[0, 0]
    p_low, p_high = np.percentile(channel_image, (0.5, 99.5))
    channel_image = np.clip(channel_image, p_low, p_high)
    axes[1].imshow(channel_image, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title(channel_titles[1])

    channel_image = predicted_image[1, 0]
    p_low, p_high = np.percentile(channel_image, (0.1, 99.5))
    channel_image = np.clip(channel_image, p_low, p_high)
    axes[2].imshow(channel_image, cmap="gray")
    axes[2].axis("off")
    axes[2].set_title(f"VS {channel_titles[2]}")

    plt.tight_layout()
    plt.show()
    break

# %% [markdown] tags=[]
"""
<div class="alert alert-success">

<h2> Checkpoint 2 </h2>
<p>When your model finishes training, please summarize hyperparameters and performance of your models in the <a href="https://docs.google.com/document/d/1Mq-yV8FTG02xE46Mii2vzPJVYSRNdeOXkeU-EKu-irE/edit?usp=sharing" target="_blank">this google doc</a></p>

</div>
"""


# %% [markdown] tags=[]
"""
<div class="alert alert-info">

<h3>Extra exercises</h3>
<b>Tune the models and explore other architectures from <a href="https://github.com/mehta-lab/VisCy/tree/main/examples/demos">VisCy</a></b>
<br>
<p>Learning goals:</p>
<ul>
    <li>Understand how data, model capacity, and training parameters control the performance of the model. Your goal is to try to underfit or overfit the model.</li>
    <li>How can we scale it up from 2D to 3D training and predictions?</li>
</ul>
</div>

"""


# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Extra Example 1: Hyperparameter tuning
#
# - Choose a model you want to train (phase2fluor or fluor2phase).
# - Set up a configuration that you think will improve the performance of the model
# - Consider modifying the learning rate and see how it changes performance
# - Use training loop illustrated in previous cells to train phase2fluor and fluor2phase models to prototype your own training loop.
# - Add code to evaluate the model using Pearson Correlation and SSIM
# As your model is training, please document hyperparameters, snapshots of predictions on validation set,
# and loss curves for your models in
# [this google doc](https://docs.google.com/document/d/1Mq-yV8FTG02xE46Mii2vzPJVYSRNdeOXkeU-EKu-irE/edit?usp=sharing)
# </div>

# %% tags=[]
##########################
######## TODO ########
##########################

tune_data = HCSDataModule(
    # Your code here (copy from above and modify as needed)
)
tune_data.setup("fit")

# Dictionary that specifies key parameters of the model.
tune_config = {
    # Your config here
}

tune_model = VSUNet(
    # Your code here (copy from above and modify as needed)
)

trainer = VSTrainer(
    # Your code here (copy from above and modify as needed)
)
trainer.fit(tune_model, datamodule=tune_data)


# Visualize the graph of fluor2phase model as image.
model_graph_tune = torchview.draw_graph(
    tune_model,
    tune_data.train_dataset[0]["source"],
    depth=2,  # adjust depth to zoom in.
    device="cpu",
)
model_graph_tune.visual_graph


# %% tags=["solution"]

##########################
######## Solution ########
##########################
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

phase2fluor_model_low_lr = VSUNet(
    architecture="UNeXt2_2D",
    model_config=phase2fluor_config.copy(),
    loss_function=MixedLoss(
        l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5
    ),  # Changed the loss function to MixedLoss L1 and MS-SSIM
    schedule="WarmupCosine",
    lr=2e-5,  # lower learning rate by factor of 10
    log_batches_per_epoch=5,  # Number of samples from each batch to log to tensorboard.
)

trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    max_epochs=n_epochs,
    log_every_n_steps=steps_per_epoch,
    logger=TensorBoardLogger(
        save_dir=log_dir,
        name="phase2fluor",
        version="phase2fluor_low_lr",
        log_graph=True,
    ),
    fast_dev_run=True,
)  # Set fast_dev_run to False to train the model.
trainer.fit(phase2fluor_model_low_lr, datamodule=phase2fluor_2D_data)
# %% [markdown]
"""
<div class="alert alert-info">
<h3>
Extra Example 2: 3D Virtual Staining
</h3>
Now, let's implement a 3D virtual staining model(Phase->Fluorescence)<br>
<b>Note:</b> This task might take longer to train +1 hr. Try it out in your free-time.

</div>
"""

# %% tags=["task"]
data_path = Path()  # TODO: Point to a 3D dataset (HEK, Neuromast)
BATCH_SIZE = 4
YX_PATCH_SIZE = (256, 256)

phase2fluor_3D_config = ...

phase2fluor_3D_data = HCSDataModule(...)

phase2fluor_3D = VSUNet(...)

trainer = VSTrainer(...)

# Start the training
trainer.fit(...)

# %% tags=["solution"]

##########################
######## Solution ########
##########################
"""
You can download the file and place it in the data folder.
https://public.czbiohub.org/comp.micro/viscy/VSCyto3D/train/raw-and-reconstructed.zarr/

You can run the following shell script:
```
cd ~/data/hek3d/training
# Download the Zarr dataset recursively (if the server supports it)
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VSCyto3D/train/raw-and-reconstructed.zarr/"
```

"""
# TODO: Point to a 3D dataset (HEK, Neuromast)
data_path = Path("./raw-and-reconstructed.zarr")
BATCH_SIZE = 4
YX_PATCH_SIZE = (384, 384)
GPU_ID = 0
n_epochs = 50

## For 3D training - VSCyto3D
source_channel = ["reconstructed-labelfree"]
target_channel = ["reconstructed-nucleus", "reconstructed-membrane"]

# Setup the new augmentations
augmentations = [
    RandWeightedCropd(
        keys=source_channel + target_channel,
        spatial_size=(-1, 512, 512),
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

phase2fluor_3D_config = dict(
    in_channels=1,
    out_channels=2,
    in_stack_depth=5,
    backbone="convnextv2_tiny",
    decoder_conv_blocks=2,
    head_expansion_ratio=4,
    stem_kernel_size=(5, 4, 4),
)
phase2fluor_3D_data = HCSDataModule(
    data_path,
    architecture="UNeXt2",
    source_channel=source_channel,
    target_channel=target_channel,
    z_window_size=5,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    yx_patch_size=YX_PATCH_SIZE,
    augmentations=augmentations,
    normalizations=normalizations,
)
phase2fluor_3D_data.setup("fit")

n_samples = len(phase2fluor_3D_data.train_dataset)
steps_per_epoch = n_samples // BATCH_SIZE  # steps per epoch.

phase2fluor_3D = VSUNet(
    architecture="UNeXt2",
    model_config=phase2fluor_3D_config.copy(),
    loss_function=MixedLoss(l1_alpha=0.5, l2_alpha=0.0, ms_dssim_alpha=0.5),
    lr=2e-4,
    schedule="WarmupCosine",
    log_batches_per_epoch=5,
)

trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    max_epochs=n_epochs,
    log_every_n_steps=steps_per_epoch,
    logger=TensorBoardLogger(
        save_dir=log_dir,
        name="phase2fluor_3D",
        version="3D_UNeXt2",
        log_graph=True,
    ),
    fast_dev_run=True,  # TODO: Set to False to run full-training
)
trainer.fit(phase2fluor_3D, datamodule=phase2fluor_3D_data)

# %% [markdown] tags=[]
"""
<div class="alert alert-success">

<h2> 
🎉 The end of the notebook 🎉
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
