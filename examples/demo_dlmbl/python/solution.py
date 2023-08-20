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


📖 As you work through parts 2 and 3, please share the layouts of the models you train and their performance with everyone via [this google doc](https://docs.google.com/document/d/1hZWSVRvt9KJEdYu7ib-vFBqAVQRYL8cWaP_vFznu7D8/edit#heading=h.n5u485pmzv2z) 📖.

<div class="alert alert-info">
Our guesstimate is that each of the three parts will take ~1.5 hours, but don't rush parts 1 and 2 if you need more time with them.
We will discuss your observations on google doc after checkpoints 2 and 3.
</div>

Before you start,

<div class="alert alert-danger">
Set your python kernel to <span style="color:black;">04-image-translation</span>
</div>
"""
# %% [markdown] <a id='1_phase2fluor'></a>
"""
# Part 1: Visualize training data using tensorboard, start training a model.
---------

Learning goals:

- Load the OME-zarr dataset and examine the channels.
- Configure and understand the data loader.
- Log some patches to tensorboard.
- Initialize a 2D U-Net model for virtual staining
- Start training the model to predict nuclei and membrane from phase.

"""

# %% Imports and paths

import matplotlib.pyplot as plt
import torch
from iohub import open_ome_zarr
import torchvision
import torchview
import os
from pathlib import Path
import numpy as np


# HCSDataModule makes it easy to load data during training. 
from viscy.light.data import HCSDataModule 
# Trainer class and UNet.
from viscy.light.engine import VSTrainer, VSUNet
from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard
from tensorboard import notebook  # for viewing tensorboard in notebook

# Paths to data and log directory
data_path = Path(
    "~/data/04_image_translation/HEK_nuclei_membrane_pyramid.zarr/"
).expanduser()

log_dir = (
    Path("~/data/04_image_translation/logs/")
    .expanduser()
)

# Create log directory if needed, and launch tensorboard
log_dir.mkdir(parents=True, exist_ok=True)

%reload_ext tensorboard
%tensorboard --logdir {log_dir}




# %% [markdown]
"""
## Load Dataset.

<div class="alert alert-info">
Task 1.1
Use <a href=https://czbiohub-sf.github.io/iohub/main/api/ngff.html#open-ome-zarr>
<code>iohub.open_ome_zarr</code></a> to read the dataset and explore several FOVs with matplotlib.
</div>

There should be 301 FOVs in the dataset (12 GB compressed).

Each FOV consists of 3 channels of 2048x2048 images,
saved in the <a href="https://ngff.openmicroscopy.org/latest/#hcs-layout">
High-Content Screening (HCS) layout</a>
specified by the Open Microscopy Environment Next Generation File Format
(OME-NGFF).

The layout on the disk is: row/col/field/pyramid_level/timepoint/channel/z/y/x.
Notice that labelling of nuclei channel is not complete - some cells are not expressing the fluorescent protein.

"""

# %%

dataset = open_ome_zarr(data_path)

print(f"Number of positions:{len(list(dataset.positions()))}")

# Use the field and pyramid_level below to visualize data.
row = "0"
col = "0"
field = "23"

# This dataset contains images at 3 resolutions.
# '0' is the highest resolution
# '1' is down-scaled 2x2,
# '2' is down-scaled 4x4.
# Such datasets are called image pyramids.
pyaramid_level = "0"

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
## Initialize data loaders and see the samples in tensorboard.

<div class="alert alert-info">
Task 1.2
Setup the data loader and log several batches to tensorboard.
</div>`

VisCy builds on top of PyTorch Lightning. PyTorch Lightning is a thin wrapper around PyTorch that allows rapid experimentation. It provides a [DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) to handle loading and processing of data during training. VisCy provides a child class, `HCSDataModule` to make it intuitve to access data stored in the HCS layout.
  
The dataloader in `HCSDataModule` returns a batch of samples. A `batch` is a list of dictionaries. The length of the list is equal to the batch size. Each dictionary consists of following key-value pairs.
- `source`: the input image, a tensor of size 1*1*Y*X
- `target`: the target image, a tensor of size 2*1*Y*X
- `index` : the tuple of (location of field in HCS layout, time, and z-slice) of the sample.

"""

# %%

BATCH_SIZE = 42 # 42 is a perfectly reasonable batch size. After all, it is the answer to the ultimate question of life, the universe and everything.
# More seriously, batch size does not have to be a power of 2. 
# See: https://sebastianraschka.com/blog/2022/batch-size-2.html

data_module = HCSDataModule(
    data_path,
    source_channel="Phase",
    target_channel=["Nuclei", "Membrane"],
    z_window_size=1,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    architecture="2D",
    yx_patch_size=(512, 512), # larger patch size makes it easy to see augmentations.
    augment = False # Turn off augmentation for now.
)
data_module.setup("fit")

print(
    f"FOVs in training set: {len(data_module.train_dataset)}, FOVs in validation set:{len(data_module.val_dataset)}"
)

# %% Define a function to write a batch to tensorboard log.
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
    batch_phase = batch['source'][:,:,0,:,:]  # batch_size x z_size x Y x X tensor. 
    batch_membrane = batch['target'][:,1,0,:,:].unsqueeze(1) # batch_size x 1 x Y x X tensor.
    batch_nuclei = batch['target'][:,0,0,:,:].unsqueeze(1) # batch_size x 1 x Y x X tensor.

    p1, p99 = np.percentile(batch_membrane, (0.1, 99.9))
    batch_membrane = np.clip((batch_membrane - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_nuclei,  (0.1, 99.9))
    batch_nuclei = np.clip((batch_nuclei - p1) / (p99 - p1), 0, 1)

    p1, p99 = np.percentile(batch_phase,  (0.1, 99.9))
    batch_phase = np.clip((batch_phase - p1) / (p99 - p1), 0, 1)
    
    [N,C,H,W] = batch_phase.shape
    interleaved_images = torch.zeros((3*N,C,H,W),dtype = batch_phase.dtype)
    interleaved_images[0::3,:]=batch_phase
    interleaved_images[1::3,:]=batch_nuclei
    interleaved_images[2::3,:]=batch_membrane

    grid=torchvision.utils.make_grid(interleaved_images, nrow=3)

    # add the grid to tensorboard
    writer.add_image(card_name, grid, batchno)
    

# %% Log a batch and an epoch to tensorboard.

writer = SummaryWriter(log_dir = f"{log_dir}/view_batch")
train_dataloader = data_module.train_dataloader()

# Draw a batch and write to tensorboard. 
batch = next(iter(train_dataloader))
log_batch_tensorboard(batch, 0, writer, "augmentation/none")

for i, batch in enumerate(train_dataloader):
    log_batch_tensorboard(batch, i, writer, "augmentation/none")
writer.close()

# %% Use the following if you need to bring up the tensorboard session again
# notebook.list()
# notebook.display(port=6006, height=800)

# %% [markdown]
"""
## View augmentations using tensorboard.

<div class="alert alert-info">
Task 1.3
Turn on augmentation and view the batch in tensorboard.
"""
# %% 
# data_module.augment = ...
# data_module.batch_size = ...
# ... # Feel free to adjust a few other parameters of data_module.
# data_module.setup("fit")
# train_dataloader = data_module.train_dataloader()
# # Draw batches and write to tensorboard
# writer = SummaryWriter(log_dir = f"{log_dir}/view_batch")
# for i, batch in enumerate(train_dataloader):
#     log_batch_tensorboard(...)
# writer.close()

# %% tags=["solution"]
data_module.augment = True
data_module.batch_size = 21
data_module.split_ratio = 0.8
data_module.setup("fit")

train_dataloader = data_module.train_dataloader()
# Draw batches and write to tensorboard
writer = SummaryWriter(log_dir = f"{log_dir}/view_batch")
for i, batch in enumerate(train_dataloader):
    log_batch_tensorboard(batch, i, writer, "augmentation/some")
writer.close()

# %% [markdown]
"""
##  Construct a 2D U-Net for image translation.
See ``viscy.unet.networks.Unet2D.Unet2d`` for configuration details.
We setup a fresh data module and instantiate the trainer class.
"""

# %% The entire training loop is contained in this block.

GPU_ID = 0
BATCH_SIZE = 32


# Dictionary that specifies key parameters of the model.
model_config = {
    "architecture": "2D",
    "in_channels": 1,
    "out_channels": 2,
    "residual": True,
    "dropout": 0.1, # dropout randomly turns off weights to avoid overfitting of the model to data.
    "task": "reg", # reg = regression task.
}

model = VSUNet(
    model_config=model_config.copy(),
    batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.mse_loss,
    schedule="WarmupCosine",
    log_num_samples=10,
)

# Reinitialize the data module. 
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
    augment = True 
)
data_module.setup("fit")
# fast_dev_run runs a single batch of data through the model to check for errors.
trainer = VSTrainer(accelerator="gpu", devices=[GPU_ID], fast_dev_run=True)

# trainer class takes the model and the data module as inputs.
trainer.fit(model, datamodule=data_module)

# %% [markdown]
"""
<div class="alert alert-info">
Task 1.4
Setup the training for ~50 epochs
</div>

Tips:
- Set ``default_root_dir`` to store the logs and checkpoints
in a specific directory.
"""

# %% tags=["solution"]
wider_config = model_config | {"num_filters": [24, 48, 96, 192, 384]}

model = model = VSUNet(
    model_config=wider_config.copy(),
    batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.mse_loss, # mean square error.
    schedule="WarmupCosine",
    log_num_samples=10,
)

n_samples = len(data_module.train_dataset)
steps_per_epoch = n_samples // BATCH_SIZE
n_epochs = 50

trainer = VSTrainer(
    accelerator="gpu",
    max_epochs=n_epochs,
    log_every_n_steps= steps_per_epoch,
    default_root_dir=Path(log_dir,"phase2fluor"),
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

# %% 

# visualize graph. 
model_graph = torchview.draw_graph(model, data_module.train_dataset[0]['source'], depth=2, device="cpu")
# Increase the depth to zoom in.

graph = model_graph.visual_graph
graph

# %% tags = ["solution"]

GPU_ID = 0
BATCH_SIZE = 32

data_module = HCSDataModule(
    data_path,
    source_channel="Nuclei",
    target_channel="Phase",
    z_window_size=1,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    architecture="2D",
    yx_patch_size=(256, 256), 
    augment = True 
)
data_module.setup("fit")


# Dictionary that specifies key parameters of the model.
model_config = {
    "architecture": "2D",
    "in_channels": 1,
    "out_channels": 1,
    "residual": True,
    "dropout": 0.1, # dropout randomly turns off weights to avoid overfitting of the model to data.
    "task": "reg", # reg = regression task.
}

wider_config = model_config | {"num_filters": [24, 48, 96, 192, 384]}


model = VSUNet(
    model_config=wider_config.copy(),
    batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.mae_loss,
    schedule="WarmupCosine",
    log_num_samples=10,
)



n_samples = len(data_module.train_dataset)
steps_per_epoch = n_samples // BATCH_SIZE
n_epochs = 50

trainer = VSTrainer(
    accelerator="gpu",
    max_epochs=n_epochs,
    log_every_n_steps= steps_per_epoch,
    default_root_dir=Path(log_dir,"fluor2phase"),
)

trainer.fit(model, datamodule=data_module)

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

- Tweak model hyperparameters, primarily depth
- Adjust batch size to fully utilize the VRAM
"""

# %% [markdown]
"""
<div class="alert alert-success">
Checkpoint 3

Congratulations! You have trained several image translation models now!
Please summarize hyperparameters and performance of your models in [this google doc](https://docs.google.com/document/d/1hZWSVRvt9KJEdYu7ib-vFBqAVQRYL8cWaP_vFznu7D8/edit#heading=h.n5u485pmzv2z)
</div>
"""
# %% 

# data_module_augmented = HCSDataModule(
#     data_path,
#     source_channel="Phase",
#     target_channel=["Nuclei", "Membrane"],
#     z_window_size=1,
#     split_ratio=0.8,
#     batch_size=21,
#     num_workers=8,
#     architecture="2D",
#     yx_patch_size=(512, 512),
#     augment = True # Turn on augmentations.
# )

# data_module_augmented.setup("fit")