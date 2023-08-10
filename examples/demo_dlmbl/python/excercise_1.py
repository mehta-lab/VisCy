# %% [markdown]
"""
# Image translation excercise part 1

In this exercise, we will solve an image translation task of
reconstructing nuclei and membrane markers from phase images of cells.
Here, the source domain is label-free microscopy (average material density),
and the target domain is fluorescence microscopy (fluorophore density).

Learning goals of part 1:

- Load the and visualize the images from OME-Zarr
- Configure the data loaders
- Initialize a 2D U-Net model for virtual staining


<div class="alert alert-danger">
Set your python kernel to <code>004-image-translation</code>
</div>
"""

# %%
import matplotlib.pyplot as plt
import torch
from iohub import open_ome_zarr
from tensorboard import notebook
from torchview import draw_graph

from viscy.light.data import HCSDataModule
from viscy.light.engine import VSTrainer, VSUNet

BATCH_SIZE = 32

# %% [markdown]
"""
Load Dataset.

<div class="alert alert-info">
Task 1.1

Use <a href=https://czbiohub-sf.github.io/iohub/main/api/ngff.html#open-ome-zarr>
<code>iohub.open_ome_zarr</code></a> to read the dataset.

There should be 301 FOVs in the dataset (9.3 GB compressed).

Each FOV consists of 3 channels of 2048x2048 images,
saved in the <a href="https://ngff.openmicroscopy.org/latest/#hcs-layout">
High-Content Screening (HCS) layout</a>
specified by the Open Microscopy Environment Next Generation File Format
(OME-NGFF).

Run <code>open_ome_zarr?</code> in a cell to see the docstring.

"""

# %%
# set dataset path here
data_path = (
    "/hpc/projects/comp.micro/virtual_staining/datasets/dlmbl/HEK_nuclei_membrane.zarr"
)

dataset = open_ome_zarr(data_path)

print(len(list(dataset.positions())))


# %% [markdown]
"""
View images with matplotlib.

The layout on the disk is: row/col/field/resolution/timepoint/channel/z/y/x.


Note that labelling is not perfect,
as some cells are not expressing the fluorophore.
"""

# %%

row = "0"
col = "0"
field = "0"
resolution = "0" # 0 is the highest resolution, 1 is 2x2 binned, 2 is 4x4 binned, etc.
image = dataset[f'{row}/{col}/{field}/{resolution}'].numpy()
print(image.shape)

figure, axes = plt.subplots(1, 3, figsize=(9, 3))

for ax, channel in zip(axes, image[0, :, 0]):
    ax.imshow(channel, cmap="gray")
    ax.axis("off")

plt.tight_layout()

# %% [markdown]
"""
Configure the data loaders for training and validation.
"""

# %%
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
<div class="alert alert-info">
Task 1.2

Validate that the data can be loaded in batches correctly.
</div>
"""

# %%
train_dataloader = data_module.train_dataloader()

for i, batch in enumerate(train_dataloader):
    ...
    # check some batches and break
    break

# %% tags=["solution"]
train_dataloader = data_module.train_dataloader()

for i, batch in enumerate(train_dataloader):
    print(f"Batch {i}:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            summary = (v.shape, v.dtype)
        else:
            summary = v
        print(k, summary)
    if i > 2:
        break

# %% [markdown]
"""
Construct a 2D U-Net for image translation.

See ``viscy.unet.networks.Unet2D.Unet2d`` for configuration details.
Increase the ``depth`` in ``draw_graph`` to zoom in.
"""

# %%
model_config = {
    "architecture": "2D",
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
model_graph = draw_graph(model, model.example_input_array, depth=2)
graph = model_graph.visual_graph
graph

# %% [markdown]
"""
Configure trainer class.
Here we use the ``fast_dev_run`` flag to run a sanity check first.
"""

# %%
trainer = VSTrainer(accelerator="gpu", fast_dev_run=True)

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
    accelerator="gpu", max_epochs=20, log_every_n_steps=10, default_root_dir=""
)

trainer.fit(model, datamodule=data_module)

# %% [markdown]
"""
Launch TensorBoard with:

```
%load_ext tensorboard
%tensorboard --logdir /path/to/lightning_logs
```
"""

# %%
notebook.list()

# %%
notebook.display(port=6006, height=800)

# %% [markdown]
"""
<div class="alert alert-success">
Checkpoint 1

Now the training has started,
we can come back after a while and evaluate the performance!
</div>
"""

# %%
