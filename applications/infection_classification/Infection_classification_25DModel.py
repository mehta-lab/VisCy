# %%
import torch
import lightning.pytorch as pl
import torch.nn as nn

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from viscy.transforms import RandWeightedCropd
from viscy.transforms import NormalizeSampled
from viscy.data.hcs import HCSDataModule
from viscy.scripts.infection_phenotyping.classify_infection_25D import SemanticSegUNet25D

from iohub.ngff import open_ome_zarr

# %% Create a dataloader and visualize the batches.

# Set the path to the dataset
dataset_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2023_11_08_Opencell_infection/OC43_infection_timelapse_trainVal.zarr"

# find ratio of background, uninfected and infected pixels
zarr_input = open_ome_zarr(
    dataset_path,
    layout="hcs",
    mode="r+",
)
in_chan_names = zarr_input.channel_names

num_pixels_bkg = 0
num_pixels_uninf = 0
num_pixels_inf = 0
num_pixels = 0
for well_id, well_data in zarr_input.wells():
    well_name, well_no = well_id.split("/")

    for pos_name, pos_data in well_data.positions():
        data = pos_data.data
        T,C,Z,Y,X = data.shape
        out_data = data.numpy()
        for time in range(T):
            Inf_mask = out_data[time,in_chan_names.index("Inf_mask"),...]
            # Calculate the number of pixels valued 0, 1, and 2 in 'Inf_mask'
            num_pixels_bkg = num_pixels_bkg + (Inf_mask == 0).sum()
            num_pixels_uninf = num_pixels_uninf + (Inf_mask == 1).sum()
            num_pixels_inf = num_pixels_inf + (Inf_mask == 2).sum()
            num_pixels = num_pixels + Z*X*Y

pixel_ratio_1 = [num_pixels/num_pixels_bkg, num_pixels/num_pixels_uninf, num_pixels/num_pixels_inf]
pixel_ratio_sum = sum(pixel_ratio_1)
pixel_ratio = [ratio / pixel_ratio_sum for ratio in pixel_ratio_1]

# %% craete data module

# Create an instance of HCSDataModule
data_module = HCSDataModule(
    dataset_path,
    source_channel=["Phase", "HSP90"],
    target_channel=["Inf_mask"],
    yx_patch_size=[512, 512],
    split_ratio=0.8,
    z_window_size=5,
    architecture="2.5D",
    num_workers=3,
    batch_size=32,
    normalizations=[
        NormalizeSampled(
            keys=["Phase","HSP90"],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
    augmentations=[
        RandWeightedCropd(
            num_samples=4,
            spatial_size=[-1, 512, 512],
            keys=["Phase","HSP90"],
            w_key="Inf_mask",
        )
    ],
)

# Prepare the data
data_module.prepare_data()

# Setup the data
data_module.setup(stage="fit")

# Create a dataloader
train_dm = data_module.train_dataloader()

val_dm = data_module.val_dataloader()

# Visualize the dataset and the batch using napari
# Set the display
# os.environ['DISPLAY'] = ':1'

# # Create a napari viewer
# viewer = napari.Viewer()

# # Add the dataset to the viewer
# for batch in dataloader:
#     if isinstance(batch, dict):
#         for k, v in batch.items():
#             if isinstance(v, torch.Tensor):
#                 viewer.add_image(v.cpu().numpy().astype(np.float32))

# # Start the napari event loop
# napari.run()


# %% Define the logger
logger = TensorBoardLogger(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/mantis_phase_hsp90/",
    name="logs",
)

# Pass the logger to the Trainer
trainer = pl.Trainer(
    logger=logger,
    max_epochs=200,
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/mantis_phase_hsp90/logs/",
    log_every_n_steps=1,
    devices=1,  # Set the number of GPUs to use. This avoids run-time exception from distributed training when the node has multiple GPUs
)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/mantis_phase_hsp90/logs/",
    filename="checkpoint_{epoch:02d}",
    save_top_k=-1,
    verbose=True,
    monitor="loss/validate",
    mode="min",
)

# Add the checkpoint callback to the trainer
trainer.callbacks.append(checkpoint_callback)

# Fit the model
model = SemanticSegUNet25D(
    in_channels=2,
    out_channels=3,
    loss_function=nn.CrossEntropyLoss(weight=torch.tensor(pixel_ratio)),
)

print(model)

# %%
# Run training.

trainer.fit(model, data_module)

# %%
