# %%
import torch
import lightning.pytorch as pl
import torch.nn as nn

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from viscy.transforms import RandWeightedCropd, NormalizeSampled, RandScaleIntensityd, RandGaussianSmoothd
from viscy.data.hcs import HCSDataModule
from viscy.scripts.infection_phenotyping.classify_infection_2D import SemanticSegUNet2D

from iohub.ngff import open_ome_zarr

# %% Create a dataloader and visualize the batches.

# Set the path to the dataset
dataset_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/4-human_annotation/train_data.zarr"

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

# %%
# Create an instance of HCSDataModule
data_module = HCSDataModule(
    dataset_path,
    source_channel=["TXR_Density3D", "Phase3D"],
    target_channel=["Inf_mask"],
    yx_patch_size=[128, 128],
    split_ratio=0.7,
    z_window_size=1,
    architecture="2D",
    num_workers=1,
    batch_size=256,
    normalizations=[
        NormalizeSampled(
            keys=["Phase3D", "TXR_Density3D"],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
    augmentations=[
        RandWeightedCropd(
            num_samples=16,
            spatial_size=[-1, 128, 128],
            keys=["TXR_Density3D", "Phase3D", "Inf_mask"],
            w_key="Inf_mask",
        ),
        RandScaleIntensityd(
            keys=["TXR_Density3D", "Phase3D"],
            factors=[0.5, 0.5],
            prob=0.5,
        ),
        RandGaussianSmoothd(
            keys=["TXR_Density3D", "Phase3D"],
            prob=0.5,
            sigma_x=[0.5, 1.0],
            sigma_y=[0.5, 1.0],
            sigma_z=[0.5, 1.0],
        ),
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
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/5-infection_classifier/0-model_training/",
    name="logs",
)

# Pass the logger to the Trainer
trainer = pl.Trainer(
    logger=logger,
    max_epochs=500,
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/5-infection_classifier/0-model_training/logs/",
    log_every_n_steps=1,
    devices=1,  # Set the number of GPUs to use. This avoids run-time exception from distributed training when the node has multiple GPUs
)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/5-infection_classifier/0-model_training/logs/",
    filename="checkpoint_{epoch:02d}",
    save_top_k=-1,
    verbose=True,
    monitor="loss/validate",
    mode="min",
)

# Add the checkpoint callback to the trainer
trainer.callbacks.append(checkpoint_callback)

# Fit the model
model = SemanticSegUNet2D(
    in_channels=2,
    out_channels=3,
    loss_function=nn.CrossEntropyLoss(weight=torch.tensor(pixel_ratio)),
)

print(model)
# %%
# Run training.

trainer.fit(model, data_module)

# %%
