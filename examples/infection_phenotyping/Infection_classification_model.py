# %%
import torch
from viscy.data.hcs import HCSDataModule

import numpy as np
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchview
from typing import Literal, Sequence
from skimage.exposure import rescale_intensity
from matplotlib.cm import get_cmap

# import napari
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from monai.transforms import (
    RandRotate,
    Resize,
    Zoom,
    Flip,
    RandFlip,
    RandZoom,
    RandRotate90,
    RandRotate,
    RandAffine,
    Rand2DElastic,
    Rand3DElastic,
    RandGaussianNoise,
    RandGaussianNoised,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.losses import DiceLoss
from viscy.light.engine import VSUNet
from viscy.unet.networks.Unet2D import Unet2d
from viscy.data.hcs import Sample

# %% Create a dataloader and visualize the batches.
# Set the path to the dataset
dataset_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2023_09_28_DENV_A2_infMarked.zarr"

# Create an instance of HCSDataModule
data_module = HCSDataModule(
    dataset_path,
    source_channel=["Sensor"],
    target_channel=["Inf_mask"],
    yx_patch_size=[128, 128],
    split_ratio=0.8,
    z_window_size=1,
    architecture="2D",
    num_workers=1,
    batch_size=12,
    augmentations=[],
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

# %% use 2D Unet and Lightning module


# Train the model
# Create a TensorBoard logger
class LightningUNet(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        lr: float = 1e-3,
        loss_function: nn.CrossEntropyLoss = None,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 2,
        log_samples_per_batch: int = 1,
    ):
        super(LightningUNet, self).__init__()
        self.unet_model = Unet2d(in_channels=in_channels, out_channels=out_channels)
        self.lr = lr
        self.loss_function = loss_function if loss_function else nn.CrossEntropyLoss()
        self.schedule = schedule
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.unet_model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: Sample, batch_idx: int):

        # Extract the input and target from the batch
        source = batch["source"]
        target = batch["target"]
        pred = self.forward(source)

        # Convert the target image to one-hot encoding
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=4).permute(
            0, 4, 1, 2, 3
        )
        target_one_hot = target_one_hot.float()  # Convert target to float type
        # Calculate the loss
        train_loss = self.loss_function(pred, target_one_hot)
        if batch_idx < self.log_batches_per_epoch:
            self.training_step_outputs.extend(
                self._detach_sample((source, target_one_hot, pred))
            )
        self.log(
            "loss/train",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return train_loss

    def validation_step(self, batch: Sample, batch_idx: int):

        # Extract the input and target from the batch
        source = batch["source"]
        target = batch["target"]
        pred = self.forward(source)

        # Convert the target image to one-hot encoding
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=4).permute(
            0, 4, 1, 2, 3
        )
        target_one_hot = target_one_hot.float()  # Convert target to float type
        # Calculate the loss
        loss = self.loss_function(pred, target_one_hot)
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(
                self._detach_sample((source, target, pred))
            )
        self.log(
            "loss/validate",
            loss,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return loss

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        source = self._predict_pad(batch["source"])
        return self._predict_pad.inverse(self.forward(source))

    def on_train_epoch_end(self):
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        self._log_samples("val_samples", self.validation_step_outputs)
        self.validation_step_outputs = []

    def _detach_sample(self, imgs: Sequence[Tensor]):
        num_samples = 2  # min(imgs[0].shape[0], self.log_samples_per_batch)
        return [
            [np.squeeze(img[i].detach().cpu().numpy().max(axis=1)) for img in imgs]
            for i in range(num_samples)
        ]

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        images_grid = []
        for sample_images in imgs:
            images_row = []
            for i, image in enumerate(sample_images):
                cm_name = "gray" if i == 0 else "inferno"
                if image.ndim == 2:
                    image = image[np.newaxis]
                for channel in image:
                    channel = rescale_intensity(channel, out_range=(0, 1))
                    render = get_cmap(cm_name)(channel, bytes=True)[..., :3]
                    images_row.append(render)
            images_grid.append(np.concatenate(images_row, axis=1))
        grid = np.concatenate(images_grid, axis=0)
        self.logger.experiment.add_image(
            key, grid, self.current_epoch, dataformats="HWC"
        )


# %% Define the logger
logger = TensorBoardLogger(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/Infection_phenotyping_data/logs",
    name="infection_classification_model",
)

# Pass the logger to the Trainer
trainer = pl.Trainer(
    logger=logger,
    max_epochs=50,
    default_root_dir="/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/Infection_phenotyping_data/logs",
    log_every_n_steps=1,
)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/Infection_phenotyping_data/checkpoints",
    filename="checkpoint_{epoch:02d}",
    save_top_k=-1,
    verbose=True,
    monitor="loss/validate",
    mode="min",
)

# Add the checkpoint callback to the trainer
trainer.callbacks.append(checkpoint_callback)

# Fit the model
model = LightningUNet(
    in_channels=1,
    out_channels=4,
    loss_function=nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.4, 0.4, 0.1])),
)
trainer.fit(model, data_module)


# %% test the model on the test set
test_datapath = "/hpc/projects/intracellular_dashboard/viral-sensor/2023_12_08-BJ5a-calibration/5_classify/2023_12_08_BJ5a_pAL040_72HPI_Calibration_1.zarr"

test_dm = HCSDataModule(
    test_datapath,
    source_channel=["Sensor", "Nuclei_mask"],
)
# Load the predict dataset
test_dataloader = test_dm.test_dataloader()

# Set the model to evaluation mode
unet_model.eval()

# Create a list to store the predictions
predictions = []

# Iterate over the test batches
for batch in test_dataloader:
    # Extract the input from the batch
    input_data = batch["source"]

    # Forward pass through the model
    output = unet_model(input_data)

    # Append the predictions to the list
    predictions.append(output.detach().cpu().numpy())

# Convert the predictions to a numpy array
predictions = np.stack(predictions)

# Save the predictions as added channel in zarr format
# use iohub or viscy to save the predictions!!!
zarr.save("predictions.zarr", predictions)
