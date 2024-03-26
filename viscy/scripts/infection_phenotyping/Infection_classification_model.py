# %%
import torch
from viscy.data.hcs import HCSDataModule

import numpy as np
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics

# import torchview
from typing import Literal, Sequence
from skimage.exposure import rescale_intensity
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.cm import get_cmap

# import napari
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from pytorch_lightning.callbacks import ModelCheckpoint

# from monai.losses import DiceLoss
from monai.transforms import DivisiblePad

# from viscy.light.engine import VSUNet
from viscy.unet.networks.Unet2D import Unet2d
from viscy.data.hcs import Sample
from viscy.transforms import RandWeightedCropd, RandGaussianNoised
from viscy.transforms import NormalizeSampled

# %% Create a dataloader and visualize the batches.

# Set the path to the dataset
dataset_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2024_02_13_DENV_3infMarked_trainVal.zarr"

# Create an instance of HCSDataModule
data_module = HCSDataModule(
    dataset_path,
    source_channel=["Sensor", "Phase"],
    target_channel=["Inf_mask"],
    yx_patch_size=[128, 128],
    split_ratio=0.8,
    z_window_size=1,
    architecture="2D",
    num_workers=1,
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

# %%

# Define a 2D UNet model for semantic segmentation as a lightning module.


class SemanticSegUNet2D(pl.LightningModule):
    # Model for semantic segmentation.
    def __init__(
        self,
        in_channels: int,  # Number of input channels
        out_channels: int,  # Number of output channels
        lr: float = 1e-3,  # Learning rate
        loss_function: nn.Module = nn.CrossEntropyLoss(),  # Loss function
        schedule: Literal[
            "WarmupCosine", "Constant"
        ] = "Constant",  # Learning rate schedule
        log_batches_per_epoch: int = 2,  # Number of batches to log per epoch
        log_samples_per_batch: int = 2,  # Number of samples to log per batch
        checkpoint_path: str = None,  # Path to the checkpoint
    ):
        super(SemanticSegUNet2D, self).__init__()  # Call the superclass initializer
        # Initialize the UNet model
        self.unet_model = Unet2d(in_channels=in_channels, out_channels=out_channels)
        self.lr = lr  # Set the learning rate
        # Set the loss function to CrossEntropyLoss if none is provided
        self.loss_function = loss_function if loss_function else nn.CrossEntropyLoss()
        self.schedule = schedule  # Set the learning rate schedule
        self.log_batches_per_epoch = (
            log_batches_per_epoch  # Set the number of batches to log per epoch
        )
        self.log_samples_per_batch = (
            log_samples_per_batch  # Set the number of samples to log per batch
        )
        self.training_step_outputs = []  # Initialize the list of training step outputs
        self.validation_step_outputs = (
            []
        )  # Initialize the list of validation step outputs

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
            state_dict.pop("loss_function.weight", None)  # Remove the unexpected key
            self.load_state_dict(state_dict)  # loading only weights

    # Define the forward pass
    def forward(self, x):
        return self.unet_model(x)  # Pass the input through the UNet model

    # Define the optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr
        )  # Use the Adam optimizer
        return optimizer

    # Define the training step
    def training_step(self, batch: Sample, batch_idx: int):
        source = batch["source"]  # Extract the source from the batch
        target = batch["target"]  # Extract the target from the batch
        pred = self.forward(source)  # Make a prediction using the source
        # Convert the target to one-hot encoding
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=3).permute(
            0, 4, 1, 2, 3
        )
        target_one_hot = target_one_hot.float()  # Convert the target to float type
        train_loss = self.loss_function(pred, target_one_hot)  # Calculate the loss
        # Log the training step outputs if the batch index is less than the number of batches to log per epoch
        if batch_idx < self.log_batches_per_epoch:
            self.training_step_outputs.extend(
                self._detach_sample((source, target_one_hot, pred))
            )
        # Log the training loss
        self.log(
            "loss/train",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return train_loss  # Return the training loss

    def validation_step(self, batch: Sample, batch_idx: int):
        source = batch["source"]  # Extract the source from the batch
        target = batch["target"]  # Extract the target from the batch
        pred = self.forward(source)  # Make a prediction using the source
        # Convert the target to one-hot encoding
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=3).permute(
            0, 4, 1, 2, 3
        )
        target_one_hot = target_one_hot.float()  # Convert the target to float type
        loss = self.loss_function(pred, target_one_hot)  # Calculate the loss
        # Log the validation step outputs if the batch index is less than the number of batches to log per epoch
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(
                self._detach_sample((source, target_one_hot, pred))
            )
        # Log the validation loss
        self.log(
            "loss/validate", loss, sync_dist=True, add_dataloader_idx=False, logger=True
        )
        return loss  # Return the validation loss

    def on_predict_start(self):
        """Pad the input shape to be divisible by the downsampling factor.
        The inverse of this transform crops the prediction to original shape.
        """
        down_factor = 2**self.unet_model.num_blocks
        self._predict_pad = DivisiblePad((0, 0, down_factor, down_factor))

    # Define the prediction step
    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        source = self._predict_pad(batch["source"])  # Pad the source
        logits = self._predict_pad.inverse(
            self.forward(source)
        )  # Predict and remove padding.
        prob_map = F.softmax(logits, dim=1)  # Calculate the probabilities
        return prob_map  # return the probabilities for computing metrics.

    # Define what happens at the end of a training epoch
    def on_train_epoch_end(self):
        self._log_samples(
            "train_samples", self.training_step_outputs
        )  # Log the training samples
        self.training_step_outputs = []  # Reset the list of training step outputs

    # Define what happens at the end of a validation epoch
    def on_validation_epoch_end(self):
        self._log_samples(
            "val_samples", self.validation_step_outputs
        )  # Log the validation samples
        self.validation_step_outputs = []  # Reset the list of validation step outputs
        # TODO: Log the confusion matrix

    # Define a method to detach a sample
    def _detach_sample(self, imgs: Sequence[Tensor]):
        # Detach the images and convert them to numpy arrays
        num_samples = 3
        return [
            [np.squeeze(img[i].detach().cpu().numpy().max(axis=1)) for img in imgs]
            for i in range(num_samples)
        ]

    # Define a method to log samples
    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        images_grid = []  # Initialize the list of image grids
        for sample_images in imgs:  # For each sample image
            images_row = []  # Initialize the list of image rows
            for i, image in enumerate(
                sample_images
            ):  # For each image in the sample images
                cm_name = "gray" if i == 0 else "inferno"  # Set the colormap name
                if image.ndim == 2:  # If the image is 2D
                    image = image[np.newaxis]  # Add a new axis
                for channel in image:  # For each channel in the image
                    channel = rescale_intensity(
                        channel, out_range=(0, 1)
                    )  # Rescale the intensity of the channel
                    render = get_cmap(cm_name)(channel, bytes=True)[
                        ..., :3
                    ]  # Render the channel
                    images_row.append(
                        render
                    )  # Append the render to the list of image rows
            images_grid.append(
                np.concatenate(images_row, axis=1)
            )  # Append the concatenated image rows to the list of image grids
        grid = np.concatenate(images_grid, axis=0)  # Concatenate the image grids
        # Log the image grid
        self.logger.experiment.add_image(
            key, grid, self.current_epoch, dataformats="HWC"
        )


# %% Define the logger
logger = TensorBoardLogger(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/",
    name="logs_wPhase",
)

# Pass the logger to the Trainer
trainer = pl.Trainer(
    logger=logger,
    max_epochs=100,
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase",
    log_every_n_steps=1,
    devices=1,  # Set the number of GPUs to use. This avoids run-time exception from distributed training when the node has multiple GPUs
)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase/",
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
    loss_function=nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.25, 0.7])),
)

print(model)
# %%
# Run training.

trainer.fit(model, data_module)

# %%
