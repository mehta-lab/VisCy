
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchview
import torchvision
from iohub import open_ome_zarr
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

# pytorch lightning wrapper for Tensorboard.
from tensorboard import notebook  # for viewing tensorboard in notebook
from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard

# HCSDataModule makes it easy to load data during training.
from viscy.light.data import HCSDataModule

# Trainer class and UNet.
from viscy.light.engine import VSTrainer, VSUNet



# SEED AND GET PATHS
seed_everything(7, workers=True)

# Paths to data and log directory
data_path = Path(
    Path("/home/ankitr/MBL-Project/Data/cropped_dataset_v3.zarr")
)

log_dir = Path("/home/ankitr/MBL-Project/Logs")

# Create log directory if needed, and launch tensorboard
log_dir.mkdir(parents=True, exist_ok=True)



# OPEN IMAGE
dataset = open_ome_zarr(data_path)

print(f"Number of positions: {len(list(dataset.positions()))}")



# TENSORBOARD WRITING FUNCTION
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



# SET MODEL PARAMETERS
# The entire training loop is contained in this cell.
GPU_ID = 0
BATCH_SIZE = 6
YX_PATCH_SIZE = (512, 512)


# Dictionary that specifies key parameters of the model.
phase2fluor_config_21D_AR = {
    # "architecture": "2.5D",
    "num_filters": [24, 48, 96, 192, 384],
    "in_channels": 1,
    "out_channels": 2,
    "in_stack_depth": 5,
    "residual": True,
    "dropout": 0.1,  # dropout randomly turns off weights to avoid overfitting of the model to data.
    "task": "reg",  # reg = regression task.
}

phase2fluor_model_21D_AR = VSUNet(
    "2.5D",
    model_config=phase2fluor_config_21D_AR.copy(),
    # batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.l1_loss,
    schedule="WarmupCosine",
    log_num_samples=3,  # Number of samples from each batch to log to tensorboard.
    example_input_yx_shape=YX_PATCH_SIZE,
    lr=2e-4,
)

# Reinitialize the data module.
phase2fluor_data_21D_AR = HCSDataModule(
    data_path,
    source_channel="Phase3D",
    target_channel=["GFP EX488 EM525-45", "mCherry EX561 EM600-37"],
    z_window_size=5,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    architecture="21D_AR",
    yx_patch_size=YX_PATCH_SIZE,
    augment=True,
    normalize_source=True,
    train_noise_std=2.0,
    train_patches_per_stack=2,
)
phase2fluor_data_21D_AR.setup("fit")



# TRAIN MODEL
GPU_ID = 0
n_samples = len(phase2fluor_data_21D_AR.train_dataset)
steps_per_epoch = n_samples // BATCH_SIZE  # steps per epoch.
n_epochs = 50

trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    max_epochs=n_epochs,
    log_every_n_steps=steps_per_epoch // 2,
    # log losses and image samples 2 times per epoch.
    default_root_dir=Path(
        log_dir, "phase2fluor"
    ),  # lightning trainer transparently saves logs and model checkpoints in this directory.
)

# Log graph
trainer.logger.log_graph(phase2fluor_model_21D_AR, phase2fluor_data_21D_AR.train_dataset[0]["source"])
# Launch training.
trainer.fit(phase2fluor_model_21D_AR, datamodule=phase2fluor_data_21D_AR)