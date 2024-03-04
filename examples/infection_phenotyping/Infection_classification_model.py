
# %%
import torch
import sys
from viscy.data.hcs import HCSDataModule
# from lightning.pytorch import CustomDataset
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningDataModule
# import cv2
import numpy as np
import torch.nn as nn
import torchvision.models as models
import lightning.pytorch as pl
import torch.nn.functional as F
from viscy.light.engine import VSUNet

# %% Create a dataloader and visualize the batches.
# Set the path to the dataset
dataset_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2023_09_28_DENV_A2_infMarked.zarr"

# Create an instance of HCSDataModule
data_module = HCSDataModule(dataset_path, source_channel=['Sensor','Nucl_mask'], target_channel=['Inf_mask'],yx_patch_size=[256, 256], split_ratio=0.8, z_window_size=1, architecture = '2D')

# Prepare the data
data_module.prepare_data()

# Setup the data
data_module.setup(stage = "fit")

# Create a dataloader
dataloader = data_module.train_dataloader()

# Visualize the dataset and the batch using napari
import napari
from pytorch_lightning.loggers import TensorBoardLogger
# import os

# Set the display
# os.environ['DISPLAY'] = ':1'

# Create a napari viewer
viewer = napari.Viewer()

# Add the dataset to the viewer
for batch in dataloader:
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                viewer.add_image(v.cpu().numpy().astype(np.float32))

# Start the napari event loop
napari.run()

# %% use 2D Unet from viscy with a softmax layer at end for 4 label classification
# use for image translation from instance segmentation to annotated image

# use diceloss function from here: https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector. This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

unet_model = VSUNet(
    architecture='2D', 
    loss_function=dice_loss, 
    lr=1e-3, 
    example_input_xy_shape=(64,64)
    )

# Define the optimizer
optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-3)

# Iterate over the batches
for batch in dataloader:
    # Extract the input and target from the batch
    input_data, target = batch['source'], batch['target']

    # Forward pass through the model
    output = unet_model(input_data)

    # Apply softmax activation to the output
    output = F.softmax(output, dim=1)

    # Calculate the loss
    loss = dice_loss(output, target)

    # Perform backpropagation and update the model's parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#%% use the batch for training the unet model using the lightning module
    
# Train the model
# Create a TensorBoard logger
logger = TensorBoardLogger("logs", name="infection_classification_model")

# Pass the logger to the Trainer
trainer = pl.Trainer(gpus=1, logger=logger)

# Fit the model
trainer.fit(unet_model, data_module)

# %% test the model on the test set
# Load the test dataset
test_dataloader = data_module.test_dataloader()

# Set the model to evaluation mode
unet_model.eval()

# Create a list to store the predictions
predictions = []

# Iterate over the test batches
for batch in test_dataloader:
    # Extract the input from the batch
    input_data = batch['source']

    # Forward pass through the model
    output = unet_model(input_data)

    # Append the predictions to the list
    predictions.append(output.detach().cpu().numpy())

# Convert the predictions to a numpy array
predictions = np.stack(predictions)

# Save the predictions as added channel in zarr format
zarr.save('predictions.zarr', predictions)
