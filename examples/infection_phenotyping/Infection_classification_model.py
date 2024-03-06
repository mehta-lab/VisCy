
# %%
import torch
from viscy.data.hcs import HCSDataModule

import numpy as np
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F

import napari
from pytorch_lightning.loggers import TensorBoardLogger
from monai.transforms import Compose, RandRotate, Resize, Zoom, Flip, RandFlip, RandZoom, RandRotate90, RandRotate, RandAffine, Rand2DElastic, Rand3DElastic, RandGaussianNoise, RandGaussianNoised

# %% Create a dataloader and visualize the batches.
# Set the path to the dataset
dataset_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2023_09_28_DENV_A2_infMarked.zarr"

# Create an instance of HCSDataModule
data_module = HCSDataModule(dataset_path, source_channel=['Sensor','Nucl_mask'], target_channel=['Inf_mask'],yx_patch_size=[128,128], split_ratio=0.8, z_window_size=1, architecture = '2D')

# Prepare the data
data_module.prepare_data()

# Setup the data
data_module.setup(stage = "fit")

# Create a dataloader
dataloader = data_module.train_dataloader()

# Visualize the dataset and the batch using napari
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

# load 2D UNet from viscy
# unet_model = VSUNet(
#     architecture='2D', 
#     model_config={"in_channels": 2, "out_channels": 1},
#     loss_function=dice_loss, 
#     lr=1e-3, 
#     example_input_xy_shape=(128,128),
#     )

# Define the data augmentations
# Define the augmentations
# transforms = Compose([
#     RandRotate(range_x=15, prob=0.5),
#     Resize(spatial_size=[64, 64],mode='linear'),
#     Zoom([0.5,2], mode='bilinear'),
#     Flip(spatial_axis=[0,1]),
#     RandFlip(spatial_axis=[0,1], prob=0.5),
#     RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
#     RandRotate90(spatial_axes=(0,1), prob=0.2, max_k=3),
#     RandGaussianNoise(prob=0.5),
# ])

transforms = Compose([
    RandRotate(range_x=15, prob=0.5),
    Flip(spatial_axis=[0,1]),
    RandFlip(spatial_axis=[0,1], prob=0.5),
    RandRotate90(spatial_axes=(0,1), prob=0.2, max_k=3),
    RandGaussianNoise(prob=0.5),
])

# create a small unet for image translation which accepts two input images (a label image and a microscopy image) and outputs one label image
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Define the decoder part of the U-Net architecture
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.Softmax(dim=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Apply the encoder to the input
        x = self.encoder(x)

        # Apply the decoder to the output of the encoder
        x = self.decoder(x)

        return x

unet_model = UNet(in_channels=2, out_channels=1)

# Define the optimizer
optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-3)

#%% Iterate over the batches
for batch in dataloader:
    # Extract the input and target from the batch
    input_data, target = batch['source'], batch['target']

    # Apply the augmentations to your data
    augmented_input = transforms(input_data)
    viewer.add_image(augmented_input.cpu().numpy().astype(np.float32))

    # Forward pass through the model
    output = unet_model(augmented_input)

    # Apply softmax activation to the output
    output = F.softmax(output, dim=1)

    # Calculate the loss
    loss = dice_loss(output, target)

    # Perform backpropagation and update the model's parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Visualize sample of the augmented data using napari
# for i in range(augmented_input.shape[0]):
#     viewer.add_image(augmented_input[i].cpu().numpy().astype(np.float32))


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
# use iohub or viscy to save the predictions!!!
zarr.save('predictions.zarr', predictions)
