
# %%
import torch
from viscy.data.hcs import HCSDataModule

import numpy as np
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F

import napari
from pytorch_lightning.loggers import TensorBoardLogger
from monai.transforms import RandRotate, Resize, Zoom, Flip, RandFlip, RandZoom, RandRotate90, RandRotate, RandAffine, Rand2DElastic, Rand3DElastic, RandGaussianNoise, RandGaussianNoised
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.losses import DiceLoss
from viscy.light.engine import VSUNet

# %% Create a dataloader and visualize the batches.
# Set the path to the dataset
dataset_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2023_09_28_DENV_A2_infMarked.zarr"

# Create an instance of HCSDataModule
data_module = HCSDataModule(
    dataset_path, 
    source_channel=['Sensor','Nucl_mask'], 
    target_channel=['Inf_mask'],
    yx_patch_size=[128,128], 
    split_ratio=0.8, 
    z_window_size=1, 
    architecture = '2D',
    num_workers=1,
    batch_size=12,
    augmentations=[],
)

# Prepare the data
data_module.prepare_data()

# Setup the data
data_module.setup(stage = "fit")

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

# %% use 2D Unet from viscy with a softmax layer at end for 4 label classification
# use for image translation from instance segmentation to annotated image
    
# load 2D UNet from viscy
unet_model = VSUNet(
    architecture='2D', 
    model_config={"in_channels": 2, "out_channels": 4, "task": "reg"}, 
    lr=1e-3,
)

# Define the optimizer
optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-3)

#%% Iterate over the batches
for batch in train_dm:
    # Extract the input and target from the batch
    input_data, target = batch['source'], batch['target']
    # viewer.add_image(input_data.cpu().numpy().astype(np.float32))

    # Forward pass through the model
    output = unet_model(input_data)

    # Calculate the loss
    loss = DiceLoss()(output, target)

    # Perform backpropagation and update the model's parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

for batch in val_dm:
    # Extract the input and target from the batch
    input_data, target = batch['source'], batch['target']

    # Forward pass through the model
    output = unet_model(input_data)

    # Calculate the loss
    loss = DiceLoss()(output, target)


# Visualize sample of the augmented data using napari
# for i in range(augmented_input.shape[0]):
#     viewer.add_image(augmented_input[i].cpu().numpy().astype(np.float32))


#%% use the batch for training the unet model using the lightning module
    
# Train the model
# Create a TensorBoard logger
class LightningUNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(LightningUNet, self).__init__()
        self.unet_model = UNet(in_channels, out_channels)

    def forward(self, x):
        return self.unet_model(x)

    def training_step(self, batch, batch_idx):
        input_data, target = batch['source'], batch['target']
        output = self(input_data)
        loss = DiceLoss()(output, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data, target = batch['source'], batch['target']
        output = self(input_data)
        loss = DiceLoss()(output, target)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Create an instance of the LightningUNet class
unet_model = LightningUNet(in_channels=2, out_channels=1)

# Define the logger
logger = TensorBoardLogger("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/Infection_phenotyping_data/logs", name="infection_classification_model")

# Pass the logger to the Trainer
trainer = pl.Trainer(logger=logger, max_epochs=30, default_root_dir="/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/Infection_phenotyping_data/logs", log_every_n_steps=1)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/Infection_phenotyping_data/checkpoints',
    filename='checkpoint_{epoch:02d}',
    save_top_k=-1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

# Add the checkpoint callback to the trainer
trainer.callbacks.append(checkpoint_callback)

# Fit the model
trainer.fit(unet_model, data_module)

# %% test the model on the test set
test_datapath = '/hpc/projects/intracellular_dashboard/viral-sensor/2023_12_08-BJ5a-calibration/5_classify/2023_12_08_BJ5a_pAL040_72HPI_Calibration_1.zarr'

test_dm = HCSDataModule(
    test_datapath, 
    source_channel=['Sensor','Nuclei_mask'],
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
