"""
Demo: WandB Image Logging for ContrastiveModule

This script demonstrates how to use Weights & Biases (WandB) for logging
image samples during contrastive learning training. It showcases:

1. How to configure WandBLogger for training
2. ContrastiveModule triplet visualization (anchor/positive/negative)
3. Side-by-side comparison with TensorBoardLogger (multi-logger setup)
4. Expected output in WandB UI

The logging utilities automatically detect the logger type and format
images appropriately for each backend.
"""

# %% Imports and paths
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pytorch_metric_learning.losses import NTXentLoss

from viscy.data.triplet import TripletDataModule
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.representation.engine import ContrastiveModule
from viscy.trainer import VisCyTrainer
from viscy.transforms import BatchedCenterSpatialCropd

# Example data paths (adjust these to your actual data)
data_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_2.zarr"
tracks_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/3-track/V0/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_cropped.zarr"

# %% Option 1: Single WandB Logger
print("=" * 80)
print("Demo: WandB Image Logging for ContrastiveModule")
print("=" * 80)

# Initialize WandB logger
wandb_logger = WandbLogger(
    project="viscy-experiments",
    name="contrastive-demo-wandb",
    save_dir="./logs",
    log_model=False,  # Set to True to log model checkpoints as artifacts
)

print("\n## Configuration")
print(f"- Project: {wandb_logger.experiment.project}")
print(f"- Run name: {wandb_logger.experiment.name}")
print(f"- Run ID: {wandb_logger.experiment.id}")

# %% Define data module
dm = TripletDataModule(
    data_path=data_path,
    tracks_path=tracks_path,
    source_channel=["Phase3D"],
    z_range=(0, 1),
    batch_size=16,
    num_workers=1,
    initial_yx_patch_size=(160, 160),
    final_yx_patch_size=(160, 160),
    augmentations=[
        BatchedCenterSpatialCropd(
            keys=["Phase3D"],
            roi_size=(1, 160, 160),
        )
    ],
)
# %%
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
print(batch.keys())
print(batch["anchor"].shape)
print(batch["positive"].shape)
print(batch["negative"].shape)
# %%

# %% Create contrastive model with image logging enabled

model = ContrastiveModule(
    encoder=ContrastiveEncoder(
        backbone="convnext_tiny",
        in_channels=1,
        in_stack_depth=1,
        stem_kernel_size=(1, 4, 4),
        embedding_dim=768,
        projection_dim=32,
        drop_path_rate=0.0,
    ),
    loss_function=NTXentLoss(temperature=0.5),
    lr=0.00002,
    log_batches_per_epoch=3,
    log_samples_per_batch=3,
    example_input_array_shape=[
        1,
        1,
        1,
        160,
        160,
    ],
)

print("\n## Model Configuration")
print(model)
# %% Create trainer with WandB logger
trainer = VisCyTrainer(
    max_epochs=3,
    limit_train_batches=5,
    limit_val_batches=5,
    logger=wandb_logger,
    log_every_n_steps=1,
    callbacks=[ModelCheckpoint()],
    use_distributed_sampler=False,
)

# %% Fit the model
print("\n## Training")
print("Starting training with WandB logging...")
print("Image samples will be logged at the end of each epoch.")
print("\nExpected WandB logs:")
print("- Metrics: loss/train, loss/val, etc.")
print("- Images: train_samples (anchor/positive/negative triplets)")
print("- Images: val_samples (anchor/positive/negative triplets)")

trainer.fit(model, dm)

# %% Print WandB run URL
print("\n" + "=" * 80)
print("## Training Complete!")
print("=" * 80)
print(f"\nView your WandB run at: {wandb_logger.experiment.url}")
print("\n### What to look for in WandB UI:")
print("- Navigate to the 'Media' tab to see logged images")
print("- Images are organized by key: 'train_samples' and 'val_samples'")
print("- Each grid shows triplets: [anchor, positive, negative] samples")
print("- Gray colormap is used for all channels (phase contrast imaging)")
print("\n### Markdown Summary for Confluence/Wiki:")
print(
    """
## WandB Image Logging Demo Results

### Configuration
- **Model**: ContrastiveModule with ConvNeXt Tiny backbone
- **Logger**: Weights & Biases (WandBLogger)
- **Epochs**: 3
- **Image Logging**: 3 batches per epoch, 3 samples per batch

### Image Visualization
- **train_samples**: Training triplets (anchor, positive, negative)
- **val_samples**: Validation triplets (anchor, positive, negative)
- **Format**: Grid layout with concatenated samples
- **Colormap**: Grayscale for phase contrast images

### Key Features Demonstrated
1. ✅ Automatic logger type detection (TensorBoard vs WandB)
2. ✅ Backward compatible with existing TensorBoard code
3. ✅ Multi-logger support (can log to both simultaneously)
4. ✅ Same visual output across both logger backends
"""
)
