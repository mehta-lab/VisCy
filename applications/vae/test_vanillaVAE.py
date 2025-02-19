# %%
import logging

import matplotlib.pyplot as plt
import torch
import torchview

from viscy.data.triplet import TripletDataModule
from viscy.representation.vanilla_vae import PythaeLightningVAE

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Test config
config = {
    "data": {
        "path": "/hpc/projects/organelle_phenotyping/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/registered_chunked.zarr",
        "tracks_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.2-tracking/track.zarr",
        "source_channel": ["Phase3D"],
        "z_range": [31, 36],
        "initial_yx_patch_size": [272, 272],
        "final_yx_patch_size": [192, 192],
        "split_ratio": 0.8,
        "time_interval": "any",
    },
    "model": {"latent_dim": 32},
    "training": {
        "batch_size": 4,  # Smaller batch size for testing
        "num_workers": 2,  # Fewer workers for testing
        "lr": 1e-4,
        "schedule": "Constant",  # Simpler schedule for testing
    },
}

# Initialize data module
data_module = TripletDataModule(
    data_path=config["data"]["path"],
    tracks_path=config["data"]["tracks_path"],
    source_channel=config["data"]["source_channel"],
    z_range=config["data"]["z_range"],
    initial_yx_patch_size=config["data"]["initial_yx_patch_size"],
    final_yx_patch_size=config["data"]["final_yx_patch_size"],
    split_ratio=config["data"]["split_ratio"],
    batch_size=config["training"]["batch_size"],
    num_workers=config["training"]["num_workers"],
)

# Calculate input dimensions
input_dim = (
    len(config["data"]["source_channel"]),  # Number of channels
    config["data"]["z_range"][1] - config["data"]["z_range"][0],  # Z dimension
    config["data"]["final_yx_patch_size"][0],  # Y dimension
    config["data"]["final_yx_patch_size"][1],  # X dimension
)

# Initialize model
model = PythaeLightningVAE(
    input_dim=input_dim,
    latent_dim=config["model"]["latent_dim"],
    lr=config["training"]["lr"],
    schedule=config["training"]["schedule"],
)

_logger.info("Model architecture:")
_logger.info(model)

# %%
# Test datamodule
_logger.info("Testing datamodule...")
data_module.setup("fit")
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

# Check batch structure
_logger.info("\nBatch structure:")
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        _logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")
    else:
        _logger.info(f"{key}: type={type(value)}")


# Plot sample images
def plot_batch_samples(batch, num_samples=None):
    """Plot a few samples from the batch."""
    anchor_images = batch["anchor"]

    if num_samples is None:
        num_samples = anchor_images.shape[0]
    else:
        assert (
            anchor_images.shape[0] == num_samples
        ), f"Expected {num_samples} samples, got {anchor_images.shape[0]}"

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))

    for i in range(num_samples):
        # For 3D images (C, Z, H, W), take middle Z slice
        if len(anchor_images[i].shape) == 4:
            img = anchor_images[
                i,
                :,
                anchor_images.shape[1] // 2,
            ]
        else:
            img = anchor_images[i]

        # If multiple channels, only show first channel
        if img.shape[0] >= 1:
            img = img[0]

        axes[i].imshow(img.numpy(), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.show()


_logger.info("\nPlotting sample images from batch:")
plot_batch_samples(batch)

# %%
# Test model graph
_logger.info("\nGenerating model graph...")

# Create input for visualization
sample_tensor = batch["anchor"]

# Define visualization parameters
viz_params = {
    "save_graph": True,
    "expand_nested": True,  # Show internal structure
    "depth": 3,  # Control level of detail
    "device": "cpu",
    "graph_dir": "./model_graphs",
}

# Visualize individual components
_logger.info("\nGenerating component graphs...")

# Encoder
_logger.info("Generating encoder graph...")
encoder_graph = torchview.draw_graph(
    model.model.encoder,
    input_data=sample_tensor,
    graph_name="vae_encoder",
    **viz_params,
)
_logger.info(f"Encoder graph saved to ./model_graphs/vae_encoder.png")

# Decoder (using encoder output for proper input shape)
_logger.info("Generating decoder graph...")
with torch.no_grad():
    encoder_output = model.model.encoder(sample_tensor)
    latent_sample = encoder_output.embedding

decoder_graph = torchview.draw_graph(
    model.model.decoder,
    input_data=latent_sample,
    graph_name="vae_decoder",
    **viz_params,
)
_logger.info(f"Decoder graph saved to ./model_graphs/vae_decoder.png")

# Print model summary
_logger.info("\nModel Summary:")
_logger.info(f"Input shape: {sample_tensor.shape}")
_logger.info(f"\nEncoder Architecture:")
_logger.info(model.model.encoder)
_logger.info(f"\nDecoder Architecture:")
_logger.info(model.model.decoder)


# Print parameter counts
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


_logger.info(f"\nParameter counts:")
_logger.info(f"Complete VAE parameters: {count_parameters(model):,}")
_logger.info(f"  ├─ Encoder parameters: {count_parameters(model.model.encoder):,}")
_logger.info(f"  └─ Decoder parameters: {count_parameters(model.model.decoder):,}")

# %%
