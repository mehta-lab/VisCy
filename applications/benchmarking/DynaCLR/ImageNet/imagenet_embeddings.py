"""
Generate embeddings using a pre-trained ImageNet model and save them to a zarr store
using VisCy Trainer and EmbeddingWriter callback.
"""

import importlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import click
import timm
import torch
import yaml
from lightning.pytorch import LightningModule

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import EmbeddingWriter
from viscy.trainer import VisCyTrainer

logger = logging.getLogger(__name__)


class ImageNetModule(LightningModule):
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        channel_reduction_methods: Optional[
            Dict[str, Literal["middle_slice", "mean", "max"]]
        ] = None,
        channel_names: Optional[List[str]] = None,
    ):
        """Initialize the ImageNet module.

        Args:
            model_name: Name of the pre-trained ImageNet model to use
            channel_reduction_methods: Dict mapping channel names to reduction methods:
                - "middle_slice": Take the middle slice along the depth dimension
                - "mean": Average across the depth dimension
                - "max": Take the maximum value across the depth dimension
            channel_names: List of channel names corresponding to the input channels
        """
        super().__init__()
        self.channel_reduction_methods = channel_reduction_methods or {}
        self.channel_names = channel_names or []

        try:
            torch.set_float32_matmul_precision("high")
            self.model = timm.create_model(model_name, pretrained=True)
            self.model.eval()
        except ImportError:
            raise ImportError("Please install the timm library: pip install timm")

    def _reduce_5d_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce 5D input (B, C, D, H, W) to 4D (B, C, H, W) using specified methods.

        Args:
            x: 5D input tensor

        Returns:
            4D tensor after applying reduction methods
        """
        if x.dim() != 5:
            return x

        B, C, D, H, W = x.shape
        result = torch.zeros((B, C, H, W), device=x.device)

        # Process all channels at once for each reduction method to minimize loops
        middle_slice_indices = []
        mean_indices = []
        max_indices = []

        # Group channels by reduction method
        for c in range(C):
            channel_name = (
                self.channel_names[c] if c < len(self.channel_names) else f"channel_{c}"
            )
            method = self.channel_reduction_methods.get(channel_name, "middle_slice")

            if method == "mean":
                mean_indices.append(c)
            elif method == "max":
                max_indices.append(c)
            else:  # Default to middle_slice for any unknown method
                middle_slice_indices.append(c)

        # Apply middle_slice reduction to all relevant channels at once
        if middle_slice_indices:
            indices = torch.tensor(middle_slice_indices, device=x.device)
            result[:, indices] = x[:, indices, D // 2]

        # Apply mean reduction to all relevant channels at once
        if mean_indices:
            indices = torch.tensor(mean_indices, device=x.device)
            result[:, indices] = x[:, indices].mean(dim=2)

        # Apply max reduction to all relevant channels at once
        if max_indices:
            indices = torch.tensor(max_indices, device=x.device)
            result[:, indices] = x[:, indices].max(dim=2)[0]

        return result

    def _convert_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input tensor to 3-channel RGB format as needed.

        Args:
            x: Input tensor with 1, 2, or 3+ channels

        Returns:
            3-channel tensor suitable for ImageNet models
        """
        if x.shape[1] == 3:
            return x
        elif x.shape[1] == 1:
            # Convert to RGB by repeating the channel 3 times
            return x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            # Normalize each channel independently to handle different scales
            B, _, H, W = x.shape
            x_3ch = torch.zeros((B, 3, H, W), device=x.device, dtype=x.dtype)

            # Normalize each channel to 0-1 range
            ch0 = x[:, 0:1]
            ch1 = x[:, 1:2]

            ch0_min = ch0.reshape(B, -1).min(dim=1, keepdim=True)[0].reshape(B, 1, 1, 1)
            ch0_max = ch0.reshape(B, -1).max(dim=1, keepdim=True)[0].reshape(B, 1, 1, 1)
            ch0_range = ch0_max - ch0_min + 1e-7  # Add epsilon for numerical stability
            ch0_norm = (ch0 - ch0_min) / ch0_range

            ch1_min = ch1.reshape(B, -1).min(dim=1, keepdim=True)[0].reshape(B, 1, 1, 1)
            ch1_max = ch1.reshape(B, -1).max(dim=1, keepdim=True)[0].reshape(B, 1, 1, 1)
            ch1_range = ch1_max - ch1_min + 1e-7  # Add epsilon for numerical stability
            ch1_norm = (ch1 - ch1_min) / ch1_range

            # Create blended RGB channels - map each normalized channel to different colors
            x_3ch[:, 0] = ch0_norm.squeeze(1)  # R channel from first input
            x_3ch[:, 1] = ch1_norm.squeeze(1)  # G channel from second input
            x_3ch[:, 2] = 0.5 * (
                ch0_norm.squeeze(1) + ch1_norm.squeeze(1)
            )  # B channel as blend

            return x_3ch
        else:
            # For more than 3 channels, use the first 3
            return x[:, :3]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Extract features from the input images.

        Returns:
            Dictionary with features, properly shaped empty projections tensor, and index information
        """
        x = batch["anchor"]

        # Handle 5D input (B, C, D, H, W) using configured reduction methods
        if x.dim() == 5:
            x = self._reduce_5d_input(x)

        # Convert input to RGB format
        x = self._convert_to_rgb(x)

        # Get embeddings
        with torch.no_grad():
            features = self.model.forward_features(x)

            # Average pooling to get feature vector
            if features.dim() > 2:
                features = features.mean(dim=[2, 3])

        # Return features and empty projections with correct batch dimension
        return {
            "features": features,
            "projections": torch.zeros((features.shape[0], 0), device=features.device),
            "index": batch["index"],
        }


def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_normalization_from_config(norm_config):
    """Load a normalization transform from a configuration dictionary."""
    class_path = norm_config["class_path"]
    init_args = norm_config.get("init_args", {})

    # Split module and class name
    module_path, class_name = class_path.rsplit(".", 1)

    # Import the module
    module = importlib.import_module(module_path)

    # Get the class
    transform_class = getattr(module, class_name)

    # Instantiate the transform
    return transform_class(**init_args)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="convnext_tiny",
    help="Name of the pre-trained ImageNet model to use",
)
def main(config, model):
    """Extract ImageNet embeddings and save to zarr format using VisCy Trainer."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load config file
    cfg = load_config(config)
    logger.info(f"Loaded configuration from {config}")

    # Prepare datamodule parameters
    dm_params = {}

    # Add data and tracks paths from the paths section
    if "paths" not in cfg:
        raise ValueError("Configuration must contain a 'paths' section")

    if "data_path" not in cfg["paths"]:
        raise ValueError(
            "Data path is required in the configuration file (paths.data_path)"
        )
    dm_params["data_path"] = cfg["paths"]["data_path"]

    if "tracks_path" not in cfg["paths"]:
        raise ValueError(
            "Tracks path is required in the configuration file (paths.tracks_path)"
        )
    dm_params["tracks_path"] = cfg["paths"]["tracks_path"]

    # Add datamodule parameters
    if "datamodule" not in cfg:
        raise ValueError("Configuration must contain a 'datamodule' section")

    # Prepare normalizations
    if (
        "normalizations" not in cfg["datamodule"]
        or not cfg["datamodule"]["normalizations"]
    ):
        raise ValueError(
            "Normalizations are required in the configuration file (datamodule.normalizations)"
        )

    norm_configs = cfg["datamodule"]["normalizations"]
    normalizations = [load_normalization_from_config(norm) for norm in norm_configs]
    dm_params["normalizations"] = normalizations

    # Copy all other datamodule parameters
    for param, value in cfg["datamodule"].items():
        if param != "normalizations":
            # Handle patch sizes
            if param == "patch_size":
                dm_params["initial_yx_patch_size"] = value
                dm_params["final_yx_patch_size"] = value
            else:
                dm_params[param] = value

    # Set up the data module
    logger.info("Setting up data module")
    dm = TripletDataModule(**dm_params)

    # Get model parameters for handling 5D inputs
    channel_reduction_methods = {}

    if "model" in cfg and "channel_reduction_methods" in cfg["model"]:
        channel_reduction_methods = cfg["model"]["channel_reduction_methods"]

    # Initialize ImageNet model with reduction settings
    logger.info(f"Loading ImageNet model: {model}")
    model_module = ImageNetModule(
        model_name=model,
        channel_reduction_methods=channel_reduction_methods,
        channel_names=dm_params.get("source_channel", []),
    )

    # Get dimensionality reduction parameters from config
    phate_kwargs = None
    pca_kwargs = None

    if "embedding" in cfg:
        # Check for both capitalization variants and normalize
        if "phate_kwargs" in cfg["embedding"]:
            phate_kwargs = cfg["embedding"]["phate_kwargs"]

        if "umap_kwargs" in cfg["embedding"]:
            cfg["embedding"]["umap_kwargs"]

        if "pca_kwargs" in cfg["embedding"]:
            pca_kwargs = cfg["embedding"]["pca_kwargs"]

    # Check if output path exists and should be overwritten
    if "output_path" not in cfg["paths"]:
        raise ValueError(
            "Output path is required in the configuration file (paths.output_path)"
        )

    output_path = Path(cfg["paths"]["output_path"])
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    overwrite = False
    if "execution" in cfg and "overwrite" in cfg["execution"]:
        overwrite = cfg["execution"]["overwrite"]
    elif output_path.exists():
        logger.warning(f"Output path {output_path} already exists, will overwrite")
        overwrite = True

    # Set up EmbeddingWriter callback
    embedding_writer = EmbeddingWriter(
        output_path=output_path,
        phate_kwargs=phate_kwargs,
        pca_kwargs=pca_kwargs,
        overwrite=overwrite,
    )

    # Set up and run VisCy trainer
    logger.info("Setting up VisCy trainer")
    trainer = VisCyTrainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[embedding_writer],
        inference_mode=True,
    )

    logger.info(f"Running prediction and saving to {output_path}")
    trainer.predict(model_module, datamodule=dm)

    # Save configuration if requested
    save_config_flag = True
    show_config_flag = True

    if "execution" in cfg:
        if "save_config" in cfg["execution"]:
            save_config_flag = cfg["execution"]["save_config"]
        if "show_config" in cfg["execution"]:
            show_config_flag = cfg["execution"]["show_config"]

    # Save configuration if requested
    if save_config_flag:
        config_path = os.path.join(output_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path}")

    # Display configuration if requested
    if show_config_flag:
        click.echo("\nConfiguration used:")
        click.echo("-" * 40)
        for key, value in cfg.items():
            click.echo(f"{key}:")
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list) and subkey == "normalizations":
                        click.echo(f"  {subkey}:")
                        for norm in subvalue:
                            click.echo(f"    - class_path: {norm['class_path']}")
                            click.echo(f"      init_args: {norm['init_args']}")
                    else:
                        click.echo(f"  {subkey}: {subvalue}")
            else:
                click.echo(f"  {value}")
        click.echo("-" * 40)

    logger.info("Done!")


if __name__ == "__main__":
    main()
