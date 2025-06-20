"""
Generate embeddings using the OpenPhenom model and save them to a zarr store
using VisCy Trainer and EmbeddingWriter callback.
"""

import importlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import click
import torch
import yaml
from lightning.pytorch import LightningModule
from transformers import AutoModel

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import EmbeddingWriter
from viscy.trainer import VisCyTrainer


class OpenPhenomModule(LightningModule):
    def __init__(
        self,
        channel_reduction_methods: Optional[
            Dict[str, Literal["middle_slice", "mean", "max"]]
        ] = None,
        channel_names: Optional[List[str]] = None,
    ):
        """Initialize the OpenPhenom module.

        Parameters
        ----------
        channel_reduction_methods : dict, optional
            Dictionary mapping channel names to reduction methods:
            - "middle_slice": Take the middle slice along the depth dimension
            - "mean": Average across the depth dimension
            - "max": Take the maximum value across the depth dimension
        channel_names : list of str, optional
            List of channel names corresponding to the input channels

        Notes
        -----
        The module uses the OpenPhenom model from HuggingFace for generating embeddings.
        """
        super().__init__()

        self.channel_reduction_methods = channel_reduction_methods or {}
        self.channel_names = channel_names or []

        try:
            torch.set_float32_matmul_precision("high")
            self.model = AutoModel.from_pretrained(
                "recursionpharma/OpenPhenom", trust_remote_code=True
            )
            self.model.eval()
        except ImportError:
            raise ImportError(
                "Please install the OpenPhenom dependencies: "
                "pip install transformers"
            )

    def on_predict_start(self):
        # Move model to GPU when prediction starts
        self.model.to(self.device)

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

        # Apply reduction method for each channel
        for c in range(C):
            channel_name = (
                self.channel_names[c] if c < len(self.channel_names) else f"channel_{c}"
            )
            # Default to middle slice if not specified
            method = self.channel_reduction_methods.get(channel_name, "middle_slice")

            if method == "middle_slice":
                result[:, c] = x[:, c, D // 2]
            elif method == "mean":
                result[:, c] = x[:, c].mean(dim=1)
            elif method == "max":
                result[:, c] = x[:, c].max(dim=1)[0]
            else:
                # Fallback to middle slice for unknown methods
                result[:, c] = x[:, c, D // 2]

        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Extract features from the input images.

        Returns:
            Dictionary with features, projections (None), and index information
        """
        x = batch["anchor"]

        # OpenPhenom expects [B, C, H, W] but our data might be [B, C, D, H, W]
        # If 5D input, handle according to specified reduction methods
        if x.dim() == 5:
            x = self._reduce_5d_input(x)

        # Convert to uint8 as OpenPhenom expects uint8 inputs
        if x.dtype != torch.uint8:
            x = (
                ((x - x.min()) / (x.max() - x.min()) * 255)
                .clamp(0, 255)
                .to(torch.uint8)
            )

        # Get embeddings
        self.model.return_channelwise_embeddings = False
        features = self.model.predict(x)
        # Create empty projections tensor with same batch size as features
        # This ensures the EmbeddingWriter can process it
        projections = torch.zeros((features.shape[0], 0), device=features.device)

        return {
            "features": features,
            "projections": projections,
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
def main(config):
    """Extract OpenPhenom embeddings and save to zarr format using VisCy Trainer."""
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

    # Initialize OpenPhenom model with reduction settings
    logger.info("Loading OpenPhenom model")
    model = OpenPhenomModule(
        channel_reduction_methods=channel_reduction_methods,
        channel_names=dm_params.get("source_channel", []),
    )

    # Get dimensionality reduction parameters from config
    phate_kwargs = None
    pca_kwargs = None

    if "embedding" in cfg:
        if "phate_kwargs" in cfg["embedding"]:
            phate_kwargs = cfg["embedding"]["phate_kwargs"]
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
    trainer.predict(model, datamodule=dm)

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
