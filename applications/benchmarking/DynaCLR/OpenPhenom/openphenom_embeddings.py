"""
Generate embeddings using the OpenPhenom model and save them to a zarr store
using VisCy Trainer and EmbeddingWriter callback.
"""

import importlib
import logging
import os
from pathlib import Path

import click
import torch
import yaml
from lightning.pytorch import LightningModule
from transformers import AutoModel

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import EmbeddingWriter
from viscy.trainer import VisCyTrainer


class OpenPhenomModule(LightningModule):
    def __init__(self):
        super().__init__()
        try:
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Extract features from the input images.

        Returns:
            Dictionary with features, projections (None), and index information
        """
        x = batch["anchor"]

        # OpenPhenom expects [B, C, H, W] but our data might be [B, C, D, H, W]
        # If 5D input, take middle slice or average across D
        if x.dim() == 5:
            x = x[:, :, x.shape[2] // 2, :, :]

        # Convert to uint8 as OpenPhenom expects uint8 inputs
        if x.dtype != torch.uint8:
            x = (
                ((x - x.min()) / (x.max() - x.min()) * 255)
                .clamp(0, 255)
                .to(torch.uint8)
            )

        # Get embeddings
        self.model.return_channelwise_embeddings = False
        with torch.no_grad():
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

    # Initialize OpenPhenom model
    logger.info("Loading OpenPhenom model")
    model = OpenPhenomModule()

    # Get dimensionality reduction parameters from config
    phate_kwargs = None
    umap_kwargs = None
    pca_kwargs = None
    reductions = None

    if "embedding" in cfg:
        if "phate_kwargs" in cfg["embedding"]:
            phate_kwargs = cfg["embedding"]["phate_kwargs"]
        if "umap_kwargs" in cfg["embedding"]:
            umap_kwargs = cfg["embedding"]["umap_kwargs"]
        if "pca_kwargs" in cfg["embedding"]:
            pca_kwargs = cfg["embedding"]["pca_kwargs"]
        if "reductions" in cfg["embedding"]:
            reductions = cfg["embedding"]["reductions"]

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
        umap_kwargs=umap_kwargs,
        pca_kwargs=pca_kwargs,
        reductions=reductions,
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
