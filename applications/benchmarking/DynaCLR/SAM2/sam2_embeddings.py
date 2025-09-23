import importlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import click
import torch
import yaml
from lightning.pytorch import LightningModule
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.exposure import rescale_intensity

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import EmbeddingWriter
from viscy.trainer import VisCyTrainer


class SAM2Module(LightningModule):
    def __init__(
        self,
        model_name: str = "facebook/sam2-hiera-base-plus",
        channel_reduction_methods: Optional[
            Dict[str, Literal["middle_slice", "mean", "max"]]
        ] = None,
        channel_names: Optional[List[str]] = None,
        middle_slice_index: Optional[int] = None,
    ):
        """
        SAM2 module for feature extraction.

        Parameters
        ----------
        model_name : str, optional
            SAM2 model name from HuggingFace Model Hub (default: "facebook/sam2-hiera-base-plus").
        channel_reduction_methods : dict[str, {"middle_slice", "mean", "max"}], optional
            Dictionary mapping channel names to reduction methods for 5D inputs (default: None, uses "middle_slice").
        channel_names : list of str, optional
            List of channel names corresponding to input channels (default: None).
        middle_slice_index : int, optional
            Specific z-slice index to use for "middle_slice" reduction (default: None, uses D//2).

        """
        super().__init__()
        self.model_name = model_name
        self.channel_reduction_methods = channel_reduction_methods or {}
        self.channel_names = channel_names or []
        self.middle_slice_index = middle_slice_index

        torch.set_float32_matmul_precision("high")
        self.model = None  # Initialize in on_predict_start when device is set

    def on_predict_start(self):
        """
        Initialize model with proper device when prediction starts.
        
        Notes
        -----
        This method is called automatically by Lightning when prediction begins.
        It ensures the SAM2 model is properly initialized on the correct device.
        """
        if self.model is None:
            self.model = SAM2ImagePredictor.from_pretrained(
                self.model_name, device=self.device
            )

    def _reduce_5d_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce 5D input (B, C, D, H, W) to 4D (B, C, H, W) using specified methods.

        Parameters
        ----------
        x : torch.Tensor
            5D input tensor with shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            4D tensor after applying reduction methods with shape (B, C, H, W).
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
            slice_idx = self.middle_slice_index if self.middle_slice_index is not None else D // 2
            result[:, indices] = x[:, indices, slice_idx]

        # Apply mean reduction to all relevant channels at once
        if mean_indices:
            indices = torch.tensor(mean_indices, device=x.device)
            result[:, indices] = x[:, indices].mean(dim=2)

        # Apply max reduction to all relevant channels at once
        if max_indices:
            indices = torch.tensor(max_indices, device=x.device)
            result[:, indices] = x[:, indices].max(dim=2)[0]

        return result

    def _convert_to_rgb(self, x: torch.Tensor) -> list:
        """
        Convert input tensor to 3-channel RGB format as needed for SAM2.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with 1, 2, or 3+ channels and shape (B, C, H, W).

        Returns
        -------
        list of numpy.ndarray
            List of numpy arrays in HWC format for SAM2 processing.
        """
        # Convert to RGB and scale to [0, 255] range for SAM2
        if x.shape[1] == 1:
            x_rgb = x.repeat(1, 3, 1, 1) * 255.0
        elif x.shape[1] == 2:
            x_3ch = torch.zeros(
                (x.shape[0], 3, x.shape[2], x.shape[3]), device=x.device
            )
            x[:, 0] = rescale_intensity(x[:, 0], out_range="uint8")
            x[:, 1] = rescale_intensity(x[:, 1], out_range="uint8")

            x_3ch[:, 0] = x[:, 0]
            x_3ch[:, 1] = x[:, 1]
            x_3ch[:, 2] = 0.5 * (x[:, 0] + x[:, 1])  # B channel as blend

        elif x.shape[1] == 3:
            x_rgb = rescale_intensity(x, out_range="uint8")
        else:
            # More than 3 channels, normalize first 3 and scale
            x_3ch = x[:, :3]
            x_rgb = rescale_intensity(x_3ch, out_range="uint8")

        # Convert to list of numpy arrays in HWC format for SAM2
        return [
            x_rgb[i].cpu().numpy().transpose(1, 2, 0) for i in range(x_rgb.shape[0])
        ]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Extract features from the input images.

        Parameters
        ----------
        batch : dict
            Batch dictionary containing "anchor" key with input tensors.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the dataloader (default: 0).

        Returns
        -------
        dict
            Dictionary containing:
            - "features": Extracted features tensor
            - "projections": Empty tensor for compatibility (B, 0)
            - "index": Batch index information
        """
        x = batch["anchor"]

        # Handle 5D input (B, C, D, H, W) using configured reduction methods
        if x.dim() == 5:
            x = self._reduce_5d_input(x)

        # Convert input to RGB format and get list of numpy arrays in HWC format for SAM2
        image_list = self._convert_to_rgb(x)
        self.model.set_image_batch(image_list)

        # Extract features
        # features_0 = self.model._features["image_embed"].mean(dim=(2, 3))
        # features_1 = self.model._features["high_res_feats"][0].mean(dim=(2, 3))
        # features_2 = self.model._features["high_res_feats"][1].mean(dim=(2, 3))
        # features = torch.concat([features_0, features_1, features_2], dim=1)
        features = self.model._features["high_res_feats"][0].mean(dim=(2, 3))

        # Return features and empty projections with correct batch dimension
        return {
            "features": features,
            "projections": torch.zeros((features.shape[0], 0), device=features.device),
            "index": batch["index"],
        }


def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Parameters
    ----------
    config_file : str or Path
        Path to the YAML configuration file.
        
    Returns
    -------
    dict
        Configuration dictionary loaded from the YAML file.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_normalization_from_config(norm_config):
    """
    Load a normalization transform from a configuration dictionary.
    
    Parameters
    ----------
    norm_config : dict
        Configuration dictionary containing "class_path" and optional "init_args".
        
    Returns
    -------
    object
        Instantiated normalization transform object.
    """
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
    """
    Extract SAM2 embeddings and save to zarr format using VisCy Trainer.
    
    Parameters
    ----------
    config : str or Path
        Path to the YAML configuration file containing all parameters for
        data loading, model configuration, and output settings.
    """
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

    class_path = cfg["datamodule_class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    datamodule_class = getattr(module, class_name)
    dm = datamodule_class(**dm_params)
    
    # Get model parameters for handling 5D inputs
    channel_reduction_methods = {}
    middle_slice_index = None

    if "model" in cfg:
        if "channel_reduction_methods" in cfg["model"]:
            channel_reduction_methods = cfg["model"]["channel_reduction_methods"]
        if "middle_slice_index" in cfg["model"]:
            middle_slice_index = cfg["model"]["middle_slice_index"]

    # Initialize SAM2 model with reduction settings
    logger.info("Loading SAM2 model")
    model = SAM2Module(
        model_name=cfg["model"]["model_name"],
        channel_reduction_methods=channel_reduction_methods,
        middle_slice_index=middle_slice_index,
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
