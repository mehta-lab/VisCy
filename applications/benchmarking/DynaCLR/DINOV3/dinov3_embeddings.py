import importlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import click
import numpy as np
import torch
import yaml
from lightning.pytorch import LightningModule
from PIL import Image
from skimage.exposure import rescale_intensity
from transformers import AutoImageProcessor, AutoModel

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import EmbeddingWriter
from viscy.trainer import VisCyTrainer


class DINOv3Module(LightningModule):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        channel_reduction_methods: Optional[
            Dict[str, Literal["middle_slice", "mean", "max"]]
        ] = None,
        channel_names: Optional[List[str]] = None,
        pooling_method: Literal["mean", "max", "cls_token"] = "mean",  
        middle_slice_index: Optional[int] = None,
    ):
        """
        DINOv3 module for feature extraction.

        Parameters
        ----------
        model_name : str, optional
            DINOv3 model name from HuggingFace Model Hub (default: "facebook/dinov3-vitb16-pretrain-lvd1689m").
        channel_reduction_methods : dict[str, {"middle_slice", "mean", "max"}], optional
            Dictionary mapping channel names to reduction methods for 5D inputs (default: None, uses "middle_slice").
        channel_names : list of str, optional
            List of channel names corresponding to input channels (default: None).
        pooling_method : Literal["mean", "max", "cls_token"], optional
            Method to pool spatial tokens from the model output (default: "mean").
        middle_slice_index : int, optional
            Specific z-slice index to use for "middle_slice" reduction (default: None, uses D//2).

        """
        super().__init__()
        self.model_name = model_name
        self.channel_reduction_methods = channel_reduction_methods or {}
        self.channel_names = channel_names or []
        self.pooling_method = pooling_method
        self.middle_slice_index = middle_slice_index

        torch.set_float32_matmul_precision("high")
        self.model = None
        self.processor = None

    def on_predict_start(self):
        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)

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

        # Group channels by reduction method
        middle_slice_indices = []
        mean_indices = []
        max_indices = []

        for c in range(C):
            channel_name = (
                self.channel_names[c] if c < len(self.channel_names) else f"channel_{c}"
            )
            method = self.channel_reduction_methods.get(channel_name, "middle_slice")

            if method == "mean":
                mean_indices.append(c)
            elif method == "max":
                max_indices.append(c)
            else:  # Default to middle_slice
                middle_slice_indices.append(c)

        # Apply reductions
        if middle_slice_indices:
            indices = torch.tensor(middle_slice_indices, device=x.device)
            slice_idx = self.middle_slice_index if self.middle_slice_index is not None else D // 2
            result[:, indices] = x[:, indices, slice_idx]

        if mean_indices:
            indices = torch.tensor(mean_indices, device=x.device)
            result[:, indices] = x[:, indices].mean(dim=2)

        if max_indices:
            indices = torch.tensor(max_indices, device=x.device)
            result[:, indices] = x[:, indices].max(dim=2)[0]

        return result

    def _convert_to_pil_images(self, x: torch.Tensor) -> List[Image.Image]:
        """
        Convert tensor to list of PIL Images for DINOv3 processing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, H, W).

        Returns
        -------
        list of PIL.Image.Image
            List of PIL Images ready for DINOv3 processing.
        """
        images = []
        
        for b in range(x.shape[0]):
            img_tensor = x[b]  # (C, H, W)
            
            if img_tensor.shape[0] == 1:
                # Single channel - convert to grayscale PIL
                img_array = img_tensor[0].cpu().numpy()
                # Normalize to 0-255
                img_normalized = ((img_array - img_array.min()) / 
                                (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_normalized, mode='L')
                
            elif img_tensor.shape[0] == 2:
                # Two channels - create RGB with blend in blue
                img_array = img_tensor.cpu().numpy()
                rgb_array = np.zeros((img_array.shape[1], img_array.shape[2], 3), dtype=np.uint8)
                
                # Normalize each channel to 0-255
                ch0_norm = rescale_intensity(img_array[0], out_range=(0, 255)).astype(np.uint8)
                ch1_norm = rescale_intensity(img_array[1], out_range=(0, 255)).astype(np.uint8)
                
                rgb_array[:, :, 0] = ch0_norm  # Red
                rgb_array[:, :, 1] = ch1_norm  # Green  
                rgb_array[:, :, 2] = (ch0_norm + ch1_norm) // 2  # Blue as blend
                
                pil_img = Image.fromarray(rgb_array, mode='RGB')
                
            elif img_tensor.shape[0] == 3:
                # Three channels - direct RGB
                img_array = img_tensor.cpu().numpy().transpose(1, 2, 0)  # HWC
                img_normalized = rescale_intensity(img_array, out_range=(0, 255)).astype(np.uint8)
                pil_img = Image.fromarray(img_normalized, mode='RGB')
                
            else:
                # More than 3 channels - use first 3
                img_array = img_tensor[:3].cpu().numpy().transpose(1, 2, 0)  # HWC
                img_normalized = rescale_intensity(img_array, out_range=(0, 255)).astype(np.uint8)
                pil_img = Image.fromarray(img_normalized, mode='RGB')
            
            images.append(pil_img)
        
        return images

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool spatial features from DINOv3 tokens.
        
        Parameters
        ----------
        features : torch.Tensor
            Token features with shape (B, num_tokens, hidden_dim).
            
        Returns
        -------
        torch.Tensor
            Pooled features with shape (B, hidden_dim).
        """
        if self.pooling_method == "cls_token":
            # For ViT models, first token is usually CLS token
            if "vit" in self.model_name.lower():
                return features[:, 0, :]  # CLS token
            else:
                # For ConvNeXt, no CLS token, fall back to mean
                return features.mean(dim=1)
                
        elif self.pooling_method == "max":
            return features.max(dim=1)[0]
        else:  # mean pooling
            return features.mean(dim=1)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["anchor"]

        # Handle 5D input (B, C, D, H, W)
        if x.dim() == 5:
            x = self._reduce_5d_input(x)

        # Convert to PIL Images for DINOv3 processing
        pil_images = self._convert_to_pil_images(x)
        
        # Batch process all images at once for better GPU utilization
        inputs = self.processor(pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_features = outputs.last_hidden_state  # (B, num_tokens, hidden_dim)
            features = self._pool_features(token_features)  # (B, hidden_dim)

        return {
            "features": features,
            "projections": torch.zeros((features.shape[0], 0), device=features.device),
            "index": batch["index"],
        }


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_normalization_from_config(norm_config):
    class_path = norm_config["class_path"]
    init_args = norm_config.get("init_args", {})

    module_path, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_path)

    transform_class = getattr(module, class_name)

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
    Extract DINOv3 embeddings and save to zarr format using VisCy Trainer.
    
    Parameters
    ----------
    config : str or Path
        Path to the YAML configuration file containing all parameters for
        data loading, model configuration, and output settings.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cfg = load_config(config)
    logger.info(f"Loaded configuration from {config}")

    dm_params = {}

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

    if "datamodule" not in cfg:
        raise ValueError("Configuration must contain a 'datamodule' section")

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

    for param, value in cfg["datamodule"].items():
        if param != "normalizations":
            # Handle patch sizes
            if param == "patch_size":
                dm_params["initial_yx_patch_size"] = value
                dm_params["final_yx_patch_size"] = value
            else:
                dm_params[param] = value

    logger.info("Setting up data module")
    dm = TripletDataModule(**dm_params)

    # Get model parameters
    model_name = cfg["model"].get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    pooling_method = cfg["model"].get("pooling_method", "mean")
    channel_reduction_methods = cfg["model"].get("channel_reduction_methods", {})
    channel_names = cfg["model"].get("channel_names", [])
    middle_slice_index = cfg["model"].get("middle_slice_index", None)

    # Initialize DINOv3 model
    logger.info(f"Loading DINOv3 model: {model_name}")
    model = DINOv3Module(
        model_name=model_name,
        pooling_method=pooling_method,
        channel_reduction_methods=channel_reduction_methods,
        channel_names=channel_names,
        middle_slice_index=middle_slice_index,
    )

    phate_kwargs = None
    pca_kwargs = None

    if "embedding" in cfg:
        if "phate_kwargs" in cfg["embedding"]:
            phate_kwargs = cfg["embedding"]["phate_kwargs"]
        if "pca_kwargs" in cfg["embedding"]:
            pca_kwargs = cfg["embedding"]["pca_kwargs"]

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

    embedding_writer = EmbeddingWriter(
        output_path=output_path,
        phate_kwargs=phate_kwargs,
        pca_kwargs=pca_kwargs,
        overwrite=overwrite,
    )

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