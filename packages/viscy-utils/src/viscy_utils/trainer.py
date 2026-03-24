"""VisCy Trainer with custom subcommands."""

import logging
from pathlib import Path
from typing import Literal

import torch
from iohub import open_ome_zarr
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized
from torch.onnx import OperatorExportTypes

from viscy_utils.meta_utils import generate_fg_masks, generate_normalization_metadata
from viscy_utils.precompute import precompute_array

_logger = logging.getLogger("lightning.pytorch")


class VisCyTrainer(Trainer):
    """Extended Trainer with preprocessing, export, and conversion subcommands."""

    def preprocess(
        self,
        data_path: Path,
        channel_names: list[str] | Literal[-1] = -1,
        num_workers: int = 1,
        block_size: int = 32,
        compute_otsu: bool = False,
        otsu_grid_spacing: int = 8,
        compute_fg_masks: bool = False,
        fg_mask_channels: list[str] | None = None,
        fg_mask_key: str = "fg_mask",
        model: LightningModule | None = None,
    ):
        """Compute dataset statistics for normalization.

        Parameters
        ----------
        data_path : Path
            Path to the HCS OME-Zarr dataset.
        channel_names : list[str] | Literal[-1], optional
            Channel names to compute statistics for, by default -1.
        num_workers : int, optional
            Number of CPU workers, by default 1.
        block_size : int, optional
            Block size to subsample images, by default 32.
        compute_otsu : bool, optional
            Whether to compute Otsu thresholds for Spotlight loss,
            by default False.
        otsu_grid_spacing : int, optional
            Grid spacing for Otsu sampling (denser than default), by default 8.
        compute_fg_masks : bool, optional
            Whether to precompute binary foreground masks from Otsu
            thresholds, by default False. Requires ``compute_otsu=True``.
        fg_mask_channels : list[str] or None, optional
            Channel names to compute FG masks for. Defaults to all channels
            that had Otsu thresholds computed (``channel_names``).
        fg_mask_key : str, optional
            Zarr array key for the mask, by default ``"fg_mask"``.
        model : LightningModule, optional
            Ignored placeholder, by default None.
        """
        if model is not None:
            _logger.warning("Ignoring model configuration during preprocessing.")
        with open_ome_zarr(data_path, layout="hcs", mode="r") as dataset:
            channel_indices = (
                [dataset.channel_names.index(c) for c in channel_names] if channel_names != -1 else channel_names
            )
            resolved_channel_names = (
                [dataset.channel_names[i] for i in channel_indices] if channel_names != -1 else dataset.channel_names
            )
        generate_normalization_metadata(
            zarr_dir=data_path,
            num_workers=num_workers,
            channel_ids=channel_indices,
            grid_spacing=block_size,
            compute_otsu=compute_otsu,
            otsu_grid_spacing=otsu_grid_spacing,
        )
        if compute_fg_masks:
            if not compute_otsu:
                raise ValueError("compute_fg_masks requires compute_otsu=True")
            mask_channels = fg_mask_channels if fg_mask_channels is not None else resolved_channel_names
            generate_fg_masks(
                zarr_dir=data_path,
                channel_names=mask_channels,
                fg_mask_key=fg_mask_key,
                num_workers=num_workers,
            )

    def export(
        self,
        model: LightningModule,
        export_path: Path,
        ckpt_path: Path,
        format: str = "onnx",
    ):
        """Export the model for deployment.

        Parameters
        ----------
        model : LightningModule
            Module to export.
        export_path : Path
            Output file name.
        ckpt_path : Path
            Model checkpoint path.
        format : str, optional
            Format (currently only ONNX is supported), by default "onnx".
        """
        if not format.lower() == "onnx":
            raise NotImplementedError(f"Export format '{format}'")
        model = _maybe_unwrap_optimized(model)
        self.strategy._lightning_module = model
        model.load_state_dict(torch.load(ckpt_path, weights_only=True)["state_dict"])
        model.eval()
        model.to_onnx(
            export_path,
            input_sample=model.example_input_array,
            export_params=True,
            opset_version=18,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {
                    0: "batch_size",
                    1: "channels",
                    3: "num_rows",
                    4: "num_cols",
                },
                "output": {
                    0: "batch_size",
                    1: "channels",
                    3: "num_rows",
                    4: "num_cols",
                },
            },
        )
        _logger.info(f"ONNX exported at {export_path}")

    def precompute(
        self,
        data_path: Path,
        output_path: Path,
        channel_names: list[str],
        subtrahends: list[Literal["mean"] | float],
        divisors: list[Literal["std"] | tuple[float, float]],
        image_array_key: str = "0",
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
        model: LightningModule | None = None,
    ):
        """Precompute normalized images.

        Parameters
        ----------
        data_path : Path
            Path to the HCS OME-Zarr dataset.
        output_path : Path
            Path to the output.
        channel_names : list[str]
            Channel names to normalize.
        subtrahends : list
            Subtrahend for each channel.
        divisors : list
            Divisor for each channel.
        image_array_key : str, optional
            Key of the image array, by default "0".
        include_wells : list[str] or None, optional
            Wells to include.
        exclude_fovs : list[str] or None, optional
            FOVs to exclude.
        model : LightningModule, optional
            Ignored placeholder.
        """
        precompute_array(
            data_path=data_path,
            output_path=output_path,
            channel_names=channel_names,
            subtrahends=subtrahends,
            divisors=divisors,
            image_array_key=image_array_key,
            include_wells=include_wells,
            exclude_fovs=exclude_fovs,
        )

    def convert_to_anndata(
        self,
        embeddings_path: Path,
        output_anndata_path: Path,
        overwrite: bool = False,
        model: LightningModule | None = None,
    ):
        """Convert an xarray dataset to an anndata dataset.

        Parameters
        ----------
        embeddings_path : Path
            Path to the embeddings dataset.
        output_anndata_path : Path
            Path to the output anndata dataset.
        overwrite : bool, optional
            Whether to overwrite existing output, by default False.
        model : LightningModule, optional
            Ignored placeholder.
        """
        from viscy_utils.evaluation.annotation import convert

        if model is not None:
            _logger.warning("Ignoring model configuration during conversion to AnnData.")

        convert(
            embeddings_ds=embeddings_path,
            output_path=output_anndata_path,
            overwrite=overwrite,
            return_anndata=False,
        )
        _logger.info(f"Anndata saved at: {output_anndata_path}")
