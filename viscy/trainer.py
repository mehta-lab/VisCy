import logging
from pathlib import Path
from typing import Literal

import torch
from iohub import open_ome_zarr
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized
from torch.onnx import OperatorExportTypes

from viscy.preprocessing.precompute import precompute_array
from viscy.utils.meta_utils import generate_normalization_metadata

_logger = logging.getLogger("lightning.pytorch")


class VisCyTrainer(Trainer):
    def preprocess(
        self,
        data_path: Path,
        channel_names: list[str] | Literal[-1] = -1,
        num_workers: int = 1,
        block_size: int = 32,
        model: LightningModule | None = None,
    ):
        """
        Compute dataset statistics before training or testing for normalization.

        Parameters
        ----------
        data_path : Path
            Path to the HCS OME-Zarr dataset
        channel_names : list[str] | Literal[-1], optional
            Channel names to compute statistics for, by default -1
        num_workers : int, optional
            Number of CPU workers, by default 1
        block_size : int, optional
            Block size to subsample images, by default 32
        model: LightningModule, optional
            Ignored placeholder, by default None
        """
        if model is not None:
            _logger.warning("Ignoring model configuration during preprocessing.")
        with open_ome_zarr(data_path, layout="hcs", mode="r") as dataset:
            channel_indices = (
                [dataset.channel_names.index(c) for c in channel_names]
                if channel_names != -1
                else channel_names
            )
        generate_normalization_metadata(
            zarr_dir=data_path,
            num_workers=num_workers,
            channel_ids=channel_indices,
            grid_spacing=block_size,
        )

    def export(
        self,
        model: LightningModule,
        export_path: Path,
        ckpt_path: Path,
        format: str = "onnx",
    ):
        """
        Export the model for deployment (currently only ONNX is supported).

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
