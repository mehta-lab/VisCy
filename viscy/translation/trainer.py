import logging
from pathlib import Path
from typing import Literal, Sequence, Union

import torch
from iohub import open_ome_zarr
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized
from torch.onnx import OperatorExportTypes

from viscy.utils.meta_utils import generate_normalization_metadata


class VSTrainer(Trainer):
    def preprocess(
        self,
        data_path: Path,
        channel_names: Union[list[str], Literal[-1]] = -1,
        num_workers: int = 1,
        block_size: int = 32,
        model: LightningModule = None,
        datamodule: LightningDataModule = None,
        dataloaders: Sequence = None,
    ):
        """Compute dataset statistics before training or testing for normalization.

        :param Path data_path: Path to the HCS OME-Zarr dataset
        :param Union[list[str], Literal[ channel_names: channel names,
            defaults to -1 (all channels)
        :param int num_workers: number of workers, defaults to 1
        :param int block_size: sampling block size, defaults to 32
        :param LightningModule model: place holder for model, ignored
        :param LightningDataModule datamodule: place holder for datamodule, ignored
        :param Sequence dataloaders: place holder for dataloaders, ignored
        """
        if model or dataloaders or datamodule:
            logging.debug("Ignoring model and data configs during preprocessing.")
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
        export_path: str,
        ckpt_path: str,
        format="onnx",
        datamodule: LightningDataModule = None,
        dataloaders: Sequence = None,
    ):
        """Export the model for deployment (currently only ONNX is supported).

        :param LightningModule model: module to export
        :param str export_path: output file name
        :param str ckpt_path: model checkpoint
        :param str format: format (currently only ONNX is supported), defaults to "onnx"
        :param LightningDataModule datamodule: placeholder for datamodule,
            defaults to None
        :param Sequence dataloaders: placeholder for dataloaders, defaults to None
        """
        if dataloaders or datamodule:
            logging.debug("Ignoring datamodule and dataloaders during export.")
        if not format.lower() == "onnx":
            raise NotImplementedError(f"Export format '{format}'")
        model = _maybe_unwrap_optimized(model)
        self.strategy._lightning_module = model
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
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
        logging.info(f"ONNX exported at {export_path}")
