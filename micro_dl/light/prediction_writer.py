from typing import Literal, Sequence
import logging
import os

from iohub.ngff import open_ome_zarr, _pad_shape
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
import torch

from micro_dl.light.data import Sample


class HCSPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_store: str,
        write_input: bool = False,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        """Callback to store virtual staining predictions as HCS OME-Zarr.

        :param str output_store: Path to the zarr store to store output
        :param bool write_input: Write the source and target channels too
            (must be writing to a new store),
            defaults to False
        :param Literal['batch', 'epoch', 'batch_and_epoch'] write_interval:
            When to write, defaults to "batch"
        """
        super().__init__(write_interval)
        source_channel = "source"
        target_channel = "target"
        prediction_channel = "prediction"
        if os.path.exists(output_store):
            if write_input:
                raise FileExistsError(
                    "Cannot write input to an existing store. Aborting."
                )
            else:
                self.plate = open_ome_zarr(output_store, mode="r+")
                self.plate.append_channel(prediction_channel, resize_arrays=True)
        else:
            channel_names = [prediction_channel]
            if write_input:
                channel_names += [source_channel, target_channel]
            self.plate = open_ome_zarr(
                output_store, layout="hcs", mode="a", channel_names=channel_names
            )
        logging.info(f"Writing prediction to: '{self.plate.zgroup.store.path}'.")
        if write_input:
            self.source_index = self.plate.get_channel_index(source_channel)
            self.target_index = self.plate.get_channel_index(target_channel)
        self.prediction_index = self.plate.get_channel_index(prediction_channel)
        self.write_input = write_input

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: torch.Tensor,
        batch_indices: Sequence[int] | None,
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        for sample_index, _ in enumerate(batch["index"][0]):
            self.write_sample(batch, prediction[sample_index], sample_index)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.plate.close()

    def write_sample(
        self, batch: Sample, sample_prediction: torch.Tensor, sample_index: int
    ) -> None:
        sample_prediction = sample_prediction.cpu().numpy()
        img_name, t_index, z_index = [batch["index"][i][sample_index] for i in range(3)]
        t_index = int(t_index)
        z_index = int(z_index)
        if img_name not in self.plate.zgroup:
            _, row_name, col_name, pos_name, arr_name = img_name.split("/")
            position = self.plate.create_position(row_name, col_name, pos_name)
            shape = [1] + list(sample_prediction.shape)
            shape[1] = len(position.channel_names)
            image = position.create_zeros(
                arr_name,
                shape=shape,
                dtype=sample_prediction.dtype,
                chunks=_pad_shape(tuple(shape[-2:]), 5),
            )
        else:
            image = self.plate[img_name]
        if image.shape[0] <= t_index or image.shape[1] <= z_index:
            image.resize(
                max(t_index + 1, image.shape[0]),
                image.channels,
                max(z_index + 1, image.shape[1]),
                *image.shape[-2:],
            )
        if self.write_input:
            # FIXME: should write center sclice of source
            image[t_index, self.source_index, z_index] = batch["source"][
                sample_index
            ].cpu()[0, 0]
            image[t_index, self.target_index, z_index] = batch["target"][
                sample_index
            ].cpu()[0, 0]
        image[t_index, self.prediction_index, z_index] = sample_prediction[0, 0]
