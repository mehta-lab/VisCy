import logging
import os
from typing import Literal, Sequence

import torch
from iohub.ngff import ImageArray, _pad_shape, open_ome_zarr
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from numpy.typing import DTypeLike

from viscy.light.data import Sample


def _resize_image(image: ImageArray, t_index: int, z_index: int):
    """Resize image array if incoming T and Z index is not within bounds."""
    if image.shape[0] <= t_index or image.shape[2] <= z_index:
        logging.debug(
            f"Resizing image '{image.name}' {image.shape} for T={t_index}, Z={z_index}."
        )
        image.resize(
            max(t_index + 1, image.shape[0]),
            image.channels,
            max(z_index + 1, image.shape[1]),
            *image.shape[-2:],
        )


class HCSPredictionWriter(BasePredictionWriter):
    """Callback to store virtual staining predictions as HCS OME-Zarr.

    :param str output_store: Path to the zarr store to store output
    :param bool write_input: Write the source and target channels too
        (must be writing to a new store),
        defaults to False
    :param Literal['batch', 'epoch', 'batch_and_epoch'] write_interval:
        When to write, defaults to "batch"
    """

    def __init__(
        self,
        output_store: str,
        write_input: bool = False,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.output_store = output_store
        self.write_input = write_input

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        source_channel: list[str] = trainer.datamodule.source_channel
        target_channel: list[str] = trainer.datamodule.target_channel
        prediction_channel = [ch + "_prediction" for ch in target_channel]
        if os.path.exists(self.output_store):
            if self.write_input:
                raise FileExistsError(
                    "Cannot write input to an existing store. Aborting."
                )
            else:
                self.plate = open_ome_zarr(self.output_store, mode="r+")
                for _, pos in self.plate.positions():
                    for ch in prediction_channel:
                        pos.append_channel(ch, resize_arrays=True)
        else:
            channel_names = prediction_channel
            if self.write_input:
                channel_names = source_channel + target_channel + channel_names
            self.plate = open_ome_zarr(
                self.output_store, layout="hcs", mode="a", channel_names=channel_names
            )
        logging.info(f"Writing prediction to: '{self.plate.zgroup.store.path}'.")
        if self.write_input:
            self.source_index = self._get_channel_indices(source_channel)
            self.target_index = self._get_channel_indices(target_channel)
        self.prediction_index = self._get_channel_indices(prediction_channel)

    def _get_channel_indices(self, channel_names: list[str]) -> list[int]:
        return [self.plate.get_channel_index(ch) for ch in channel_names]

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
        logging.debug(f"Writing batch {batch_idx}.")
        for sample_index, _ in enumerate(batch["index"][0]):
            self.write_sample(batch, prediction[sample_index], sample_index)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.plate.close()

    def write_sample(
        self, batch: Sample, sample_prediction: torch.Tensor, sample_index: int
    ) -> None:
        logging.debug(f"Writing sample {sample_index}.")
        sample_prediction = sample_prediction.cpu().numpy()
        img_name, t_index, z_index = [batch["index"][i][sample_index] for i in range(3)]
        t_index = int(t_index)
        z_index = int(z_index)
        image = self._create_image(
            img_name, sample_prediction.shape, sample_prediction.dtype
        )
        _resize_image(image, t_index, z_index)
        if self.write_input:
            # FIXME: should write center sclice of source
            image[t_index, self.source_index, z_index] = batch["source"][
                sample_index
            ].cpu()[:, 0]
            image[t_index, self.target_index, z_index] = batch["target"][
                sample_index
            ].cpu()[:, 0]
        # write C1YX
        image.oindex[t_index, self.prediction_index, z_index] = sample_prediction[:, 0]

    def _create_image(self, img_name: str, shape: tuple[int], dtype: DTypeLike):
        if img_name in self.plate.zgroup:
            return self.plate[img_name]
        logging.debug(f"Creating image '{img_name}'")
        _, row_name, col_name, pos_name, arr_name = img_name.split("/")
        position = self.plate.create_position(row_name, col_name, pos_name)
        shape = [1] + list(shape)
        shape[1] = len(position.channel_names)
        return position.create_zeros(
            arr_name,
            shape=shape,
            dtype=dtype,
            chunks=_pad_shape(tuple(shape[-2:]), 5),
        )
