import logging
import os
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from iohub.ngff import ImageArray, _pad_shape, open_ome_zarr
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from numpy.typing import DTypeLike, NDArray

from viscy.light.data import HCSDataModule, Sample

__all__ = ["HCSPredictionWriter"]
_logger = logging.getLogger("lightning.pytorch")


def _resize_image(image: ImageArray, t_index: int, z_slice: slice) -> None:
    """Resize image array if incoming (1, C, Z, Y, X) stack is not within bounds."""
    if image.shape[0] <= t_index or image.shape[2] < z_slice.stop:
        _logger.debug(
            f"Resizing image '{image.name}' {image.shape} for "
            f"T={t_index}, Z-sclice={z_slice}."
        )
        image.resize(
            max(t_index + 1, image.shape[0]),
            image.channels,
            max(z_slice.stop, image.shape[2]),
            *image.shape[-2:],
        )


def _blend_in(old_stack: NDArray, new_stack: NDArray, z_slice: slice) -> None:
    if z_slice.start == 0:
        return new_stack
    depth = z_slice.stop - z_slice.start
    # relevant predictions to integrate
    samples = min(z_slice.start + 1, depth)
    factors = []
    for i in reversed(list(range(depth))):
        factors.append(min(i + 1, samples))
    _logger.debug(f"Blending with factors {factors}.")
    factors = np.array(factors)[np.newaxis :, np.newaxis, np.newaxis]
    return old_stack * (factors - 1) / factors + new_stack / factors


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
        dm: HCSDataModule = trainer.datamodule
        self.z_padding = dm.z_window_size // 2 if dm.target_2d else 0
        _logger.debug(f"Setting Z padding to {self.z_padding}")
        source_channel = dm.source_channel
        target_channel = dm.target_channel
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
                channel_names = source_channel + channel_names
            self.plate = open_ome_zarr(
                self.output_store, layout="hcs", mode="a", channel_names=channel_names
            )
        _logger.info(f"Writing prediction to: '{self.plate.zgroup.store.path}'.")
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
        batch_indices: Optional[Sequence[int]],
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        _logger.debug(f"Writing batch {batch_idx}.")
        for sample_index, _ in enumerate(batch["index"][0]):
            self.write_sample(batch, prediction[sample_index], sample_index)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.plate.close()

    def write_sample(
        self, batch: Sample, sample_prediction: torch.Tensor, sample_index: int
    ) -> None:
        _logger.debug(f"Writing sample {sample_index}.")
        sample_prediction = sample_prediction.cpu().numpy()
        img_name, t_index, z_index = [batch["index"][i][sample_index] for i in range(3)]
        t_index = int(t_index)
        z_index = int(z_index)
        # account for lost slices in 2.5D
        z_index += self.z_padding
        z_slice = slice(z_index, z_index + sample_prediction.shape[-3])
        image = self._create_image(
            img_name, sample_prediction.shape, sample_prediction.dtype
        )
        _resize_image(image, t_index, z_slice)
        if self.write_input:
            source_stack = batch["source"][sample_index].cpu()
            center_slice_index = source_stack.shape[-3] // 2
            image[t_index, self.source_index, z_index] = source_stack[
                :, center_slice_index
            ]
            if "target" in batch:
                image[t_index, self.target_index, z_index] = batch["target"][
                    sample_index
                ][:, center_slice_index].cpu()
        # write CZYX
        if self.z_padding == 0 and sample_prediction.shape[-3] > 1:
            old_stack = image.oindex[t_index, self.prediction_index, z_slice]
            sample_prediction = _blend_in(old_stack, sample_prediction, z_slice)
        image.oindex[t_index, self.prediction_index, z_slice] = sample_prediction

    def _create_image(self, img_name: str, shape: tuple[int], dtype: DTypeLike):
        if img_name in self.plate.zgroup:
            return self.plate[img_name]
        _logger.debug(f"Creating image '{img_name}'")
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
