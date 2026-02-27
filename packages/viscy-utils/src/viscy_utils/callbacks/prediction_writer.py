"""HCS prediction writer callback for OME-Zarr storage.

Stores virtual staining predictions in HCS OME-Zarr format,
with optional blending of overlapping depth slices.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np
import torch
from iohub.ngff import ImageArray, Plate, Position, TransformationMeta, open_ome_zarr
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from numpy.typing import DTypeLike, NDArray

if TYPE_CHECKING:
    from viscy_data import HCSDataModule, Sample

__all__ = ["HCSPredictionWriter"]
_logger = logging.getLogger("lightning.pytorch")


def _pad_shape(shape: tuple[int, ...], target: int = 5) -> tuple[int, ...]:
    """Pad shape tuple to a target length.

    Vendored from ``iohub.ngff.nodes._pad_shape()``.

    Parameters
    ----------
    shape : tuple of int
        The shape tuple to pad.
    target : int, optional
        Target length to pad to, by default 5.

    Returns
    -------
    tuple of int
        Padded shape tuple with leading 1s.
    """
    pad = target - len(shape)
    return (1,) * pad + shape


def _resize_image(image: ImageArray, t_index: int, z_slice: slice) -> None:
    """Resize image array if incoming stack is not within bounds.

    Parameters
    ----------
    image : ImageArray
        The image array to potentially resize.
    t_index : int
        Time index for the incoming data.
    z_slice : slice
        Z-slice range for the incoming data.
    """
    if image.shape[0] <= t_index or image.shape[2] < z_slice.stop:
        _logger.debug(f"Resizing image '{image.name}' {image.shape} for T={t_index}, Z-sclice={z_slice}.")
        image.resize(
            (
                max(t_index + 1, image.shape[0]),
                image.channels,
                max(z_slice.stop, image.shape[2]),
                *image.shape[-2:],
            )
        )


def _blend_in(old_stack: NDArray, new_stack: NDArray, z_slice: slice) -> NDArray:
    """Blend a new stack of images into an old stack over a specified range of slices.

    This function blends the ``new_stack`` of images into the ``old_stack`` over the
    range specified by ``z_slice``. The blending is done using a weighted average where
    the weights are determined by the position within the range of slices. If the start
    of ``z_slice`` is 0, the function returns the ``new_stack`` unchanged.

    Parameters
    ----------
    old_stack : NDArray
        The original stack of images to be blended.
    new_stack : NDArray
        The new stack of images to blend into the original stack.
    z_slice : slice
        A slice object indicating the range of slices over which to perform
        the blending. The start and stop attributes determine the range.

    Returns
    -------
    NDArray
        The blended stack of images. If ``z_slice.start`` is 0, returns
        ``new_stack`` unchanged.
    """
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

    Parameters
    ----------
    output_store : str
        Path to the zarr store to store output.
    write_input : bool, optional
        Write the source and target channels too
        (must be writing to a new store), by default False.
    write_interval : {'batch', 'epoch', 'batch_and_epoch'}, optional
        When to write, by default "batch".
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
        self._dataset_scale = None

    def _get_scale_metadata(self, metadata_store: os.PathLike | None) -> None:
        """Read scale metadata from an existing zarr store.

        Parameters
        ----------
        metadata_store : PathLike or None
            Path to the zarr store containing scale metadata.
        """
        if metadata_store is not None:
            with open_ome_zarr(metadata_store, mode="r") as metadata_store:
                if isinstance(metadata_store, Position):
                    self._dataset_scale = [TransformationMeta(type="scale", scale=metadata_store.scale)]
                elif isinstance(metadata_store, Plate):
                    for _, pos in metadata_store.positions():
                        self._dataset_scale = [TransformationMeta(type="scale", scale=pos.scale)]
                        break
                _logger.debug(f"Dataset scale {self._dataset_scale}.")

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set up the output zarr store for writing predictions.

        Parameters
        ----------
        trainer : Trainer
            The Lightning trainer instance.
        pl_module : LightningModule
            The Lightning module being used for prediction.
        """
        dm: HCSDataModule = trainer.datamodule
        self._get_scale_metadata(dm.data_path)
        self.z_padding = dm.z_window_size // 2 if dm.target_2d else 0
        _logger.debug(f"Setting Z padding to {self.z_padding}")
        source_channel = dm.source_channel
        target_channel = dm.target_channel
        prediction_channel = [ch + "_prediction" for ch in target_channel]
        if os.path.exists(self.output_store):
            if self.write_input:
                raise FileExistsError("Cannot write input to an existing store. Aborting.")
            else:
                with open_ome_zarr(self.output_store, mode="r+") as plate:
                    for _, pos in plate.positions():
                        for ch in prediction_channel:
                            pos.append_channel(ch, resize_arrays=True)
                self.plate = open_ome_zarr(self.output_store, mode="r+")
        else:
            channel_names = prediction_channel
            if self.write_input:
                channel_names = source_channel + channel_names
            self.plate = open_ome_zarr(
                self.output_store,
                layout="hcs",
                mode="a",
                channel_names=channel_names,
            )
        _logger.info(f"Writing prediction to: '{self.plate.zgroup.store.root}'.")
        if self.write_input:
            self.source_index = self._get_channel_indices(source_channel)
            self.target_index = self._get_channel_indices(target_channel)
        self.prediction_index = self._get_channel_indices(prediction_channel)

    def _get_channel_indices(self, channel_names: list[str]) -> list[int]:
        """Get channel indices from the plate for given channel names.

        Parameters
        ----------
        channel_names : list of str
            Channel names to look up.

        Returns
        -------
        list of int
            Corresponding channel indices.
        """
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
        """Write predictions at the end of each batch.

        Parameters
        ----------
        trainer : Trainer
            The Lightning trainer instance.
        pl_module : LightningModule
            The Lightning module.
        prediction : torch.Tensor
            Model predictions for the batch.
        batch_indices : sequence of int or None
            Indices of the batch samples.
        batch : Sample
            The input batch.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int
            Index of the current dataloader.
        """
        _logger.debug(f"Writing batch {batch_idx}.")
        for sample_index, _ in enumerate(batch["index"][0]):
            self.write_sample(batch, prediction[sample_index], sample_index)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Close the zarr store after prediction completes.

        Parameters
        ----------
        trainer : Trainer
            The Lightning trainer instance.
        pl_module : LightningModule
            The Lightning module.
        """
        self.plate.close()

    def write_sample(
        self,
        batch: Sample,
        sample_prediction: torch.Tensor,
        sample_index: int,
    ) -> None:
        """Write a single sample prediction to the zarr store.

        Parameters
        ----------
        batch : Sample
            The input batch containing index metadata.
        sample_prediction : torch.Tensor
            Prediction tensor for this sample.
        sample_index : int
            Index of this sample within the batch.
        """
        _logger.debug(f"Writing sample {sample_index}.")
        sample_prediction = sample_prediction.cpu().numpy()
        img_name, t_index, z_index = [batch["index"][i][sample_index] for i in range(3)]
        t_index = int(t_index)
        z_index = int(z_index)
        # account for lost slices in 2.5D
        z_index += self.z_padding
        z_slice = slice(z_index, z_index + sample_prediction.shape[-3])
        image = self._create_image(img_name, sample_prediction.shape, sample_prediction.dtype)
        _resize_image(image, t_index, z_slice)
        if self.write_input:
            source_stack = batch["source"][sample_index].cpu()
            center_slice_index = source_stack.shape[-3] // 2
            image[t_index, self.source_index, z_index] = source_stack[:, center_slice_index]
            if "target" in batch:
                image[t_index, self.target_index, z_index] = batch["target"][sample_index][:, center_slice_index].cpu()
        # write CZYX
        if self.z_padding == 0 and sample_prediction.shape[-3] > 1:
            old_stack = image.oindex[t_index, self.prediction_index, z_slice]
            sample_prediction = _blend_in(old_stack, sample_prediction, z_slice)
        image.oindex[t_index, self.prediction_index, z_slice] = sample_prediction

    def _create_image(self, img_name: str, shape: tuple[int], dtype: DTypeLike):
        """Create or retrieve an image in the zarr store.

        Parameters
        ----------
        img_name : str
            Hierarchical path name for the image.
        shape : tuple of int
            Shape of the prediction array.
        dtype : DTypeLike
            Data type for the array.

        Returns
        -------
        ImageArray
            The created or existing image array.
        """
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
            transform=self._dataset_scale,
        )
