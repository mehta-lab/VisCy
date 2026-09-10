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

from viscy_utils.prediction_metadata import (
    PREDICTION_COMPLETE_KEY,
    clear_completion,
    completion_marker,
    mark_complete,
    prediction_run,
    same_run,
    tzyx_shape,
)
from viscy_utils.tensor_utils import to_numpy

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
        _logger.debug(f"Resizing image '{image.path}' {image.shape} for T={t_index}, Z-slice={z_slice}.")
        image.resize(
            (
                max(t_index + 1, image.shape[0]),
                image.channels,
                max(z_slice.stop, image.shape[2]),
                *image.shape[-2:],
            )
        )


def _blend_in(
    old_stack: torch.Tensor | NDArray,
    new_stack: torch.Tensor | NDArray,
    z_slice: slice,
) -> torch.Tensor | NDArray:
    """Blend a new stack into an old stack over a Z range with linear feathering.

    Supports both torch tensors (5D: B,C,Z,Y,X) for in-memory sliding window
    prediction and numpy arrays (4D: C,Z,Y,X) for out-of-core HCSPredictionWriter.

    Parameters
    ----------
    old_stack : torch.Tensor or NDArray
        Existing prediction stack to blend into.
    new_stack : torch.Tensor or NDArray
        New prediction stack to blend in.
    z_slice : slice
        Z-range of the new stack within the full volume.

    Returns
    -------
    torch.Tensor or NDArray
        Blended stack. Returns ``new_stack`` unchanged if ``z_slice.start == 0``.
    """
    if z_slice.start == 0:
        return new_stack
    depth = z_slice.stop - z_slice.start
    samples = min(z_slice.start + 1, depth)
    factors = []
    for i in reversed(list(range(depth))):
        factors.append(min(i + 1, samples))
    _logger.debug(f"Blending with factors {factors}.")
    if isinstance(old_stack, torch.Tensor):
        factors = torch.tensor(factors, dtype=old_stack.dtype, device=old_stack.device)
        factors = factors.view(1, 1, -1, 1, 1)
    else:
        factors = np.array(factors)[np.newaxis, :, np.newaxis, np.newaxis]
    return old_stack * (factors - 1) / factors + new_stack / factors


class HCSPredictionWriter(BasePredictionWriter):
    """Callback to store virtual staining predictions as HCS OME-Zarr.

    Single-process only: completion is tracked per process and channels are
    appended to the store without cross-rank coordination, so a multi-device
    predict is refused before the store is opened.

    Parameters
    ----------
    output_store : str
        Path to the zarr store to store output.
    overwrite : bool, optional
        When True, overwrite existing prediction channels in the output
        store instead of raising an error. Default False.
    write_input : bool, optional
        Write the source and target channels too
        (must be writing to a new store), by default False.
    write_interval : {'batch', 'epoch', 'batch_and_epoch'}, optional
        When to write, by default "batch".
    z_reduction : {'blend', 'center'}, optional
        How to combine the depth windows that cover one output plane, by
        default "blend".

        ``'blend'`` feathers each window in with :func:`_blend_in`. With the
        usual stride-1 depth windows that recurrence is the incremental mean,
        so every plane ends up the *unweighted mean of ``z_window_size``
        forward passes*. That is a real sharpness cost for a generative model:
        on an iPSC-trained pix2pix3d membrane prediction the mean over 8
        windows halves the mid-band power relative to a single pass
        (0.045 -> 0.021 of ground truth), and because all 8 share one XY grid
        it cancels none of the decoder's XY lattice.

        ``'center'`` writes each window from its center plane onward, so a
        later window overwrites everything but that center plane and the
        volume is assembled one plane per forward pass with no averaging.
        The leading ``z_window_size // 2`` planes come from the first window
        and the trailing ``z_window_size - 1 - z_window_size // 2`` from the
        last, since no window is centered on them. For odd ``z_window_size``
        that trailing count equals ``z_window_size // 2`` -- at ``w=5`` it is
        2, not 1. Every dynacell leaf that sets ``z_window_size`` uses an odd
        value.

        Both modes assume windows arrive in increasing ``z`` order within a
        ``(position, timepoint)`` -- ``'blend'`` because its factors are
        derived from ``z_slice.start``, ``'center'`` because it relies on
        last-write-wins. The predict dataloader is sequential and iterates
        ``z`` innermost, which is what makes that hold.
    checkpoint_path : str or None, optional
        Checkpoint the model predicts with. Its path and content hash are
        recorded in every FOV's completion marker, so a resume can tell
        predictions made with different weights apart and refuses to mix
        them in one store. Default None records no checkpoint.
    """

    def __init__(
        self,
        output_store: str,
        overwrite: bool = False,
        write_input: bool = False,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        z_reduction: Literal["blend", "center"] = "blend",
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__(write_interval)
        if z_reduction not in ("blend", "center"):
            raise ValueError(f"z_reduction must be 'blend' or 'center', got {z_reduction!r}")
        self.output_store = output_store
        self.overwrite = overwrite
        self.write_input = write_input
        self.z_reduction = z_reduction
        self.checkpoint_path = checkpoint_path
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
        if trainer.world_size > 1:
            raise NotImplementedError(
                f"HCSPredictionWriter tracks completion per process and appends channels without "
                f"cross-rank coordination; run predict on a single device (got world_size={trainer.world_size})."
            )
        dm: HCSDataModule = trainer.datamodule
        self._get_scale_metadata(dm.data_path)
        self.z_padding = dm.z_window_size // 2 if dm.target_2d else 0
        _logger.debug(f"Setting Z padding to {self.z_padding}")
        source_channel = dm.source_channel
        target_channel = dm.target_channel
        prediction_channel = [ch + "_prediction" for ch in target_channel]
        self._prediction_channels = prediction_channel
        # Hashes the checkpoint once; every FOV's marker records this identity.
        self._run = prediction_run(
            array_key=dm.array_key,
            z_window_size=dm.z_window_size,
            z_reduction=self.z_reduction,
            checkpoint_path=self.checkpoint_path,
        )
        window_arrays = dm.predict_dataset.window_arrays
        self._source_shapes = {f"/{array.path}": tzyx_shape(array) for array in window_arrays}
        # Array dimensions grow before writes, so only a successful write of every
        # distinct (T, Z-window) establishes completion, including overlapping windows.
        self._written_windows: dict[str, NDArray[np.bool_]] = {
            f"/{array.path}": np.zeros((array.frames, array.slices - dm.z_window_size + 1), dtype=bool)
            for array in window_arrays
        }
        run_positions = {array.path.rsplit("/", 1)[0] for array in window_arrays}
        if os.path.exists(self.output_store):
            if self.write_input:
                raise FileExistsError("Cannot write input to an existing store. Aborting.")
            else:
                self.plate = open_ome_zarr(self.output_store, mode="r+")
                # Validate all positions before mutating any.
                positions = dict(self.plate.positions())
                needs_append: list[tuple[str, list[str]]] = []
                overwritten: list[str] = []
                mixed: list[str] = []
                oversized: list[str] = []
                channel_orders: set[tuple[str, ...]] = set()
                for name, pos in positions.items():
                    existing = set(pos.channel_names)
                    missing = [ch for ch in prediction_channel if ch not in existing]
                    channel_orders.add((*pos.channel_names, *missing))
                    for ch in prediction_channel:
                        if ch in existing and not self.overwrite:
                            self.plate.close()
                            raise FileExistsError(
                                f"Channel '{ch}' already exists in "
                                f"'{self.output_store}'. "
                                f"Set overwrite=True to replace."
                            )
                        elif ch in existing and self.overwrite:
                            _logger.info(
                                "Overwriting existing channel '%s' in '%s'.",
                                ch,
                                self.output_store,
                            )
                    if missing:
                        needs_append.append((name, missing))
                    if name in run_positions:
                        overwritten.append(name)
                        if self._outruns_source(pos, name, dm.array_key):
                            oversized.append(name)
                    elif self._cannot_share(pos, prediction_channel):
                        mixed.append(name)
                if oversized:
                    self.plate.close()
                    raise ValueError(
                        f"{len(oversized)} FOVs in '{self.output_store}' hold more timepoints or depth "
                        f"slices than their source (e.g. {oversized[:3]}); arrays only grow, so the stale "
                        "planes would survive every rewrite yet the FOVs would be marked complete. "
                        "Predict into a new output store."
                    )
                if mixed:
                    self.plate.close()
                    raise ValueError(
                        f"{len(mixed)} FOVs outside this run already hold {prediction_channel} "
                        f"in '{self.output_store}' that were predicted before completion markers "
                        f"existed or with a different checkpoint or settings (e.g. {mixed[:3]}); "
                        "predict into a new output store instead of mixing them."
                    )
                if len(channel_orders) > 1:
                    self.plate.close()
                    raise ValueError(
                        f"Positions of '{self.output_store}' would disagree on channel order after "
                        f"appending {prediction_channel} ({sorted(channel_orders)}); the writer addresses "
                        "channels by one plate-wide index. Predict into a new output store."
                    )
                for name, channels in needs_append:
                    for ch in channels:
                        positions[name].append_channel(ch, resize_arrays=True)
                if needs_append:
                    # Plate.channel_names is cached from the first position when the
                    # store is opened; reopen so channel indices and positions created
                    # later in this run see the appended channel.
                    self.plate.close()
                    self.plate = open_ome_zarr(self.output_store, mode="r+")
                    positions = {name: self.plate[name] for name in overwritten}
                # This run replaces these channels' voxels, so an interrupted run
                # must never reuse their old completion. FOVs outside the run
                # (e.g. excluded on resume) keep theirs.
                for name in overwritten:
                    clear_completion(positions[name], prediction_channel)
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
        sample_prediction = to_numpy(sample_prediction)
        img_name, t_index, z_index = [batch["index"][i][sample_index] for i in range(3)]
        t_index = int(t_index)
        z_index = int(z_index)
        window_z_index = z_index
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
            if self.z_reduction == "blend":
                old_stack = image.oindex[t_index, self.prediction_index, z_slice]
                sample_prediction = _blend_in(old_stack, sample_prediction, z_slice)
            else:
                # Drop the planes before this window's center: the window centered on
                # each of them writes it later and wins. The first window keeps all of
                # them because nothing is centered on the leading planes.
                keep = 0 if z_index == 0 else sample_prediction.shape[-3] // 2
                sample_prediction = sample_prediction[..., keep:, :, :]
                z_slice = slice(z_slice.start + keep, z_slice.stop)
        image.oindex[t_index, self.prediction_index, z_slice] = sample_prediction
        written = self._written_windows[img_name]
        written[t_index, window_z_index] = True
        if written.all():
            position = self.plate[img_name.rsplit("/", 1)[0]]
            marker = completion_marker(self._source_shapes[img_name], self._run)
            mark_complete(position, self._prediction_channels, marker)

    def _outruns_source(self, position: Position, name: str, array_key: str) -> bool:
        """Return whether the existing output array outruns this run's source in T or Z.

        Parameters
        ----------
        position : Position
            Existing output position that this run rewrites.
        name : str
            Plate-relative position name.
        array_key : str
            Array level this run writes.

        Returns
        -------
        bool
            True when the array exists and has more frames or more depth slices
            than the source; no run writes beyond either extent, so the excess
            would keep stale voxels under a fresh completion marker.
        """
        try:
            output = position[array_key]
        except KeyError:
            return False
        frames, slices = self._source_shapes[f"/{name}/{array_key}"][:2]
        return output.frames > frames or output.slices > slices

    def _cannot_share(self, position: Position, channels: list[str]) -> bool:
        """Return whether ``position`` holds any of ``channels`` from a run other than this one.

        Parameters
        ----------
        position : Position
            Existing output position outside this run.
        channels : list of str
            This run's prediction channels.

        Returns
        -------
        bool
            True when the position carries a channel without any completion
            attribute (written before markers existed, so unverifiable), or a
            marker written with other weights or settings, or in a layout this
            version cannot read.
        """
        completed = position.zattrs.get(PREDICTION_COMPLETE_KEY)
        if completed is None:
            return any(channel in position.channel_names for channel in channels)
        return any(channel in completed and not same_run(completed[channel], self._run) for channel in channels)

    def _create_image(self, img_name: str, shape: tuple[int, ...], dtype: DTypeLike):
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
        # ``img_name in self.plate.zgroup`` does not reliably hit
        # in-progress positions (zarr3 nested-path lookups + the
        # leading slash in img_name combine to miss entries created
        # earlier in this same writer). On multi-timepoint inputs the
        # second batch for a given FOV would then re-enter the create
        # branch and crash with FileExistsError. Try-open is robust to
        # both cases.
        try:
            return self.plate[img_name]
        except KeyError:
            pass
        _logger.debug(f"Creating image '{img_name}'")
        _, row_name, col_name, pos_name, arr_name = img_name.split("/")
        position = self.plate.create_position(row_name, col_name, pos_name)
        # An empty marker from the first write on distinguishes an interrupted FOV
        # from one written before completion markers existed.
        clear_completion(position, self._prediction_channels)
        shape = [1] + list(shape)
        shape[1] = len(position.channel_names)
        return position.create_zeros(
            arr_name,
            shape=shape,
            dtype=dtype,
            chunks=_pad_shape(tuple(shape[-2:]), 5),
            transform=self._dataset_scale,
        )
