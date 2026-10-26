"""HCS prediction writer callback for OME-Zarr storage.

Stores virtual staining predictions in HCS OME-Zarr format,
with optional blending of overlapping depth slices.
"""

from __future__ import annotations

import functools
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np
import torch
from iohub.ngff import ImageArray, Plate, Position, TransformationMeta, open_ome_zarr
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from numpy.typing import DTypeLike, NDArray

from viscy_utils.prediction_metadata import (
    PREDICTION_COMPLETE_KEY,
    canonical_z_planes,
    checkpoint_signature,
    completion_marker,
    mark_complete,
    mark_started,
    outruns,
    predicted_z_planes,
    prediction_run,
    record_z_planes,
    same_run,
    tzyx_shape,
    z_planes_sha256_12,
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

    Supports torch tensors (B,C,Z,Y,X or C,Z,Y,X) for in-memory prediction and
    numpy arrays (C,Z,Y,X) for HCSPredictionWriter's per-window writes.

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
        factors = _torch_factors(tuple(factors), old_stack.dtype, old_stack.device)
    else:
        factors = np.array(factors)[np.newaxis, :, np.newaxis, np.newaxis]
    return old_stack * (factors - 1) / factors + new_stack / factors


@functools.lru_cache(maxsize=64)
def _torch_factors(factors: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Blend factors as a ``(Z, 1, 1)`` tensor, cached: a per-window host-to-device copy would sync every window."""
    # A normal tensor even when first built under predict's inference mode, so a later
    # grad-enabled caller can still save it for backward.
    with torch.inference_mode(False):
        return torch.tensor(factors, dtype=dtype, device=device).view(-1, 1, 1)


def _store_into(volume: torch.Tensor, z_slice: slice, planes: torch.Tensor) -> None:
    """Write ``planes`` into ``volume[:, z_slice]``, rounding to the volume dtype as numpy does.

    Torch rounds to float32 and float64 exactly like numpy. It converts float64 to
    float16 through float32, though, which can round twice, so other dtypes are cast
    by numpy on the host.

    Parameters
    ----------
    volume : torch.Tensor
        ``(C, Z, Y, X)`` volume buffer.
    z_slice : slice
        Planes of ``volume`` to write.
    planes : torch.Tensor
        ``(C, Z, Y, X)`` planes, on the volume's device.
    """
    if volume.dtype not in (torch.float32, torch.float64) and planes.dtype != volume.dtype:
        cast = planes.cpu().numpy().astype(torch.empty(0, dtype=volume.dtype).numpy().dtype)
        planes = torch.from_numpy(cast).to(volume.device)
    volume[:, z_slice] = planes


class HCSPredictionWriter(BasePredictionWriter):
    """Callback to store virtual staining predictions as HCS OME-Zarr.

    Single-process only: completion is tracked per process and channels are
    appended to the store without cross-rank coordination, so a multi-device
    predict is refused before the store is opened.

    Each (position, timepoint) volume is assembled in memory from its depth
    windows and written once, when its last window arrives (or at predict end),
    instead of a zarr read-modify-write per window; the stored bytes are the
    same. The write runs on a background thread while the next volume is
    predicted, one at a time and in order; a FOV's completion marker is written
    on that thread after its last volume, and a write error is raised at the
    next batch or at predict end. ``write_input`` and plane-restricted predicts (``predict_z_planes``) keep
    the per-window writes: the first stores slices beside each window, and the
    second writes single planes, which a volume flush would fill in between.

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
        them in one store. The file's size and mtime are taken when the
        writer is constructed -- the same instantiation pass that loads the
        model's weights -- and ``on_predict_start`` refuses a file changed
        since, so the recorded hash names the bytes the model loaded rather
        than whatever sits at the path later. Default None records no
        checkpoint.
    settings_sha256_12 : str or None, optional
        Hash of the other settings that shape the predicted voxels (model
        inference arguments, input normalization, precision), recorded in the
        marker alongside the checkpoint so a resume with the same weights but
        other settings is refused as well. The submitter computes it from the
        resolved config (``submit_benchmark_job.prediction_settings_sha256_12``);
        the writer cannot derive it from live objects. Default None records
        none.
    """

    def __init__(
        self,
        output_store: str,
        overwrite: bool = False,
        write_input: bool = False,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        z_reduction: Literal["blend", "center"] = "blend",
        checkpoint_path: str | None = None,
        settings_sha256_12: str | None = None,
    ) -> None:
        super().__init__(write_interval)
        if z_reduction not in ("blend", "center"):
            raise ValueError(f"z_reduction must be 'blend' or 'center', got {z_reduction!r}")
        self.output_store = output_store
        self.overwrite = overwrite
        self.write_input = write_input
        self.z_reduction = z_reduction
        self.checkpoint_path = checkpoint_path
        self._checkpoint_signature = None if checkpoint_path is None else checkpoint_signature(checkpoint_path)
        self.settings_sha256_12 = settings_sha256_12
        self._dataset_scale = None
        # Set by on_predict_start; None until then, so a failure during setup closes nothing.
        self.plate: Plate | None = None
        self._flush_pool: ThreadPoolExecutor | None = None

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
        if (
            self.checkpoint_path is not None
            and checkpoint_signature(self.checkpoint_path) != self._checkpoint_signature
        ):
            raise RuntimeError(
                f"Checkpoint '{self.checkpoint_path}' changed since this writer was constructed alongside the "
                "model, so its hash would not name the weights actually loaded. Restart the predict."
            )
        # Plane-restricted predict (HCSDataModule.predict_z_planes): only the listed
        # planes are predicted, the rest keep the fill value, and the planes are part
        # of the run identity so a resume never mixes them with another selection.
        self._z_planes = dm.predict_dataset.z_planes
        # write_input stores source/target slices beside each window, and a plane selection
        # writes single planes that a [first, last] flush would fill in between: both write
        # per window.
        self._write_once = not self.write_input and self._z_planes is None
        self._volumes: dict[tuple[str, int], tuple[torch.Tensor, int, int]] = {}
        # Volume writes (and the markers after them) run here, in submission order.
        self._flush_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prediction-flush")
        self._pending: Future | None = None
        # Hashes the checkpoint once; every FOV's marker records this identity.
        self._run = prediction_run(
            array_key=dm.array_key,
            z_window_size=dm.z_window_size,
            z_reduction=self.z_reduction,
            checkpoint_path=self.checkpoint_path,
            settings_sha256_12=self.settings_sha256_12,
            z_planes_sha256_12=None if self._z_planes is None else z_planes_sha256_12(self._z_planes),
        )
        window_arrays = dm.predict_dataset.window_arrays
        self._source_shapes = {f"/{array.path}": tzyx_shape(array) for array in window_arrays}
        # Array dimensions grow before writes, so only a successful write of every
        # distinct (T, Z-window) establishes completion, including overlapping windows.
        # A plane-restricted run marks the windows it skips as written up front.
        self._written_windows: dict[str, NDArray[np.bool_]] = {}
        for array in window_arrays:
            written = np.zeros((array.frames, array.slices - dm.z_window_size + 1), dtype=bool)
            if self._z_planes is not None:
                written[:] = True
                for t, planes in enumerate(self._z_planes[array.path.rsplit("/", 1)[0]]):
                    written[t, list(planes)] = False
            self._written_windows[f"/{array.path}"] = written
        run_positions = {array.path.rsplit("/", 1)[0] for array in window_arrays}
        existed = os.path.exists(self.output_store)
        if existed:
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
                foreign: list[str] = []
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
                        if self._holds_other_planes(pos, name, prediction_channel):
                            foreign.append(name)
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
                if foreign:
                    self.plate.close()
                    raise ValueError(
                        f"{len(foreign)} FOVs in '{self.output_store}' hold {prediction_channel} that another run "
                        f"predicted on other planes (e.g. {foreign[:3]}); a plane-restricted predict rewrites only "
                        "its own planes, so remove that store (or predict into a new one) first."
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
                # A started marker names this run as the writer of a channel's voxels.
                # Run positions get one for every prediction channel: this run
                # replaces their voxels, so an interrupted run must never reuse an
                # old completion, and the next resume sees unfinished FOVs, not
                # legacy ones. FOVs outside the run (e.g. excluded on resume) keep
                # their markers, except for the channels appended below: those hold
                # zeros this run allocated, and without a marker a later resume
                # could not tell them from data predicted before markers existed.
                started = {name: prediction_channel for name in overwritten}
                for name, channels in needs_append:
                    started.setdefault(name, channels)
                    for ch in channels:
                        positions[name].append_channel(ch, resize_arrays=True)
                if needs_append:
                    # Plate.channel_names is cached from the first position when the
                    # store is opened; reopen so channel indices and positions created
                    # later in this run see the appended channel.
                    self.plate.close()
                    self.plate = open_ome_zarr(self.output_store, mode="r+")
                    positions = {name: self.plate[name] for name in started}
                for name, channels in started.items():
                    mark_started(positions[name], channels, self._run)
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
        # Read now: once volume writes run on the flush thread, this thread leaves the store alone.
        self._store_dtypes: dict[str, torch.dtype] = {}
        if existed:
            for img_name in self._source_shapes:
                try:
                    array = self.plate[img_name]
                except KeyError:
                    continue  # created by this run, in the prediction's dtype
                self._store_dtypes[img_name] = torch.from_numpy(np.empty(0, dtype=array.dtype)).dtype

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
        if self._pending is not None and self._pending.done():
            self._pending.result()  # raise a failed volume write here, not volumes later
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
        try:
            for img_name, t_index in list(self._volumes):
                self._flush_volume(img_name, t_index, complete=False)
        finally:
            # Even when a write failed.
            self._close()
        if self._pending is not None:
            self._pending.result()

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Close the store when predict fails, e.g. on a volume write that failed mid-run.

        Parameters
        ----------
        trainer : Trainer
            The Lightning trainer instance.
        pl_module : LightningModule
            The Lightning module.
        exception : BaseException
            The exception that stopped predict.
        """
        self._close()

    def _close(self) -> None:
        """Let the last volume write finish, then close the store; safe after a partial setup or a second call."""
        if self._flush_pool is not None:
            self._flush_pool.shutdown(wait=True)
        if self.plate is not None:
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
        img_name, t_index, z_index = [batch["index"][i][sample_index] for i in range(3)]
        t_index = int(t_index)
        z_index = int(z_index)
        window_z_index = z_index
        # account for lost slices in 2.5D
        z_index += self.z_padding
        z_slice = slice(z_index, z_index + sample_prediction.shape[-3])
        written = self._written_windows[img_name]
        if written[t_index, window_z_index]:
            # A repeat would blend into a volume already flushed (a zero buffer) or blend twice.
            raise RuntimeError(f"Window z={window_z_index} of {img_name} t={t_index} arrived twice.")
        # Blend reads and the write go to the in-memory volume, or to the store per window.
        if self._write_once:
            # The volume stays on the prediction's device and reaches the host once, at its flush.
            sample_prediction = sample_prediction.detach()
            if sample_prediction.dtype == torch.bfloat16:
                sample_prediction = sample_prediction.float()  # to_numpy's cast
            dest, lead = self._buffer_volume(img_name, t_index, sample_prediction, z_slice), (slice(None),)
        else:
            sample_prediction = to_numpy(sample_prediction)
            image = self._create_image(img_name, sample_prediction.shape, sample_prediction.dtype)
            _resize_image(image, t_index, z_slice)
            dest, lead = image.oindex, (t_index, self.prediction_index)
        if self.write_input:
            source_stack = batch["source"][sample_index].cpu()
            center_slice_index = source_stack.shape[-3] // 2
            image[t_index, self.source_index, z_index] = source_stack[:, center_slice_index]
            if "target" in batch:
                image[t_index, self.target_index, z_index] = batch["target"][sample_index][:, center_slice_index].cpu()
        # write CZYX
        if self.z_padding == 0 and sample_prediction.shape[-3] > 1:
            if self.z_reduction == "blend":
                old_stack = dest[(*lead, z_slice)]
                if self._write_once and z_slice.start != 0:
                    # The numpy blend promotes to float64 (integer factors); so does this one.
                    # The first window is stored as predicted, so it needs no float64 copies.
                    old_stack, sample_prediction = old_stack.double(), sample_prediction.double()
                sample_prediction = _blend_in(old_stack, sample_prediction, z_slice)
            else:
                # Drop the planes before this window's center: the window centered on
                # each of them writes it later and wins. The first window keeps all of
                # them because nothing is centered on the leading planes.
                keep = 0 if z_index == 0 else sample_prediction.shape[-3] // 2
                sample_prediction = sample_prediction[..., keep:, :, :]
                z_slice = slice(z_slice.start + keep, z_slice.stop)
        # The buffer has the store's dtype, so this casts like the zarr write and blending
        # reads back what the store would hold.
        if self._write_once:
            _store_into(dest, z_slice, sample_prediction)
        else:
            dest[(*lead, z_slice)] = sample_prediction
        written[t_index, window_z_index] = True
        if self._write_once:
            if written[t_index].all():
                self._flush_volume(img_name, t_index, complete=written.all())
        elif written.all():
            self._mark_complete(img_name)

    def _mark_complete(self, img_name: str) -> None:
        """Write the completion marker of the FOV holding ``img_name``; every window is in the store."""
        name = img_name.strip("/").rsplit("/", 1)[0]
        position = self.plate[name]
        planes = None if self._z_planes is None else self._z_planes[name]
        record_z_planes(position, self._prediction_channels, planes)
        marker = completion_marker(self._source_shapes[img_name], self._run)
        mark_complete(position, self._prediction_channels, marker)

    def _buffer_volume(
        self, img_name: str, t_index: int, sample_prediction: torch.Tensor, z_slice: slice
    ) -> torch.Tensor:
        """Return the in-memory CZYX volume of ``(img_name, t_index)``, allocating it on first use.

        Parameters
        ----------
        img_name : str
            Hierarchical path name of the source image.
        t_index : int
            Timepoint of the volume.
        sample_prediction : torch.Tensor
            First window of the volume; sets channel count, YX shape, device, and the dtype
            unless the store already holds this array (then its dtype, as the per-window
            writes would cast to it).
        z_slice : slice
            Planes of the incoming window. The flush writes from the first plane any
            window reached to the last, so planes no window predicts (``target_2d``
            padding, unselected planes) keep what the store holds, as with per-window
            writes.

        Returns
        -------
        torch.Tensor
            Zero-filled ``(C, Z, Y, X)`` buffer spanning the source depth, like a new zarr array.
        """
        key = (img_name, t_index)
        if key not in self._volumes:
            depth = self._source_shapes[img_name][1]
            shape = (sample_prediction.shape[0], depth, *sample_prediction.shape[-2:])
            dtype = self._store_dtypes.get(img_name, sample_prediction.dtype)
            volume = torch.zeros(shape, dtype=dtype, device=sample_prediction.device)
            self._volumes[key] = (volume, z_slice.start, z_slice.stop)
        volume, start, stop = self._volumes[key]
        if z_slice.stop > volume.shape[1]:
            # A prediction deeper than its window: grow like the per-window resize does.
            extra = volume.new_zeros((volume.shape[0], z_slice.stop - volume.shape[1], *volume.shape[2:]))
            volume = torch.cat([volume, extra], dim=1)
        self._volumes[key] = (volume, min(start, z_slice.start), max(stop, z_slice.stop))
        return volume

    def _flush_volume(self, img_name: str, t_index: int, complete: bool) -> None:
        """Copy one buffered volume to the host and queue its write, over the planes its windows reached.

        Parameters
        ----------
        img_name : str
            Hierarchical path name of the source image.
        t_index : int
            Timepoint of the volume.
        complete : bool
            The volume is the FOV's last: mark the FOV complete once it is written.
        """
        volume, start, stop = self._volumes.pop((img_name, t_index))
        planes = volume[:, start:stop].cpu().numpy()
        if self._pending is not None:
            # One write in flight bounds the host copies held to two volumes.
            self._pending.result()
        self._pending = self._flush_pool.submit(self._write_volume, img_name, t_index, planes, start, complete)

    def _write_volume(self, img_name: str, t_index: int, planes: NDArray, start: int, complete: bool) -> None:
        """Write planes ``start:`` of one volume to the store, then the FOV's marker if ``complete`` (flush thread)."""
        stop = start + planes.shape[1]
        image = self._create_image(img_name, (planes.shape[0], stop, *planes.shape[2:]), planes.dtype)
        _resize_image(image, t_index, slice(start, stop))
        image.oindex[t_index, self.prediction_index, slice(start, stop)] = planes
        if complete:
            self._mark_complete(img_name)

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
            True when the array exists and :func:`outruns` the source.
        """
        try:
            output = position[array_key]
        except KeyError:
            return False
        return outruns(output, self._source_shapes[f"/{name}/{array_key}"])

    def _cannot_share(self, position: Position, channels: list[str]) -> bool:
        """Return whether ``position`` holds any of ``channels`` that this run cannot vouch for.

        Parameters
        ----------
        position : Position
            Existing output position outside this run.
        channels : list of str
            This run's prediction channels.

        Returns
        -------
        bool
            True when a channel present in the position has no marker of its
            own (written before markers existed, so unverifiable even if other
            channels are marked), or a marker written with other weights or
            settings, or in a layout this version cannot read.
        """
        completed = position.zattrs.get(PREDICTION_COMPLETE_KEY, {})
        return any(
            channel in position.channel_names and not same_run(completed.get(channel), self._run)
            for channel in channels
        )

    def _holds_other_planes(self, position: Position, name: str, channels: list[str]) -> bool:
        """Return whether a plane-restricted run would leave another run's planes in ``position``.

        Overwrite rewrites in place and this run writes only its own planes, so a
        channel another run wrote survives off them unless that run predicted exactly
        these planes (e.g. the same slab with a new checkpoint). A full-volume run
        rewrites every plane and never holds any.
        """
        if self._z_planes is None or not self._cannot_share(position, channels):
            return False
        planes = canonical_z_planes(self._z_planes[name])
        return any(
            predicted_z_planes(position, channel) != planes for channel in channels if channel in position.channel_names
        )

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
        # A started marker from the first write on distinguishes an interrupted FOV
        # from one written before completion markers existed.
        mark_started(position, self._prediction_channels, self._run)
        shape = [1] + list(shape)
        if self._z_planes is not None:
            # Allocate the full source extent: the planes this run skips must still
            # exist (at the fill value) so the store aligns plane by plane with its
            # source, instead of ending at the last predicted plane.
            shape[0], shape[2] = self._source_shapes[img_name][:2]
        shape[1] = len(position.channel_names)
        return position.create_zeros(
            arr_name,
            shape=shape,
            dtype=dtype,
            chunks=_pad_shape(tuple(shape[-2:]), 5),
            transform=self._dataset_scale,
        )
