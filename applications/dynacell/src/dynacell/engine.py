"""Dynacell LightningModules for virtual staining benchmarks.

Provides :class:`DynacellUNet` for supervised regression and
:class:`DynacellFlowMatching` for flow-matching generative staining.
"""

import inspect
import itertools
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import DivisiblePad
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR

from dynacell.celldiff_wrapper import CELLDiff3DVS
from viscy_data import Sample
from viscy_models import Unet3d, UNeXt2
from viscy_models.celldiff import CELLDiffNet, UNetViT3D
from viscy_utils.log_images import detach_sample, log_image_grid

_ARCHITECTURE: dict[str, type[nn.Module]] = {
    "UNetViT3D": UNetViT3D,
    "FNet3D": Unet3d,
    "UNeXt2": UNeXt2,
}


def _configure_adamw_scheduler(
    module: LightningModule,
    model: nn.Module,
    lr: float,
    schedule: str,
) -> tuple[list, list]:
    """Build AdamW optimizer with WarmupCosine or Constant LR schedule.

    Shared by :class:`DynacellUNet` and :class:`DynacellFlowMatching`.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if schedule == "WarmupCosine":
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=3,
            t_total=module.trainer.estimated_stepping_batches,
            warmup_multiplier=1e-3,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    elif schedule == "Constant":
        scheduler = ConstantLR(optimizer, factor=1, total_iters=module.trainer.max_epochs)
    else:
        raise ValueError(f"Unknown schedule {schedule!r}, expected 'WarmupCosine' or 'Constant'")
    return [optimizer], [scheduler]


def _aggregate_validation_losses(
    validation_losses: list[list[tuple[Tensor, int]]],
) -> Tensor:
    """Compute sample-weighted mean loss across dataloaders.

    Parameters
    ----------
    validation_losses : list of list of (Tensor, int)
        Per-dataloader list of ``(scalar_loss, batch_size)`` tuples
        accumulated during validation.

    Returns
    -------
    Tensor
        Scalar weighted mean loss.
    """
    dl_means: list[Tensor] = []
    dl_totals: list[Tensor] = []
    for dl_batches in validation_losses:
        losses, sizes = zip(*dl_batches)
        sizes_t = torch.tensor(sizes, dtype=torch.float, device=losses[0].device)
        dl_means.append((torch.stack(losses) * sizes_t).sum() / sizes_t.sum())
        dl_totals.append(sizes_t.sum())
    total_n = torch.stack(dl_totals).sum()
    weighted = torch.stack([m * n for m, n in zip(dl_means, dl_totals)]).sum()
    return weighted / total_n


def _make_divisible_pad(model: nn.Module) -> DivisiblePad:
    """Build a DivisiblePad matching the model's spatial downsampling axes.

    Parameters
    ----------
    model : nn.Module
        A model with ``num_blocks`` and optionally ``downsamples_z``.

    Returns
    -------
    DivisiblePad
        Pads YX (and Z if ``downsamples_z``) to the nearest multiple of
        ``2**num_blocks``.
    """
    down_factor = 2**model.num_blocks
    if getattr(model, "downsamples_z", False):
        return DivisiblePad((0, down_factor, down_factor, down_factor))
    return DivisiblePad((0, 0, down_factor, down_factor))


def _center_crop_to_shape(tensor: Tensor, spatial_shape: tuple[int, ...]) -> Tensor:
    """Center-crop trailing spatial dimensions to the requested shape."""
    slices = [slice(None)] * tensor.ndim
    start_dim = tensor.ndim - len(spatial_shape)
    for dim, size in enumerate(spatial_shape, start=start_dim):
        current = tensor.shape[dim]
        if current < size:
            raise ValueError(f"Cannot crop dimension {dim} from {current} to {size}")
        start = (current - size) // 2
        slices[dim] = slice(start, start + size)
    return tensor[tuple(slices)]


class DynacellUNet(LightningModule):
    """Supervised regression U-Net for benchmark virtual staining.

    Parameters
    ----------
    architecture : {"UNetViT3D", "FNet3D", "UNeXt2"}
        Architecture key selecting the backbone.
    model_config : dict | None
        Keyword arguments forwarded to the backbone constructor.
    loss_function : nn.Module | None
        Loss function. Defaults to ``nn.MSELoss()``.
    lr : float
        Learning rate.
    schedule : {"WarmupCosine", "Constant"}
        LR scheduler type.
    log_batches_per_epoch : int
        Batches to log images per epoch.
    log_samples_per_batch : int
        Samples per batch to log.
    example_input_yx_shape : Sequence[int]
        YX shape for example input (used by FNet3D for graph logging).
        Ignored when the model provides ``input_spatial_size``.
    ckpt_path : str | None
        Path to a checkpoint to load **weights only** at construction time.
        Intended for inference (predict/test), not training resumption —
        optimizer state, epoch counters, and scheduler state are not
        restored.
    """

    def __init__(
        self,
        architecture: Literal["UNetViT3D", "FNet3D", "UNeXt2"] = "UNetViT3D",
        model_config: dict | None = None,
        loss_function: nn.Module | None = None,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        example_input_yx_shape: Sequence[int] = (256, 256),
        predict_method: Literal["full_image", "sliding_window"] = "full_image",
        predict_overlap: tuple[int, int, int] = (4, 256, 256),
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_function", "ckpt_path"])
        if model_config is None:
            model_config = {}
        net_class = _ARCHITECTURE.get(architecture)
        if net_class is None:
            raise ValueError(f"Architecture {architecture!r} not in {set(_ARCHITECTURE)}")
        self.model = net_class(**model_config)
        self.loss_function = loss_function if loss_function is not None else nn.MSELoss()
        self.lr = lr
        self.schedule = schedule
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.predict_method = predict_method
        self.predict_overlap = predict_overlap

        self.training_step_outputs: list = []
        # Each entry is a list of (loss, batch_size) tuples for weighted aggregation.
        self.validation_losses: list[list[tuple[Tensor, int]]] = []
        self.validation_step_outputs: list = []

        # Cache fg_mask compatibility to avoid per-batch inspect.signature().
        sig = inspect.signature(self.loss_function.forward)
        self._loss_accepts_fg_mask = "fg_mask" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        # Build example_input_array for graph logging (TensorBoard/W&B).
        in_channels = model_config.get("in_channels") or 1
        if hasattr(self.model, "input_spatial_size"):
            # UNetViT3D: must use exact spatial dims.
            d, h, w = self.model.input_spatial_size
        else:
            # FNet3D: flexible spatial, use in_stack_depth + user YX.
            d = model_config.get("in_stack_depth") or 5
            h, w = example_input_yx_shape
        self.example_input_array = torch.rand(1, in_channels, d, h, w)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu")["state_dict"])

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass through the model.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor
            Model output.
        """
        return self.model(x)

    def _compute_loss(self, pred: Tensor, target: Tensor, batch: Sample) -> Tensor:
        """Compute loss, optionally passing fg_mask to the loss function."""
        if "fg_mask" in batch:
            if not self._loss_accepts_fg_mask:
                raise TypeError(
                    f"{type(self.loss_function).__name__} does not accept 'fg_mask'. "
                    f"Use SpotlightLoss or remove fg_mask_key from the data config."
                )
            return self.loss_function(pred, target, fg_mask=batch["fg_mask"])
        return self.loss_function(pred, target)

    def training_step(self, batch: Sample, batch_idx: int) -> Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : Sample
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            Training loss.
        """
        source = batch["source"]
        target = batch["target"]
        pred = self.forward(source)
        loss = self._compute_loss(pred, target, batch)
        if batch_idx < self.log_batches_per_epoch:
            self.training_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        return loss

    def validation_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        """Execute a single validation step.

        Parameters
        ----------
        batch : Sample
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        source: Tensor = batch["source"]
        target: Tensor = batch["target"]
        pred = self.forward(source)
        loss = self._compute_loss(pred, target, batch)
        if dataloader_idx + 1 > len(self.validation_losses):
            self.validation_losses.append([])
        self.validation_losses[dataloader_idx].append((loss.detach(), source.shape[0]))
        self.log(
            f"loss/val/{dataloader_idx}",
            loss,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))

    def on_predict_start(self) -> None:
        """Build the divisible-pad transform for tiled inference."""
        self._predict_pad = _make_divisible_pad(self.model)

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Execute a single prediction step.

        Pads the input tile to the nearest multiple of the model's downsampling
        factor, runs the forward pass, then crops back to the original shape.

        Parameters
        ----------
        batch : Sample
            Input batch. Only ``"source"`` is used.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index, defaults to 0.

        Returns
        -------
        Tensor
            Model prediction, cropped to the input spatial shape.
        """
        source = batch["source"]
        original_shape = source.shape[2:]
        source = self._predict_pad(source)
        if self.predict_method == "full_image":
            prediction = self.forward(source)
        elif self.predict_method == "sliding_window":
            prediction = self.predict_sliding_window(source, overlap_size=self.predict_overlap)
        else:
            raise ValueError(
                f"Unknown predict_method: {self.predict_method!r}. Choose 'full_image' or 'sliding_window'."
            )
        return _center_crop_to_shape(prediction, original_shape)

    def on_train_epoch_end(self):
        """Log training image samples."""
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        """Log validation samples and aggregate loss weighted by batch size."""
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)
        if self.validation_losses:
            self.log("loss/validate", _aggregate_validation_losses(self.validation_losses), sync_dist=True)
        self.validation_step_outputs.clear()
        self.validation_losses.clear()

    def configure_optimizers(self):
        """Configure AdamW optimizer with LR scheduler."""
        return _configure_adamw_scheduler(self, self.model, self.lr, self.schedule)

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        """Log image grid to the active logger."""
        if not imgs or not self.trainer.is_global_zero or self.logger is None:
            return
        log_image_grid(self.logger, key, imgs, self.current_epoch)

    def predict_sliding_window(self, source: Tensor, overlap_size: tuple[int, int, int] = (4, 256, 256)) -> Tensor:
        """Run sliding-window inference over a large input volume.

        Overlapping regions are averaged across all covering patches.

        Parameters
        ----------
        source : Tensor
            Input tensor of shape ``(B, C, D, H, W)``.
        overlap_size : tuple of int
            Overlap in ``(D, H, W)`` between adjacent patches.

        Returns
        -------
        Tensor
            Prediction with the same spatial shape as ``source``.
        """
        spatial = source.shape[-3:]
        patch_spatial = tuple(self.model.input_spatial_size)
        n_spatial = 3
        overlap = tuple(overlap_size)

        for i in range(n_spatial):
            S, P, ov = spatial[i], patch_spatial[i], overlap[i]
            if S < P:
                raise ValueError(f"spatial dim {i} size {S} must be >= patch size {P}")
            if not (0 <= ov < P):
                raise ValueError(f"overlap at dim {i} must satisfy 0 <= overlap < patch (got {ov} vs {P})")

        prediction_sum = torch.zeros_like(source)
        prediction_count = torch.zeros_like(source)

        start_lists = []
        for i in range(n_spatial):
            S, P, ov = spatial[i], patch_spatial[i], overlap[i]
            stride = P - ov
            last = S - P
            starts = [0]
            while starts[-1] + stride < last:
                starts.append(starts[-1] + stride)
            if starts[-1] != last:
                starts.append(last)
            start_lists.append(starts)

        with torch.no_grad():
            for starts in itertools.product(*start_lists):
                slicer: list = [slice(None)] * source.ndim
                for i, st in enumerate(starts):
                    slicer[-(n_spatial - i)] = slice(st, st + patch_spatial[i])
                patch_out = self.forward(source[tuple(slicer)])
                prediction_sum[tuple(slicer)] += patch_out
                prediction_count[tuple(slicer)] += 1

        if not torch.all(prediction_count > 0):
            raise RuntimeError("sliding window left uncovered voxels")
        return prediction_sum / prediction_count


class DynacellFlowMatching(LightningModule):
    """Flow-matching LightningModule for generative virtual staining.

    Wraps :class:`~dynacell.celldiff_wrapper.CELLDiff3DVS` for training,
    validation image logging, and ODE-based prediction.  The flow-matching
    loss is computed entirely inside ``CELLDiff3DVS.forward``; no external
    loss function is needed.

    Parameters
    ----------
    net_config : dict or None
        Keyword arguments forwarded to ``CELLDiffNet``.
    transport_config : dict or None
        Keyword arguments forwarded to ``CELLDiff3DVS`` (excluding ``net``).
        Supports ``path_type``, ``prediction``, ``loss_weight``, ``train_eps``,
        ``sample_eps``.
    lr : float
        Learning rate for AdamW optimizer.
    schedule : {"WarmupCosine", "Constant"}
        Learning rate schedule.
    log_batches_per_epoch : int
        Number of batches per epoch to accumulate for image logging.
    log_samples_per_batch : int
        Number of samples per batch to log.
    num_generate_steps : int
        Number of ODE steps for prediction inference.
    num_log_steps : int
        Number of ODE steps for validation image generation (cheaper than
        ``num_generate_steps``).
    compute_validation_loss : bool
        Whether to compute and log flow-matching validation loss on the
        validation loader. Disabled by default to preserve the previous
        cheaper validation behavior.
    predict_method : {"denoise", "generate", "sliding_window", "iterative"}
        Prediction generation method.  ``"generate"`` runs single-patch ODE
        (default, matches standard HCS tile workflow).
    predict_overlap : int or tuple of int
        Overlap for sliding-window prediction.
    ckpt_path : str | None
        Path to a checkpoint to load **weights only** at construction time.
        Intended for inference (predict/test), not training resumption —
        optimizer state, epoch counters, and scheduler state are not
        restored.  Bypasses LightningCLI's checkpoint hparam merging, so
        predict-time settings (``predict_method``, ``predict_overlap``,
        etc.) are taken from the config rather than the checkpoint.
    """

    def __init__(
        self,
        net_config: dict | None = None,
        transport_config: dict | None = None,
        lr: float = 1e-4,
        schedule: Literal["WarmupCosine", "Constant"] = "WarmupCosine",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        num_generate_steps: int = 100,
        num_log_steps: int = 10,
        compute_validation_loss: bool = False,
        predict_method: Literal["denoise", "generate", "sliding_window", "iterative"] = "generate",
        predict_overlap: int | tuple[int, int, int] = 256,
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["predict_method", "predict_overlap", "num_generate_steps", "num_log_steps", "ckpt_path"]
        )
        net = CELLDiffNet(**(net_config or {}))
        self.model = CELLDiff3DVS(net, **(transport_config or {}))
        self.lr = lr
        self.schedule = schedule
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.num_generate_steps = num_generate_steps
        self.num_log_steps = num_log_steps
        self.compute_validation_loss = compute_validation_loss
        self.predict_method = predict_method
        self.predict_overlap = predict_overlap
        self._training_step_outputs: list = []
        self._validation_losses: list[list[tuple[Tensor, int]]] = []
        self._val_log_batch: tuple[Tensor, Tensor] | None = None
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu")["state_dict"])

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Compute flow-matching training loss for one batch.

        Parameters
        ----------
        batch : dict
            Must contain ``"source"`` and ``"target"`` tensors.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            Scalar flow-matching loss.
        """
        phase: Tensor = batch["source"]
        target: Tensor = batch["target"]
        loss = self.model(phase, target)
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=phase.shape[0],
        )
        if batch_idx < self.log_batches_per_epoch:
            self._training_step_outputs.extend(detach_sample((phase, target), self.log_samples_per_batch))
        return loss

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Capture validation samples and optionally compute loss."""
        if batch_idx == 0 and self._val_log_batch is None:
            n = self.log_samples_per_batch
            self._val_log_batch = (
                batch["source"][:n].clone(),
                batch["target"][:n].clone(),
            )
        if not self.compute_validation_loss:
            return
        phase: Tensor = batch["source"]
        target: Tensor = batch["target"]
        loss = self.model(phase, target)
        if dataloader_idx + 1 > len(self._validation_losses):
            self._validation_losses.append([])
        self._validation_losses[dataloader_idx].append((loss.detach(), phase.shape[0]))
        self.log(
            f"loss/val/{dataloader_idx}",
            loss,
            sync_dist=True,
            batch_size=phase.shape[0],
        )

    def on_train_epoch_end(self) -> None:
        """Log training image samples at end of epoch."""
        self._log_samples("train_samples", self._training_step_outputs)
        self._training_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        """Generate ODE samples from captured validation batch and log."""
        super().on_validation_epoch_end()
        if self._val_log_batch is not None:
            if self.logger is not None:
                phase_log, target_log = self._val_log_batch
                n = min(self.log_samples_per_batch, phase_log.shape[0])
                generated = self.model.generate(phase_log[:n], num_steps=self.num_log_steps)
                gen_samples = detach_sample((phase_log[:n], target_log[:n], generated), n)
                self._log_samples("val_generated_samples", gen_samples)
            self._val_log_batch = None
        if self._validation_losses:
            self.log("loss/validate", _aggregate_validation_losses(self._validation_losses), sync_dist=True)
        self._validation_losses.clear()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Generate virtual staining for one batch via ODE sampling.

        Pads source if smaller than ``input_spatial_size``, dispatches to
        the configured predict method, then crops back to the original shape.

        Parameters
        ----------
        batch : dict
            Must contain ``"source"`` tensor.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.

        Returns
        -------
        Tensor
            Generated fluorescence, cropped to original spatial shape.
        """
        source: Tensor = batch["source"]
        original_shape = source.shape[2:]

        # Pad source if any spatial dim is smaller than input_spatial_size.
        patch_size = self.model.net.input_spatial_size
        min_size = tuple(patch_size)
        if any(s < p for s, p in zip(source.shape[2:], min_size)):
            pad: list[int] = []
            for s, p in zip(reversed(source.shape[2:]), reversed(min_size)):
                pad.extend([0, max(0, p - s)])
            source = F.pad(source, pad, mode="replicate")

        if self.predict_method == "denoise":
            prediction = self.model.denoise_sliding_window(source, overlap_size=self.predict_overlap)
        elif self.predict_method == "generate":
            prediction = self.model.generate(source, num_steps=self.num_generate_steps)
        elif self.predict_method == "sliding_window":
            prediction = self.model.generate_sliding_window(source, num_steps=self.num_generate_steps)
        elif self.predict_method == "iterative":
            prediction = self.model.generate_iterative(
                source,
                num_steps=self.num_generate_steps,
                overlap_size=self.predict_overlap,
            )
        else:
            raise ValueError(
                f"Unknown predict_method: {self.predict_method!r}. "
                "Choose 'denoise', 'generate', 'sliding_window', or 'iterative'."
            )

        return prediction[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]

    def configure_optimizers(self):
        """Configure AdamW optimizer with LR scheduler."""
        return _configure_adamw_scheduler(self, self.model, self.lr, self.schedule)

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]) -> None:
        """Log image grid to the active logger."""
        if not imgs or not self.trainer.is_global_zero or self.logger is None:
            return
        log_image_grid(self.logger, key, imgs, self.current_epoch)
