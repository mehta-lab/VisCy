"""Dynacell LightningModules for virtual staining benchmarks.

Provides :class:`DynacellUNet` for supervised regression,
:class:`DynacellFlowMatching` for flow-matching generative staining, and
:class:`DynacellGAN` for adversarial (LSGAN + L1) virtual staining.
"""

import inspect
import itertools
import logging
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import DivisiblePad
from torch import Tensor, nn
from torch.optim import AdamW

from dynacell.celldiff_wrapper import CELLDiff3DVS
from viscy_data import Sample
from viscy_models import Unet3d, UNeXt2
from viscy_models.celldiff import CELLDiffNet, UNetViT3D
from viscy_models.gan import MultiScalePatchGAN3D, lsgan_d_loss, lsgan_g_loss
from viscy_models.unet.fcmae import FullyConvolutionalMAE
from viscy_utils.log_images import detach_sample, log_image_grid
from viscy_utils.optimizers import configure_adamw_scheduler

_logger = logging.getLogger("lightning.pytorch")

_ARCHITECTURE: dict[str, type[nn.Module]] = {
    "UNetViT3D": UNetViT3D,
    "FNet3D": Unet3d,
    "UNeXt2": UNeXt2,
    "fcmae": FullyConvolutionalMAE,
}


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
    architecture : {"UNetViT3D", "FNet3D", "UNeXt2", "fcmae"}
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
    encoder_only : bool, default False
        When True, ``ckpt_path`` must be set, and only the
        ``model.encoder.*`` weights are loaded (decoder/head stay at fresh
        init). Intended for finetuning from an FCMAE-pretrained encoder.
        Only supported for ``architecture='fcmae'``.

        Note: on resumed runs (via trainer-level ``--ckpt_path``), this
        pre-load still fires in ``__init__`` before Lightning restores
        the resume checkpoint, and the resume state overwrites it. The
        file at ``ckpt_path`` must therefore remain accessible for the
        lifetime of any run based on a pretrained leaf.
    """

    def __init__(
        self,
        architecture: Literal["UNetViT3D", "FNet3D", "UNeXt2", "fcmae"] = "UNetViT3D",
        model_config: dict | None = None,
        loss_function: nn.Module | None = None,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        warmup_steps: int = 3,
        warmup_multiplier: float = 1e-3,
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        example_input_yx_shape: Sequence[int] = (256, 256),
        predict_method: Literal["full_image", "sliding_window"] = "full_image",
        predict_overlap: tuple[int, int, int] = (4, 256, 256),
        ckpt_path: str | None = None,
        encoder_only: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_function", "ckpt_path", "encoder_only"])
        if model_config is None:
            model_config = {}
        net_class = _ARCHITECTURE.get(architecture)
        if net_class is None:
            raise ValueError(f"Architecture {architecture!r} not in {set(_ARCHITECTURE)}")
        self.model = net_class(**model_config)
        self.loss_function = loss_function if loss_function is not None else nn.MSELoss()
        self.lr = lr
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.warmup_multiplier = warmup_multiplier
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

        if encoder_only:
            if ckpt_path is None:
                raise ValueError("DynacellUNet(encoder_only=True) requires ckpt_path to be set")
            if not isinstance(self.model, FullyConvolutionalMAE):
                raise ValueError(f"encoder_only is only supported for architecture='fcmae', got {architecture!r}")
            state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")["state_dict"]
            prefix = "model.encoder."
            encoder_weights = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
            self.model.encoder.load_state_dict(encoder_weights, strict=True)
            _logger.info(f"Loaded {len(encoder_weights)} encoder parameters from {ckpt_path}")
        elif ckpt_path is not None:
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
        return configure_adamw_scheduler(
            self,
            self.model,
            self.lr,
            self.schedule,
            warmup_steps=self.warmup_steps,
            warmup_multiplier=self.warmup_multiplier,
        )

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

        # Accumulators are allocated lazily from the first patch output so
        # their channel dimension matches the model's out_channels (which can
        # differ from source's in_channels, e.g. 1 phase in -> 2 target out).
        prediction_sum: Tensor | None = None
        prediction_count: Tensor | None = None

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
                if prediction_sum is None:
                    out_shape = list(source.shape)
                    out_shape[1] = patch_out.shape[1]
                    prediction_sum = torch.zeros(out_shape, device=source.device, dtype=patch_out.dtype)
                    prediction_count = torch.zeros(out_shape, device=source.device, dtype=patch_out.dtype)
                prediction_sum[tuple(slicer)] += patch_out
                prediction_count[tuple(slicer)] += 1

        if prediction_sum is None:
            raise RuntimeError("sliding window produced no patches")
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
        Prediction generation method. ``"generate"`` runs single-patch ODE
        (default, matches standard HCS tile workflow). ``"sliding_window"``
        partitions the volume into **non-overlapping** tiles (ignores
        ``predict_overlap``; passing a non-zero overlap raises so users
        aren't silently misled). ``"iterative"`` slides overlapping tiles
        with velocity anchoring — use this when you want
        ``predict_overlap`` to apply. ``"denoise"`` uses the noise-space
        overlap tiler.
    predict_overlap : int or tuple of int
        Overlap for ``denoise`` and ``iterative``. Ignored by
        ``sliding_window``; must be ``0`` or ``[0, 0, 0]`` when
        ``predict_method='sliding_window'``.
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
        warmup_steps: int = 3,
        warmup_multiplier: float = 1e-3,
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
        self.warmup_steps = warmup_steps
        self.warmup_multiplier = warmup_multiplier
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
            # generate_sliding_window partitions into non-overlapping tiles
            # and does NOT consume predict_overlap. A non-zero overlap means
            # the user wants overlapping tiled inference — route them to
            # `iterative`, which anchors overlapping regions via velocity.
            overlap = self.predict_overlap
            overlap_values = (overlap,) * 3 if isinstance(overlap, int) else tuple(overlap)
            if any(o > 0 for o in overlap_values):
                raise ValueError(
                    "predict_method='sliding_window' uses non-overlapping tiles and "
                    f"ignores predict_overlap (got {overlap_values}). "
                    "Use predict_method='iterative' for overlap-anchored tiled inference, "
                    "or set predict_overlap=[0, 0, 0] to acknowledge the non-overlapping behavior."
                )
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
        return configure_adamw_scheduler(
            self,
            self.model,
            self.lr,
            self.schedule,
            warmup_steps=self.warmup_steps,
            warmup_multiplier=self.warmup_multiplier,
        )

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]) -> None:
        """Log image grid to the active logger."""
        if not imgs or not self.trainer.is_global_zero or self.logger is None:
            return
        log_image_grid(self.logger, key, imgs, self.current_epoch)


class DynacellGAN(LightningModule):
    """Adversarial virtual-staining LightningModule (LSGAN + L1).

    Pairs a regression-style generator (default ``UNetViT3D``) with a
    multi-scale 3D PatchGAN discriminator. The training step alternates a
    discriminator update and a generator update per batch using Lightning's
    manual-optimization API, so the module sets
    ``automatic_optimization = False``.

    The discriminator is conditional: it consumes the concatenation of
    ``source`` and either the real target or the generator output along the
    channel dimension. The generator loss combines an LSGAN adversarial term
    (MSE-to-real for the fake pair) with an L1 reconstruction term scaled by
    ``lambda_l1``.

    The ``requires_grad`` flags on the generator and discriminator are
    toggled at each phase of the training step so each backward pass only
    populates ``.grad`` on the parameters being optimized. After the
    discriminator update we additionally call ``opt_d.zero_grad`` so that
    after a full ``training_step`` no discriminator parameter retains a
    non-zero accumulated gradient — this invariant is verified by unit
    tests.

    Parameters
    ----------
    architecture : {"UNetViT3D"}
        Generator architecture key. Looked up in the shared
        :data:`_ARCHITECTURE` registry.
    generator_config : dict or None
        Keyword arguments forwarded to the generator constructor.
    discriminator_config : dict or None
        Keyword arguments forwarded to :class:`MultiScalePatchGAN3D`.
    lambda_l1 : float
        Weight of the L1 reconstruction loss in the generator objective.
    lr_g : float
        Learning rate for the generator optimizer.
    lr_d : float
        Learning rate for the discriminator optimizer.
    schedule : {"WarmupCosine"}
        Learning rate schedule. Only ``"WarmupCosine"`` is supported because
        manual optimization expects a step-interval scheduler.
    warmup_steps : int
        Number of warmup steps for the WarmupCosine schedule.
    warmup_multiplier : float
        Initial LR multiplier at step 0 of the WarmupCosine schedule.
    log_batches_per_epoch : int
        Maximum number of batches per epoch to accumulate for image logging.
    log_samples_per_batch : int
        Number of samples per batch to log.
    example_input_yx_shape : Sequence of int
        YX shape used to build ``example_input_array`` for graph logging
        when the generator does not advertise an ``input_spatial_size``.
    predict_method : {"full_image"}
        Prediction method. Only ``"full_image"`` is supported (the
        discriminator is not exposed at inference time and the generator is
        a fixed-input-size ViT-bottleneck UNet).
    predict_overlap : tuple of int
        Reserved for future tiled inference; currently unused at predict.
    ckpt_path : str or None
        Optional path to a Lightning checkpoint to load **weights only** at
        construction time. Intended for inference (predict/test), not
        training resumption — optimizer state, epoch counters, and
        scheduler state are not restored.
    """

    def __init__(
        self,
        architecture: Literal["UNetViT3D"] = "UNetViT3D",
        generator_config: dict | None = None,
        discriminator_config: dict | None = None,
        lambda_l1: float = 100.0,
        lr_g: float = 3e-4,
        lr_d: float = 3e-4,
        schedule: Literal["WarmupCosine"] = "WarmupCosine",
        warmup_steps: int = 8500,
        warmup_multiplier: float = 1e-3,
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        example_input_yx_shape: Sequence[int] = (512, 512),
        predict_method: Literal["full_image"] = "full_image",
        predict_overlap: tuple[int, int, int] = (4, 256, 256),
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        # Lightning's manual-optimization API: required because the GAN
        # alternates two optimizers per training_step.
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["ckpt_path"])

        net_class = _ARCHITECTURE.get(architecture)
        if net_class is None:
            raise ValueError(f"Architecture {architecture!r} not in {set(_ARCHITECTURE)}")
        self.generator = net_class(**(generator_config or {}))
        self.discriminator = MultiScalePatchGAN3D(**(discriminator_config or {}))

        self.lambda_l1 = lambda_l1
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.warmup_multiplier = warmup_multiplier
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.predict_method = predict_method
        self.predict_overlap = predict_overlap

        self.training_step_outputs: list = []
        # Each entry is a list of (loss, batch_size) tuples for weighted aggregation.
        self.validation_losses: list[list[tuple[Tensor, int]]] = []
        self.validation_step_outputs: list = []

        # Build example_input_array for graph logging (TensorBoard/W&B).
        gen_cfg = generator_config or {}
        in_channels = gen_cfg.get("in_channels") or 1
        if hasattr(self.generator, "input_spatial_size"):
            d, h, w = self.generator.input_spatial_size
        else:
            d = gen_cfg.get("in_stack_depth") or 5
            h, w = example_input_yx_shape
        self.example_input_array = torch.rand(1, in_channels, d, h, w)

        # Mirror DynacellUNet's end-of-init weight load so Phase 4 predict
        # leaves with ``model.init_args.ckpt_path`` actually load weights.
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu")["state_dict"])

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool) -> None:
        """Toggle ``requires_grad`` on every parameter of ``module``.

        Parameters
        ----------
        module : nn.Module
            Module whose parameters should have ``requires_grad`` set.
        value : bool
            New value for ``requires_grad``.
        """
        for p in module.parameters():
            p.requires_grad = value

    def forward(self, x: Tensor) -> Tensor:
        """Run a generator-only forward pass (inference contract).

        The discriminator is not exposed at inference time; ``forward``
        always returns the generator output.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor
            Generator output.
        """
        return self.generator(x)

    def training_step(self, batch: Sample, batch_idx: int) -> None:
        """Run one alternating D/G optimization step.

        The discriminator is updated first using a detached generator
        forward (so D's loss has no gradient path into G), then the
        generator is updated using the LSGAN adversarial term + L1.
        ``requires_grad`` is toggled per phase so each backward populates
        ``.grad`` only on the side being trained, and the discriminator's
        gradients are explicitly cleared after its update so the G step
        starts with no residual D gradients.

        Parameters
        ----------
        batch : Sample
            Batch dict with ``"source"`` and ``"target"`` tensors.
        batch_idx : int
            Batch index within the epoch.
        """
        source = batch["source"]
        target = batch["target"]
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        # --- D step (D updates; G frozen, no-grad forward) ---
        self._set_requires_grad(self.generator, False)
        self._set_requires_grad(self.discriminator, True)
        with torch.no_grad():
            pred = self.generator(source)
        d_real = self.discriminator(torch.cat([source, target], dim=1))
        d_fake = self.discriminator(torch.cat([source, pred], dim=1))
        d_loss = lsgan_d_loss(d_real, d_fake)
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()
        # Clear D grads after the D update so the G-step backward starts
        # from a clean slate on D; this makes the "no D grads after G step"
        # invariant verifiable in tests and avoids carrying D-step grads
        # forward should anything else (a callback, etc.) inspect ``.grad``.
        opt_d.zero_grad(set_to_none=True)

        # --- G step (G updates; D frozen, fwd-only) ---
        self._set_requires_grad(self.generator, True)
        self._set_requires_grad(self.discriminator, False)
        pred = self.generator(source)
        d_fake_for_g = self.discriminator(torch.cat([source, pred], dim=1))
        adv_loss = lsgan_g_loss(d_fake_for_g)
        l1_loss = F.l1_loss(pred, target)
        g_loss = adv_loss + self.lambda_l1 * l1_loss
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        # WarmupCosine is step-based and ignored by Lightning's automatic
        # scheduler machinery when ``automatic_optimization = False``.
        sch_g.step()
        sch_d.step()

        # Accumulate samples for image grid logging — mirror DynacellUNet.
        if batch_idx < self.log_batches_per_epoch:
            self.training_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))

        self.log_dict(
            {
                "loss/d_train": d_loss,
                "loss/g_train": g_loss,
                "loss/g_adv_train": adv_loss,
                "loss/g_l1_train": l1_loss,
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=source.size(0),
        )

    def validation_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute generator-only L1 validation loss and capture samples.

        Mirrors :class:`DynacellUNet`'s aggregation pattern so
        ``loss/validate`` is an epoch-aggregated weighted mean — the value
        ``ModelCheckpoint(monitor="loss/validate")`` keys off.

        Parameters
        ----------
        batch : Sample
            Batch dict with ``"source"`` and ``"target"`` tensors.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Index of the validation dataloader.

        Returns
        -------
        Tensor
            Scalar L1 loss on this batch.
        """
        source: Tensor = batch["source"]
        target: Tensor = batch["target"]
        pred = self.generator(source)
        l1 = F.l1_loss(pred, target)
        if dataloader_idx + 1 > len(self.validation_losses):
            self.validation_losses.append([])
        self.validation_losses[dataloader_idx].append((l1.detach(), source.shape[0]))
        self.log(
            f"loss/val/{dataloader_idx}",
            l1,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))
        return l1

    def on_train_epoch_end(self) -> None:
        """Log accumulated training image samples and reset the buffer."""
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        """Log validation samples and the aggregated ``loss/validate`` alias."""
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)
        if self.validation_losses:
            self.log(
                "loss/validate",
                _aggregate_validation_losses(self.validation_losses),
                sync_dist=True,
            )
        self.validation_step_outputs.clear()
        self.validation_losses.clear()

    def configure_optimizers(self):
        """Configure two AdamW optimizers and step-interval WarmupCosine schedulers.

        ``configure_adamw_scheduler`` already builds a WarmupCosine
        ``{"scheduler", "interval": "step"}`` dict, so we call it once per
        optimizer and concatenate the resulting single-entry lists. This
        keeps the two-optimizer return shape Lightning expects for manual
        optimization without re-implementing the schedule construction.

        Returns
        -------
        tuple[list, list]
            ``([opt_g, opt_d], [sch_g_dict, sch_d_dict])``.

        Raises
        ------
        ValueError
            If ``schedule`` is not ``"WarmupCosine"``.
        """
        if self.schedule != "WarmupCosine":
            raise ValueError(f"DynacellGAN only supports schedule='WarmupCosine', got {self.schedule!r}")
        # Reuse the helper rather than rebuilding the scheduler in-place so
        # any future change to the warmup contract stays in one location.
        # The helper returns ``[opt], [{"scheduler", "interval"}]`` for
        # WarmupCosine — we just concatenate two such pairs.
        opt_g = AdamW(self.generator.parameters(), lr=self.lr_g)
        opt_d = AdamW(self.discriminator.parameters(), lr=self.lr_d)
        t_total = self.trainer.estimated_stepping_batches
        sch_g = WarmupCosineSchedule(
            opt_g,
            warmup_steps=self.warmup_steps,
            t_total=t_total,
            warmup_multiplier=self.warmup_multiplier,
        )
        sch_d = WarmupCosineSchedule(
            opt_d,
            warmup_steps=self.warmup_steps,
            t_total=t_total,
            warmup_multiplier=self.warmup_multiplier,
        )
        return (
            [opt_g, opt_d],
            [
                {"scheduler": sch_g, "interval": "step"},
                {"scheduler": sch_d, "interval": "step"},
            ],
        )

    def on_predict_start(self) -> None:
        """Build the divisible-pad transform matching the generator."""
        self._predict_pad = _make_divisible_pad(self.generator)

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Run a generator-only prediction step with divisible padding.

        Pads the input tile to the nearest multiple of the generator's
        downsampling factor, runs the generator forward pass, then crops
        back to the original spatial shape. The discriminator is not
        exposed at inference.

        Parameters
        ----------
        batch : Sample
            Batch dict. Only ``"source"`` is used.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.

        Returns
        -------
        Tensor
            Generator prediction cropped to the input spatial shape.
        """
        source = batch["source"]
        original_shape = source.shape[2:]
        source = self._predict_pad(source)
        if self.predict_method == "full_image":
            prediction = self.generator(source)
        else:
            raise ValueError(f"Unknown predict_method: {self.predict_method!r}. Choose 'full_image'.")
        return _center_crop_to_shape(prediction, original_shape)

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]) -> None:
        """Log image grid to the active logger."""
        if not imgs or not self.trainer.is_global_zero or self.logger is None:
            return
        log_image_grid(self.logger, key, imgs, self.current_epoch)
