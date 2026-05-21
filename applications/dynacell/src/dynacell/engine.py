"""Dynacell LightningModules for virtual staining benchmarks.

Provides :class:`DynacellUNet` for supervised regression,
:class:`DynacellFlowMatching` for flow-matching generative staining, and
:class:`DynacellGAN` for adversarial (LSGAN + L1) virtual staining.
"""

import copy
import inspect
import itertools
import logging
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from monai.transforms import DivisiblePad
from torch import Tensor, nn

from dynacell.celldiff_wrapper import CELLDiff3DVS
from viscy_data import Sample
from viscy_models import Unet3d, UNeXt2
from viscy_models.celldiff import CELLDiffNet, UNetViT3D
from viscy_models.gan import (
    MultiScalePatchGAN3D,
    lsgan_d_loss,
    lsgan_g_loss,
    nonsat_d_loss,
    nonsat_g_loss,
    r1_penalty,
    r2_penalty,
    rpgan_d_loss,
    rpgan_g_loss,
)
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


def _log_samples(module: LightningModule, key: str, imgs: Sequence[Sequence[np.ndarray]]) -> None:
    """Log a list of detached image samples to the experiment logger at rank 0."""
    if not imgs or not module.trainer.is_global_zero or module.logger is None:
        return
    log_image_grid(module.logger, key, imgs, module.current_epoch)


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
        _log_samples(self, "train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        """Log validation samples and aggregate loss weighted by batch size."""
        super().on_validation_epoch_end()
        _log_samples(self, "val_samples", self.validation_step_outputs)
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
        _log_samples(self, "train_samples", self._training_step_outputs)
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
                _log_samples(self, "val_generated_samples", gen_samples)
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


class DynacellGAN(LightningModule):
    """Adversarial virtual-staining LightningModule.

    Pairs a regression-style generator (default ``UNetViT3D``) with a
    multi-scale 3D PatchGAN discriminator. The training step alternates a
    discriminator update and a generator update per batch using Lightning's
    manual-optimization API.

    Supports three adversarial loss families via ``loss_type``:

    - ``"lsgan"`` (default, legacy): MSE-to-real / MSE-to-zero.
    - ``"nonsat"``: non-saturating softplus loss (StyleGAN2 convention).
    - ``"rpgan"``: relativistic pairing loss (R3GAN, NeurIPS 2024).

    Optional modernization knobs (all default OFF for legacy safety; the
    14 existing leaves that compose ``pix2pix3d_unetvit_fit.yml`` are
    therefore unaffected by the wiring of these knobs):

    - R1 / R2 zero-centered gradient penalties on a lazy schedule
      (``r1_every`` D-steps, with StyleGAN2-style unbiased ``* r1_every``
      rescaling on the loss contribution).
    - Generator weight EMA with half-life parametrized via ``ema_kimg``.
    - LeCam regularization with sync_dist'd batch-mean EMA buffers.

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
    loss_type : {"lsgan", "nonsat", "rpgan"}
        Adversarial loss family. Default ``"lsgan"`` matches the
        pre-modernization recipe. ``"nonsat"`` is the StyleGAN2 default;
        ``"rpgan"`` requires nonzero ``r1_gamma`` and (recommended)
        nonzero ``r2_gamma`` for convergence.
    lambda_adv : float
        Weight on the adversarial term in the generator objective.
        Defaults to ``1.0`` (legacy-equivalent).
    r1_gamma : float
        R1 gradient-penalty weight (Mescheder 2018). ``0.0`` disables R1.
    r2_gamma : float
        R2 gradient-penalty weight on fake samples (R3GAN). ``0.0``
        disables R2.
    r1_every : int
        Lazy schedule: apply R1 / R2 every ``r1_every`` D-steps with a
        ``* r1_every`` unbiased rescaling factor. Default ``16`` matches
        StyleGAN2-ADA's ``D_reg_interval``. Only consulted when
        ``r1_gamma > 0 or r2_gamma > 0``.
    ema_kimg : float or None
        Generator EMA half-life in thousands of images. ``None`` (default)
        disables EMA entirely; no shadow submodule is constructed.
        ``10.0`` matches StyleGAN2's 256² default; with global batch ``B``
        the per-step decay is ``0.5 ** (B / (ema_kimg * 1000))``.
    lecam_gamma : float
        LeCam regularization weight (Tseng et al. 2021). ``0.0`` disables
        LeCam entirely; no EMA buffers are registered.
    lecam_decay : float
        EMA decay for LeCam's running D output statistics. Default
        ``0.9`` matches the ``google/lecam-gan`` reference.
    use_ema_at_predict : bool
        When True (default) AND ``generator_ema`` exists, ``predict_step``
        / ``forward`` use the EMA generator. Set False to force
        raw-generator predictions from a modernized checkpoint without
        editing the predict overlay.
    lr_g : float
        Learning rate for the generator optimizer.
    lr_d : float
        Learning rate for the discriminator optimizer.
    schedule : {"WarmupCosine"}
        Learning rate schedule. Only ``"WarmupCosine"`` is supported.
    warmup_steps : int
        Number of warmup steps for the WarmupCosine schedule.
    warmup_multiplier : float
        Initial LR multiplier at step 0.
    log_batches_per_epoch : int
        Maximum number of batches per epoch to accumulate for image logging.
    log_samples_per_batch : int
        Number of samples per batch to log.
    example_input_yx_shape : Sequence of int
        YX shape used to build ``example_input_array`` for graph logging
        when the generator does not advertise an ``input_spatial_size``.
    predict_method : {"full_image"}
        Prediction method. Only ``"full_image"`` is supported.
    predict_overlap : tuple of int
        Reserved for future tiled inference; currently unused at predict.
    ckpt_path : str or None
        Optional path to a Lightning checkpoint to load weights from at
        construction time. Loaded with ``strict=False`` so pre-modernization
        checkpoints (which lack ``generator_ema.*`` / ``_d_step_count`` /
        ``_lecam_ema_*``) load cleanly. When the checkpoint has no
        ``generator_ema.*`` keys but EMA is enabled, the EMA submodule is
        seeded from the loaded generator weights (so inference matches the
        non-EMA inference path on legacy checkpoints).
    """

    def __init__(
        self,
        architecture: Literal["UNetViT3D"] = "UNetViT3D",
        generator_config: dict | None = None,
        discriminator_config: dict | None = None,
        lambda_l1: float = 100.0,
        loss_type: Literal["lsgan", "nonsat", "rpgan"] = "lsgan",
        lambda_adv: float = 1.0,
        r1_gamma: float = 0.0,
        r2_gamma: float = 0.0,
        r1_every: int = 16,
        ema_kimg: float | None = None,
        lecam_gamma: float = 0.0,
        lecam_decay: float = 0.9,
        use_ema_at_predict: bool = True,
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
        if loss_type not in ("lsgan", "nonsat", "rpgan"):
            raise ValueError(f"Unknown loss_type {loss_type!r}; expected lsgan|nonsat|rpgan.")
        if loss_type == "rpgan" and (r1_gamma <= 0.0 or r2_gamma <= 0.0):
            raise ValueError(
                "RpGAN requires nonzero r1_gamma AND r2_gamma for convergence on sharp "
                "distributions (R3GAN Theorem 3.1). "
                f"Got r1_gamma={r1_gamma}, r2_gamma={r2_gamma}."
            )
        if ema_kimg is not None and ema_kimg <= 0.0:
            raise ValueError(
                f"ema_kimg must be > 0 (or None to disable EMA); got {ema_kimg}. "
                "ema_kimg=0 would freeze EMA at init weights with no signal."
            )
        self.generator = net_class(**(generator_config or {}))
        self.discriminator = MultiScalePatchGAN3D(**(discriminator_config or {}))

        self.lambda_l1 = lambda_l1
        self.loss_type = loss_type
        self.lambda_adv = lambda_adv
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma
        self.r1_every = r1_every
        self.ema_kimg = ema_kimg
        self.lecam_gamma = lecam_gamma
        self.lecam_decay = lecam_decay
        self.use_ema_at_predict = use_ema_at_predict
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.warmup_multiplier = warmup_multiplier
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.predict_method = predict_method
        self.predict_overlap = predict_overlap

        # D-step counter for lazy R1 schedule. self.global_step would
        # advance by 2 per training_step (D opt + G opt) so it can't be used
        # directly. Buffer => DDP broadcasts at init and it persists in
        # checkpoints.
        self.register_buffer("_d_step_count", torch.tensor(0, dtype=torch.long))

        # Generator EMA shadow. Custom deepcopy (not timm) so we control
        # device/dtype precisely. requires_grad_(False) keeps EMA params out
        # of opt_g and out of DDP's gradient reducer.
        if ema_kimg is not None:
            self.generator_ema = copy.deepcopy(self.generator)
            self.generator_ema.requires_grad_(False)
        else:
            self.generator_ema = None

        # LeCam EMA buffers — only registered when LeCam is enabled.
        if lecam_gamma > 0.0:
            self.register_buffer("_lecam_ema_real", torch.tensor(0.0))
            self.register_buffer("_lecam_ema_fake", torch.tensor(0.0))

        self.training_step_outputs: list = []
        # Two accumulators: raw-generator + EMA-generator validation losses.
        # The second one is only populated when generator_ema exists.
        self.validation_losses_raw: list[list[tuple[Tensor, int]]] = []
        self.validation_losses_ema: list[list[tuple[Tensor, int]]] = []
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

        if ckpt_path is not None:
            state = torch.load(ckpt_path, weights_only=True, map_location="cpu")["state_dict"]
            # strict=False: pre-modernization checkpoints don't carry
            # generator_ema.* / _d_step_count / _lecam_ema_* keys. Filter
            # missing-key warnings to expected-missing prefixes; anything else
            # is a genuine state-dict mismatch (renamed layer, dropped module,
            # wrong checkpoint kind). RAISE rather than warn — silent
            # half-loaded models burn hours of training before users notice.
            incompat = self.load_state_dict(state, strict=False)
            expected_missing = ("generator_ema.", "_d_step_count", "_lecam_ema_")
            unexpected_missing = [k for k in incompat.missing_keys if not k.startswith(expected_missing)]
            if unexpected_missing:
                raise RuntimeError(
                    f"Checkpoint {ckpt_path!r} is missing keys that are not part of the "
                    f"modernization additions: {unexpected_missing}. The model would load "
                    "with partially random weights. If this is intentional, drop those "
                    "submodules from the model config or use an explicit migration step."
                )
            if incompat.unexpected_keys:
                _logger.warning(
                    "Checkpoint %s has unexpected keys ignored by strict=False load: %s",
                    ckpt_path,
                    incompat.unexpected_keys,
                )
            # Seed EMA from loaded generator when ckpt has no EMA section,
            # so inference paths return the loaded weights (not the
            # random-init deepcopy from __init__). RAISE on partial-EMA ckpt
            # to catch a corrupted save (e.g., crash mid-checkpoint, or a
            # future buffer added to generator_ema that an old ckpt lacks) —
            # silently half-seeding would produce nonsense at the partial layers.
            if self.generator_ema is not None:
                ema_keys_in_ckpt = sum(1 for k in state if k.startswith("generator_ema."))
                ema_keys_expected = len(self.generator_ema.state_dict())
                if ema_keys_in_ckpt == 0:
                    self.generator_ema.load_state_dict(self.generator.state_dict())
                    _logger.info(
                        "Checkpoint %s has no generator_ema.* keys; seeded EMA shadow from loaded generator weights.",
                        ckpt_path,
                    )
                elif ema_keys_in_ckpt != ema_keys_expected:
                    raise RuntimeError(
                        f"Checkpoint {ckpt_path!r} has {ema_keys_in_ckpt} generator_ema.* "
                        f"keys but the EMA submodule expects {ema_keys_expected}. "
                        "Partial EMA loads would leave random-init values in the missing "
                        "EMA layers and silently produce wrong inference outputs."
                    )

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

    def _inference_generator(self) -> nn.Module:
        """Return the generator used for inference paths.

        Returns the EMA shadow if it exists AND ``use_ema_at_predict`` is
        True; otherwise the raw generator. Used by ``forward`` and
        ``predict_step``.
        """
        if self.generator_ema is not None and self.use_ema_at_predict:
            return self.generator_ema
        return self.generator

    def forward(self, x: Tensor) -> Tensor:
        """Run a generator-only forward pass (inference contract).

        The discriminator is not exposed at inference time; ``forward``
        returns the EMA generator's output when available, otherwise the
        raw generator.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor
            Generator output (EMA generator if available).
        """
        return self._inference_generator()(x)

    def _adv_d_loss(self, d_real: list[Tensor], d_fake: list[Tensor]) -> Tensor:
        """Dispatch the configured adversarial D loss across loss families."""
        if self.loss_type == "lsgan":
            return lsgan_d_loss(d_real, d_fake)
        if self.loss_type == "nonsat":
            return nonsat_d_loss(d_real, d_fake)
        # rpgan
        return rpgan_d_loss(d_real, d_fake)

    def _adv_g_loss(self, d_real: list[Tensor] | None, d_fake: list[Tensor]) -> Tensor:
        """Dispatch the configured adversarial G loss.

        For RpGAN, ``d_real`` must be freshly computed against the
        post-D-update discriminator (not reused from the D step).
        """
        if self.loss_type == "lsgan":
            return lsgan_g_loss(d_fake)
        if self.loss_type == "nonsat":
            return nonsat_g_loss(d_fake)
        # rpgan
        if d_real is None:
            raise ValueError("RpGAN G loss requires fresh d_real logits; got None.")
        return rpgan_g_loss(d_real, d_fake)

    def training_step(self, batch: Sample, batch_idx: int) -> None:
        """Run one alternating D/G optimization step.

        The discriminator is updated first using a detached generator
        forward (so D's loss has no gradient path into G), then the
        generator is updated. ``requires_grad`` is toggled per phase so
        each backward populates ``.grad`` only on the side being trained.

        Lazy R1 / R2 gradient penalties fire every ``r1_every`` D-steps
        (tracked via ``_d_step_count``, NOT ``self.global_step`` which
        advances per opt.step() and would fire R1 every 8 batches with two
        opt steps per batch). LeCam regularization is added inline using
        sync_dist'd batch means so EMA buffers stay synchronized across
        DDP ranks without explicit ``dist.all_reduce`` on the buffers.

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
        real_pair = torch.cat([source, target], dim=1)
        fake_pair = torch.cat([source, pred], dim=1)
        d_real = self.discriminator(real_pair)
        d_fake = self.discriminator(fake_pair)
        d_loss = self._adv_d_loss(d_real, d_fake)

        # Increment D-step counter BEFORE the lazy reg check, then check
        # if R1 or R2 is enabled AND we're on a lazy-reg step.
        self._d_step_count += 1
        r1_value: Tensor | None = None
        r2_value: Tensor | None = None
        do_lazy_reg = (self.r1_gamma > 0.0 or self.r2_gamma > 0.0) and int(self._d_step_count) % self.r1_every == 0
        if do_lazy_reg:
            # Mescheder R1 grad-of-grad needs fp32; under Lightning bf16-mixed,
            # autocast() injects bf16 into D forwards which is numerically fragile
            # for create_graph=True. Disable autocast around the penalty.
            with torch.amp.autocast(device_type=source.device.type, enabled=False):
                if self.r1_gamma > 0.0:
                    real_fp32 = real_pair.detach().float()
                    r1_value = r1_penalty(self.discriminator, real_fp32)
                    # (γ/2) is Mescheder's standard formula factor;
                    # `* r1_every` is the separate StyleGAN2 unbiased rescaling.
                    d_loss = d_loss + (self.r1_gamma / 2) * r1_value * self.r1_every
                if self.r2_gamma > 0.0:
                    fake_fp32 = fake_pair.detach().float()
                    r2_value = r2_penalty(self.discriminator, fake_fp32)
                    d_loss = d_loss + (self.r2_gamma / 2) * r2_value * self.r1_every

        if self.lecam_gamma > 0.0:
            # Use Lightning's all_gather for cross-rank mean so all ranks
            # see identical scalar -> identical EMA buffer update.
            real_mean = torch.stack([d.mean() for d in d_real]).mean().detach()
            fake_mean = torch.stack([d.mean() for d in d_fake]).mean().detach()
            if self.trainer is not None and self.trainer.world_size > 1:
                real_mean = self.all_gather(real_mean).mean()
                fake_mean = self.all_gather(fake_mean).mean()
            self._lecam_ema_real.mul_(self.lecam_decay).add_(real_mean * (1.0 - self.lecam_decay))
            self._lecam_ema_fake.mul_(self.lecam_decay).add_(fake_mean * (1.0 - self.lecam_decay))
            # Multi-scale LeCam: relu hinge on each scale, averaged.
            lecam_per_scale = [
                F.relu(real - self._lecam_ema_fake).pow(2).mean() + F.relu(self._lecam_ema_real - fake).pow(2).mean()
                for real, fake in zip(d_real, d_fake, strict=True)
            ]
            d_loss = d_loss + self.lecam_gamma * torch.stack(lecam_per_scale).mean()

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()
        # Clear D grads so the no-D-grads-after-G-step invariant is verifiable.
        opt_d.zero_grad(set_to_none=True)

        # --- G step (G updates; D frozen, fwd-only) ---
        self._set_requires_grad(self.generator, True)
        self._set_requires_grad(self.discriminator, False)
        pred = self.generator(source)
        d_fake_for_g = self.discriminator(torch.cat([source, pred], dim=1))
        # RpGAN G loss is relativistic — needs fresh d_real against the
        # POST-D-update discriminator (R3GAN Trainer.py convention).
        if self.loss_type == "rpgan":
            d_real_for_g = self.discriminator(torch.cat([source, target], dim=1))
            adv_loss = self._adv_g_loss(d_real_for_g, d_fake_for_g)
        else:
            adv_loss = self._adv_g_loss(None, d_fake_for_g)
        l1_loss = F.l1_loss(pred, target)
        g_loss = self.lambda_adv * adv_loss + self.lambda_l1 * l1_loss
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        # Generator EMA update — uses the StyleGAN2 formula
        # decay = 0.5 ** (global_batch_size / (ema_kimg * 1000)). World-size
        # multiplier matters here because all ranks see identical post-step
        # generator weights (DDP-synced via opt_g.step()) and apply the
        # identical EMA update — so the shadow stays in lockstep.
        if self.generator_ema is not None:
            bs = source.shape[0]
            if self.trainer is not None and self.trainer.world_size > 1:
                bs = bs * self.trainer.world_size
            decay = 0.5 ** (bs / max(self.ema_kimg * 1000.0, 1e-8))
            with torch.no_grad():
                for p_ema, p in zip(
                    self.generator_ema.parameters(),
                    self.generator.parameters(),
                    strict=True,
                ):
                    p_ema.lerp_(p.detach(), 1.0 - decay)
                for b_ema, b in zip(
                    self.generator_ema.buffers(),
                    self.generator.buffers(),
                    strict=True,
                ):
                    b_ema.copy_(b)

        # WarmupCosine is step-based and ignored by Lightning's automatic
        # scheduler machinery when ``automatic_optimization = False``.
        sch_g.step()
        sch_d.step()

        if batch_idx < self.log_batches_per_epoch:
            self.training_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))

        log_payload: dict[str, Tensor] = {
            "loss/d_train": d_loss,
            "loss/g_train": g_loss,
            "loss/g_adv_train": adv_loss,
            "loss/g_l1_train": l1_loss,
        }
        self.log_dict(
            log_payload,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=source.size(0),
        )
        # Sparse R1 / R2 logging: only on the steps they fire. sync_dist=False
        # because all DDP ranks fire on the same _d_step_count (deterministic)
        # so there's no rank-mismatch risk; sync_dist=True would invite
        # deadlock-on-skip if any rank ever stops firing.
        if r1_value is not None:
            self.log("reg/r1", r1_value.detach(), on_step=True, on_epoch=True, sync_dist=False)
        if r2_value is not None:
            self.log("reg/r2", r2_value.detach(), on_step=True, on_epoch=True, sync_dist=False)

    def validation_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute generator L1 validation loss(es) and capture samples.

        Always runs the raw generator forward and accumulates into
        ``validation_losses_raw`` (drives the back-compat ``loss/validate``
        alias). When ``generator_ema`` exists, ALSO runs the EMA generator
        forward and accumulates into ``validation_losses_ema`` (drives
        ``loss/validate_ema``). Modernized leaves should
        ``monitor: loss/validate_ema`` once EMA is enabled.

        Logged sample grids use the EMA generator's prediction when
        available — matches what ``predict_step`` will produce.

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
            Scalar L1 loss on this batch (raw generator).
        """
        source: Tensor = batch["source"]
        target: Tensor = batch["target"]
        # Raw generator pass (always).
        pred_raw = self.generator(source)
        l1_raw = F.l1_loss(pred_raw, target)
        if dataloader_idx + 1 > len(self.validation_losses_raw):
            self.validation_losses_raw.append([])
        self.validation_losses_raw[dataloader_idx].append((l1_raw.detach(), source.shape[0]))
        self.log(
            f"loss/val/{dataloader_idx}",
            l1_raw,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        # EMA generator pass (only when EMA submodule exists).
        pred_for_samples = pred_raw
        if self.generator_ema is not None:
            with torch.no_grad():
                pred_ema = self.generator_ema(source)
            l1_ema = F.l1_loss(pred_ema, target)
            if dataloader_idx + 1 > len(self.validation_losses_ema):
                self.validation_losses_ema.append([])
            self.validation_losses_ema[dataloader_idx].append((l1_ema.detach(), source.shape[0]))
            self.log(
                f"loss/val_ema/{dataloader_idx}",
                l1_ema,
                sync_dist=True,
                batch_size=source.shape[0],
            )
            pred_for_samples = pred_ema
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(
                detach_sample((source, target, pred_for_samples), self.log_samples_per_batch)
            )
        return l1_raw

    def on_train_epoch_end(self) -> None:
        """Log accumulated training image samples and reset the buffer."""
        _log_samples(self, "train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        """Log validation samples and ``loss/validate`` (raw) + ``loss/validate_ema``."""
        super().on_validation_epoch_end()
        _log_samples(self, "val_samples", self.validation_step_outputs)
        # Back-compat alias: legacy leaves' ModelCheckpoint(monitor="loss/validate")
        # keys off the raw-generator val loss. Always emit it.
        if self.validation_losses_raw:
            self.log(
                "loss/validate",
                _aggregate_validation_losses(self.validation_losses_raw),
                sync_dist=True,
            )
        # Modernized alias: only emit when EMA generator exists. Modernized
        # leaves switch ModelCheckpoint(monitor="loss/validate_ema").
        if self.generator_ema is not None and self.validation_losses_ema:
            self.log(
                "loss/validate_ema",
                _aggregate_validation_losses(self.validation_losses_ema),
                sync_dist=True,
            )
        self.validation_step_outputs.clear()
        self.validation_losses_raw.clear()
        self.validation_losses_ema.clear()

    def configure_optimizers(self):
        """Build two AdamW optimizers + WarmupCosine schedulers via the shared helper."""
        [opt_g], [sch_g] = configure_adamw_scheduler(
            self,
            self.generator,
            self.lr_g,
            self.schedule,
            warmup_steps=self.warmup_steps,
            warmup_multiplier=self.warmup_multiplier,
        )
        [opt_d], [sch_d] = configure_adamw_scheduler(
            self,
            self.discriminator,
            self.lr_d,
            self.schedule,
            warmup_steps=self.warmup_steps,
            warmup_multiplier=self.warmup_multiplier,
        )
        return [opt_g, opt_d], [sch_g, sch_d]

    def on_predict_start(self) -> None:
        """Build the divisible-pad transform matching the generator.

        Also logs which generator (raw vs EMA) will be used at inference, so
        users notice a silent fallback when EMA is unexpectedly disabled
        (e.g., a predict overlay that forgot to set ``ema_kimg`` on a
        modernized checkpoint, or ``use_ema_at_predict=False``).
        """
        self._predict_pad = _make_divisible_pad(self.generator)
        which = "EMA" if (self.generator_ema is not None and self.use_ema_at_predict) else "raw"
        _logger.info(
            "DynacellGAN predict: using %s generator (ema_kimg=%s, use_ema_at_predict=%s, generator_ema=%s)",
            which,
            self.ema_kimg,
            self.use_ema_at_predict,
            "present" if self.generator_ema is not None else "absent",
        )

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
            # Inference uses EMA generator when available (and use_ema_at_predict=True).
            prediction = self._inference_generator()(source)
        else:
            raise ValueError(f"Unknown predict_method: {self.predict_method!r}. Choose 'full_image'.")
        return _center_crop_to_shape(prediction, original_shape)
