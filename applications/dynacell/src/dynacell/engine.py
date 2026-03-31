"""Dynacell LightningModule for supervised virtual staining benchmarks."""

import inspect
import logging
from typing import Literal, Sequence

import numpy as np
import torch
from lightning.pytorch import LightningModule
from monai.optimizers import WarmupCosineSchedule
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR

from viscy_data import Sample
from viscy_models import Unet3d
from viscy_models.celldiff import UNetViT3D
from viscy_utils.log_images import detach_sample, log_image_grid

_ARCHITECTURE: dict[str, type[nn.Module]] = {
    "UNetViT3D": UNetViT3D,
    "FNet3D": Unet3d,
}

_logger = logging.getLogger("lightning.pytorch")


class DynacellUNet(LightningModule):
    """Supervised regression U-Net for benchmark virtual staining.

    Parameters
    ----------
    architecture : {"UNetViT3D", "FNet3D"}
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
        Checkpoint path to load model weights.
    """

    def __init__(
        self,
        architecture: Literal["UNetViT3D", "FNet3D"] = "UNetViT3D",
        model_config: dict | None = None,
        loss_function: nn.Module | None = None,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        example_input_yx_shape: Sequence[int] = (256, 256),
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
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
        self.training_step_outputs: list = []
        self.validation_losses: list[list] = []
        self.validation_step_outputs: list = []

        # Cache fg_mask compatibility to avoid per-batch inspect.signature().
        sig = inspect.signature(self.loss_function.forward)
        self._loss_accepts_fg_mask = "fg_mask" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        # Build example_input_array for TensorBoard graph logging.
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

    def training_step(self, batch: Sample | Sequence[Sample], batch_idx: int):
        """Execute a single training step.

        Parameters
        ----------
        batch : Sample | Sequence[Sample]
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            Training loss.
        """
        losses = []
        batch_size = 0
        if not isinstance(batch, Sequence):
            batch = [batch]
        for b in batch:
            source = b["source"]
            target = b["target"]
            pred = self.forward(source)
            loss = self._compute_loss(pred, target, b)
            losses.append(loss)
            batch_size += source.shape[0]
            if batch_idx < self.log_batches_per_epoch:
                self.training_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))
        loss_step = torch.stack(losses).mean()
        self.log(
            "loss/train",
            loss_step.to(self.device),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss_step

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
        self.validation_losses[dataloader_idx].append(loss.detach())
        self.log(
            f"loss/val/{dataloader_idx}",
            loss.to(self.device),
            sync_dist=True,
            batch_size=source.shape[0],
        )
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        """Not implemented in Stage 2.

        Raises
        ------
        NotImplementedError
            Prediction requires DivisiblePad and tiled inference (Stage 3).
        """
        raise NotImplementedError(
            "Prediction is not supported in Dynacell v1. "
            "The HCS predict pipeline passes full-FOV spatial sizes "
            "which are incompatible with UNetViT3D/FNet3D without "
            "DivisiblePad and tiled inference. See Stage 3."
        )

    def on_train_epoch_end(self):
        """Log training image samples."""
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        """Log validation samples and aggregate loss."""
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)
        loss_means = [torch.tensor(losses).mean() for losses in self.validation_losses]
        self.log(
            "loss/validate",
            torch.tensor(loss_means).mean().to(self.device),
            sync_dist=True,
        )
        self.validation_step_outputs.clear()
        self.validation_losses.clear()

    def configure_optimizers(self):
        """Configure AdamW optimizer with LR scheduler."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        if self.schedule == "WarmupCosine":
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=3,
                t_total=self.trainer.max_epochs,
                warmup_multiplier=1e-3,
            )
        elif self.schedule == "Constant":
            scheduler = ConstantLR(optimizer, factor=1, total_iters=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        """Log image grid to the active logger."""
        if not self.trainer.is_global_zero or self.logger is None:
            return
        log_image_grid(self.logger, key, imgs, self.current_epoch)
