"""CellDiff end-to-end 3D virtual staining LightningModule."""

import itertools
from typing import Literal, Sequence

import wandb

import numpy as np
import torch
from lightning.pytorch import LightningModule
from monai.optimizers import WarmupCosineSchedule
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR

from viscy_models import UNetViT3D
from viscy_utils.log_images import detach_sample, render_images


class CELLDiffE2E(LightningModule):
    """End-to-end 3D virtual staining with UNetViT3D.

    Parameters
    ----------
    model_config : dict or None
        Keyword arguments forwarded to ``UNetViT3D``.
        Defaults to UNetViT3D's own defaults when None.
    loss_function : nn.Module or None
        Loss function for training and validation.
        Defaults to MSELoss when None.
    lr : float
        Learning rate for AdamW optimizer.
    schedule : {"WarmupCosine", "Constant"}
        Learning rate schedule.
    log_batches_per_epoch : int
        Number of batches per epoch to accumulate for image logging.
    log_samples_per_batch : int
        Number of samples per batch to log.
    """

    def __init__(
        self,
        model_config: dict | None = None,
        loss_function: nn.Module | None = None,
        lr: float = 1e-4,
        schedule: Literal["WarmupCosine", "Constant"] = "WarmupCosine",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        predict_overlap: tuple[int, int, int] = (4, 256, 256),
    ) -> None:
        super().__init__()
        if model_config is None:
            model_config = {}
        self.model = UNetViT3D(**model_config)
        self.loss_function = loss_function if loss_function is not None else nn.MSELoss()
        self.lr = lr
        self.schedule = schedule
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self._model_config = model_config
        self.predict_overlap = predict_overlap
        self._training_step_outputs: list = []
        self._validation_step_outputs: list = []
        self._validation_losses: list[list[Tensor]] = []

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor
            Predicted tensor of shape ``(B, C, D, H, W)``.
        """
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : dict
            Must contain ``"source"`` and ``"target"`` tensors.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            Training loss.
        """
        source: Tensor = batch["source"]
        target: Tensor = batch["target"]
        pred = self.forward(source)
        loss = self.loss_function(pred, target)
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
        if batch_idx < self.log_batches_per_epoch:
            self._training_step_outputs.extend(
                detach_sample((source, target, pred), self.log_samples_per_batch)
            )
        return loss

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Execute a single validation step.

        Parameters
        ----------
        batch : dict
            Must contain ``"source"`` and ``"target"`` tensors.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Index of the current validation dataloader.
        """
        source: Tensor = batch["source"]
        target: Tensor = batch["target"]
        pred = self.forward(source)
        loss = self.loss_function(pred, target)
        while dataloader_idx >= len(self._validation_losses):
            self._validation_losses.append([])
        self._validation_losses[dataloader_idx].append(loss.detach())
        self.log(
            f"loss/val/{dataloader_idx}",
            loss,
            sync_dist=True,
            batch_size=source.shape[0],
        )
        if batch_idx < self.log_batches_per_epoch:
            self._validation_step_outputs.extend(
                detach_sample((source, target, pred), self.log_samples_per_batch)
            )

    def on_train_epoch_end(self) -> None:
        """Log training image samples at end of epoch."""
        self._log_samples("train_samples", self._training_step_outputs)
        self._training_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        """Log validation images and aggregate loss at end of epoch."""
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self._validation_step_outputs)
        self._validation_step_outputs.clear()
        loss_means = [torch.stack(losses).mean() for losses in self._validation_losses]
        self.log(
            "loss/validate",
            torch.stack(loss_means).mean(),
            sync_dist=True,
        )
        self._validation_losses.clear()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Execute a single prediction step.

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
            Model prediction.
        """
        source: Tensor = batch["source"]
        original_shape = source.shape[2:]
        patch_size = self.model.input_spatial_size  # [D, H, W]

        # Pad to at least patch_size using replicate padding if any dim is too small
        pad = []
        for s, p in zip(reversed(source.shape[2:]), reversed(patch_size)):
            pad.extend([0, max(0, p - s)])
        if any(p > 0 for p in pad):
            source = torch.nn.functional.pad(source, pad, mode="replicate")
        overlap = self.predict_overlap
        padded_shape = list(source.shape[2:])

        prediction_sum = torch.zeros_like(source)
        prediction_count = torch.zeros_like(source)

        start_lists = []
        for i in range(3):
            S, P, O = padded_shape[i], patch_size[i], overlap[i]
            stride = P - O
            starts = sorted(set(range(0, S - P, stride)) | {S - P})
            start_lists.append(starts)

        for starts in itertools.product(*start_lists):
            slicer = (slice(None), slice(None)) + tuple(
                slice(s, s + patch_size[i]) for i, s in enumerate(starts)
            )
            patch_pred = self.forward(source[slicer])
            prediction_sum[slicer] += patch_pred
            prediction_count[slicer] += 1

        prediction = prediction_sum / prediction_count
        return prediction[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]

    def configure_optimizers(self):
        """Configure AdamW optimizer with WarmupCosine or Constant LR schedule."""
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
        else:
            raise ValueError(f"Unknown schedule: {self.schedule!r}. Choose 'WarmupCosine' or 'Constant'.")
        return [optimizer], [scheduler]

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]) -> None:
        """Log image samples to W&B."""
        if not imgs or self.logger is None:
            return
        grid = render_images(imgs)

        self.logger.experiment.log({key: wandb.Image(grid)}, step=self.current_epoch)
