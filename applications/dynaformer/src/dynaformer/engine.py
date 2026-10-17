"""LightningModule for decoupled spatio-temporal virtual staining."""

import logging
from typing import Literal, Sequence

import torch
from lightning.pytorch import LightningModule
from monai.optimizers import WarmupCosineSchedule
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR

from dynaformer.models.dynaformer import SpatioTemporalModel
from viscy_utils.log_images import detach_sample, log_image_grid

_logger = logging.getLogger("lightning.pytorch")

__all__ = ["DynaformerModule"]


class DynaformerModule(LightningModule):
    """Train a :class:`~dynaformer.models.SpatioTemporalModel` on movie clips.

    Optimizes a dense image objective over the predicted marker plus any
    auxiliary per-frame heads, which share the same encoder and temporal module.

    Parameters
    ----------
    model : SpatioTemporalModel
        The composed encoder, temporal module, decoder and heads.
    loss_function : nn.Module | None
        Image loss over ``(B * T, C, Z, Y, X)`` tensors. Defaults to an equally
        weighted L1 and squared error, which is what the reference configuration
        uses; ``None`` selects that default.
    lr : float
        Learning rate for freshly initialized parameters, by default 3e-4.
    encoder_lr : float | None
        Learning rate for the frame encoder. Set this well below ``lr`` when the
        encoder is a pretrained backbone — fine-tuning it at the head learning
        rate destroys the pretrained features. ``None`` trains everything at
        ``lr``, by default ``None``.
    weight_decay : float
        AdamW weight decay, applied only to parameters with more than one
        dimension, by default 0.
    schedule : {"WarmupCosine", "Constant"}
        Learning rate schedule, by default ``"Constant"``.
    warmup_steps : int
        WarmupCosine only: steps spent ramping the learning rate, by default 3.
    warmup_multiplier : float
        WarmupCosine only: learning rate multiplier at step 0, by default 1e-3.
    ckpt_path : str | None
        Checkpoint to load weights from, by default ``None``.
    log_batches_per_epoch : int
        Number of batches to log example images from, by default 4.
    log_samples_per_batch : int
        Number of clips per logged batch, by default 1.
    log_frames_per_sample : int
        Number of evenly spaced frames shown per logged clip, by default 4.
    example_input_shape : Sequence[int] | None
        ``(B, T, C, Z, Y, X)`` shape used to trace the graph. ``None`` skips
        tracing, by default ``None``.
    """

    def __init__(
        self,
        model: SpatioTemporalModel,
        loss_function: nn.Module | None = None,
        lr: float = 3e-4,
        encoder_lr: float | None = None,
        weight_decay: float = 0.0,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        warmup_steps: int = 3,
        warmup_multiplier: float = 1e-3,
        ckpt_path: str | None = None,
        log_batches_per_epoch: int = 4,
        log_samples_per_batch: int = 1,
        log_frames_per_sample: int = 4,
        example_input_shape: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_function", "ckpt_path"])
        self.model = model
        self.loss_function = loss_function if loss_function is not None else _L1AndMSELoss()
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.warmup_multiplier = warmup_multiplier
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.log_frames_per_sample = log_frames_per_sample
        self.training_step_outputs: list = []
        self.validation_step_outputs: list = []
        self.validation_losses: list[list[Tensor]] = []
        if example_input_shape is not None:
            self.example_input_array = torch.rand(*example_input_shape)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, weights_only=True, map_location="cpu")["state_dict"]
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing:
                _logger.info(f"Checkpoint is missing keys (new modules): {missing}")
            if unexpected:
                _logger.warning(f"Checkpoint has unexpected keys: {unexpected}")

    def forward(self, x: Tensor) -> tuple[Tensor | None, dict[str, Tensor]]:
        """Run the model on a ``(B, T, C, Z, Y, X)`` movie."""
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        """Advance each head's loss-weight schedule."""
        for head in self.model.heads.values():
            head.step(self.current_epoch)

    def _compute_losses(self, batch: dict, stage: str) -> tuple[Tensor, Tensor | None]:
        """Run the model and combine the image loss with every head's loss.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Total loss and the predicted movie (``None`` without a decoder).
        """
        prediction, head_outputs = self.forward(batch["source"])
        total = torch.zeros((), device=self.device)

        if prediction is not None:
            target = batch["target"]
            image_loss = self.loss_function(prediction.flatten(0, 1), target.flatten(0, 1))
            self.log(f"loss/image/{stage}", image_loss, sync_dist=True, batch_size=target.shape[0])
            total = total + image_loss

        for name, head in self.model.heads.items():
            y = batch[head.batch_key]
            y_hat = head_outputs[name]
            head_loss = head.compute_loss(y_hat, y)
            head.log_metrics({"loss": head_loss, "logits": y_hat, "y": y}, self.log, stage)
            total = total + head.get_weight() * head_loss

        return total, prediction

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Run one optimization step."""
        loss, prediction = self._compute_losses(batch, "train")
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["source"].shape[0],
        )
        if batch_idx < self.log_batches_per_epoch and prediction is not None:
            self.training_step_outputs.extend(self._detach_clip(batch, prediction))
        return loss

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Score one validation batch."""
        loss, prediction = self._compute_losses(batch, "val")
        if dataloader_idx + 1 > len(self.validation_losses):
            self.validation_losses.append([])
        self.validation_losses[dataloader_idx].append(loss.detach())
        self.log(
            f"loss/val/{dataloader_idx}",
            loss,
            sync_dist=True,
            batch_size=batch["source"].shape[0],
        )
        if batch_idx < self.log_batches_per_epoch and prediction is not None:
            self.validation_step_outputs.extend(self._detach_clip(batch, prediction))

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Predict a movie and every head's per-frame output."""
        prediction, head_outputs = self.forward(batch["source"])
        return {"prediction": prediction, "heads": head_outputs, "index": batch.get("index")}

    def on_train_epoch_end(self) -> None:
        """Log training example images."""
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Log validation example images and the epoch validation loss."""
        self._log_samples("val_samples", self.validation_step_outputs)
        if self.validation_losses:
            per_loader = [torch.stack(losses).mean() for losses in self.validation_losses]
            self.log("loss/validate", torch.stack(per_loader).mean(), sync_dist=True)
        self.validation_step_outputs.clear()
        self.validation_losses.clear()

    def _detach_clip(self, batch: dict, prediction: Tensor) -> list[list]:
        """Build image rows showing evenly spaced frames of a clip.

        Each row is one clip; columns run source frames, then target frames, then
        predicted frames, so drift between neighboring frames is visible at a
        glance — the failure mode a decoder without temporal context shows.
        """
        frames = prediction.shape[1]
        indices = torch.linspace(0, frames - 1, min(self.log_frames_per_sample, frames)).round().long()
        views = (
            [batch["source"][:, t] for t in indices]
            + [batch["target"][:, t] for t in indices]
            + [prediction[:, t] for t in indices]
        )
        return detach_sample(views, self.log_samples_per_batch)

    def _log_samples(self, key: str, images: list) -> None:
        """Send an image grid to the logger from rank zero only."""
        if not images or self.logger is None or not self.trainer.is_global_zero:
            return
        log_image_grid(self.logger, key, images, self.current_epoch)

    def _parameter_groups(self) -> list[dict]:
        """Split parameters into encoder and new-parameter learning-rate groups.

        Weight decay is skipped for parameters with one dimension or fewer
        (biases and normalization scales), the standard transformer recipe.
        """
        encoder_ids = {id(p) for p in self.model.encoder.parameters()}
        groups: dict[tuple[float, float], list[nn.Parameter]] = {}
        encoder_lr = self.lr if self.encoder_lr is None else self.encoder_lr
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            lr = encoder_lr if id(parameter) in encoder_ids else self.lr
            decay = 0.0 if parameter.ndim <= 1 else self.weight_decay
            groups.setdefault((lr, decay), []).append(parameter)
        return [{"params": params, "lr": lr, "weight_decay": decay} for (lr, decay), params in sorted(groups.items())]

    def configure_optimizers(self) -> tuple[list, list]:
        """Build AdamW with per-group learning rates and the configured schedule."""
        groups = self._parameter_groups()
        optimizer = torch.optim.AdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
        if self.encoder_lr is not None:
            _logger.info(f"Discriminative fine-tuning: encoder_lr={self.encoder_lr}, lr={self.lr}")
        if self.schedule == "WarmupCosine":
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=self.warmup_steps,
                t_total=self.trainer.estimated_stepping_batches,
                warmup_multiplier=self.warmup_multiplier,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        if self.schedule == "Constant":
            return [optimizer], [ConstantLR(optimizer, factor=1, total_iters=self.trainer.max_epochs)]
        raise ValueError(f"Unknown schedule {self.schedule!r}, expected 'WarmupCosine' or 'Constant'.")


class _L1AndMSELoss(nn.Module):
    """Equally weighted absolute and squared error.

    The default image objective. Kept as a module rather than a lambda so it
    survives checkpointing and shows up in the saved hyperparameters.
    """

    def __init__(self, l1_weight: float = 1.0, mse_weight: float = 1.0) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Return the weighted sum of L1 and squared error."""
        residual = preds - target
        return self.l1_weight * residual.abs().mean() + self.mse_weight * residual.pow(2).mean()
