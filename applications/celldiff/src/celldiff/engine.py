"""CELLDiff 3D virtual staining LightningModule."""

from typing import Literal, Sequence

import wandb

import numpy as np
import torch
from lightning.pytorch import LightningModule
from monai.optimizers import WarmupCosineSchedule
from torch import Tensor
from torch.optim.lr_scheduler import ConstantLR

from viscy_models.cell_diff.celldff import CELLDiffNet, CELLDiff3DVS
from viscy_utils.log_images import detach_sample, render_images


class CELLDiff(LightningModule):
    """LightningModule for flow-matching 3D virtual staining with CELLDiff3DVS.

    Wraps :class:`~viscy_models.cell_diff.celldff.CELLDiff3DVS` for training,
    validation, and sliding-window inference.  The flow-matching loss is computed
    entirely inside ``CELLDiff3DVS.forward``; no external loss function is needed.

    Parameters
    ----------
    net_config : dict or None
        Keyword arguments forwarded to ``CELLDiffNet``.
        Defaults to ``CELLDiffNet``'s own defaults when ``None``.
    transport_config : dict or None
        Keyword arguments forwarded to ``CELLDiff3DVS`` (excluding ``net``).
        Supports ``path_type``, ``prediction``, ``loss_weight``, ``train_eps``,
        ``sample_eps``.  Defaults to ``CELLDiff3DVS``'s own defaults when ``None``.
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
    predict_overlap : int or tuple of int
        Overlap in each spatial dimension for sliding-window prediction.
        A single int applies the same overlap to all three dimensions.
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
        num_log_steps: int = 100,
        predict_method: Literal["non_overlapping", "sliding_window"] = "sliding_window",
        predict_overlap: int | tuple[int, int, int] = 256,
    ) -> None:
        super().__init__()
        net = CELLDiffNet(**(net_config or {}))
        self.model = CELLDiff3DVS(net, **(transport_config or {}))
        self.lr = lr
        self.schedule = schedule
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.num_generate_steps = num_generate_steps
        self.num_log_steps = num_log_steps
        self.predict_method = predict_method
        self.predict_overlap = predict_overlap
        self._training_step_outputs: list = []
        self._validation_step_outputs: list = []
        self._validation_losses: list[list[Tensor]] = []
        self._val_log_batch: tuple[Tensor, Tensor] | None = None

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Compute flow-matching training loss for one batch.

        Parameters
        ----------
        batch : dict
            Must contain ``"source"`` (phase) and ``"target"`` (fluorescence) tensors.
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
            self._training_step_outputs.extend(
                detach_sample((phase, target), self.log_samples_per_batch)
            )
        return loss

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        # no validation loss for flow matching.
        pass

    def on_train_epoch_end(self) -> None:
        """Log training image samples at end of epoch."""
        self._log_samples("train_samples", self._training_step_outputs)
        self._training_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        """Log validation images and generates a small number of predictions via ODE sampling
        (``num_log_steps`` steps) for qualitative inspection.
        """
        super().on_validation_epoch_end()

        if self._val_log_batch is not None and self.logger is not None:
            phase_log, target_log = self._val_log_batch
            n = min(self.log_samples_per_batch, phase_log.shape[0])
            generated = self.model.generate(phase_log[:n], num_steps=self.num_log_steps)
            gen_samples = detach_sample((phase_log[:n], target_log[:n], generated), n)
            self._log_samples("val_generated_samples", gen_samples)
            self._val_log_batch = None

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Generate virtual staining for one batch via sliding-window ODE sampling.

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
            Generated fluorescence of the same spatial shape as the input.
        """
        source: Tensor = batch["source"]
        original_shape = source.shape[2:]

        patch_size = self.model.net.input_spatial_size  # [D, H, W]
        min_size = (patch_size,) * 3 if isinstance(patch_size, int) else tuple(patch_size)
        if any(s < p for s, p in zip(source.shape[2:], min_size)):
            pad = []
            for s, p in zip(reversed(source.shape[2:]), reversed(min_size)):
                pad.extend([0, max(0, p - s)])
            source = torch.nn.functional.pad(source, pad, mode="replicate")

        if self.predict_method == "non_overlapping":
            prediction = self.model.generate_non_overlapping(
                source,
                num_steps=self.num_generate_steps,
            )
        elif self.predict_method == "sliding_window":
            prediction = self.model.generate_sliding_window(
                source,
                num_steps=self.num_generate_steps,
                overlap_size=self.predict_overlap,
            )
        else:
            raise ValueError(f"Unknown predict_method: {self.predict_method!r}. Choose 'non_overlapping' or 'sliding_window'.")

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