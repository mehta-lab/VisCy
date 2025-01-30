import logging
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from tarrow.models import TimeArrowNet
from tarrow.models.losses import DecorrelationLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau

from viscy.utils.log_images import render_images

logger = logging.getLogger(__name__)


class TarrowModule(LightningModule):
    """Lightning Module wrapper for TimeArrowNet.

    Parameters
    ----------
    backbone : str, default="unet"
        Dense network architecture
    projection_head : str, default="minimal_batchnorm"
        Dense projection head architecture
    classification_head : str, default="minimal"
        Classification head architecture
    n_frames : int, default=2
        Number of input frames
    n_features : int, default=16
        Number of output features from the backbone
    n_input_channels : int, default=1
        Number of input channels
    symmetric : bool, default=False
        If True, use permutation-equivariant classification head
    learning_rate : float, default=1e-4
        Learning rate for optimizer
    weight_decay : float, default=1e-6
        Weight decay for optimizer
    lambda_decorrelation : float, default=0.01
        Prefactor of decorrelation loss
    lr_scheduler : str, default="cyclic"
        Learning rate scheduler ('plateau' or 'cyclic')
    lr_patience : int, default=50
        Patience for learning rate scheduler
    cam_size : tuple or int, optional
        Size of the class activation map (H, W). If None, use input size.
    log_batches_per_epoch : int, default=8
        Number of batches to log samples from during training
    log_samples_per_batch : int, default=1
        Number of samples to log from each batch
    """

    def __init__(
        self,
        backbone="unet",
        projection_head="minimal_batchnorm",
        classification_head="minimal",
        n_frames=2,
        n_features=16,
        n_input_channels=1,
        symmetric=False,
        learning_rate=1e-4,
        weight_decay=1e-6,
        lambda_decorrelation=0.01,
        lr_scheduler="cyclic",
        lr_patience=50,
        cam_size=None,
        log_batches_per_epoch=8,
        log_samples_per_batch=1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.model = TimeArrowNet(
            backbone=backbone,
            projection_head=projection_head,
            classification_head=classification_head,
            n_frames=n_frames,
            n_features=n_features,
            n_input_channels=n_input_channels,
            symmetric=symmetric,
        )

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.criterion_decorr = DecorrelationLoss()

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        """Log sample images to TensorBoard.

        Parameters
        ----------
        key : str
            Key for logging
        imgs : Sequence[Sequence[np.ndarray]]
            List of image pairs to log
        """
        grid = render_images(imgs, cmaps=["gray"] * 2)  # Only 2 timepoints
        self.logger.experiment.add_image(
            key, grid, self.current_epoch, dataformats="HWC"
        )

    def _log_step_samples(self, batch_idx, images, stage: Literal["train", "val"]):
        """Log samples from a batch.

        Parameters
        ----------
        batch_idx : int
            Index of current batch
        images : torch.Tensor
            Batch of images with shape (B, T, C, H, W)
        stage : str
            Either "train" or "val"
        """
        if batch_idx < self.log_batches_per_epoch:
            # Get first n samples from batch
            n = min(self.log_samples_per_batch, images.shape[0])
            samples = images[:n].detach().cpu().numpy()

            # Split into pairs of timepoints
            pairs = [(sample[0], sample[1]) for sample in samples]

            output_list = (
                self.training_step_outputs
                if stage == "train"
                else self.validation_step_outputs
            )
            output_list.extend(pairs)

    def forward(self, x):
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_frames, channels, height, width)

        Returns
        -------
        tuple
            Tuple of (output, projection) where:
            - output is the classification logits
            - projection is the feature space projection
        """
        return self.model(x, mode="both")

    def _shared_step(self, batch, batch_idx, step="train"):
        """Shared step for training and validation.

        Parameters
        ----------
        batch : tuple
            Tuple of (images, labels)
        batch_idx : int
            Index of the current batch
        step : str, default="train"
            Current step type ("train" or "val")

        Returns
        -------
        torch.Tensor
            Combined loss (classification + decorrelation)
        """
        x, y = batch

        # Log sample images
        self._log_step_samples(batch_idx, x, step)

        out, pro = self(x)

        if out.ndim > 2:
            y = torch.broadcast_to(
                y.unsqueeze(1).unsqueeze(1), (y.shape[0],) + out.shape[-2:]
            )
            loss = self.criterion(out, y)
            loss = torch.mean(loss, tuple(range(1, loss.ndim)))
            y = y[:, 0, 0]
            u_avg = torch.mean(out, tuple(range(2, out.ndim)))
        else:
            u_avg = out
            loss = self.criterion(out, y)

        pred = torch.argmax(u_avg.detach(), 1)
        loss = torch.mean(loss)

        # decorrelation loss
        pro_batched = pro.flatten(0, 1)
        loss_decorr = self.criterion_decorr(pro_batched)
        loss_all = loss + self.hparams.lambda_decorrelation * loss_decorr

        acc = torch.mean((pred == y).float())

        # Main classification loss
        self.log(f"loss/{step}_loss", loss, prog_bar=True)
        # Decorrelation loss for feature space
        self.log(f"loss/{step}_loss_decorr", loss_decorr, prog_bar=True)
        # Classification accuracy
        self.log(f"metric/{step}_accuracy", acc, prog_bar=True)
        # Ratio of positive predictions (class 1) - useful to detect class imbalance
        self.log(f"metric/{step}_pred1_ratio", pred.sum().float() / len(pred))

        return loss_all

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.2,
                patience=self.hparams.lr_patience,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss/val_loss",
                    "interval": "epoch",
                },
            }
        elif self.hparams.lr_scheduler == "cyclic":
            # Get dataloader length accounting for DDP
            dataloader = self.trainer.datamodule.train_dataloader()
            steps_per_epoch = len(dataloader)

            # Account for gradient accumulation and multiple GPUs
            if self.trainer.accumulate_grad_batches:
                steps_per_epoch = (
                    steps_per_epoch // self.trainer.accumulate_grad_batches
                )

            total_steps = steps_per_epoch * self.trainer.max_epochs

            scheduler = CyclicLR(
                optimizer,
                base_lr=self.hparams.learning_rate,
                max_lr=self.hparams.learning_rate * 10,
                cycle_momentum=False,
                step_size_up=total_steps // 2,  # Half the total steps for one cycle
                scale_mode="cycle",
                scale_fn=lambda x: 0.9**x,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

    def on_train_epoch_end(self):
        """Log collected training samples at end of epoch."""
        if self.training_step_outputs:
            self._log_samples("train_samples", self.training_step_outputs)
            self.training_step_outputs = []

    def on_validation_epoch_end(self):
        """Log collected validation samples at end of epoch."""
        if self.validation_step_outputs:
            self._log_samples("val_samples", self.validation_step_outputs)
            self.validation_step_outputs = []
