import logging
from typing import Any, Dict, Literal, Tuple, Type

import torch
from pytorch_lightning import LightningModule
from pythae.models import BaseAE, BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.vae import VAE, VAEConfig
from pythae.data.datasets import BaseDataset
from torch import Tensor

_logger = logging.getLogger("lightning.pytorch")


class PythaeLightningVAE(LightningModule):
    """Base class for Pythae-based VAE models in Lightning.

    This class provides a wrapper around any Pythae VAE model to use with PyTorch Lightning.
    It handles training, validation, prediction, and logging in a consistent way.

    Parameters
    ----------
    input_dim : Tuple[int, ...]
        Input dimensions (channels, depth, height, width)
    latent_dim : int
        Dimension of the latent space
    lr : float, optional
        Learning rate, by default 1e-3
    schedule : Literal["WarmupCosine", "Constant"]
        Learning rate schedule, by default "Constant"
    """

    def __init__(
        self,
        input_dim: Tuple[int, ...],
        latent_dim: int = 10,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize Pythae's VAE model
        model_config = VAEConfig(input_dim=input_dim, latent_dim=latent_dim)
        self.model = VAE(model_config)
        self.lr = lr
        self.schedule = schedule

    def _wrap_input(self, x: Tensor) -> BaseDataset:
        """Wrap input tensor in Pythae's BaseDataset format.

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        BaseDataset
            Wrapped input in Pythae's format
        """
        # Create dummy labels (zeros) with same batch size as the data
        dummy_labels = torch.zeros(x.shape[0])
        return BaseDataset(data=x, labels=dummy_labels)

    def forward(self, x: Tensor) -> ModelOutput:
        # Wrap input in BaseDataset format
        x_wrapped = self._wrap_input(x)
        return self.model(x_wrapped)

    def training_step(self, batch, batch_idx):
        x = batch["anchor"]
        output = self(x)

        # Get individual losses
        recon_loss = output.recon_loss
        kld = output.kld
        loss = output.loss

        # Log metrics
        self.log(
            "train/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train/kld", kld, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["anchor"]
        output = self(x)

        # Get individual losses
        recon_loss = output.recon_loss
        kld = output.kld
        loss = output.loss

        # Log metrics
        self.log(
            "val/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("val/kld", kld, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        if self.schedule == "WarmupCosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
            }
        return optimizer

    def predict_step(self, batch, batch_idx):
        x = batch["anchor"]
        output = self(x)
        return {
            "latent": output.z,
            "reconstruction": output.recon_x,
            "index": batch["index"],
        }
