"""Foundation model LightningModule for frozen inference (and future fine-tuning)."""

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from dynaclr.engine import ContrastivePrediction
from viscy_data._typing import TripletSample


class FoundationModule(LightningModule):
    """Lightning wrapper around a foundation model for prediction.

    Parameters
    ----------
    model : nn.Module
        A foundation model (e.g. ``DINOv3Model``, ``OpenPhenomModel``)
        returning ``(features, projections)``.
    lr : float
        Learning rate for future fine-tuning, by default ``1e-4``.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return features and projections."""
        return self.model(x)

    def predict_step(self, batch: TripletSample, batch_idx: int, dataloader_idx: int = 0) -> ContrastivePrediction:
        """Extract embeddings from anchor images.

        Calls ``model.preprocess_2d`` (if available) to convert raw
        dataloader output before the backbone forward pass.  Dataloaders
        that already produce ``(B, 3, H, W)`` tensors can use a model
        without ``preprocess_2d``.
        """
        x = batch["anchor"]
        if hasattr(self.model, "preprocess_2d"):
            x = self.model.preprocess_2d(x)
        features, projections = self.model(x)
        return {
            "features": features,
            "projections": projections,
            "index": batch["index"],
        }

    def configure_optimizers(self):
        """Return AdamW optimizer (placeholder for fine-tuning)."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
