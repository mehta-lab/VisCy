"""Linear probing of trained encoder based on cell state labels."""

import logging
from pprint import pformat
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from numpy.typing import NDArray
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)

_logger = logging.getLogger("lightning.pytorch")


def _test_metrics(pred: Tensor, y: Tensor, num_classes: int) -> dict[str, float]:
    """Test metrics for the linear classifier.

    Parameters
    ----------
    pred : Tensor
        Predicted logits
    y : Tensor
        Labels
    num_classes : int
        Number of classes

    Returns
    -------
    dict[str, float]
        Metrics
    """
    # TODO: add more metrics
    metrics = {}
    metrics.update(
        {
            f"accuracy_{average}": multiclass_accuracy(
                pred, y, num_classes, average=average
            ).item(),
            f"f1_{average}": multiclass_f1_score(
                pred, y, num_classes, average=average
            ).item(),
        }
        for average in ["macro", "weighted"]
    )
    return metrics


class LinearProbingDataModule(LightningDataModule):
    def __init__(
        self,
        embeddings: Tensor,
        labels: Tensor,
        split_ratio: tuple[int, int, int],
        batch_size: int,
    ) -> None:
        """Data module for linear probing.

        Parameters
        ----------
        embeddings : Tensor
            Input embeddings
        labels : Tensor
            Annotation labels
        split_ratio : tuple[int, int, int]
            Train/validate/test split ratio, must sum to 1.
        batch_size : int
            Batch sizes
        """
        super().__init__()
        if not embeddings.shape[0] == labels.shape[0]:
            raise ValueError("Number of samples in embeddings and labels must match.")
        if sum(split_ratio) != 1.0:
            raise ValueError("Split ratio must sum to 1.")
        embeddings = embeddings.float()
        labels = labels.float()
        self.dataset = TensorDataset(embeddings, labels)
        self.split_ratio = split_ratio
        self.batch_size = batch_size

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        n = len(self.dataset)
        train_size = int(n * self.split_ratio[0])
        val_size = int(n * self.split_ratio[1])
        test_size = n - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(
                self.dataset, [train_size, val_size, test_size]
            )
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class LinearClassifier(LightningModule):
    def __init__(self, in_features: int, out_features: int, lr: float) -> None:
        """Linear classifier.

        Parameters
        ----------
        in_features : int
            Number of input feature channels
        out_features : int
            Number of output feature channels (number of classes)
        lr : float
            Learning rate
        """
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        pred = self(x)
        self.log("loss/val", self.loss(pred, y))

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        return optim.AdamW(self.parameters())

    def on_test_start(self) -> None:
        self.test_labels: list[Tensor] = []
        self.test_predictions: list[Tensor] = []

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        pred = self(x)
        self.test_labels.append(y)
        self.test_predictions.append(pred)

    def on_test_epoch_end(self) -> None:
        y = torch.cat(self.test_labels)
        pred = torch.cat(self.test_predictions)
        num_classes = self.fc.out_features
        _logger.info("Test metrics:\n" + pformat(_test_metrics(pred, y, num_classes)))

    def predict_step(self, x: Tensor) -> Tensor:
        logits = self(x)
        return torch.argmax(logits, dim=1)


def train_and_test_linear_classifier(
    embeddings: NDArray,
    labels: NDArray,
    split_ratio: tuple[int, int, int] = (0.4, 0.2, 0.4),
    batch_size: int = 1024,
    lr: float = 1e-3,
    train_max_epochs: int = 10,
    **trainer_kwargs,
) -> None:
    """Train and test a linear classifier.

    Parameters
    ----------
    embeddings : NDArray
        Input embeddings, shape (n_samples, n_features).
    labels : NDArray
        Annotation labels, shape (n_samples,).
    split_ratio : tuple[int, int, int], optional
        Train/validate/test split ratio, by default (0.4, 0.2, 0.4)
    batch_size : int, optional
        Batch size, by default 1024
    lr : float, optional
        Learning rate, by default 1e-3
    train_max_epochs : int, optional
        Maximum number of training epochs, by default 10
    **trainer_kwargs
        Additional keyword arguments for the Lightning Trainer class.
    """
    if not isinstance(embeddings, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("Input embeddings and labels must be NumPy arrays.")
    if not embeddings.ndim == 2:
        raise ValueError("Input embeddings must have 2 dimensions.")
    if not labels.ndim == 1:
        raise ValueError("Labels must have 1 dimension.")
    embeddings = torch.from_numpy(embeddings)
    labels_onehot = torch.nn.functional.one_hot(torch.from_numpy(labels))
    data = LinearProbingDataModule(embeddings, labels_onehot, split_ratio, batch_size)
    model = LinearClassifier(embeddings.shape[1], labels_onehot.shape[1], lr)
    trainer = Trainer(max_epochs=train_max_epochs, **trainer_kwargs)
    trainer.fit(model, data)
    trainer.test(model, data)
