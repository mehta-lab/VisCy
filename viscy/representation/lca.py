"""Linear probing of trained encoder based on cell state labels."""

import logging
from pprint import pformat
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from numpy.typing import NDArray
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)

_logger = logging.getLogger("lightning.pytorch")


def _test_metrics(preds: Tensor, target: Tensor, num_classes: int) -> dict[str, float]:
    """Test metrics for the linear classifier.

    Parameters
    ----------
    preds : Tensor
        Predicted logits, shape (n_samples, n_classes)
    target : Tensor
        Labels, shape (n_samples,)
    num_classes : int
        Number of classes

    Returns
    -------
    dict[str, float]
        Test metrics
    """
    # TODO: add more metrics
    metrics = {}
    for average in ["macro", "weighted"]:
        metrics[f"accuracy_{average}"] = multiclass_accuracy(
            preds, target, num_classes, average=average
        ).item()
        metrics[f"f1_{average}"] = multiclass_f1_score(
            preds, target, num_classes, average=average
        ).item()
    return metrics


class LinearProbingDataModule(LightningDataModule):
    def __init__(
        self,
        embeddings: Tensor,
        labels: Tensor,
        split_ratio: tuple[int, int, int],
        batch_size: int,
        use_smote: bool = False,
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
        use_smote : bool, optional
           Whether to apply SMOTE to the training data, by default False
        """
        super().__init__()
        if not embeddings.shape[0] == labels.shape[0]:
            raise ValueError("Number of samples in embeddings and labels must match.")
        if sum(split_ratio) != 1.0:
            raise ValueError("Split ratio must sum to 1.")
        self.dataset = TensorDataset(embeddings.float(), labels.long())
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.use_smote = use_smote
        self.test_indices = None

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

        self.test_indices = self.test_dataset.indices

        if self.use_smote and stage == "fit":
            train_embeddings, train_labels = zip(
                *[(x, y) for x, y in self.train_dataset]
            )
            train_embeddings = torch.stack(train_embeddings)
            train_labels = torch.tensor(train_labels)

            smote = SMOTE()
            resampled_embeddings, resampled_labels = smote.fit_resample(
                train_embeddings.numpy(), train_labels.numpy()
            )

            self.train_dataset = TensorDataset(
                torch.from_numpy(resampled_embeddings).float(),
                torch.from_numpy(resampled_labels).long(),
            )

        elif not self.use_smote and stage == "fit":
            _logger.warning("SMOTE is disabled. Proceeding without oversampling.")

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
        self.test_predictions_cache = None

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

    def _fit_step(self, batch, stage: str) -> Tensor:
        x, y = batch
        preds = self(x)
        target = nn.functional.one_hot(y, num_classes=preds.shape[1]).float()
        loss = self.loss(preds, target)
        self.log(f"loss/{stage}", loss, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._fit_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int) -> None:
        _ = self._fit_step(batch, stage="val")

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        return optim.AdamW(self.parameters())

    def on_test_start(self) -> None:
        self.test_labels: list[Tensor] = []
        self.test_predictions: list[Tensor] = []

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        preds = self(x)
        self.test_labels.append(y)
        self.test_predictions.append(preds)

    def on_test_epoch_end(self) -> None:
        y = torch.cat(self.test_labels)
        preds = torch.cat(self.test_predictions)
        num_classes = self.fc.out_features
        test_class_preds = torch.argmax(preds, dim=1)
        self.test_predictions_cache = test_class_preds.cpu()
        _logger.info("Test metrics:\n" + pformat(_test_metrics(preds, y, num_classes)))

    def predict_step(self, x: Tensor) -> Tensor:
        logits = self(x)
        return torch.argmax(logits, dim=1)


def train_and_test_linear_classifier(
    embeddings: NDArray,
    labels: NDArray,
    num_classes: int,
    trainer: Trainer,
    split_ratio: tuple[int, int, int] = (0.4, 0.2, 0.4),
    batch_size: int = 1024,
    lr: float = 1e-3,
    use_smote: bool = False,
    save_predictions: bool = False,
    csv_path: str = None,
    merged_df: pd.DataFrame = None,
) -> None:
    """Train and test a linear classifier.

    Parameters
    ----------
    embeddings : NDArray
        Input embeddings, shape (n_samples, n_features).
    labels : NDArray
        Annotation labels, shape (n_samples,).
    num_classes : int
        Number of classes.
    trainer : Trainer
        Lightning Trainer object for training and testing.
        Define the number of epochs, logging, etc.
    split_ratio : tuple[int, int, int], optional
        Train/validate/test split ratio, by default (0.4, 0.2, 0.4)
    batch_size : int, optional
        Batch size, by default 1024
    lr : float, optional
        Learning rate, by default 1e-3
    use_smote : bool, optional
       Whether to apply SMOTE to the training data, by default False
    save_predictions : bool, optional
       Whether to save predictions to CSV, by default False
    csv_path : str, optional
       Path to save predictions CSV, by default None.
    merged_df : pd.DataFrame, optional
       DataFrame containing the initial input data, used for outputting correct predictions
    """
    if not isinstance(embeddings, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("Input embeddings and labels must be NumPy arrays.")
    if not embeddings.ndim == 2:
        raise ValueError("Input embeddings must have 2 dimensions.")
    if not labels.ndim == 1:
        raise ValueError("Labels must have 1 dimension.")
    embeddings = torch.from_numpy(embeddings)
    data = LinearProbingDataModule(
        embeddings, torch.from_numpy(labels), split_ratio, batch_size
    )
    model = LinearClassifier(embeddings.shape[1], num_classes, lr)
    trainer.fit(model, data)
    trainer.test(model, data)

    test_preds = model.test_predictions_cache

    if save_predictions:
        if csv_path is None or merged_df is None:
            raise ValueError(
                "csv_path and merged_df must be provided if save_predictions is True."
            )
        else:
            test_indices = data.test_indices

            label_mapping = {0: "background", 1: "uninfected", 2: "infected"}
            y_test_pred_mapped = [
                label_mapping[label] for label in test_preds.cpu().numpy()
            ]

            predicted_labels_df = pd.DataFrame(
                {
                    "id": merged_df.loc[test_indices, "id"].values,
                    "track_id": merged_df.loc[test_indices, "track_id"].values,
                    "fov_name": merged_df.loc[test_indices, "fov_name"].values,
                    "Predicted_Label": y_test_pred_mapped,
                }
            )

            predicted_labels_df.to_csv(csv_path, index=False)
