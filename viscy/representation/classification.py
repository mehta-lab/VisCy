from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from torch import nn
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.utils.log_images import render_images


class ClassificationPredictionWriter(BasePredictionWriter):
    """Prediction writer callback for saving classification outputs to CSV.

    Collects predictions from all batches and writes them to a CSV file at the
    end of each epoch. Converts tensor outputs to numpy arrays for storage.

    Parameters
    ----------
    output_path : Path
        Path to the output CSV file.
    """

    def __init__(self, output_path: Path) -> None:
        super().__init__("epoch")
        if Path(output_path).exists():
            raise FileExistsError(f"Output path {output_path} already exists.")
        self.output_path = output_path

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: list[dict[str, Any]],
        batch_indices: list[int],
    ) -> None:
        """Write all predictions to CSV file at epoch end.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer instance.
        pl_module : LightningModule
            Lightning module being trained.
        predictions : list[dict[str, Any]]
            List of prediction dictionaries from all batches.
        batch_indices : list[int]
            Indices of batches processed during prediction.
        """
        all_predictions = []
        for prediction in predictions:
            for key, value in prediction.items():
                if isinstance(value, torch.Tensor):
                    prediction[key] = value.detach().cpu().numpy().flatten()
            all_predictions.append(pd.DataFrame(prediction))
        pd.concat(all_predictions).to_csv(self.output_path, index=False)


class ClassificationModule(LightningModule):
    """Binary classification module using pre-trained contrastive encoder.

    Adapts a contrastive encoder for binary classification by replacing the
    final linear layer and adding classification-specific training logic.
    Computes binary cross-entropy loss and tracks accuracy and F1-score metrics.

    Parameters
    ----------
    encoder : ContrastiveEncoder
        Contrastive encoder model.
    lr : float | None
        Learning rate.
    loss : nn.Module | None
        Loss function. By default, BCEWithLogitsLoss with positive weight of 1.0.
    """

    def __init__(
        self,
        encoder: ContrastiveEncoder,
        lr: float | None,
        loss: nn.Module | None = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0)),
    ) -> None:
        super().__init__()
        self.stem = encoder.stem
        self.backbone = encoder.encoder
        self.backbone.head.fc = nn.Linear(768, 1)
        self.loss = loss
        self.lr = lr
        self.example_input_array = torch.rand(2, 1, 15, 160, 160)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stem and backbone for classification.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, depth, height, width).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch_size, 1) for binary classification.
        """
        x = self.stem(x)
        return self.backbone(x)

    def on_fit_start(self) -> None:
        """Initialize example storage lists at start of training.

        Creates empty lists to store training and validation examples for
        visualization logging during the training process.
        """
        self.train_examples = []
        self.val_examples = []

    def _fit_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, loss_on_step: bool
    ) -> tuple[torch.Tensor, np.ndarray]:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = binary_accuracy(y_hat, y)
        f1 = binary_f1_score(y_hat, y)
        self.log(f"loss/{stage}", loss, on_step=loss_on_step, on_epoch=True)
        self.log_dict(
            {f"metric/accuracy/{stage}": acc, f"metric/f1_score/{stage}": f1},
            on_step=False,
            on_epoch=True,
        )
        return loss, x[0, 0, x.shape[2] // 2].detach().cpu().numpy()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute single training step with loss computation and logging.

        Parameters
        ----------
        batch : tuple
            Training batch containing (inputs, targets).
        batch_idx : int
            Index of current batch within epoch.

        Returns
        -------
        torch.Tensor
            Training loss for backpropagation.
        """
        loss, example = self._fit_step(batch, "train", loss_on_step=True)
        if batch_idx < 4:
            self.train_examples.append([example])
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute single validation step with metrics computation.

        Parameters
        ----------
        batch : tuple
            Validation batch containing (inputs, targets).
        batch_idx : int
            Index of current batch within epoch.

        Returns
        -------
        torch.Tensor
            Validation loss for monitoring.
        """
        loss, example = self._fit_step(batch, "val", loss_on_step=False)
        if batch_idx < 4:
            self.val_examples.append([example])
        return loss

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Execute prediction step with sigmoid activation for probabilities.

        Parameters
        ----------
        batch : tuple
            Prediction batch containing (inputs, targets, indices).
        batch_idx : int
            Index of current batch.
        dataloader_idx : int or None, optional
            Index of dataloader when multiple dataloaders used.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing indices, labels, and sigmoid probabilities.
        """
        x, y, indices = batch
        y_hat = nn.functional.sigmoid(self(x))
        indices["label"] = y
        indices["prediction"] = y_hat
        return indices

    def _log_images(self, examples: list[list[np.ndarray]], stage: str) -> None:
        image = render_images(examples)
        self.logger.experiment.add_image(
            f"{stage}/examples",
            image,
            global_step=self.current_epoch,
            dataformats="HWC",
        )

    def on_train_epoch_end(self) -> None:
        """Log training examples and clear storage at epoch end.

        Renders and logs training examples to tensorboard, then clears the
        examples list for the next epoch.
        """
        self._log_images(self.train_examples, "train")
        self.train_examples.clear()

    def on_validation_epoch_end(self) -> None:
        """Log validation examples and clear storage at epoch end.

        Renders and logs validation examples to tensorboard, then clears the
        examples list for the next epoch.
        """
        self._log_images(self.val_examples, "val")
        self.val_examples.clear()

    def configure_optimizers(self) -> torch.optim.AdamW:
        """Configure AdamW optimizer for training.

        Returns
        -------
        torch.optim.AdamW
            AdamW optimizer with specified learning rate.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
