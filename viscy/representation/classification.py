from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from torch import nn
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score

from viscy.representation.contrastive import ContrastiveEncoder
from viscy.utils.log_images import render_images


class ClassificationPredictionWriter(BasePredictionWriter):
    def __init__(self, output_path: Path):
        super().__init__("epoch")
        if Path(output_path).exists():
            raise FileExistsError(f"Output path {output_path} already exists.")
        self.output_path = output_path

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        all_predictions = []
        for prediction in predictions:
            for key, value in prediction.items():
                if isinstance(value, torch.Tensor):
                    prediction[key] = value.detach().cpu().numpy().flatten()
            all_predictions.append(pd.DataFrame(prediction))
        pd.concat(all_predictions).to_csv(self.output_path, index=False)


class ClassificationModule(LightningModule):
    def __init__(
        self,
        encoder: ContrastiveEncoder,
        lr: float | None,
        loss: nn.Module | None = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0)),
        example_input_array_shape: tuple[int, ...] = (2, 1, 15, 160, 160),
    ):
        super().__init__()
        self.stem = encoder.stem
        self.backbone = encoder.encoder
        self.backbone.head.fc = nn.Linear(768, 1)
        self.loss = loss
        self.lr = lr
        self.example_input_array = torch.rand(example_input_array_shape)

    def forward(self, x):
        x = self.stem(x)
        return self.backbone(x)

    def on_fit_start(self):
        self.train_examples = []
        self.val_examples = []

    def _fit_step(self, batch, stage: str, loss_on_step: bool):
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

    def training_step(self, batch, batch_idx: int):
        loss, example = self._fit_step(batch, "train", loss_on_step=True)
        if batch_idx < 4:
            self.train_examples.append([example])
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, example = self._fit_step(batch, "val", loss_on_step=False)
        if batch_idx < 4:
            self.val_examples.append([example])
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int | None = None):
        x, y, indices = batch
        y_hat = nn.functional.sigmoid(self(x))
        indices["label"] = y
        indices["prediction"] = y_hat
        return indices

    def _log_images(self, examples, stage):
        image = render_images(examples)
        self.logger.experiment.add_image(
            f"{stage}/examples",
            image,
            global_step=self.current_epoch,
            dataformats="HWC",
        )

    def on_train_epoch_end(self):
        self._log_images(self.train_examples, "train")
        self.train_examples.clear()

    def on_validation_epoch_end(self):
        self._log_images(self.val_examples, "val")
        self.val_examples.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
