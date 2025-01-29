import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from tarrow.models import TimeArrowNet
from tarrow.models.losses import DecorrelationLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau


class TarrowModule(LightningModule):
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
        **kwargs,
    ):
        """Lightning Module wrapper for TimeArrowNet.

        Args:
            backbone: Dense network architecture
            projection_head: Dense projection head architecture
            classification_head: Classification head architecture
            n_frames: Number of input frames
            n_features: Number of output features from the backbone
            n_input_channels: Number of input channels
            symmetric: If True, use permutation-equivariant classification head
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            lambda_decorrelation: Prefactor of decorrelation loss
            lr_scheduler: Learning rate scheduler ('plateau' or 'cyclic')
            lr_patience: Patience for learning rate scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = TimeArrowNet(
            backbone=backbone,
            projection_head=projection_head,
            classification_head=classification_head,
            n_frames=n_frames,
            n_features=n_features,
            n_input_channels=n_input_channels,
            symmetric=symmetric,
            device="cpu",  # Let Lightning handle device placement
        )

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.criterion_decorr = DecorrelationLoss()

    def forward(self, x):
        return self.model(x, mode="both")

    def _shared_step(self, batch, batch_idx, phase="train"):
        x, y = batch
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

        self.log(f"{phase}_loss", loss, prog_bar=True)
        self.log(f"{phase}_loss_decorr", loss_decorr, prog_bar=True)
        self.log(f"{phase}_accuracy", acc, prog_bar=True)
        self.log(f"{phase}_pred1_ratio", pred.sum().float() / len(pred))

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
                    "monitor": "val_loss",
                },
            }
        elif self.hparams.lr_scheduler == "cyclic":
            scheduler = CyclicLR(
                optimizer,
                base_lr=self.hparams.learning_rate,
                max_lr=self.hparams.learning_rate * 10,
                cycle_momentum=False,
                step_size_up=self.trainer.estimated_stepping_batches,
                scale_mode="cycle",
                scale_fn=lambda x: 0.9**x,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
