import torch
import torch.nn as nn
import torchvision
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback
from tarrow.models import TimeArrowNet
from tarrow.models.losses import DecorrelationLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.utils.data import DataLoader


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
        **kwargs,
    ):
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
        )

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.criterion_decorr = DecorrelationLoss()

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

    def embedding(self, x):
        """Get dense embeddings from the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_frames, channels, height, width)

        Returns
        -------
        torch.Tensor
            Dense embeddings from the backbone network
        """
        return self.model.embedding(x)


class TarrowVisualizationCallback(Callback):
    """Callback for visualizing cells and embeddings in TensorBoard.

    Parameters
    ----------
    dataset : Dataset
        Dataset to visualize
    max_samples : int, default=100
        Maximum number of samples to visualize
    log_every_n_epochs : int, default=3
        How often to log visualizations
    cam_size : tuple or int, optional
        Size for class activation maps. If None, use original size
    """

    def __init__(self, dataset, max_samples=100, log_every_n_epochs=3, cam_size=None):
        """
        Parameters
        ----------
        dataset : Dataset
            Dataset to visualize
        max_samples : int, default=100
            Maximum number of samples to visualize
        log_every_n_epochs : int, default=3
            How often to log visualizations
        cam_size : tuple or int, optional
            Size for class activation maps. If None, use original size
        """
        super().__init__()
        self.dataset = dataset
        self.max_samples = max_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.cam_size = cam_size

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            # Get samples from dataset
            loader = DataLoader(
                self.dataset,
                batch_size=min(32, self.max_samples),
                shuffle=True,
            )
            batch = next(iter(loader))
            images, labels = batch
            images = images.to(pl_module.device)

            # Get embeddings
            with torch.no_grad():
                embeddings = pl_module.embedding(images)
                out, _ = pl_module(images)
                preds = torch.argmax(out, dim=1)

            # Log images
            grid = torchvision.utils.make_grid(
                images[:, 0],  # First timepoint
                nrow=8,
                normalize=True,
                value_range=(images.min(), images.max()),
            )
            trainer.logger.experiment.add_image(
                "cells/timepoint1",
                grid,
                trainer.current_epoch,
            )

            grid = torchvision.utils.make_grid(
                images[:, 1],  # Second timepoint
                nrow=8,
                normalize=True,
                value_range=(images.min(), images.max()),
            )
            trainer.logger.experiment.add_image(
                "cells/timepoint2",
                grid,
                trainer.current_epoch,
            )

            # Log embeddings
            trainer.logger.experiment.add_embedding(
                embeddings.reshape(len(embeddings), -1),
                metadata=[
                    f"label={l.item()}, pred={p.item()}" for l, p in zip(labels, preds)
                ],
                label_img=images[:, 0],  # Use first timepoint as label image
                global_step=trainer.current_epoch,
            )

            # Log CAMs if cam_size is provided
            if self.cam_size is not None and hasattr(pl_module.model, "get_cam"):
                cam = pl_module.model.get_cam(images, size=self.cam_size)
                grid = torchvision.utils.make_grid(
                    cam.unsqueeze(1),  # Add channel dimension
                    nrow=8,
                    normalize=True,
                )
                trainer.logger.experiment.add_image(
                    "cells/cam",
                    grid,
                    trainer.current_epoch,
                )
