from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from matplotlib.cm import get_cmap
from monai.optimizers import WarmupCosineSchedule
from skimage.exposure import rescale_intensity
from torch.optim.lr_scheduler import ConstantLR

from micro_dl.torch_unet.networks.Unet25D import Unet25d
from micro_dl.torch_unet.utils.model import ModelDefaults25D, define_model


class PhaseToNuc25D(LightningModule):
    def __init__(
        self,
        model_config: dict = {},
        batch_size: int = 16,
        loss_function: nn.Module = None,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_num_samples: int = 8,
        example_input_yx_shape: Sequence[int] = (256, 256),
    ) -> None:
        """Regression U-Net module for virtual staining.

        Parameters
        ----------
        model : nn.Module
            U-Net model
        max_epochs : int, optional
            Max epochs in fitting, by default 100
        loss_function : nn.Module, optional
            Loss function module, by default L2 (mean squared error)
        lr : float, optional
            Learning rate, by default 1e-3
        schedule: Literal["WarmupCosine", "Constant"], optional
            Learning rate scheduler, by default 'Constant'
        """
        super().__init__()
        self.model = define_model(Unet25d, ModelDefaults25D(), model_config)
        self.batch_size = batch_size
        self.loss_function = loss_function if loss_function else F.mse_loss
        self.lr = lr
        self.schedule = schedule
        self.log_num_samples = log_num_samples
        self.training_step_outputs = []
        self.validation_step_outputs = []
        # required to log the graph
        self.example_input_array = torch.rand(
            1, 1, (model_config.get("in_stack_depth") or 5), *example_input_yx_shape
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        source = batch["source"]
        target = batch["target"]
        pred = self.forward(source)
        loss = self.loss_function(pred, target)
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        if batch_idx < self.log_num_samples:
            self.training_step_outputs.append(
                self._detach_sample((source, target, pred))
            )
        return loss

    def validation_step(self, batch, batch_idx):
        source = batch["source"]
        target = batch["target"]
        pred = self.forward(source)
        loss = self.loss_function(pred, target)
        self.log("val_loss", loss, batch_size=self.batch_size, sync_dist=True)
        if batch_idx < self.log_num_samples:
            self.validation_step_outputs.append(
                self._detach_sample((source, target, pred))
            )

    def on_train_epoch_end(self):
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        self._log_samples("val_samples", self.validation_step_outputs)
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.schedule == "WarmupCosine":
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=3,
                t_total=self.trainer.max_epochs,
                warmup_multiplier=1e-3,
            )
        elif self.schedule == "Constant":
            scheduler = ConstantLR(
                optimizer, factor=1, total_iters=self.trainer.max_epochs
            )
        return [optimizer], [scheduler]

    @staticmethod
    def _detach_sample(imgs: Sequence[torch.Tensor]):
        return [
            np.squeeze(img[0].detach().cpu().numpy().max(axis=(0, 1))) for img in imgs
        ]

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        images_grid = []
        for sample_images in imgs:
            images_row = []
            for im, cm_name in zip(sample_images, ["gray"] + ["inferno"] * 2):
                im = rescale_intensity(im, out_range=(0, 1))
                rendered_im = get_cmap(cm_name)(im, bytes=True)[..., :3]
                images_row.append(rendered_im)
            images_grid.append(np.concatenate(images_row, axis=1))
        grid = np.concatenate(images_grid, axis=0)
        self.logger.experiment.add_image(
            key, grid, self.current_epoch, dataformats="HWC"
        )
