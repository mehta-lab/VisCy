import logging
from typing import Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from cellpose.models import CellposeModel
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized
from matplotlib.cm import get_cmap
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import DivisiblePad
from skimage.exposure import rescale_intensity
from torch.onnx import OperatorExportTypes
from torch.optim.lr_scheduler import ConstantLR
from torchmetrics.functional import (
    accuracy,
    cosine_similarity,
    dice,
    jaccard_index,
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
    r2_score,
    structural_similarity_index_measure,
)

from viscy.evaluation.evaluation_metrics import mean_average_precision
from viscy.light.data import Sample
from viscy.unet.networks.Unet25D import Unet25d
from viscy.unet.utils.model import ModelDefaults25D, define_model


class VSTrainer(Trainer):
    def export(
        self,
        model: LightningModule,
        export_path: str,
        ckpt_path: str,
        format="onnx",
        datamodule: LightningDataModule = None,
        dataloaders: Sequence = None,
    ):
        """Export the model for deployment (currently only ONNX is supported).

        :param LightningModule model: module to export
        :param str export_path: output file name
        :param str ckpt_path: model checkpoint
        :param str format: format (currently only ONNX is supported), defaults to "onnx"
        :param LightningDataModule datamodule: placeholder for datamodule,
            defaults to None
        :param Sequence dataloaders: placeholder for dataloaders, defaults to None
        """
        if dataloaders or datamodule:
            logging.debug("Ignoring datamodule and dataloaders during export.")
        if not format.lower() == "onnx":
            raise NotImplementedError(f"Export format '{format}'")
        model = _maybe_unwrap_optimized(model)
        self.strategy._lightning_module = model
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        model.eval()
        model.to_onnx(
            export_path,
            input_sample=model.example_input_array,
            export_params=True,
            opset_version=18,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {
                    0: "batch_size",
                    1: "channels",
                    3: "num_rows",
                    4: "num_cols",
                },
                "output": {
                    0: "batch_size",
                    1: "channels",
                    3: "num_rows",
                    4: "num_cols",
                },
            },
        )
        logging.info(f"ONNX exported at {export_path}")


class VSUNet(LightningModule):
    """Regression U-Net module for virtual staining.

    :param dict model_config: model config,
        defaults to :py:class:`viscy.unet.utils.model.ModelDefaults25D`
    :param int batch_size: batch size, defaults to 16
    :param Callable[[torch.Tensor, torch.Tensor], torch.Tensor] loss_function:
        loss function in training/validation, defaults to L2 (mean squared error)
    :param float lr: learning rate in training, defaults to 1e-3
    :param Literal['WarmupCosine', 'Constant'] schedule:
        learning rate scheduler, defaults to "Constant"
    :param int log_num_samples:
        number of image samples to log each training/validation epoch, defaults to 8
    :param Sequence[int] example_input_yx_shape:
        XY shape of the example input for network graph tracing, defaults to (256, 256)
    :param str test_cellpose_model_path:
        path to the CellPose model for testing segmentation, defaults to None
    """

    def __init__(
        self,
        model_config: dict = {},
        batch_size: int = 16,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_num_samples: int = 8,
        example_input_yx_shape: Sequence[int] = (256, 256),
        test_cellpose_model_path: str = None,
    ) -> None:
        super().__init__()
        self.model = define_model(Unet25d, ModelDefaults25D(), model_config)
        # TODO: handle num_outputs in metrics
        # self.out_channels = self.model.terminal_block.out_filters
        self.batch_size = batch_size
        self.loss_function = loss_function if loss_function else F.mse_loss
        self.lr = lr
        self.schedule = schedule
        self.log_num_samples = log_num_samples
        self.training_step_outputs = []
        self.validation_step_outputs = []
        # required to log the graph
        self.example_input_array = torch.rand(
            1,
            1,
            (model_config.get("in_stack_depth") or 5),
            *example_input_yx_shape,
        )
        self.test_cellpose_model_path = test_cellpose_model_path

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Sample, batch_idx: int):
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

    def validation_step(self, batch: Sample, batch_idx: int):
        source = batch["source"]
        target = batch["target"]
        pred = self.forward(source)
        loss = self.loss_function(pred, target)
        self.log("loss/validate", loss, batch_size=self.batch_size, sync_dist=True)
        if batch_idx < self.log_num_samples:
            self.validation_step_outputs.append(
                self._detach_sample((source, target, pred))
            )

    def test_step(self, batch: Sample, batch_idx: int):
        source = batch["source"]
        target = batch["target"][:, 0]
        pred = self.forward(source)[:, 0]
        # FIXME: Only works for batch size 1 and the first channel
        self._log_regression_metrics(pred, target)
        if "mask" in batch:
            self._log_segmentation_metrics(batch, pred)
        img_names, ts, zs = batch["index"]
        self.log_dict(
            {
                "position": int(img_names[0].split("/")[-2]),
                "time": ts[0],
                "slice": zs[0],
            },
            on_step=True,
            on_epoch=False,
        )

    def _log_regression_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        # paired image translation metrics
        self.log_dict(
            {
                # regression
                "test_metrics/MAE": mean_absolute_error(pred, target),
                "test_metrics/MSE": mean_squared_error(pred, target),
                "test_metrics/cosine": cosine_similarity(
                    pred, target, reduction="mean"
                ),
                "test_metrics/pearson": pearson_corrcoef(
                    pred.flatten(), target.flatten()
                ),
                "test_metrics/r2": r2_score(pred.flatten(), target.flatten()),
                # image perception
                "test_metrics/SSIM": structural_similarity_index_measure(
                    pred, target, gaussian_kernel=False, kernel_size=21
                ),
            },
            on_step=True,
            on_epoch=True,
        )

    def _log_segmentation_metrics(self, batch: Sample, pred: torch.Tensor):
        pred_mask = torch.from_numpy(
            self.cellpose_model.eval(
                pred.cpu().numpy(), channels=[0, 0], diameter=None
            )[0].astype(np.int16)[np.newaxis]
        ).to(self.device)
        target_mask = batch["mask"]
        self.log_dict(
            {
                # semantic segmentation
                "test_metrics/accuracy": accuracy(pred_mask, target_mask),
                "test_metrics/dice": dice(pred_mask, target_mask),
                "test_metrics/jaccard": jaccard_index(
                    pred_mask, target_mask, task="binary"
                ),
                "test_metrics/mAP": mean_average_precision(pred_mask, target_mask),
            },
            on_step=True,
            on_epoch=True,
        )

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        source = self._predict_pad(batch["source"])
        return self._predict_pad.inverse(self.forward(source))

    def on_train_epoch_end(self):
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        self._log_samples("val_samples", self.validation_step_outputs)
        self.validation_step_outputs = []

    def on_test_start(self):
        """Load CellPose model for segmentation."""
        if self.test_cellpose_model_path is not None:
            self.cellpose_model = CellposeModel(
                model_type=self.test_cellpose_model_path
            )

    def on_predict_start(self):
        """Pad the input shape to be divisible by the downsampling factor.
        The inverse of this transform crops the prediction to original shape.
        """
        down_factor = 2**self.model.num_blocks
        self._predict_pad = DivisiblePad((0, 0, down_factor, down_factor))

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
        return [np.squeeze(img[0].detach().cpu().numpy().max(axis=(1))) for img in imgs]

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        images_grid = []
        for sample_images in imgs:
            images_row = []
            for i, image in enumerate(sample_images):
                cm_name = "gray" if i == 0 else "inferno"
                if image.ndim == 2:
                    image = image[np.newaxis]
                for channel in image:
                    channel = rescale_intensity(channel, out_range=(0, 1))
                    render = get_cmap(cm_name)(channel, bytes=True)[..., :3]
                    images_row.append(render)
            images_grid.append(np.concatenate(images_row, axis=1))
        grid = np.concatenate(images_grid, axis=0)
        self.logger.experiment.add_image(
            key, grid, self.current_epoch, dataformats="HWC"
        )
