import logging
import os
from typing import Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from imageio import imwrite
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
from viscy.unet.networks.Unet2D import Unet2d
from viscy.unet.networks.Unet21D import Unet21d
from viscy.unet.networks.Unet25D import Unet25d

try:
    from cellpose.models import CellposeModel
except ImportError:
    CellposeModel = None


_UNET_ARCHITECTURE = {
    "2D": Unet2d,
    "2.1D": Unet21d,
    "2.5D": Unet25d,
}


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
    :param float test_cellpose_diameter:
        diameter parameter of the CellPose model for testing segmentation,
        defaults to None
    :param bool test_evaluate_cellpose:
        evaluate the performance of the CellPose model instead of the trained model
        in test stage, defaults to False
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
        test_cellpose_diameter: float = None,
        test_evaluate_cellpose: bool = False,
    ) -> None:
        super().__init__()
        arch = model_config.pop("architecture")
        net_class = _UNET_ARCHITECTURE.get(arch)
        if not arch:
            raise ValueError(f"Architecture {arch} not in {_UNET_ARCHITECTURE.keys()}")
        self.model = net_class(**model_config)
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
        if arch == "2D":
            example_depth = 1
        else:
            example_depth = model_config.get("in_stack_depth") or 5
        self.example_input_array = torch.rand(
            1,
            1,
            example_depth,
            *example_input_yx_shape,
        )
        self.test_cellpose_model_path = test_cellpose_model_path
        self.test_cellpose_diameter = test_cellpose_diameter
        self.test_evaluate_cellpose = test_evaluate_cellpose

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
        if self.test_evaluate_cellpose:
            pred = target
        else:
            pred = self.forward(source)[:, 0]
            # FIXME: Only works for batch size 1 and the first channel
            self._log_regression_metrics(pred, target)
        img_names, ts, zs = batch["index"]
        position = float(img_names[0].split("/")[-2])
        self.log_dict(
            {
                "position": position,
                "time": float(ts[0]),
                "slice": float(zs[0]),
            },
            on_step=True,
            on_epoch=False,
        )
        if "labels" in batch:
            pred_labels = self._cellpose_predict(
                pred, f"p{int(position)}_t{ts[0]}_z{zs[0]}"
            )
            self._log_segmentation_metrics(pred_labels, batch["labels"][0])
        else:
            self._log_segmentation_metrics(None, None)

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
                    pred.flatten() * 1e4, target.flatten() * 1e4
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

    def _cellpose_predict(self, pred: torch.Tensor, name: str) -> torch.ShortTensor:
        pred_labels_np = self.cellpose_model.eval(
            pred.cpu().numpy(), channels=[0, 0], diameter=self.test_cellpose_diameter
        )[0].astype(np.int16)
        imwrite(os.path.join(self.logger.log_dir, f"{name}.png"), pred_labels_np)
        return torch.from_numpy(pred_labels_np).to(self.device)

    def _log_segmentation_metrics(
        self, pred_labels: torch.ShortTensor, target_labels: torch.ShortTensor
    ):
        compute = pred_labels is not None
        if compute:
            pred_binary = pred_labels > 0
            target_binary = target_labels > 0
            coco_metrics = mean_average_precision(pred_labels, target_labels)
            logging.debug(coco_metrics)
        self.log_dict(
            {
                # semantic segmentation
                "test_metrics/accuracy": accuracy(
                    pred_binary, target_binary, task="binary"
                )
                if compute
                else -1,
                "test_metrics/dice": dice(pred_binary, target_binary)
                if compute
                else -1,
                "test_metrics/jaccard": jaccard_index(
                    pred_binary, target_binary, task="binary"
                )
                if compute
                else -1,
                "test_metrics/mAP": coco_metrics["map"] if compute else -1,
                "test_metrics/mAP_50": coco_metrics["map_50"] if compute else -1,
                "test_metrics/mAP_75": coco_metrics["map_75"] if compute else -1,
                "test_metrics/mAR_100": coco_metrics["mar_100"] if compute else -1,
            },
            on_step=True,
            on_epoch=False,
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
        if CellposeModel is None:
            raise ImportError(
                "CellPose not installed. "
                "Please install the metrics dependency with "
                '`pip install viscy".[metrics]"`'
            )
        if self.test_cellpose_model_path is not None:
            self.cellpose_model = CellposeModel(
                model_type=self.test_cellpose_model_path, device=self.device
            )

    def on_predict_start(self):
        """Pad the input shape to be divisible by the downsampling factor.
        The inverse of this transform crops the prediction to original shape.
        """
        down_factor = 2**self.model.num_blocks
        self._predict_pad = DivisiblePad((0, 0, down_factor, down_factor))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
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
