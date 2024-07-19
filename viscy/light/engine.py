import logging
import os
from typing import Literal, Sequence, Union

import numpy as np
import torch
from imageio import imwrite
from lightning.pytorch import LightningModule
from matplotlib.pyplot import get_cmap
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import DivisiblePad, Rotate90
from skimage.exposure import rescale_intensity
from torch import Tensor, nn
from torch.nn import functional as F
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

from viscy.data.hcs import Sample
from viscy.evaluation.evaluation_metrics import mean_average_precision, ms_ssim_25d
from viscy.unet.networks.fcmae import FullyConvolutionalMAE
from viscy.unet.networks.Unet2D import Unet2d
from viscy.unet.networks.Unet25D import Unet25d
from viscy.unet.networks.unext2 import UNeXt2

try:
    from cellpose.models import CellposeModel
except ImportError:
    CellposeModel = None


_UNET_ARCHITECTURE = {
    "2D": Unet2d,
    "UNeXt2": UNeXt2,
    "2.5D": Unet25d,
    "fcmae": FullyConvolutionalMAE,
    "UNeXt2_2D": FullyConvolutionalMAE,
}


class MixedLoss(nn.Module):
    """Mixed reconstruction loss.
    Adapted from Zhao et al, https://arxiv.org/pdf/1511.08861.pdf
    Reduces to simple distances if only one weight is non-zero.

    :param float l1_alpha: L1 loss weight, defaults to 0.5
    :param float l2_alpha: L2 loss weight, defaults to 0.0
    :param float ms_dssim_alpha: MS-DSSIM weight, defaults to 0.5
    """

    def __init__(
        self, l1_alpha: float = 0.5, l2_alpha: float = 0.0, ms_dssim_alpha: float = 0.5
    ):
        super().__init__()
        if not any([l1_alpha, l2_alpha, ms_dssim_alpha]):
            raise ValueError("Loss term weights cannot be all zero!")
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.ms_dssim_alpha = ms_dssim_alpha

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, preds, target):
        loss = 0
        if self.l1_alpha:
            # the gaussian in the reference is not used
            # because the SSIM here uses a uniform window
            loss += F.l1_loss(preds, target) * self.l1_alpha
        if self.l2_alpha:
            loss += F.mse_loss(preds, target) * self.l2_alpha
        if self.ms_dssim_alpha:
            ms_ssim = ms_ssim_25d(preds, target, clamp=True)
            # the 1/2 factor in the original DSSIM is not used
            # since the MS-SSIM here is stabilized with ReLU
            loss += (1 - ms_ssim) * self.ms_dssim_alpha
        return loss


class VSUNet(LightningModule):
    """Regression U-Net module for virtual staining.

    :param dict model_config: model config,
        defaults to :py:class:`viscy.unet.utils.model.ModelDefaults25D`
    :param Union[nn.Module, MixedLoss] loss_function:
        loss function in training/validation,
        if a dictionary, should specify weights of each term
        ('l1_alpha', 'l2_alpha', 'ssim_alpha')
        defaults to L2 (mean squared error)
    :param float lr: learning rate in training, defaults to 1e-3
    :param Literal['WarmupCosine', 'Constant'] schedule:
        learning rate scheduler, defaults to "Constant"
    :param str chkpt_path: path to the checkpoint to load weights, defaults to None
    :param int log_batches_per_epoch:
        number of batches to log each training/validation epoch,
        has to be smaller than steps per epoch, defaults to 8
    :param int log_samples_per_batch:
        number of samples to log each training/validation batch,
        has to be smaller than batch size, defaults to 1
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
    :param bool test_time_augmentations:
        apply test time augmentations in test stage, defaults to False
    :param Literal['mean', 'median', 'product'] tta_type:
        type of test time augmentations aggregation, defaults to "mean"
    """

    def __init__(
        self,
        architecture: Literal["2D", "UNeXt2", "2.5D", "3D", "fcmae", "UNeXt2_2D"],
        model_config: dict = {},
        loss_function: Union[nn.Module, MixedLoss] = None,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        freeze_encoder: bool = False,
        ckpt_path: str = None,
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        example_input_yx_shape: Sequence[int] = (256, 256),
        test_cellpose_model_path: str = None,
        test_cellpose_diameter: float = None,
        test_evaluate_cellpose: bool = False,
        test_time_augmentations: bool = False,
        tta_type: Literal["mean", "median", "product"] = "mean",
    ) -> None:
        super().__init__()
        net_class = _UNET_ARCHITECTURE.get(architecture)
        if not net_class:
            raise ValueError(
                f"Architecture {architecture} not in {_UNET_ARCHITECTURE.keys()}"
            )
        self.model = net_class(**model_config)
        # TODO: handle num_outputs in metrics
        # self.out_channels = self.model.terminal_block.out_filters
        self.loss_function = loss_function if loss_function else nn.MSELoss()
        self.lr = lr
        self.schedule = schedule
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.training_step_outputs = []
        self.validation_losses = []
        self.validation_step_outputs = []
        # required to log the graph
        if architecture == "2D":
            example_depth = 1
        else:
            example_depth = model_config.get("in_stack_depth") or 5
        self.example_input_array = torch.rand(
            1,
            model_config.get("in_channels") or 1,
            example_depth,
            *example_input_yx_shape,
        )
        self.test_cellpose_model_path = test_cellpose_model_path
        self.test_cellpose_diameter = test_cellpose_diameter
        self.test_evaluate_cellpose = test_evaluate_cellpose
        self.test_time_augmentations = test_time_augmentations
        self.tta_type = tta_type
        self.freeze_encoder = freeze_encoder
        self._original_shape_yx = None
        if ckpt_path is not None:
            self.load_state_dict(
                torch.load(ckpt_path)["state_dict"]
            )  # loading only weights

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Sample | Sequence[Sample], batch_idx: int):
        losses = []
        batch_size = 0
        if not isinstance(batch, Sequence):
            batch = [batch]
        for b in batch:
            source = b["source"]
            target = b["target"]
            pred = self.forward(source)
            loss = self.loss_function(pred, target)
            losses.append(loss)
            batch_size += source.shape[0]
            if batch_idx < self.log_batches_per_epoch:
                self.training_step_outputs.extend(
                    self._detach_sample((source, target, pred))
                )
        loss_step = torch.stack(losses).mean()
        self.log(
            "loss/train",
            loss_step.to(self.device),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss_step

    def validation_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        source: Tensor = batch["source"]
        target: Tensor = batch["target"]
        pred = self.forward(source)
        loss = self.loss_function(pred, target)
        if dataloader_idx + 1 > len(self.validation_losses):
            self.validation_losses.append([])
        self.validation_losses[dataloader_idx].append(loss.detach())
        self.log(
            f"loss/val/{dataloader_idx}",
            loss.to(self.device),
            sync_dist=True,
            batch_size=source.shape[0],
        )
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(
                self._detach_sample((source, target, pred))
            )

    def test_step(self, batch: Sample, batch_idx: int):
        source = batch["source"]
        target = batch["target"]
        center_index = target.shape[-3] // 2
        center_slice = slice(center_index, center_index + 1)
        target = target[:, 0, center_slice]
        if self.test_evaluate_cellpose:
            pred = target
        else:
            pred = self.forward(source)[:, 0, center_slice]
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

    def _log_regression_metrics(self, pred: Tensor, target: Tensor):
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

    def _cellpose_predict(self, pred: Tensor, name: str) -> torch.ShortTensor:
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
                "test_metrics/accuracy": (
                    accuracy(pred_binary, target_binary, task="binary")
                    if compute
                    else -1
                ),
                "test_metrics/dice": (
                    dice(pred_binary, target_binary) if compute else -1
                ),
                "test_metrics/jaccard": (
                    jaccard_index(pred_binary, target_binary, task="binary")
                    if compute
                    else -1
                ),
                "test_metrics/mAP": coco_metrics["map"] if compute else -1,
                "test_metrics/mAP_50": coco_metrics["map_50"] if compute else -1,
                "test_metrics/mAP_75": coco_metrics["map_75"] if compute else -1,
                "test_metrics/mAR_100": coco_metrics["mar_100"] if compute else -1,
            },
            on_step=True,
            on_epoch=False,
        )

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        source = batch["source"]
        if self.test_time_augmentations:
            prediction = self.perform_test_time_augmentations(source)
        else:
            source = self._predict_pad(source)
            prediction = self.forward(source)
            prediction = self._predict_pad.inverse(prediction)

        return prediction

    def perform_test_time_augmentations(self, source: Tensor) -> Tensor:
        """Perform test time augmentations on the input source
        and aggregate the predictions using the specified method.

        :param source: input tensor
        :return: aggregated prediction
        """

        # Save the yx coords to crop post rotations
        self._original_shape_yx = source.shape[-2:]
        predictions = []
        for i in range(4):
            augmented = self._rotate_volume(source, k=i, spatial_axes=(1, 2))
            augmented = self._predict_pad(augmented)
            augmented_prediction = self.forward(augmented)
            de_augmented_prediction = self._predict_pad.inverse(augmented_prediction)
            de_augmented_prediction = self._rotate_volume(
                de_augmented_prediction, k=4 - i, spatial_axes=(1, 2)
            )
            de_augmented_prediction = self._crop_to_original(de_augmented_prediction)

            # Undo rotation and padding
            predictions.append(de_augmented_prediction)

        if self.tta_type == "mean":
            prediction = torch.stack(predictions).mean(dim=0)
        elif self.tta_type == "median":
            prediction = torch.stack(predictions).median(dim=0).values
        elif self.tta_type == "product":
            # Perform multiplication of predictions in logarithmic space for numerical stability adding epsion to avoid log(0) case
            log_predictions = torch.stack([torch.log(p + 1e-9) for p in predictions])
            log_prediction_sum = log_predictions.sum(dim=0)
            prediction = torch.exp(log_prediction_sum)
        return prediction

    def on_train_epoch_end(self):
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)
        self.validation_step_outputs = []
        # average within each dataloader
        loss_means = [torch.tensor(losses).mean() for losses in self.validation_losses]
        self.log(
            "loss/validate",
            torch.tensor(loss_means).mean().to(self.device),
            sync_dist=True,
        )

    def on_test_start(self):
        """Load CellPose model for segmentation."""
        if CellposeModel is None:
            # raise ImportError(
            #     "CellPose not installed. "
            #     "Please install the metrics dependency with "
            #     '`pip install viscy".[metrics]"`'
            # )
            logging.warning(
                "CellPose not installed. "
                "Please install the metrics dependency with "
                '`pip install viscy"[metrics]"`'
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
        if self.freeze_encoder:
            self.model: FullyConvolutionalMAE
            self.model.encoder.requires_grad_(False)
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

    def _detach_sample(self, imgs: Sequence[Tensor]):
        num_samples = min(imgs[0].shape[0], self.log_samples_per_batch)
        return [
            [np.squeeze(img[i].detach().cpu().numpy().max(axis=1)) for img in imgs]
            for i in range(num_samples)
        ]

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

    def _rotate_volume(self, tensor: Tensor, k: int, spatial_axes: tuple) -> Tensor:
        # Padding to ensure square shape
        max_dim = max(tensor.shape[-2], tensor.shape[-1])
        pad_transform = DivisiblePad((0, 0, max_dim, max_dim))
        padded_tensor = pad_transform(tensor)

        # Rotation
        rotated_tensor = []
        rotate = Rotate90(k=k, spatial_axes=spatial_axes)
        for b in range(padded_tensor.shape[0]):  # iterate over batch
            rotated_tensor.append(rotate(padded_tensor[b]))

        # Stack the list of tensors back into a single tensor
        rotated_tensor = torch.stack(rotated_tensor)
        del padded_tensor
        # # Cropping to original shape
        return rotated_tensor

    def _crop_to_original(self, tensor: Tensor) -> Tensor:
        original_y, original_x = self._original_shape_yx
        pad_y = (tensor.shape[-2] - original_y) // 2
        pad_x = (tensor.shape[-1] - original_x) // 2
        cropped_tensor = tensor[
            ..., pad_y : pad_y + original_y, pad_x : pad_x + original_x
        ]
        return cropped_tensor


class FcmaeUNet(VSUNet):
    def __init__(self, fit_mask_ratio: float = 0.0, **kwargs):
        super().__init__(architecture="fcmae", **kwargs)
        self.fit_mask_ratio = fit_mask_ratio

    def forward(self, x: Tensor, mask_ratio: float = 0.0):
        return self.model(x, mask_ratio)

    def forward_fit(self, batch: Sample) -> tuple[Tensor]:
        source = batch["source"]
        target = batch["target"]
        pred, mask = self.forward(source, mask_ratio=self.fit_mask_ratio)
        loss = F.mse_loss(pred, target, reduction="none")
        loss = (loss.mean(2) * mask).sum() / mask.sum()
        return source, target, pred, mask, loss

    def training_step(self, batch: Sequence[Sample], batch_idx: int):
        losses = []
        batch_size = 0
        for b in batch:
            source, target, pred, mask, loss = self.forward_fit(b)
            losses.append(loss)
            batch_size += source.shape[0]
            if batch_idx < self.log_batches_per_epoch:
                self.training_step_outputs.extend(
                    self._detach_sample((source, target * mask.unsqueeze(2), pred))
                )
        loss_step = torch.stack(losses).mean()
        self.log(
            "loss/train",
            loss_step.to(self.device),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss_step

    def validation_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        source, target, pred, mask, loss = self.forward_fit(batch)
        if dataloader_idx + 1 > len(self.validation_losses):
            self.validation_losses.append([])
        self.validation_losses[dataloader_idx].append(loss.detach())
        self.log(
            f"loss/val/{dataloader_idx}",
            loss.to(self.device),
            sync_dist=True,
            batch_size=source.shape[0],
        )
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(
                self._detach_sample((source, target * mask.unsqueeze(2), pred))
            )
