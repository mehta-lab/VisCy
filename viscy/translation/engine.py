import logging
import os
import random
from collections.abc import Callable, Sequence
from typing import Any, Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from imageio import imwrite
from lightning.pytorch import LightningModule
from monai.data.utils import collate_meta_tensor
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import DivisiblePad, Rotate90
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR
from torchmetrics.functional import (
    accuracy,
    cosine_similarity,
    jaccard_index,
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
    r2_score,
    structural_similarity_index_measure,
)
from torchmetrics.functional.segmentation import dice_score
from viscy.data.combined import CombinedDataModule
from viscy.data.gpu_aug import GPUTransformDataModule
from viscy.data.typing import Sample
from viscy.translation.evaluation_metrics import mean_average_precision, ms_ssim_25d
from viscy.unet.networks.fcmae import FullyConvolutionalMAE
from viscy.unet.networks.Unet2D import Unet2d
from viscy.unet.networks.Unet25D import Unet25d
from viscy.unet.networks.unext2 import UNeXt2
from viscy.utils.log_images import detach_sample, render_images

_UNET_ARCHITECTURE = {
    "2D": Unet2d,
    "UNeXt2": UNeXt2,
    "2.5D": Unet25d,
    "fcmae": FullyConvolutionalMAE,
    "UNeXt2_2D": FullyConvolutionalMAE,
}

_logger = logging.getLogger("lightning.pytorch")


class MixedLoss(nn.Module):
    """Mixed reconstruction loss.

    Adapted from Zhao et al, https://arxiv.org/pdf/1511.08861.pdf
    Reduces to simple distances if only one weight is non-zero.

    Parameters
    ----------
    l1_alpha : float, optional
        L1 loss weight, by default 0.5
    l2_alpha : float, optional
        L2 loss weight, by default 0.0
    ms_dssim_alpha : float, optional
        MS-DSSIM weight, by default 0.5
    """

    def __init__(
        self, l1_alpha: float = 0.5, l2_alpha: float = 0.0, ms_dssim_alpha: float = 0.5
    ) -> None:
        super().__init__()
        if not any([l1_alpha, l2_alpha, ms_dssim_alpha]):
            raise ValueError("Loss term weights cannot be all zero!")
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.ms_dssim_alpha = ms_dssim_alpha

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mixed reconstruction loss.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted tensor
        target : torch.Tensor
            Target tensor

        Returns
        -------
        torch.Tensor
            Combined loss value
        """
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


class MaskedMSELoss(nn.Module):
    """Masked mean squared error loss.

    Computes MSE loss only for masked regions.
    """

    def forward(
        self, preds: torch.Tensor, original: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked MSE loss.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted tensor.
        original : torch.Tensor
            Original tensor.
        mask : torch.Tensor
            Binary mask tensor.

        Returns
        -------
        torch.Tensor
            Masked MSE loss value.
        """
        loss = F.mse_loss(preds, original, reduction="none")
        loss = (loss.mean(2) * mask).sum() / mask.sum()
        return loss


class VSUNet(LightningModule):
    """Regression U-Net module for virtual staining.

    Parameters
    ----------
    architecture : Literal["2D", "UNeXt2", "2.5D", "3D", "fcmae", "UNeXt2_2D"]
        Model architecture type.
    model_config : dict, optional
        Model config, defaults to :py:class:`viscy.unet.utils.model.ModelDefaults25D`,
        by default {}.
    loss_function : Union[nn.Module, MixedLoss], optional
        Loss function in training/validation. If a dictionary, should specify weights
        of each term ('l1_alpha', 'l2_alpha', 'ssim_alpha'), defaults to L2
        (mean squared error), by default None.
    lr : float, optional
        Learning rate in training, by default 1e-3.
    schedule : Literal['WarmupCosine', 'Constant'], optional
        Learning rate scheduler, by default "Constant".
    freeze_encoder : bool, optional
        Whether to freeze encoder weights, by default False.
    ckpt_path : str, optional
        Path to the checkpoint to load weights, by default None.
    log_batches_per_epoch : int, optional
        Number of batches to log each training/validation epoch,
        has to be smaller than steps per epoch, by default 8.
    log_samples_per_batch : int, optional
        Number of samples to log each training/validation batch,
        has to be smaller than batch size, by default 1.
    example_input_yx_shape : Sequence[int], optional
        XY shape of the example input for network graph tracing, by default (256, 256).
    test_cellpose_model_path : str, optional
        Path to the CellPose model for testing segmentation, by default None.
    test_cellpose_diameter : float, optional
        Diameter parameter of the CellPose model for testing segmentation,
        by default None.
    test_evaluate_cellpose : bool, optional
        Evaluate the performance of the CellPose model instead of the trained model
        in test stage, by default False.
    test_time_augmentations : bool, optional
        Apply test time augmentations in test stage, by default False.
    tta_type : Literal['mean', 'median', 'product'], optional
        Type of test time augmentations aggregation, by default "mean".
    """

    def __init__(
        self,
        architecture: Literal["2D", "UNeXt2", "2.5D", "3D", "fcmae", "UNeXt2_2D"],
        model_config: dict = {},
        loss_function: nn.Module | MixedLoss | None = None,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        freeze_encoder: bool = False,
        ckpt_path: str | None = None,
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        example_input_yx_shape: Sequence[int] = (256, 256),
        test_cellpose_model_path: str | None = None,
        test_cellpose_diameter: float | None = None,
        test_evaluate_cellpose: bool | None = False,
        test_time_augmentations: bool | None = False,
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
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        return self.model(x)

    def training_step(
        self, batch: Sample | Sequence[Sample], batch_idx: int
    ) -> torch.Tensor:
        """Execute single training step.

        Parameters
        ----------
        batch : Sample or Sequence[Sample]
            Training batch data.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
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
                    detach_sample((source, target, pred), self.log_samples_per_batch)
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

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Execute single validation step.

        Parameters
        ----------
        batch : Sample
            Validation batch data.
        batch_idx : int
            Batch index.
        dataloader_idx : int, default=0
            Dataloader index for multi-dataloader validation.
        """
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
                detach_sample((source, target, pred), self.log_samples_per_batch)
            )

    def test_step(self, batch: Sample, batch_idx: int) -> None:
        """Execute single test step.

        Parameters
        ----------
        batch : Sample
            Test batch data.
        batch_idx : int
            Batch index.
        """
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

    def _log_regression_metrics(self, pred: Tensor, target: Tensor) -> None:
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
    ) -> None:
        compute = pred_labels is not None
        if compute:
            pred_binary = pred_labels > 0
            target_binary = target_labels > 0
            coco_metrics = mean_average_precision(pred_labels, target_labels)
            _logger.debug(coco_metrics)
        self.log_dict(
            {
                # semantic segmentation
                "test_metrics/accuracy": (
                    accuracy(pred_binary, target_binary, task="binary")
                    if compute
                    else -1
                ),
                "test_metrics/dice_score": (
                    dice_score(
                        pred_binary.long(),
                        target_binary.long(),
                        num_classes=2,
                        input_format="index",
                    )
                    if compute
                    else -1
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

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Execute single prediction step.

        Parameters
        ----------
        batch : Sample
            Prediction batch data.
        batch_idx : int
            Batch index.
        dataloader_idx : int, default=0
            Dataloader index.

        Returns
        -------
        torch.Tensor
            Model prediction.
        """
        source = batch["source"]
        if self.test_time_augmentations:
            prediction = self.perform_test_time_augmentations(source)
        else:
            source = self._predict_pad(source)
            prediction = self.forward(source)
            prediction = self._predict_pad.inverse(prediction)

        return prediction

    def perform_test_time_augmentations(self, source: Tensor) -> Tensor:
        """Perform test time augmentations and aggregate predictions.

        Apply rotational augmentations to input source and aggregate the
        predictions using the specified method.

        Parameters
        ----------
        source : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Aggregated prediction.
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
            # Perform multiplication of predictions in logarithmic space
            # for numerical stability adding epsilon to avoid log(0) case
            log_predictions = torch.stack([torch.log(p + 1e-9) for p in predictions])
            log_prediction_sum = log_predictions.sum(dim=0)
            prediction = torch.exp(log_prediction_sum)
        return prediction

    def on_train_epoch_end(self) -> None:
        """Log training samples at end of epoch."""
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        """Log validation samples and compute average loss at end of epoch."""
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)
        # average within each dataloader
        loss_means = [torch.tensor(losses).mean() for losses in self.validation_losses]
        self.log(
            "loss/validate",
            torch.tensor(loss_means).mean().to(self.device),
            sync_dist=True,
        )
        self.validation_step_outputs.clear()
        self.validation_losses.clear()

    def on_test_start(self) -> None:
        """Load CellPose model for segmentation."""
        if self.test_cellpose_model_path is not None:
            try:
                from cellpose.models import CellposeModel

                self.cellpose_model = CellposeModel(
                    model_type=self.test_cellpose_model_path, device=self.device
                )
            except ImportError:
                raise ImportError(
                    "CellPose not installed. "
                    "Please install the metrics dependency with "
                    '`pip install viscy"[metrics]"`'
                )

    def on_predict_start(self) -> None:
        """Setup prediction padding transform.

        Pad the input shape to be divisible by the downsampling factor.
        The inverse of this transform crops the prediction to original shape.
        """
        down_factor = 2**self.model.num_blocks
        self._predict_pad = DivisiblePad((0, 0, down_factor, down_factor))

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[Any]]:
        """Configure optimizer and learning rate scheduler.

        Returns
        -------
        tuple
            Tuple containing optimizer and scheduler lists.
        """
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

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]) -> None:
        grid = render_images(imgs)
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


class AugmentedPredictionVSUNet(LightningModule):
    """Apply arbitrary collection of test-time augmentations for image translation prediction.

    Parameters
    ----------
    model : nn.Module
        The model to be used for prediction.
    forward_transforms : list[Callable[[Tensor], Tensor]]
        A collection of transforms to apply to the input image before passing it to the model.
        Each one is applied independently.
        For example, resizing the input to match the expected voxel size of the model.
    inverse_transforms : list[Callable[[Tensor], Tensor]]
        Inverse transforms to apply to the model output before reduction.
        They should be the inverse of each forward transform.
        For example, resizing the output to match the original input shape for storage.
    reduction : Literal["mean", "median"], optional
        The reduction method to apply to the predictions, by default "mean"

    Notes
    -----
    Given sample tensor ``x``,
    model instance ``model()``,
    a list of forward transforms ``[f1(), f2()]``,
    a list of inverse transforms ``[i1(), i2()]``,
    and reduction method ``reduce()``,
    the prediction is computed as follows:

        prediction = reduce(
            [
                i1(model(f1(x))),
                i2(model(f2(x))),
            ]
        )
    """

    def __init__(
        self,
        model: nn.Module,
        forward_transforms: list[Callable[[Tensor], Tensor]],
        inverse_transforms: list[Callable[[Tensor], Tensor]],
        reduction: Literal["mean", "median"] = "mean",
    ) -> None:
        super().__init__()
        down_factor = 2**model.num_blocks
        self._predict_pad = DivisiblePad((0, 0, down_factor, down_factor))
        self.model = model
        self._forward_transforms = forward_transforms
        self._inverse_transforms = inverse_transforms
        self._reduction = reduction

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        return self.model(x)

    def setup(self, stage: str) -> None:
        """Setup method for Lightning module.

        Parameters
        ----------
        stage : str
            Stage name (only 'predict' is supported).

        Raises
        ------
        NotImplementedError
            If stage is not 'predict'.
        """
        if stage != "predict":
            raise NotImplementedError(
                f"Only the 'predict' stage is supported by {type(self)}"
            )

    def _reduce_predictions(self, preds: list[Tensor]) -> Tensor:
        """Reduce multiple predictions using specified method.

        Parameters
        ----------
        preds : list[torch.Tensor]
            List of prediction tensors.

        Returns
        -------
        torch.Tensor
            Reduced prediction tensor.
        """
        prediction = torch.stack(preds, dim=0)
        if self._reduction == "mean":
            prediction = prediction.mean(dim=0)
        elif self._reduction == "median":
            prediction = prediction.median(dim=0).values
        return prediction

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Execute single prediction step with augmentations.

        Parameters
        ----------
        batch : Sample
            Prediction batch data.
        batch_idx : int
            Batch index.
        dataloader_idx : int, default=0
            Dataloader index.

        Returns
        -------
        torch.Tensor
            Aggregated prediction from augmented inputs.
        """
        source = batch["source"]
        preds = []
        for forward_t, inverse_t in zip(
            self._forward_transforms, self._inverse_transforms
        ):
            source = forward_t(source)
            source = self._predict_pad(source)
            pred = self.forward(source)
            pred = self._predict_pad.inverse(pred)
            pred = inverse_t(pred)
            preds.append(pred)
        if len(preds) == 1:
            prediction = preds[0]
        else:
            prediction = self._reduce_predictions(preds)
        return prediction


class FcmaeUNet(VSUNet):
    """Fully Convolutional Masked Autoencoder U-Net.

    Extends VSUNet to support masked autoencoder pre-training and supervised
    fine-tuning for virtual staining tasks.

    Parameters
    ----------
    fit_mask_ratio : float, default=0.0
        Masking ratio for FCMAE pre-training.
    **kwargs
        Additional arguments passed to VSUNet.
    """

    def __init__(
        self,
        fit_mask_ratio: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(architecture="fcmae", **kwargs)
        self.fit_mask_ratio = fit_mask_ratio
        self.save_hyperparameters(ignore=["loss_function"])

    def on_fit_start(self) -> None:
        """Setup data modules and validate configuration for training.

        Raises
        ------
        ValueError
            If data module configuration is incompatible with FCMAE training.
        """
        dm = self.trainer.datamodule
        if not isinstance(dm, CombinedDataModule):
            raise ValueError(
                f"Container data module type {type(dm)} "
                "is not supported for FCMAE training"
            )
        for subdm in dm.data_modules:
            if not isinstance(subdm, GPUTransformDataModule):
                raise ValueError(
                    f"Member data module type {type(subdm)} "
                    "is not supported for FCMAE training"
                )
        self.datamodules = dm.data_modules
        if self.model.pretraining and not isinstance(self.loss_function, MaskedMSELoss):
            raise ValueError(
                "MaskedMSELoss is required for FCMAE pre-training, "
                f"got {type(self.loss_function)}"
            )

    def forward(
        self, x: Tensor, mask_ratio: float = 0.0
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Forward pass with optional masking.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mask_ratio : float, default=0.0
            Masking ratio for FCMAE mode.

        Returns
        -------
        torch.Tensor or tuple
            Model output, optionally with mask if mask_ratio > 0.
        """
        return self.model(x, mask_ratio)

    def forward_fit_fcmae(
        self, batch: Sample, return_target: bool = False
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Forward pass for FCMAE pre-training.

        Parameters
        ----------
        batch : Sample
            Input batch.
        return_target : bool, default=False
            Whether to return masked target for logging.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor or None, torch.Tensor]
            Prediction, target (if requested), and loss.
        """
        x = batch["source"]
        pred, mask = self.forward(x, mask_ratio=self.fit_mask_ratio)
        loss = self.loss_function(pred, x, mask)
        if return_target:
            target = x * mask.unsqueeze(2)
        else:
            target = None
        return pred, target, loss

    def forward_fit_supervised(self, batch: Sample) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for supervised training.

        Parameters
        ----------
        batch : Sample
            Input batch containing source and target.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Prediction, target, and loss.
        """
        x = batch["source"]
        target = batch["target"]
        pred = self.forward(x)
        loss = self.loss_function(pred, target)
        return pred, target, loss

    def forward_fit_task(
        self, batch: Sample, batch_idx: int
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Forward pass for current training task.

        Automatically selects FCMAE pre-training or supervised training
        based on model configuration.

        Parameters
        ----------
        batch : Sample
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor or None, torch.Tensor]
            Prediction, target, and loss.
        """
        if self.model.pretraining:
            if batch_idx < self.log_batches_per_epoch:
                return_target = True
            pred, target, loss = self.forward_fit_fcmae(batch, return_target)
        else:
            pred, target, loss = self.forward_fit_supervised(batch)
        return pred, target, loss

    @torch.no_grad()
    def train_transform_and_collate(self, batch: list[dict[str, Tensor]]) -> Sample:
        """Apply training transforms and collate batch data.

        Parameters
        ----------
        batch : list[dict[str, torch.Tensor]]
            List of batch dictionaries from multiple data modules.

        Returns
        -------
        Sample
            Collated and transformed sample.
        """
        transformed = []
        for dataset_batch, dm in zip(batch, self.datamodules):
            dataset_batch = dm.train_gpu_transforms(dataset_batch)
            transformed.extend(dataset_batch)
        # shuffle references in place for better logging
        random.shuffle(transformed)
        return collate_meta_tensor(transformed)

    @torch.no_grad()
    def val_transform_and_collate(
        self, batch: list[Sample], dataloader_idx: int
    ) -> Tensor:
        """Apply validation transforms and collate batch data.

        Parameters
        ----------
        batch : list[Sample]
            List of samples.
        dataloader_idx : int
            Index of the validation dataloader.

        Returns
        -------
        torch.Tensor
            Collated and transformed batch.
        """
        batch = self.datamodules[dataloader_idx].val_gpu_transforms(batch)
        return collate_meta_tensor(batch)

    def training_step(self, batch: list[list[Sample]], batch_idx: int) -> Tensor:
        """Execute single training step for FCMAE.

        Parameters
        ----------
        batch : list[list[Sample]]
            Nested list of samples from multiple data modules.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        batch = self.train_transform_and_collate(batch)
        pred, target, loss = self.forward_fit_task(batch, batch_idx)
        if batch_idx < self.log_batches_per_epoch:
            self.training_step_outputs.extend(
                detach_sample(
                    (batch["source"], target, pred), self.log_samples_per_batch
                )
            )
        self.log(
            "loss/train",
            loss.to(self.device),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=pred.shape[0],
        )
        return loss

    def validation_step(
        self, batch: list[Sample], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Execute single validation step for FCMAE.

        Parameters
        ----------
        batch : list[Sample]
            List of validation samples.
        batch_idx : int
            Batch index.
        dataloader_idx : int, default=0
            Dataloader index.
        """
        batch = self.val_transform_and_collate(batch, dataloader_idx)
        pred, target, loss = self.forward_fit_task(batch, batch_idx)
        if dataloader_idx + 1 > len(self.validation_losses):
            self.validation_losses.append([])
        self.validation_losses[dataloader_idx].append(loss.detach())
        self.log(
            "loss/val", loss.to(self.device), sync_dist=True, batch_size=pred.shape[0]
        )
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(
                detach_sample(
                    (batch["source"], target, pred), self.log_samples_per_batch
                )
            )
