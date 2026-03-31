"""Cytoland LightningModules for virtual staining."""

import inspect
import logging
import os
from typing import Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from imageio import imwrite
from lightning.pytorch import LightningModule
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

from viscy_data import CombinedDataModule, GPUTransformDataModule, Sample
from viscy_models import FullyConvolutionalMAE, Unet2d, Unet3d, Unet25d, UNeXt2
from viscy_utils.callbacks.prediction_writer import _blend_in
from viscy_utils.evaluation.metrics import mean_average_precision
from viscy_utils.log_images import detach_sample, log_image_grid

_UNET_ARCHITECTURE = {
    "2D": Unet2d,
    "UNeXt2": UNeXt2,
    "2.5D": Unet25d,
    "FNet3D": Unet3d,
    "fcmae": FullyConvolutionalMAE,
    "UNeXt2_2D": FullyConvolutionalMAE,
}

_logger = logging.getLogger("lightning.pytorch")


def _make_divisible_pad(model: nn.Module) -> DivisiblePad:
    """Build a DivisiblePad that matches the model's downsampling axes."""
    down_factor = 2**model.num_blocks
    if getattr(model, "downsamples_z", False):
        return DivisiblePad((0, down_factor, down_factor, down_factor))
    return DivisiblePad((0, 0, down_factor, down_factor))


def _identity(x: Tensor) -> Tensor:
    """Identity transform (no-op)."""
    return x


def _center_crop_to_shape(tensor: Tensor, spatial_shape: tuple[int, ...]) -> Tensor:
    """Center-crop trailing spatial dimensions to the requested shape."""
    slices = [slice(None)] * tensor.ndim
    start_dim = tensor.ndim - len(spatial_shape)
    for dim, size in enumerate(spatial_shape, start=start_dim):
        current = tensor.shape[dim]
        if current < size:
            raise ValueError(f"Cannot crop dimension {dim} from {current} to {size}")
        start = (current - size) // 2
        slices[dim] = slice(start, start + size)
    return tensor[tuple(slices)]


class MaskedMSELoss(nn.Module):
    """Masked MSE loss for FCMAE pre-training."""

    def forward(self, preds, original, mask):
        """Compute masked mean squared error loss.

        Parameters
        ----------
        preds : Tensor
            Predicted tensor.
        original : Tensor
            Original tensor.
        mask : Tensor
            Binary mask tensor.

        Returns
        -------
        Tensor
            Masked MSE loss value.
        """
        loss = F.mse_loss(preds, original, reduction="none")
        loss = (loss.mean(2) * mask).sum() / mask.sum()
        return loss


class VSUNet(LightningModule):
    """Regression U-Net module for virtual staining.

    Parameters
    ----------
    architecture : Literal["2D", "UNeXt2", "2.5D", "FNet3D", "fcmae", "UNeXt2_2D"]
        Architecture type to use.
    model_config : dict
        Model configuration dictionary.
    loss_function : nn.Module | None
        Loss function for training/validation.
        Defaults to L2 (mean squared error).
    lr : float
        Learning rate, defaults to 1e-3.
    schedule : Literal["WarmupCosine", "Constant"]
        Learning rate scheduler, defaults to "Constant".
    freeze_encoder : bool
        Whether to freeze encoder weights.
    ckpt_path : str | None
        Path to checkpoint to load weights.
    log_batches_per_epoch : int
        Number of batches to log each epoch, defaults to 8.
    log_samples_per_batch : int
        Number of samples to log each batch, defaults to 1.
    example_input_yx_shape : Sequence[int]
        XY shape of example input for graph tracing, defaults to (256, 256).
    test_cellpose_model_path : str | None
        Path to CellPose model for testing segmentation.
    test_cellpose_diameter : float | None
        Diameter parameter for CellPose model.
    test_evaluate_cellpose : bool | None
        Evaluate CellPose model instead of trained model in test stage.
    test_time_augmentations : bool | None
        Apply test time augmentations in test stage.
    tta_type : Literal["mean", "median", "product"]
        Type of test time augmentations aggregation, defaults to "mean".
    """

    def __init__(
        self,
        architecture: Literal["2D", "UNeXt2", "2.5D", "FNet3D", "fcmae", "UNeXt2_2D"],
        model_config: dict | None = None,
        loss_function: nn.Module | None = None,
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
        if model_config is None:
            model_config = {}
        net_class = _UNET_ARCHITECTURE.get(architecture)
        if not net_class:
            raise ValueError(f"Architecture {architecture} not in {_UNET_ARCHITECTURE.keys()}")
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
        # Cache loss function fg_mask compatibility to avoid per-batch inspect.signature().
        sig = inspect.signature(self.loss_function.forward)
        self._loss_accepts_fg_mask = "fg_mask" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        self.test_time_augmentations = test_time_augmentations
        self.tta_type = tta_type
        self.freeze_encoder = freeze_encoder
        self._original_shape_yx = None
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu")["state_dict"])

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass through the model.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Model output.
        """
        return self.model(x)

    def _compute_loss(self, pred: Tensor, target: Tensor, batch: Sample) -> Tensor:
        """Compute loss, passing precomputed fg_mask to the loss if present.

        When ``fg_mask_key`` is set in the data config, ``batch["fg_mask"]``
        is forwarded as a keyword argument.  The loss function must accept
        ``fg_mask`` explicitly or via ``**kwargs``; standard losses like
        ``nn.MSELoss`` will raise ``TypeError`` at configuration time.
        """
        if "fg_mask" in batch:
            if not self._loss_accepts_fg_mask:
                raise TypeError(
                    f"{type(self.loss_function).__name__} does not accept 'fg_mask'. "
                    f"Use SpotlightLoss or remove fg_mask_key from the data config."
                )
            return self.loss_function(pred, target, fg_mask=batch["fg_mask"])
        return self.loss_function(pred, target)

    def training_step(self, batch: Sample | Sequence[Sample], batch_idx: int):
        """Execute a single training step.

        Parameters
        ----------
        batch : Sample | Sequence[Sample]
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
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
            loss = self._compute_loss(pred, target, b)
            losses.append(loss)
            batch_size += source.shape[0]
            if batch_idx < self.log_batches_per_epoch:
                self.training_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))
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
        """Execute a single validation step.

        Parameters
        ----------
        batch : Sample
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index, defaults to 0.
        """
        source: Tensor = batch["source"]
        target: Tensor = batch["target"]
        pred = self.forward(source)
        loss = self._compute_loss(pred, target, batch)
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
            self.validation_step_outputs.extend(detach_sample((source, target, pred), self.log_samples_per_batch))

    def test_step(self, batch: Sample, batch_idx: int):
        """Execute a single test step.

        Parameters
        ----------
        batch : Sample
            Input batch.
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
            pred_labels = self._cellpose_predict(pred, f"p{int(position)}_t{ts[0]}_z{zs[0]}")
            self._log_segmentation_metrics(pred_labels, batch["labels"][0])
        else:
            self._log_segmentation_metrics(None, None)

    def _log_regression_metrics(self, pred: Tensor, target: Tensor):
        """Log regression metrics for paired image translation."""
        # paired image translation metrics
        self.log_dict(
            {
                # regression
                "test_metrics/MAE": mean_absolute_error(pred, target),
                "test_metrics/MSE": mean_squared_error(pred, target),
                "test_metrics/cosine": cosine_similarity(pred, target, reduction="mean"),
                "test_metrics/pearson": pearson_corrcoef(pred.flatten() * 1e4, target.flatten() * 1e4),
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
        """Run CellPose segmentation on predicted image."""
        pred_labels_np = self.cellpose_model.eval(
            pred.cpu().numpy(), channels=[0, 0], diameter=self.test_cellpose_diameter
        )[0].astype(np.int16)
        imwrite(os.path.join(self.logger.log_dir, f"{name}.png"), pred_labels_np)
        return torch.from_numpy(pred_labels_np).to(self.device)

    def _log_segmentation_metrics(self, pred_labels: torch.ShortTensor, target_labels: torch.ShortTensor):
        """Log segmentation metrics comparing predictions to ground truth."""
        compute = pred_labels is not None
        if compute:
            pred_binary = pred_labels > 0
            target_binary = target_labels > 0
            coco_metrics = mean_average_precision(pred_labels, target_labels)
            _logger.debug(coco_metrics)
        self.log_dict(
            {
                # semantic segmentation
                "test_metrics/accuracy": (accuracy(pred_binary, target_binary, task="binary") if compute else -1),
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
                "test_metrics/jaccard": (jaccard_index(pred_binary, target_binary, task="binary") if compute else -1),
                "test_metrics/mAP": coco_metrics["map"] if compute else -1,
                "test_metrics/mAP_50": coco_metrics["map_50"] if compute else -1,
                "test_metrics/mAP_75": coco_metrics["map_75"] if compute else -1,
                "test_metrics/mAR_100": coco_metrics["mar_100"] if compute else -1,
            },
            on_step=True,
            on_epoch=False,
        )

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):
        """Execute a single prediction step.

        Parameters
        ----------
        batch : Sample
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index, defaults to 0.

        Returns
        -------
        Tensor
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
        """Perform test time augmentations on the input source.

        Applies rotations and aggregates predictions using the specified method.

        Parameters
        ----------
        source : Tensor
            Input tensor.

        Returns
        -------
        Tensor
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
            de_augmented_prediction = self._rotate_volume(de_augmented_prediction, k=4 - i, spatial_axes=(1, 2))
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

    def on_train_epoch_end(self):
        """Log training samples at end of epoch."""
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        """Log validation samples and average losses at end of epoch."""
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

    def on_test_start(self):
        """Load CellPose model for segmentation."""
        if self.test_cellpose_model_path is not None:
            try:
                from cellpose.models import CellposeModel

                self.cellpose_model = CellposeModel(model_type=self.test_cellpose_model_path, device=self.device)
            except ImportError:
                raise ImportError(
                    'CellPose not installed. Please install the metrics dependency with `pip install viscy"[metrics]"`'
                )

    def on_predict_start(self):
        """Pad the input shape to be divisible by the downsampling factor.

        The inverse of this transform crops the prediction to original shape.
        """
        self._predict_pad = _make_divisible_pad(self.model)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.freeze_encoder:
            if not hasattr(self.model, "encoder"):
                raise ValueError(
                    f"freeze_encoder=True requires a model with an 'encoder' attribute "
                    f"(e.g. FullyConvolutionalMAE), got {type(self.model).__name__}"
                )
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
            scheduler = ConstantLR(optimizer, factor=1, total_iters=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        """Log image sample grid to the active logger (TensorBoard or W&B)."""
        if not self.trainer.is_global_zero or self.logger is None:
            return
        log_image_grid(self.logger, key, imgs, self.current_epoch)

    def _rotate_volume(self, tensor: Tensor, k: int, spatial_axes: tuple) -> Tensor:
        """Rotate a volume tensor by k*90 degrees."""
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
        """Crop tensor back to original YX shape after rotation padding."""
        original_y, original_x = self._original_shape_yx
        pad_y = (tensor.shape[-2] - original_y) // 2
        pad_x = (tensor.shape[-1] - original_x) // 2
        cropped_tensor = tensor[..., pad_y : pad_y + original_y, pad_x : pad_x + original_x]
        return cropped_tensor


class AugmentedPredictionVSUNet(LightningModule):
    """Apply test-time augmentations and sliding window prediction for image translation.

    Parameters
    ----------
    model : nn.Module
        The model to be used for prediction.
    forward_transforms : list[Callable[[Tensor], Tensor]] or None, optional
        Transforms to apply to the input before the model. Each is applied independently.
        If None, defaults to a single identity transform.
    inverse_transforms : list[Callable[[Tensor], Tensor]] or None, optional
        Inverse transforms to apply to the model output before reduction.
        If None, defaults to a single identity transform.
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
        forward_transforms: list[Callable[[Tensor], Tensor]] | None = None,
        inverse_transforms: list[Callable[[Tensor], Tensor]] | None = None,
        reduction: Literal["mean", "median"] = "mean",
    ) -> None:
        super().__init__()
        self._predict_pad = _make_divisible_pad(model)
        self.model = model
        self._forward_transforms = forward_transforms or [_identity]
        self._inverse_transforms = inverse_transforms or [_identity]
        self._reduction = reduction

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass through the model.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Model output.
        """
        return self.model(x)

    def setup(self, stage: str) -> None:
        """Set up the module for the given stage.

        Parameters
        ----------
        stage : str
            Stage name (only "predict" is supported).

        Raises
        ------
        NotImplementedError
            If stage is not "predict".
        """
        if stage != "predict":
            raise NotImplementedError(f"Only the 'predict' stage is supported by {type(self)}")

    def _reduce_predictions(self, preds: list[Tensor]) -> Tensor:
        """Reduce multiple predictions using the configured method."""
        prediction = torch.stack(preds, dim=0)
        if self._reduction == "mean":
            prediction = prediction.mean(dim=0)
        elif self._reduction == "median":
            prediction = prediction.median(dim=0).values
        return prediction

    def _predict_with_tta(self, source: Tensor) -> Tensor:
        """Apply test-time augmentations and reduce predictions.

        Parameters
        ----------
        source : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Prediction (reduced if multiple augmentations).
        """
        preds = []
        for fwd_t, inv_t in zip(self._forward_transforms, self._inverse_transforms):
            aug_source = fwd_t(source)
            aug_source = self._predict_pad(aug_source)
            pred = self.forward(aug_source)
            pred = _center_crop_to_shape(pred, source.shape[2:])
            preds.append(inv_t(pred))
        if len(preds) == 1:
            return preds[0]
        return self._reduce_predictions(preds)

    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Execute a single prediction step with test-time augmentations.

        Parameters
        ----------
        batch : Sample
            Input batch containing "source" tensor.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index, defaults to 0.

        Returns
        -------
        Tensor
            Model prediction.
        """
        return self._predict_with_tta(batch["source"])

    def predict_sliding_windows(self, x: Tensor, out_channel: int = 2, step: int = 1) -> Tensor:
        """Run inference using sliding windows along Z with linear feathering blending.

        Produces the same results as ``viscy predict`` CLI (HCSPredictionWriter)
        since both use the same ``_blend_in`` blending algorithm.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, Z, Y, X).
        out_channel : int, optional
            Number of output channels, by default 2.
        step : int, optional
            Step size for sliding window along Z, by default 1.
            With step=1, every Z position is covered. With step>1,
            trailing positions beyond the last full window are not predicted.

        Returns
        -------
        Tensor
            Output tensor of shape (B, out_channel, Z, Y, X).

        Raises
        ------
        ValueError
            If input is not 5D, model lacks ``out_stack_depth``, or
            model's stack depth exceeds input depth.
        """
        if x.ndim != 5:
            raise ValueError(f"Expected input with 5 dimensions (B, C, Z, Y, X), got {x.shape}")
        batch_size, _, depth, height, width = x.shape
        in_stack_depth = getattr(self.model, "out_stack_depth", None)
        if in_stack_depth is None:
            raise ValueError(
                f"Model {type(self.model).__name__} does not support sliding window "
                "prediction (missing out_stack_depth attribute)."
            )
        if in_stack_depth > depth:
            raise ValueError(f"in_stack_depth {in_stack_depth} > input depth {depth}")
        out_tensor = x.new_zeros((batch_size, out_channel, depth, height, width))
        for start in range(0, depth - in_stack_depth + 1, step):
            end = start + in_stack_depth
            pred = self._predict_with_tta(x[:, :, start:end])
            z_slice = slice(start, end)
            out_tensor[:, :, z_slice] = _blend_in(out_tensor[:, :, z_slice], pred, z_slice)
        return out_tensor


class FcmaeUNet(VSUNet):
    """FCMAE-based U-Net for self-supervised pre-training and fine-tuning.

    Workflow
    --------
    1. **Pretrain** with ``fit_mask_ratio > 0`` and ``MaskedMSELoss``.
       Set ``model_config["pretraining"] = True`` (the default).
    2. **Fine-tune** by loading the pretrained checkpoint with
       ``encoder_only=True`` and ``ckpt_path=<path>``.  Set
       ``model_config["pretraining"] = False`` and change
       ``out_channels`` / loss as needed.  Optionally set
       ``freeze_encoder=True`` to freeze the encoder.

    Parameters
    ----------
    fit_mask_ratio : float
        Mask ratio for FCMAE pre-training, defaults to 0.0.
    encoder_only : bool
        When True and ``ckpt_path`` is set, load only encoder weights
        from the checkpoint (ignoring decoder/head).  Useful for
        fine-tuning with a different number of output channels.
        Defaults to False.
    freeze_encoder : bool
        Freeze encoder weights during fine-tuning (passed to VSUNet).
        Defaults to False.
    **kwargs
        Additional keyword arguments passed to VSUNet.
    """

    def __init__(
        self,
        fit_mask_ratio: float = 0.0,
        encoder_only: bool = False,
        **kwargs,
    ):
        if encoder_only:
            if "ckpt_path" not in kwargs or kwargs["ckpt_path"] is None:
                raise ValueError("encoder_only=True requires ckpt_path")
            ckpt_path = kwargs.pop("ckpt_path")
        else:
            ckpt_path = None
        super().__init__(architecture="fcmae", **kwargs)
        self.fit_mask_ratio = fit_mask_ratio
        if ckpt_path is not None:
            self._load_encoder_weights(ckpt_path)
        self.save_hyperparameters(ignore=["loss_function", "ckpt_path", "encoder_only"])

    def _load_encoder_weights(self, ckpt_path: str) -> None:
        """Load only encoder weights from a pretrained checkpoint.

        Parameters
        ----------
        ckpt_path : str
            Path to the pretrained checkpoint file.
        """
        state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")["state_dict"]
        prefix = "model.encoder."
        encoder_weights = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
        self.model.encoder.load_state_dict(encoder_weights, strict=True)
        _logger.info(f"Loaded {len(encoder_weights)} encoder parameters from {ckpt_path}")

    def on_fit_start(self):
        """Validate datamodule configuration for FCMAE training."""
        dm = self.trainer.datamodule
        if not isinstance(dm, CombinedDataModule):
            raise ValueError(f"Container data module type {type(dm)} is not supported for FCMAE training")
        for subdm in dm.data_modules:
            if not isinstance(subdm, GPUTransformDataModule):
                raise ValueError(f"Member data module type {type(subdm)} is not supported for FCMAE training")
        if self.model.pretraining and not isinstance(self.loss_function, MaskedMSELoss):
            raise ValueError(f"MaskedMSELoss is required for FCMAE pre-training, got {type(self.loss_function)}")

    def forward(self, x: Tensor, mask_ratio: float = 0.0):
        """Run forward pass with optional masking.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        mask_ratio : float
            Mask ratio for FCMAE, defaults to 0.0.

        Returns
        -------
        Tensor
            Model output.
        """
        return self.model(x, mask_ratio)

    def forward_fit_fcmae(self, batch: Sample, return_target: bool = False) -> tuple[Tensor, Tensor | None, Tensor]:
        """Forward pass for FCMAE pre-training.

        Parameters
        ----------
        batch : Sample
            Input batch.
        return_target : bool
            Whether to return the masked target.

        Returns
        -------
        tuple[Tensor, Tensor | None, Tensor]
            Prediction, optional target, and loss.
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
        """Forward pass for supervised fine-tuning.

        Parameters
        ----------
        batch : Sample
            Input batch.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Prediction, target, and loss.
        """
        x = batch["source"]
        target = batch["target"]
        pred = self.forward(x)
        loss = self._compute_loss(pred, target, batch)
        return pred, target, loss

    def forward_fit_task(self, batch: Sample, batch_idx: int) -> tuple[Tensor, Tensor | None, Tensor]:
        """Dispatch to FCMAE or supervised forward pass based on model state.

        Parameters
        ----------
        batch : Sample
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        tuple[Tensor, Tensor | None, Tensor]
            Prediction, optional target, and loss.
        """
        return_target = False
        if self.model.pretraining:
            if batch_idx < self.log_batches_per_epoch:
                return_target = True
            pred, target, loss = self.forward_fit_fcmae(batch, return_target)
        else:
            pred, target, loss = self.forward_fit_supervised(batch)
        return pred, target, loss

    @staticmethod
    def _merge_batches(batch: list[Sample] | Sample) -> Sample:
        """Merge per-dataset batches from CombinedLoader into one batch.

        Parameters
        ----------
        batch : list[Sample] | Sample
            List of per-dataset batches (training) or a single batch
            (validation).

        Returns
        -------
        Sample
            Merged batch with concatenated tensors.
        """
        if not isinstance(batch, list):
            return batch
        combined: dict[str, Tensor] = {}
        for key in batch[0]:
            vals = [b[key] for b in batch if key in b]
            if isinstance(vals[0], Tensor):
                combined[key] = torch.cat(vals, dim=0)
            else:
                combined[key] = vals[0]
        return combined

    def training_step(self, batch: list[Sample] | Sample, batch_idx: int) -> Tensor:
        """Execute a single FCMAE training step.

        Parameters
        ----------
        batch : list[Sample] | Sample
            Per-dataset batches from CombinedLoader (already transformed
            by ``CombinedDataModule.on_after_batch_transfer``).
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            Training loss.
        """
        batch = self._merge_batches(batch)
        pred, target, loss = self.forward_fit_task(batch, batch_idx)
        if batch_idx < self.log_batches_per_epoch:
            self.training_step_outputs.extend(
                detach_sample((batch["source"], target, pred), self.log_samples_per_batch)
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

    def validation_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Execute a single FCMAE validation step.

        Parameters
        ----------
        batch : Sample
            Input batch (already transformed by
            ``CombinedDataModule.on_after_batch_transfer``).
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index, defaults to 0.
        """
        pred, target, loss = self.forward_fit_task(batch, batch_idx)
        if dataloader_idx + 1 > len(self.validation_losses):
            self.validation_losses.append([])
        self.validation_losses[dataloader_idx].append(loss.detach())
        self.log("loss/val", loss.to(self.device), sync_dist=True, batch_size=pred.shape[0])
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(
                detach_sample((batch["source"], target, pred), self.log_samples_per_batch)
            )
