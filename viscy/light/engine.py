import logging
import os
from typing import Literal, Sequence, Union
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import wandb
from imageio import imwrite

# from lightning.pytorch import LightningModule
# from lightning import LightningModule
from torch.optim import Adam
from PIL import Image

import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only

from lightning.pytorch import LightningDataModule, LightningModule, Trainer

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
from viscy.representation.contrastive import ContrastiveEncoder

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


class ContrastiveModule(LightningModule):
    """Contrastive Learning Model for self-supervised learning."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        loss_function: Union[
            nn.Module, nn.CosineEmbeddingLoss, nn.TripletMarginLoss
        ] = nn.TripletMarginLoss(),
        margin: float = 0.5,
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_steps_per_epoch: int = 8,
        in_channels: int = 2,
        example_input_yx_shape: Sequence[int] = (256, 256),
        in_stack_depth: int = 15,
        stem_kernel_size: tuple[int, int, int] = (5, 3, 3),
        embedding_len: int = 256,
        predict: bool = False,
    ) -> None:
        super().__init__()

        self.loss_function = loss_function
        self.margin = margin
        self.lr = lr
        self.schedule = schedule
        self.log_steps_per_epoch = log_steps_per_epoch
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_metrics = []
        self.validation_metrics = []
        self.test_metrics = []
        self.processed_order = []

        self.encoder = ContrastiveEncoder(
            backbone=backbone,
            in_channels=in_channels,
            in_stack_depth=in_stack_depth,
            stem_kernel_size=stem_kernel_size,
            embedding_len=embedding_len,
            predict=predict,
        )

        # required to log the graph.
        self.example_input_array = torch.rand(
            1,  # batch size
            in_channels,
            in_stack_depth,
            *example_input_yx_shape,
        )

        self.images_to_log = []
        self.train_batch_counter = 0
        self.val_batch_counter = 0

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        projections = self.encoder(x)
        return projections
        # features is without projection head and projects is with projection head

    def log_feature_statistics(self, embeddings: Tensor, prefix: str):
        mean = torch.mean(embeddings, dim=0).detach().cpu().numpy()
        std = torch.std(embeddings, dim=0).detach().cpu().numpy()

        print(f"{prefix}_mean: {mean}")
        print(f"{prefix}_std: {std}")

    def print_embedding_norms(self, anchor, positive, negative, phase):
        anchor_norm = torch.norm(anchor, dim=1).mean().item()
        positive_norm = torch.norm(positive, dim=1).mean().item()
        negative_norm = torch.norm(negative, dim=1).mean().item()

        print(f"{phase}/anchor_norm: {anchor_norm}")
        print(f"{phase}/positive_norm: {positive_norm}")
        print(f"{phase}/negative_norm: {negative_norm}")

    # logs over all steps
    @rank_zero_only
    def log_metrics(self, anchor, positive, negative, phase):
        cosine_sim_pos = F.cosine_similarity(anchor, positive, dim=1).mean().item()
        cosine_sim_neg = F.cosine_similarity(anchor, negative, dim=1).mean().item()

        euclidean_dist_pos = F.pairwise_distance(anchor, positive).mean().item()
        euclidean_dist_neg = F.pairwise_distance(anchor, negative).mean().item()

        metrics = {
            f"{phase}/cosine_similarity_positive": cosine_sim_pos,
            f"{phase}/cosine_similarity_negative": cosine_sim_neg,
            f"{phase}/euclidean_distance_positive": euclidean_dist_pos,
            f"{phase}/euclidean_distance_negative": euclidean_dist_neg,
        }

        wandb.log(metrics)

        if phase == "train":
            self.training_metrics.append(metrics)
        elif phase == "val":
            self.validation_metrics.append(metrics)
        elif phase == "test":
            self.test_metrics.append(metrics)

    @rank_zero_only
    # logs only one sample from the first batch per epoch
    def log_images(self, anchor, positive, negative, epoch, step_name):
        z_idx = 7

        anchor_img_rfp = anchor[0, 0, z_idx, :, :].cpu().numpy()
        positive_img_rfp = positive[0, 0, z_idx, :, :].cpu().numpy()
        negative_img_rfp = negative[0, 0, z_idx, :, :].cpu().numpy()

        anchor_img_phase = anchor[0, 1, z_idx, :, :].cpu().numpy()
        positive_img_phase = positive[0, 1, z_idx, :, :].cpu().numpy()
        negative_img_phase = negative[0, 1, z_idx, :, :].cpu().numpy()

        # Debug prints to check the contents of the images
        print(f"Anchor RFP min: {anchor_img_rfp.min()}, max: {anchor_img_rfp.max()}")
        print(
            f"Positive RFP min: {positive_img_rfp.min()}, max: {positive_img_rfp.max()}"
        )
        print(
            f"Negative RFP min: {negative_img_rfp.min()}, max: {negative_img_rfp.max()}"
        )

        print(
            f"Anchor Phase min: {anchor_img_phase.min()}, max: {anchor_img_phase.max()}"
        )
        print(
            f"Positive Phase min: {positive_img_phase.min()}, max: {positive_img_phase.max()}"
        )
        print(
            f"Negative Phase min: {negative_img_phase.min()}, max: {negative_img_phase.max()}"
        )

        # combine the images side by side
        combined_img_rfp = np.concatenate(
            (anchor_img_rfp, positive_img_rfp, negative_img_rfp), axis=1
        )
        combined_img_phase = np.concatenate(
            (anchor_img_phase, positive_img_phase, negative_img_phase), axis=1
        )
        combined_img = np.concatenate((combined_img_rfp, combined_img_phase), axis=0)

        self.images_to_log.append(
            wandb.Image(
                combined_img, caption=f"Anchor | Positive | Negative (Epoch {epoch})"
            )
        )

        wandb.log({f"{step_name}": self.images_to_log})
        self.images_to_log = []

    def training_step(
        self,
        batch: tuple[Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Training step of the model."""

        anchor, pos_img, neg_img = batch
        emb_anchor = self.encoder(anchor)
        emb_pos = self.encoder(pos_img)
        emb_neg = self.encoder(neg_img)
        loss = self.loss_function(emb_anchor, emb_pos, emb_neg)

        self.log("train/loss_step", loss, on_step=True, prog_bar=True, logger=True)

        self.train_batch_counter += 1
        if self.train_batch_counter % self.log_steps_per_epoch == 0:
            self.log_images(
                anchor, pos_img, neg_img, self.current_epoch, "training_images"
            )

        self.log_metrics(emb_anchor, emb_pos, emb_neg, "train")
        # self.print_embedding_norms(emb_anchor, emb_pos, emb_neg, 'train')

        self.training_step_outputs.append(loss)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        epoch_loss = torch.stack(self.training_step_outputs).mean()
        self.log(
            "train/loss_epoch", epoch_loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.training_step_outputs.clear()

        if self.training_metrics:
            avg_metrics = self.aggregate_metrics(self.training_metrics, "train")
            self.log(
                "train/avg_cosine_similarity_positive",
                avg_metrics["train/cosine_similarity_positive"],
                on_epoch=True,
                logger=True,
            )
            self.log(
                "train/avg_cosine_similarity_negative",
                avg_metrics["train/cosine_similarity_negative"],
                on_epoch=True,
                logger=True,
            )
            self.log(
                "train/avg_euclidean_distance_positive",
                avg_metrics["train/euclidean_distance_positive"],
                on_epoch=True,
                logger=True,
            )
            self.log(
                "train/avg_euclidean_distance_negative",
                avg_metrics["train/euclidean_distance_negative"],
                on_epoch=True,
                logger=True,
            )
            self.training_metrics.clear()
        self.train_batch_counter = 0

    def validation_step(
        self,
        batch: tuple[Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Validation step of the model."""

        anchor, pos_img, neg_img = batch
        emb_anchor = self.encoder(anchor)
        emb_pos = self.encoder(pos_img)
        emb_neg = self.encoder(neg_img)
        loss = self.loss_function(emb_anchor, emb_pos, emb_neg)

        self.log("val/loss_step", loss, on_step=True, prog_bar=True, logger=True)

        self.val_batch_counter += 1
        if self.val_batch_counter % self.log_steps_per_epoch == 0:
            self.log_images(
                anchor, pos_img, neg_img, self.current_epoch, "validation_images"
            )

        self.log_metrics(emb_anchor, emb_pos, emb_neg, "val")

        self.validation_step_outputs.append(loss)
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        epoch_loss = torch.stack(self.validation_step_outputs).mean()
        self.log(
            "val/loss_epoch", epoch_loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.validation_step_outputs.clear()

        if self.validation_metrics:
            avg_metrics = self.aggregate_metrics(self.validation_metrics, "val")
            self.log(
                "val/avg_cosine_similarity_positive",
                avg_metrics["val/cosine_similarity_positive"],
                on_epoch=True,
                logger=True,
            )
            self.log(
                "val/avg_cosine_similarity_negative",
                avg_metrics["val/cosine_similarity_negative"],
                on_epoch=True,
                logger=True,
            )
            self.log(
                "val/avg_euclidean_distance_positive",
                avg_metrics["val/euclidean_distance_positive"],
                on_epoch=True,
                logger=True,
            )
            self.log(
                "val/avg_euclidean_distance_negative",
                avg_metrics["val/euclidean_distance_negative"],
                on_epoch=True,
                logger=True,
            )
            self.validation_metrics.clear()
        self.val_batch_counter = 0

    def test_step(
        self,
        batch: tuple[Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Test step of the model."""

        anchor, pos_img, neg_img = batch
        emb_anchor = self.encoder(anchor)
        emb_pos = self.encoder(pos_img)
        emb_neg = self.encoder(neg_img)
        loss = self.loss_function(emb_anchor, emb_pos, emb_neg)

        self.log("test/loss_step", loss, on_step=True, prog_bar=True, logger=True)

        self.log_metrics(emb_anchor, emb_pos, emb_neg, "test")

        self.test_step_outputs.append(loss)
        return {"loss": loss}

    @rank_zero_only
    def on_test_epoch_end(self) -> None:
        epoch_loss = torch.stack(self.test_step_outputs).mean()
        self.log(
            "test/loss_epoch", epoch_loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.test_step_outputs.clear()

        if self.test_metrics:
            avg_metrics = self.aggregate_metrics(self.test_metrics, "test")
            self.log(
                "test/avg_cosine_similarity_positive",
                avg_metrics["test/cosine_similarity_positive"],
                on_epoch=True,
                logger=True,
            )
            self.log(
                "test/avg_cosine_similarity_negative",
                avg_metrics["test/cosine_similarity_negative"],
                on_epoch=True,
                logger=True,
            )
            self.log(
                "test/avg_euclidean_distance_positive",
                avg_metrics["test/euclidean_distance_positive"],
                on_epoch=True,
                logger=True,
            )
            self.log(
                "test/avg_euclidean_distance_negative",
                avg_metrics["test/euclidean_distance_negative"],
                on_epoch=True,
                logger=True,
            )
            self.test_metrics.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def aggregate_metrics(self, metrics, phase):
        avg_metrics = {}
        if metrics:
            avg_metrics[f"{phase}/cosine_similarity_positive"] = sum(
                m[f"{phase}/cosine_similarity_positive"] for m in metrics
            ) / len(metrics)
            avg_metrics[f"{phase}/cosine_similarity_negative"] = sum(
                m[f"{phase}/cosine_similarity_negative"] for m in metrics
            ) / len(metrics)
            avg_metrics[f"{phase}/euclidean_distance_positive"] = sum(
                m[f"{phase}/euclidean_distance_positive"] for m in metrics
            ) / len(metrics)
            avg_metrics[f"{phase}/euclidean_distance_negative"] = sum(
                m[f"{phase}/euclidean_distance_negative"] for m in metrics
            ) / len(metrics)
        return avg_metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        print("running predict step!")
        """Prediction step for extracting embeddings."""
        x, position_info = batch
        features, projections = self.encoder(x)
        self.processed_order.extend(position_info)
        return features, projections

    # already saved, not needed again
    # def on_predict_epoch_end(self) -> None:
    #         print(f"Processed order: {self.processed_order}")
    #         rows, columns, fovs, cell_ids = [], [], [], []

    #         for position_path in self.processed_order:
    #             try:
    #                 parts = position_path.split("/")
    #                 if len(parts) < 3:
    #                     raise ValueError(f"Invalid position path: {position_path}")

    #                 row = parts[0]
    #                 column = parts[1]
    #                 fov_cell = parts[2]

    #                 fov = int(fov_cell.split("fov")[1].split("cell")[0])
    #                 cell_id = int(fov_cell.split("cell")[1])

    #                 rows.append(row)
    #                 columns.append(column)
    #                 fovs.append(fov)
    #                 cell_ids.append(cell_id)

    #             except (IndexError, ValueError) as e:
    #                 print(f"Skipping invalid position path: {position_path} with error: {e}")

    #         # Save processed order
    #         if rows and columns and fovs and cell_ids:
    #             processed_order_df = pd.DataFrame({
    #                 "Row": rows,
    #                 "Column": columns,
    #                 "FOV": fovs,
    #                 "Cell ID": cell_ids
    #             })
    #             print(f"Saving processed order DataFrame: {processed_order_df}")
    #             processed_order_df.to_csv("/hpc/mydata/alishba.imran/VisCy/viscy/applications/contrastive_phenotyping/epoch66_processed_order.csv", index=False)
    #         else:
    #             print("No valid processed orders found to save.")
