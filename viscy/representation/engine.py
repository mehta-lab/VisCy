import logging
from typing import Literal, Sequence, TypedDict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from viscy.data.typing import TrackingIndex, TripletSample
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.utils.log_images import detach_sample, render_images
from pytorch_metric_learning.losses import SelfSupervisedLoss
from pytorch_metric_learning.losses import NTXentLoss as NTXentLoss_pml

_logger = logging.getLogger("lightning.pytorch")


class NTXentLoss_viscy(torch.nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss

    From Chen et.al, https://arxiv.org/abs/2002.05709
    """

    def __init__(
        self,
        temperature=0.5,
        criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    ):
        super(NTXentLoss_viscy, self).__init__()
        self.temperature = temperature
        self.criterion = criterion

    def _get_correlated_mask(self, batch_size):
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        _logger.info(f"mask: {mask}")
        return mask

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, embeddings, labels):
        """
        embeddings = [zis, zjs]
        zis and zjs are the output projections from the two augmented views.
        Here, we assume the two augmented views are the anchor and positive samples
        """
        # Get the batch size from tensor
        batch_size = embeddings.shape[0] // 2

        zis, zjs = torch.split(embeddings, batch_size, dim=0)

        # Cosine similarity
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        # Temperature scaling
        similarity_matrix = similarity_matrix / self.temperature

        mask = self._get_correlated_mask(batch_size).to(similarity_matrix.device)

        # Mask out unwanted pairs
        similarity_matrix = similarity_matrix[mask].view(2 * batch_size, -1)

        # Calculate NT-Xent Loss as cross-entropy
        loss = self.criterion(similarity_matrix, labels)
        loss /= 2 * batch_size

        return loss


class ContrastivePrediction(TypedDict):
    features: Tensor
    projections: Tensor
    index: TrackingIndex


class ContrastiveModule(LightningModule):
    """Contrastive Learning Model for self-supervised learning."""

    def __init__(
        self,
        encoder: nn.Module | ContrastiveEncoder,
        loss_function: (
            nn.Module
            | nn.CosineEmbeddingLoss
            | nn.TripletMarginLoss
            | NTXentLoss_pml
            | NTXentLoss_viscy
        ) = nn.TripletMarginLoss(margin=0.5),
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        example_input_array_shape: Sequence[int] = (1, 2, 15, 256, 256),
    ) -> None:
        super().__init__()
        self.model = encoder
        self.loss_function = loss_function
        self.lr = lr
        self.schedule = schedule
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.example_input_array = torch.rand(*example_input_array_shape)
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: Tensor) -> Tensor:
        "Only return projected embeddings for training and validation."
        return self.model(x)[1]

    def log_feature_statistics(self, embeddings: Tensor, prefix: str):
        mean = torch.mean(embeddings, dim=0).detach().cpu().numpy()
        std = torch.std(embeddings, dim=0).detach().cpu().numpy()
        _logger.debug(f"{prefix}_mean: {mean}")
        _logger.debug(f"{prefix}_std: {std}")

    def print_embedding_norms(self, anchor, positive, negative, phase):
        anchor_norm = torch.norm(anchor, dim=1).mean().item()
        positive_norm = torch.norm(positive, dim=1).mean().item()
        negative_norm = torch.norm(negative, dim=1).mean().item()
        _logger.debug(f"{phase}/anchor_norm: {anchor_norm}")
        _logger.debug(f"{phase}/positive_norm: {positive_norm}")
        _logger.debug(f"{phase}/negative_norm: {negative_norm}")

    def _log_metrics(
        self, loss, anchor, positive, stage: Literal["train", "val"], negative=None
    ):
        self.log(
            f"loss/{stage}",
            loss.to(self.device),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        cosine_sim_pos = F.cosine_similarity(anchor, positive, dim=1).mean()
        euclidean_dist_pos = F.pairwise_distance(anchor, positive).mean()
        log_metric_dict = {
            f"metrics/cosine_similarity_positive/{stage}": cosine_sim_pos,
            f"metrics/euclidean_distance_positive/{stage}": euclidean_dist_pos,
        }

        if negative is not None:
            euclidean_dist_neg = F.pairwise_distance(anchor, negative).mean()
            cosine_sim_neg = F.cosine_similarity(anchor, negative, dim=1).mean()
            log_metric_dict[f"metrics/cosine_similarity_negative/{stage}"] = (
                cosine_sim_neg
            )
            log_metric_dict[f"metrics/euclidean_distance_negative/{stage}"] = (
                euclidean_dist_neg
            )

        self.log_dict(
            log_metric_dict,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        grid = render_images(imgs, cmaps=["gray"] * 3)
        self.logger.experiment.add_image(
            key, grid, self.current_epoch, dataformats="HWC"
        )

    def training_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        anchor_img = batch["anchor"]
        pos_img = batch["positive"]
        anchor_projection = self(anchor_img)
        positive_projection = self(pos_img)
        if isinstance(self.loss_function, (NTXentLoss_pml, NTXentLoss_viscy)):
            indices = torch.arange(
                0, anchor_projection.size(0), device=anchor_projection.device
            )
            labels = torch.cat((indices, indices))
            # Note: we assume the two augmented views are the anchor and positive samples
            embeddings = torch.cat((anchor_projection, positive_projection))
            loss = self.loss_function(embeddings, labels)
            self._log_metrics(
                loss=loss,
                anchor=anchor_projection,
                positive=positive_projection,
                negative=None,
                stage="train",
            )
            if batch_idx < self.log_batches_per_epoch:
                self.training_step_outputs.extend(
                    detach_sample((anchor_img, pos_img), self.log_samples_per_batch)
                )
        else:
            neg_img = batch["negative"]
            negative_projection = self(neg_img)
            loss = self.loss_function(
                anchor_projection, positive_projection, negative_projection
            )
            self._log_metrics(
                loss=loss,
                anchor=anchor_projection,
                positive=positive_projection,
                negative=negative_projection,
                stage="train",
            )
            if batch_idx < self.log_batches_per_epoch:
                self.training_step_outputs.extend(
                    detach_sample(
                        (anchor_img, pos_img, neg_img), self.log_samples_per_batch
                    )
                )
        return loss

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def validation_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        """Validation step of the model."""
        anchor = batch["anchor"]
        pos_img = batch["positive"]
        anchor_projection = self(anchor)
        positive_projection = self(pos_img)
        if isinstance(self.loss_function, (NTXentLoss_pml, NTXentLoss_viscy)):
            indices = torch.arange(
                0, anchor_projection.size(0), device=anchor_projection.device
            )
            labels = torch.cat((indices, indices))
            # Note: we assume the two augmented views are the anchor and positive samples
            embeddings = torch.cat((anchor_projection, positive_projection))
            loss = self.loss_function(embeddings, labels)
            self._log_metrics(
                loss=loss,
                anchor=anchor_projection,
                positive=positive_projection,
                negative=None,
                stage="val",
            )
            if batch_idx < self.log_batches_per_epoch:
                self.validation_step_outputs.extend(
                    detach_sample((anchor, pos_img), self.log_samples_per_batch)
                )
        else:
            neg_img = batch["negative"]
            negative_projection = self(neg_img)
            loss = self.loss_function(
                anchor_projection, positive_projection, negative_projection
            )
            self._log_metrics(
                loss=loss,
                anchor=anchor_projection,
                positive=positive_projection,
                negative=negative_projection,
                stage="val",
            )
            if batch_idx < self.log_batches_per_epoch:
                self.validation_step_outputs.extend(
                    detach_sample(
                        (anchor, pos_img, neg_img), self.log_samples_per_batch
                    )
                )
        return loss

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(
        self, batch: TripletSample, batch_idx, dataloader_idx=0
    ) -> ContrastivePrediction:
        """Prediction step for extracting embeddings."""
        features, projections = self.model(batch["anchor"])
        return {
            "features": features,
            "projections": projections,
            "index": batch["index"],
        }
