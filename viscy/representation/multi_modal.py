from logging import getLogger
from typing import Literal, Sequence

import torch
from pytorch_metric_learning.losses import NTXentLoss
from torch import Tensor, nn

from viscy.data.typing import TripletSample
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.representation.engine import ContrastiveModule

_logger = getLogger("lightning.pytorch")


class JointEncoders(nn.Module):
    def __init__(
        self,
        source_encoder: nn.Module | ContrastiveEncoder,
        target_encoder: nn.Module | ContrastiveEncoder,
    ) -> None:
        super().__init__()
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder

    def forward(
        self, source: Tensor, target: Tensor
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        return self.source_encoder(source), self.target_encoder(target)

    def forward_features(self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        return self.source_encoder(source)[0], self.target_encoder(target)[0]

    def forward_projections(
        self, source: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.source_encoder(source)[1], self.target_encoder(target)[1]


class JointContrastiveModule(ContrastiveModule):
    """CLIP-style model pair for self-supervised cross-modality representation learning."""

    def __init__(
        self,
        encoder: nn.Module | JointEncoders,
        loss_function: (
            nn.Module | nn.CosineEmbeddingLoss | nn.TripletMarginLoss | NTXentLoss
        ) = nn.TripletMarginLoss(margin=0.5),
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        log_embeddings: bool = False,
        embedding_log_frequency: int = 10,
        example_input_array_shape: Sequence[int] = (1, 2, 15, 256, 256),
        prediction_arm: Literal["source", "target"] = "source",
    ) -> None:
        super().__init__(
            encoder=encoder,
            loss_function=loss_function,
            lr=lr,
            schedule=schedule,
            log_batches_per_epoch=log_batches_per_epoch,
            log_samples_per_batch=log_samples_per_batch,
            log_embeddings=log_embeddings,
            embedding_log_frequency=embedding_log_frequency,
            example_input_array_shape=example_input_array_shape,
        )
        self.example_input_array = (self.example_input_array, self.example_input_array)
        self._prediction_arm = prediction_arm

    def forward(self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        return self.model.forward_projections(source, target)

    def _info_nce_style_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        indices = torch.arange(0, z1.size(0), device=z2.device)
        labels = torch.cat((indices, indices))
        embeddings = torch.cat((z1, z2))
        return self.loss_function(embeddings, labels)

    def _fit_forward_step(
        self, batch: TripletSample, batch_idx: int, stage: Literal["train", "val"]
    ) -> Tensor:
        anchor_img = batch["anchor"]
        pos_img = batch["positive"]
        anchor_source_projection, anchor_target_projection = (
            self.model.forward_projections(anchor_img[:, 0:1], anchor_img[:, 1:2])
        )
        positive_source_projection, positive_target_projection = (
            self.model.forward_projections(pos_img[:, 0:1], pos_img[:, 1:2])
        )
        # loss_source = self._info_nce_style_loss(
        #     anchor_source_projection, positive_source_projection
        # )
        # loss_target = self._info_nce_style_loss(
        #     anchor_target_projection, positive_target_projection
        # )
        loss_joint = self._info_nce_style_loss(
            anchor_source_projection, anchor_target_projection
        ) + self._info_nce_style_loss(
            positive_target_projection, positive_source_projection
        )
        # loss = loss_source + loss_target + loss_joint
        loss = loss_joint
        self._log_step_samples(batch_idx, (anchor_img, pos_img), stage)
        self._log_metrics(
            loss=loss,
            anchor=anchor_source_projection,
            positive=anchor_target_projection,
            negative=None,
            stage=stage,
        )
        return loss

    def training_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        return self._fit_forward_step(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        return self._fit_forward_step(batch=batch, batch_idx=batch_idx, stage="val")

    def on_predict_start(self) -> None:
        _logger.info(f"Using {self._prediction_arm} encoder for predictions.")
        if self._prediction_arm == "source":
            self._prediction_encoder = self.model.source_encoder
            self._prediction_channel_slice = slice(0, 1)
        elif self._prediction_arm == "target":
            self._prediction_encoder = self.model.target_encoder
            self._prediction_channel_slice = slice(1, 2)
        else:
            raise ValueError("Invalid prediction arm.")

    def predict_step(
        self, batch: TripletSample, batch_idx: int, dataloader_idx: int = 0
    ):
        features, projections = self._prediction_encoder(
            batch["anchor"][:, self._prediction_channel_slice]
        )
        return {
            "features": features,
            "projections": projections,
            "index": batch["index"],
        }
