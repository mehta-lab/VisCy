import logging
from collections.abc import Sequence
from typing import Literal, TypedDict

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from numpy.typing import NDArray
from pytorch_metric_learning.losses import NTXentLoss
from torch import Tensor, nn
from umap import UMAP
from viscy.data.typing import TrackingIndex, TripletSample
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.utils.log_images import detach_sample, render_images

_logger = logging.getLogger("lightning.pytorch")


class ContrastivePrediction(TypedDict):
    """Typed dictionary for contrastive model predictions.

    Contains features, projections, and metadata for contrastive learning
    inference outputs.
    """

    features: Tensor
    projections: Tensor
    index: TrackingIndex


class ContrastiveModule(LightningModule):
    """Contrastive Learning Model for self-supervised learning.

    Parameters
    ----------
    encoder : nn.Module | ContrastiveEncoder
        Encoder model.
    loss_function : nn.Module | nn.CosineEmbeddingLoss | nn.TripletMarginLoss | NTXentLoss
        Loss function. By default, nn.TripletMarginLoss with margin 0.5.
    lr : float
        Learning rate. By default, 1e-3.
    schedule : Literal["WarmupCosine", "Constant"]
        Schedule for learning rate. By default, "Constant".
    log_batches_per_epoch : int
        Number of batches to log. By default, 8.
    log_samples_per_batch : int
        Number of samples to log. By default, 1.
    log_embeddings : bool
        Whether to log embeddings. By default, False.
    example_input_array_shape : Sequence[int]
        Shape of example input array.
    """

    def __init__(
        self,
        encoder: nn.Module | ContrastiveEncoder,
        loss_function: (
            nn.Module | nn.CosineEmbeddingLoss | nn.TripletMarginLoss | NTXentLoss
        ) = nn.TripletMarginLoss(margin=0.5),
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        log_embeddings: bool = False,
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
        self.log_embeddings = log_embeddings

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return both features and projections.

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        tuple[Tensor, Tensor]
            Tuple of (features, projections)
        """
        return self.model(x)

    def log_feature_statistics(self, embeddings: Tensor, prefix: str):
        """Log embedding statistics for monitoring training dynamics.

        Parameters
        ----------
        embeddings : Tensor
            Embedding vectors to analyze.
        prefix : str
            Prefix for logging keys.
        """
        mean = torch.mean(embeddings, dim=0).detach().cpu().numpy()
        std = torch.std(embeddings, dim=0).detach().cpu().numpy()
        _logger.debug(f"{prefix}_mean: {mean}")
        _logger.debug(f"{prefix}_std: {std}")

    def print_embedding_norms(
        self, anchor: Tensor, positive: Tensor, negative: Tensor, phase: str
    ):
        """Log L2 norms of embeddings for triplet components.

        Parameters
        ----------
        anchor : Tensor
            Anchor embeddings.
        positive : Tensor
            Positive embeddings.
        negative : Tensor
            Negative embeddings.
        phase : str
            Training phase identifier for logging.
        """
        anchor_norm = torch.norm(anchor, dim=1).mean().item()
        positive_norm = torch.norm(positive, dim=1).mean().item()
        negative_norm = torch.norm(negative, dim=1).mean().item()
        _logger.debug(f"{phase}/anchor_norm: {anchor_norm}")
        _logger.debug(f"{phase}/positive_norm: {positive_norm}")
        _logger.debug(f"{phase}/negative_norm: {negative_norm}")

    def _log_metrics(
        self,
        loss: Tensor,
        anchor: Tensor,
        positive: Tensor,
        stage: Literal["train", "val"],
        negative: Tensor | None = None,
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

    def _log_samples(self, key: str, imgs: Sequence[Sequence[NDArray]]):
        grid = render_images(imgs, cmaps=["gray"] * 3)
        self.logger.experiment.add_image(
            key, grid, self.current_epoch, dataformats="HWC"
        )

    def _log_step_samples(self, batch_idx, samples, stage: Literal["train", "val"]):
        """Common method for logging step samples"""
        if batch_idx < self.log_batches_per_epoch:
            output_list = (
                self.training_step_outputs
                if stage == "train"
                else self.validation_step_outputs
            )
            output_list.extend(detach_sample(samples, self.log_samples_per_batch))

    def log_embedding_umap(self, embeddings: Tensor, tag: str):
        """Log UMAP visualization of embedding space to TensorBoard.

        Parameters
        ----------
        embeddings : Tensor
            High-dimensional embeddings to visualize.
        tag : str
            Tag for TensorBoard logging.
        """
        _logger.debug(f"Computing UMAP for {tag} embeddings.")
        umap = UMAP(n_components=2)
        embeddings_np = embeddings.detach().cpu().numpy()
        umap_embeddings = umap.fit_transform(embeddings_np)

        # Log UMAP embeddings to TensorBoard
        self.logger.experiment.add_embedding(
            umap_embeddings,
            global_step=self.current_epoch,
            tag=f"{tag}_umap",
        )

    def training_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        """Execute training step for contrastive learning.

        Computes triplet or NT-Xent loss based on configured loss function
        and logs training metrics.

        Parameters
        ----------
        batch : TripletSample
            Batch containing anchor, positive, and negative samples.
        batch_idx : int
            Index of current batch.

        Returns
        -------
        Tensor
            Computed contrastive loss.
        """
        anchor_img = batch["anchor"]
        pos_img = batch["positive"]
        _, anchor_projection = self(anchor_img)
        _, positive_projection = self(pos_img)
        negative_projection = None
        if isinstance(self.loss_function, NTXentLoss):
            indices = torch.arange(
                0, anchor_projection.size(0), device=anchor_projection.device
            )
            labels = torch.cat((indices, indices))
            # Note: we assume the two augmented views are the anchor and positive samples
            embeddings = torch.cat((anchor_projection, positive_projection))
            loss = self.loss_function(embeddings, labels)
            self._log_step_samples(batch_idx, (anchor_img, pos_img), "train")
        else:
            neg_img = batch["negative"]
            _, negative_projection = self(neg_img)
            loss = self.loss_function(
                anchor_projection, positive_projection, negative_projection
            )
            self._log_step_samples(batch_idx, (anchor_img, pos_img, neg_img), "train")
        self._log_metrics(
            loss=loss,
            anchor=anchor_projection,
            positive=positive_projection,
            negative=negative_projection,
            stage="train",
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Log training samples and embeddings at epoch end.

        Logs sample images and optionally computes UMAP visualization
        of embedding space for monitoring training progress.
        """
        super().on_train_epoch_end()
        self._log_samples("train_samples", self.training_step_outputs)
        # Log UMAP embeddings for validation
        if self.log_embeddings:
            embeddings = torch.cat(
                [output["embeddings"] for output in self.validation_step_outputs]
            )
            self.log_embedding_umap(embeddings, tag="train")
        self.training_step_outputs = []

    def validation_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        """Validation step of the model."""
        anchor = batch["anchor"]
        pos_img = batch["positive"]
        _, anchor_projection = self(anchor)
        _, positive_projection = self(pos_img)
        negative_projection = None
        if isinstance(self.loss_function, NTXentLoss):
            indices = torch.arange(
                0, anchor_projection.size(0), device=anchor_projection.device
            )
            labels = torch.cat((indices, indices))
            # Note: we assume the two augmented views are the anchor and positive samples
            embeddings = torch.cat((anchor_projection, positive_projection))
            loss = self.loss_function(embeddings, labels)
            self._log_step_samples(batch_idx, (anchor, pos_img), "val")
        else:
            neg_img = batch["negative"]
            _, negative_projection = self(neg_img)
            loss = self.loss_function(
                anchor_projection, positive_projection, negative_projection
            )
            self._log_step_samples(batch_idx, (anchor, pos_img, neg_img), "val")
        self._log_metrics(
            loss=loss,
            anchor=anchor_projection,
            positive=positive_projection,
            negative=negative_projection,
            stage="val",
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log validation samples and embeddings at epoch end.

        Logs sample images and optionally computes UMAP visualization
        of embedding space for monitoring validation performance.
        """
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)
        # Log UMAP embeddings for training
        if self.log_embeddings:
            embeddings = torch.cat(
                [output["embeddings"] for output in self.training_step_outputs]
            )
            self.log_embedding_umap(embeddings, tag="val")

        self.validation_step_outputs = []

    def configure_optimizers(self) -> torch.optim.AdamW:
        """Configure optimizer for contrastive learning.

        Returns
        -------
        torch.optim.AdamW
            AdamW optimizer with configured learning rate.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(
        self, batch: TripletSample, batch_idx: int, dataloader_idx: int = 0
    ) -> ContrastivePrediction:
        """Prediction step for extracting embeddings."""
        features, projections = self.model(batch["anchor"])
        return {
            "features": features,
            "projections": projections,
            "index": batch["index"],
        }
