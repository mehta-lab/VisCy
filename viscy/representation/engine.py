import logging
from typing import Literal, Sequence, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from pytorch_metric_learning.losses import NTXentLoss
from torch import Tensor, nn

from viscy.data.typing import TrackingIndex, TripletSample
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.utils.log_images import detach_sample, render_images

_logger = logging.getLogger("lightning.pytorch")


# TODO: log the embeddings every other epoch? expose a variable to control this
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
            nn.Module | nn.CosineEmbeddingLoss | nn.TripletMarginLoss | NTXentLoss
        ) = nn.TripletMarginLoss(margin=0.5),
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        embedding_log_interval: int = 1,  # Log embeddings every N epochs
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
        self.embedding_log_interval = embedding_log_interval

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

    def _log_step_samples(self, batch_idx, samples, stage: Literal["train", "val"]):
        """Common method for logging step samples"""
        if batch_idx < self.log_batches_per_epoch:
            output_list = (
                self.training_step_outputs
                if stage == "train"
                else self.validation_step_outputs
            )
            output_list.extend(detach_sample(samples, self.log_samples_per_batch))

    def log_embedding_tensorboard(
        self,
        embeddings: Tensor,
        images: Tensor,
        metadata: Sequence[list],
        tag: str,
        metadata_header: Sequence[str],
        global_step: int = 0,
    ):
        """Log embeddings with their corresponding images and metadata to TensorBoard Embedding Projector

        Args:
            embeddings: Tensor of embeddings to visualize
            images: Corresponding images for the embeddings (B, C, D, H, W) or (B, C, H, W)
                    where D is the depth dimension
            metadata: List of list with the metadata for each embedding
            tag: Name tag for the embedding visualization
            metadata_header: List of strings with the header for each metadata column
            global_step: Current training step
        """
        _logger.debug(
            f"Logging embeddings to TensorBoard Embedding Projector for {tag}"
        )
        # Store original embeddings tensor for norm calculations
        embeddings_tensor = embeddings.detach()
        # Convert to numpy only for visualization
        embeddings_numpy = embeddings_tensor.cpu().numpy()
        # Take middle slice of 3D images for visualization
        images = images.detach().cpu()
        if images.ndim == 5:  # (B, C, D, H, W)
            middle_d = images.shape[2] // 2
            images = images[:, :, middle_d]  # Now (B, C, H, W)

        # Handle different channel configurations
        if images.shape[1] > 1:
            # Create a list to store normalized channels
            normalized_channels = []
            for ch in range(images.shape[1]):
                # Convert single channel to grayscale
                ch_images = images[:, ch : ch + 1]
                # Normalize each channel independently
                ch_images = (ch_images - ch_images.min()) / (
                    ch_images.max() - ch_images.min()
                )
                normalized_channels.append(ch_images)

            # Combine channels - using first channel for red, second for green, rest averaged for blue
            combined_images = torch.zeros(
                images.shape[0], 3, images.shape[2], images.shape[3]
            )
            combined_images[:, 0] = normalized_channels[0].squeeze(1)  # Red channel
            combined_images[:, 1] = (
                normalized_channels[1].squeeze(1)
                if len(normalized_channels) > 1
                else normalized_channels[0].squeeze(1)
            )  # Green channel
            if len(normalized_channels) > 2:
                combined_images[:, 2] = (
                    torch.stack(normalized_channels[2:]).mean(dim=0).squeeze(1)
                )  # Blue channel - average of remaining channels
            else:
                combined_images[:, 2] = normalized_channels[0].squeeze(1)
        else:
            # For single channel, repeat to create grayscale
            combined_images = images.repeat(1, 3, 1, 1)
            combined_images = (combined_images - combined_images.min()) / (
                combined_images.max() - combined_images.min()
            )

        # Log a single embedding visualization with the combined image
        self.logger.experiment.add_embedding(
            embeddings_numpy,
            metadata=metadata,
            label_img=combined_images,
            global_step=global_step,
            tag=tag,
            metadata_header=metadata_header,
        )

        # Log statistics using the original tensor
        self.log(
            f"{tag}/mean_norm",
            torch.norm(embeddings_tensor, dim=1).mean(),
            on_epoch=True,
        )
        self.log(
            f"{tag}/std_norm",
            torch.norm(embeddings_tensor, dim=1).std(),
            on_epoch=True,
        )

    def _format_metadata(self, index: TrackingIndex | None) -> str:
        """Format tracking index into a metadata string."""
        if index is None:
            return "unknown"
        return f"track_{index.get('track_id', 'unknown')}:fov_{index.get('fov', 'unknown')}"

    def training_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        anchor_img = batch["anchor"]
        pos_img = batch["positive"]
        anchor_projection = self(anchor_img)
        positive_projection = self(pos_img)
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
            negative_projection = self(neg_img)
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
        super().on_train_epoch_end()
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def _prepare_embedding_metadata(
        self,
        anchor_metadata: dict,
        positive_metadata: dict | None = None,
        negative_metadata: dict | None = None,
        include_positive: bool = False,
        include_negative: bool = False,
    ) -> tuple[list[list[str]], list[str]]:
        """Prepare metadata for embedding visualization.

        Args:
            anchor_metadata: Metadata for anchor samples
            positive_metadata: Metadata for positive samples (optional)
            negative_metadata: Metadata for negative samples (optional)
            include_positive: Whether to include positive samples in metadata
            include_negative: Whether to include negative samples in metadata

        Returns:
            tuple containing:
                - metadata: List of lists containing metadata values
                - metadata_header: List of metadata field names
        """
        metadata_header = ["fov_name", "t", "id", "type"]

        def process_field(x, field):
            if field in ["t", "id"] and isinstance(x, torch.Tensor):
                return str(x.detach().cpu().item())
            return str(x)

        # Create lists for each metadata field
        metadata = [
            [str(x) for x in anchor_metadata["fov_name"]],
            [process_field(x, "t") for x in anchor_metadata["t"]],
            [process_field(x, "id") for x in anchor_metadata["id"]],
            ["anchor"] * len(anchor_metadata["fov_name"]),  # type field for anchors
        ]

        # If including positive samples, extend metadata
        if include_positive and positive_metadata is not None:
            for i, field in enumerate(metadata_header[:-1]):  # Exclude 'type' field
                metadata[i].extend(
                    [process_field(x, field) for x in positive_metadata[field]]
                )
            # Add 'positive' type for positive samples
            metadata[-1].extend(["positive"] * len(positive_metadata["fov_name"]))

        # If including negative samples, extend metadata
        if include_negative and negative_metadata is not None:
            for i, field in enumerate(metadata_header[:-1]):  # Exclude 'type' field
                metadata[i].extend(
                    [process_field(x, field) for x in negative_metadata[field]]
                )
            # Add 'negative' type for negative samples
            metadata[-1].extend(["negative"] * len(negative_metadata["fov_name"]))

        return metadata, metadata_header

    def validation_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        """Validation step of the model."""
        anchor = batch["anchor"]
        pos_img = batch["positive"]
        anchor_projection = self(anchor)
        positive_projection = self(pos_img)
        negative_projection = None

        if isinstance(self.loss_function, NTXentLoss):
            batch_size = anchor.size(0)
            indices = torch.arange(0, batch_size, device=anchor_projection.device)
            labels = torch.cat((indices, indices))
            # Note: we assume the two augmented views are the anchor and positive samples
            embeddings = torch.cat((anchor_projection, positive_projection))
            loss = self.loss_function(embeddings, labels)
            self._log_step_samples(batch_idx, (anchor, pos_img), "val")

            # Store embeddings for visualization
            if self.current_epoch % self.embedding_log_interval == 0 and batch_idx == 0:
                # Must include positive samples since we're concatenating embeddings
                metadata, metadata_header = self._prepare_embedding_metadata(
                    batch["anchor_metadata"],
                    batch["positive_metadata"],
                    include_positive=True,  # Required since we concatenate embeddings
                )
                self.val_embedding_outputs = {
                    "embeddings": embeddings.detach(),
                    "images": torch.cat((anchor, pos_img)).detach(),
                    "metadata": list(zip(*metadata)),
                    "metadata_header": metadata_header,
                }
        else:
            neg_img = batch["negative"]
            negative_projection = self(neg_img)
            loss = self.loss_function(
                anchor_projection, positive_projection, negative_projection
            )
            self._log_step_samples(batch_idx, (anchor, pos_img, neg_img), "val")

            # Store embeddings for visualization
            if self.current_epoch % self.embedding_log_interval == 0 and batch_idx == 0:
                metadata, metadata_header = self._prepare_embedding_metadata(
                    batch["anchor_metadata"],
                    batch["positive_metadata"],
                    batch["negative_metadata"],
                    include_positive=True,  # Required since we concatenate embeddings
                    include_negative=True,
                )
                self.val_embedding_outputs = {
                    "embeddings": torch.cat(
                        (anchor_projection, positive_projection, negative_projection)
                    ).detach(),
                    "images": torch.cat((anchor, pos_img, neg_img)).detach(),
                    "metadata": list(zip(*metadata)),
                    "metadata_header": metadata_header,
                }

        self._log_metrics(
            loss=loss,
            anchor=anchor_projection,
            positive=positive_projection,
            negative=negative_projection,
            stage="val",
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)

        # Log embeddings for validation on interval epochs
        if hasattr(self, "val_embedding_outputs"):
            self.log_embedding_tensorboard(
                self.val_embedding_outputs["embeddings"],
                self.val_embedding_outputs["images"],
                self.val_embedding_outputs["metadata"],
                tag="embeddings",
                metadata_header=self.val_embedding_outputs["metadata_header"],
                global_step=self.current_epoch,
            )
            delattr(self, "val_embedding_outputs")
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
