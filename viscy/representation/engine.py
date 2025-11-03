import logging
from typing import Literal, Sequence, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from pytorch_metric_learning.losses import NTXentLoss
from torch import Tensor, nn
from umap import UMAP

from viscy.data.typing import TrackingIndex, TripletSample
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.representation.vae import BetaVae25D, BetaVaeMonai
from viscy.representation.vae_logging import BetaVaeLogger
from viscy.utils.log_images import detach_sample, render_images

_logger = logging.getLogger("lightning.pytorch")


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
        log_embeddings: bool = False,
        log_negative_metrics_every_n_epochs: int = 2,
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
        self.log_negative_metrics_every_n_epochs = log_negative_metrics_every_n_epochs

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
            f"metrics/cosine_similarity/positive/{stage}": cosine_sim_pos,
            f"metrics/euclidean_distance/positive/{stage}": euclidean_dist_pos,
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
        elif isinstance(self.loss_function, NTXentLoss):
            if self.current_epoch % self.log_negative_metrics_every_n_epochs == 0:
                batch_size = anchor.size(0)

                # Cosine similarity metrics
                anchor_norm = F.normalize(anchor, dim=1)
                positive_norm = F.normalize(positive, dim=1)
                all_embeddings_norm = torch.cat([anchor_norm, positive_norm], dim=0)
                sim_matrix = torch.mm(anchor_norm, all_embeddings_norm.t())

                mask = torch.ones_like(sim_matrix, dtype=torch.bool)
                mask[range(batch_size), range(batch_size)] = False  # Exclude self
                mask[range(batch_size), range(batch_size, 2 * batch_size)] = (
                    False  # Exclude positive
                )

                negative_sims = sim_matrix[mask].view(batch_size, -1)

                mean_neg_sim = negative_sims.mean()
                sum_neg_sim = negative_sims.sum(dim=1).mean()
                margin_cosine = cosine_sim_pos - mean_neg_sim

                all_embeddings = torch.cat([anchor, positive], dim=0)
                dist_matrix = torch.cdist(anchor, all_embeddings, p=2)
                negative_dists = dist_matrix[mask].view(batch_size, -1)

                mean_neg_dist = negative_dists.mean()
                sum_neg_dist = negative_dists.sum(dim=1).mean()
                margin_euclidean = mean_neg_dist - euclidean_dist_pos

                log_metric_dict.update(
                    {
                        f"metrics/cosine_similarity/negative_mean/{stage}": mean_neg_sim,
                        f"metrics/cosine_similarity/negative_sum/{stage}": sum_neg_sim,
                        f"metrics/margin_positive/negative/{stage}": margin_cosine,
                        f"metrics/euclidean_distance/negative_mean/{stage}": mean_neg_dist,
                        f"metrics/euclidean_distance/negative_sum/{stage}": sum_neg_dist,
                        f"metrics/margin_euclidean_positive/negative/{stage}": margin_euclidean,
                    }
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

    def log_embedding_umap(self, embeddings: Tensor, tag: str):
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
        super().on_train_epoch_end()
        self._log_samples("train_samples", self.training_step_outputs)
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


class BetaVaeModule(LightningModule):
    def __init__(
        self,
        vae: nn.Module | BetaVae25D | BetaVaeMonai,
        loss_function: nn.Module | nn.MSELoss = nn.MSELoss(reduction="sum"),
        beta: float = 1.0,
        beta_schedule: Literal["linear", "cosine", "warmup"] | None = None,
        beta_min: float = 0.1,
        beta_warmup_epochs: int = 50,
        lr: float = 1e-5,
        lr_schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        example_input_array_shape: Sequence[int] = (1, 2, 30, 256, 256),
        log_enhanced_visualizations: bool = False,
        log_enhanced_visualizations_frequency: int = 30,
    ):
        super().__init__()

        self.model = vae
        self.loss_function = loss_function

        self.beta = beta
        self.beta_schedule = beta_schedule
        self.beta_min = beta_min
        self.beta_warmup_epochs = beta_warmup_epochs

        self.lr = lr
        self.lr_schedule = lr_schedule

        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch

        self.example_input_array = torch.rand(*example_input_array_shape)

        self.log_enhanced_visualizations = log_enhanced_visualizations
        self.log_enhanced_visualizations_frequency = (
            log_enhanced_visualizations_frequency
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self._min_beta = 1e-15
        self._logvar_minmax = (-20, 20)

        # Handle different parameter names for latent dimensions
        latent_dim = None
        if hasattr(self.model, "latent_dim"):
            latent_dim = self.model.latent_dim
        elif hasattr(self.model, "latent_size"):
            latent_dim = self.model.latent_size
        elif hasattr(self.model, "encoder") and hasattr(
            self.model.encoder, "latent_dim"
        ):
            latent_dim = self.model.encoder.latent_dim

        if latent_dim is not None:
            self.vae_logger = BetaVaeLogger(latent_dim=latent_dim)
        else:
            _logger.warning(
                "No latent dimension provided for BetaVaeLogger. Using default with 128 dimensions."
            )
            self.vae_logger = BetaVaeLogger()

    def setup(self, stage: str = None):
        """Setup hook to initialize device-dependent components."""
        super().setup(stage)

        # Initialize the VAE logger with proper device
        self.vae_logger.setup(device=self.device)

    def _get_current_beta(self) -> float:
        """Get current beta value based on scheduling."""
        if self.beta_schedule is None:
            return max(self.beta, self._min_beta)

        epoch = self.current_epoch

        if self.beta_schedule == "linear":
            # Linear warmup from beta_min to beta
            if epoch < self.beta_warmup_epochs:
                beta_val = (
                    self.beta_min
                    + (self.beta - self.beta_min) * epoch / self.beta_warmup_epochs
                )
                return max(beta_val, self._min_beta)
            else:
                return max(self.beta, self._min_beta)

        elif self.beta_schedule == "cosine":
            # Cosine warmup from beta_min to beta
            if epoch < self.beta_warmup_epochs:
                import math

                progress = epoch / self.beta_warmup_epochs
                beta_val = self.beta_min + (self.beta - self.beta_min) * 0.5 * (
                    1 + math.cos(math.pi * (1 - progress))
                )
                return max(beta_val, self._min_beta)
            else:
                return max(self.beta, self._min_beta)

        elif self.beta_schedule == "warmup":
            # Keep beta_min for warmup epochs, then jump to beta
            beta_val = self.beta_min if epoch < self.beta_warmup_epochs else self.beta
            return max(beta_val, self._min_beta)

        else:
            return max(self.beta, self._min_beta)

    def forward(self, x: Tensor) -> dict:
        """Forward pass through Beta-VAE."""

        original_shape = x.shape
        is_monai_2d = (
            isinstance(self.model, BetaVaeMonai)
            and hasattr(self.model, "spatial_dims")
            and self.model.spatial_dims == 2
        )
        if is_monai_2d and len(x.shape) == 5 and x.shape[2] == 1:
            x = x.squeeze(2)

        # Handle different model output formats
        model_output = self.model(x)

        recon_x = model_output.recon_x
        mu = model_output.mean
        logvar = model_output.logvar
        z = model_output.z

        if is_monai_2d and len(original_shape) == 5 and original_shape[2] == 1:
            # Convert back (B, C, H, W) to (B, C, 1, H, W)
            recon_x = recon_x.unsqueeze(2)

        current_beta = self._get_current_beta()
        batch_size = original_shape[0]

        # Use original input for loss computation to ensure shape consistency
        x_original = (
            x
            if not (is_monai_2d and len(original_shape) == 5 and original_shape[2] == 1)
            else x.unsqueeze(2)
        )
        recon_loss = self.loss_function(recon_x, x_original)
        if isinstance(self.loss_function, nn.MSELoss):
            if (
                hasattr(self.loss_function, "reduction")
                and self.loss_function.reduction == "sum"
            ):
                recon_loss = recon_loss / batch_size
            elif (
                hasattr(self.loss_function, "reduction")
                and self.loss_function.reduction == "mean"
            ):
                # Correct the over-normalization by PyTorch's mean reduction by multiplying by the number of elements per image
                num_elements_per_image = x_original[0].numel()
                recon_loss = recon_loss * num_elements_per_image

        kl_loss = -0.5 * torch.sum(
            1
            + torch.clamp(logvar, self._logvar_minmax[0], self._logvar_minmax[1])
            - mu.pow(2)
            - logvar.exp(),
            dim=1,
        )
        kl_loss = torch.mean(kl_loss)

        total_loss = recon_loss + current_beta * kl_loss

        return {
            "recon_x": recon_x,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }

    def training_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        """Training step with VAE loss computation."""

        x = batch["anchor"]
        model_output = self(x)
        loss = model_output["total_loss"]

        # Log enhanced β-VAE metrics
        self.vae_logger.log_enhanced_metrics(
            lightning_module=self, model_output=model_output, batch=batch, stage="train"
        )
        # Log samples
        self._log_step_samples(batch_idx, x, model_output["recon_x"], "train")

        return loss

    def validation_step(self, batch: TripletSample, batch_idx: int) -> Tensor:
        """Validation step with VAE loss computation."""
        x = batch["anchor"]
        model_output = self(x)
        loss = model_output["total_loss"]

        # Log enhanced β-VAE metrics
        self.vae_logger.log_enhanced_metrics(
            lightning_module=self, model_output=model_output, batch=batch, stage="val"
        )

        # Log samples
        self._log_step_samples(batch_idx, x, model_output["recon_x"], "val")

        return loss

    def _log_step_samples(
        self, batch_idx, original, reconstruction, stage: Literal["train", "val"]
    ):
        """Log sample reconstructions."""
        if batch_idx < self.log_batches_per_epoch:
            output_list = (
                self.training_step_outputs
                if stage == "train"
                else self.validation_step_outputs
            )

            # Store samples for epoch end logging
            samples = {
                "original": original.detach().cpu()[: self.log_samples_per_batch],
                "reconstruction": reconstruction.detach().cpu()[
                    : self.log_samples_per_batch
                ],
            }
            output_list.append(samples)

    def _log_samples(self, key: str, samples_list: list):
        """Log reconstruction samples at epoch end."""
        if len(samples_list) > 0:
            # Take middle z-slice for visualization
            mid_z = samples_list[0]["original"].shape[2] // 2

            originals = []
            reconstructions = []

            for sample in samples_list:
                orig = sample["original"][:, :, mid_z].numpy()
                recon = sample["reconstruction"][:, :, mid_z].numpy()

                originals.extend([orig[i] for i in range(orig.shape[0])])
                reconstructions.extend([recon[i] for i in range(recon.shape[0])])

            # Create grid with originals and reconstructions
            combined = []
            for orig, recon in zip(originals[:4], reconstructions[:4]):
                combined.append([orig, recon])

            grid = render_images(combined, cmaps=["gray", "gray"])
            self.logger.experiment.add_image(
                key, grid, self.current_epoch, dataformats="HWC"
            )

    def on_train_epoch_end(self) -> None:
        """Log training samples at epoch end."""
        super().on_train_epoch_end()
        self._log_samples("train_reconstructions", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        """Log validation samples at epoch end."""
        super().on_validation_epoch_end()
        self._log_samples("val_reconstructions", self.validation_step_outputs)
        self.validation_step_outputs = []

        if (
            self.log_enhanced_visualizations
            and self.current_epoch % self.log_enhanced_visualizations_frequency == 0
            and self.current_epoch > 0
        ):
            self._log_enhanced_visualizations()

    def _log_enhanced_visualizations(self):
        """Log enhanced β-VAE visualizations."""
        try:
            val_dataloaders = self.trainer.val_dataloaders
            if val_dataloaders is None:
                val_dataloader = None
            elif isinstance(val_dataloaders, list):
                val_dataloader = val_dataloaders[0] if val_dataloaders else None
            else:
                val_dataloader = val_dataloaders

            if val_dataloader is None:
                _logger.warning("No validation dataloader available for visualizations")
                return

            _logger.info(
                f"Logging enhanced β-VAE visualizations at epoch {self.current_epoch}"
            )

            self.vae_logger.log_latent_traversal(
                lightning_module=self, n_dims=8, n_steps=11
            )
            self.vae_logger.log_latent_interpolation(
                lightning_module=self, n_pairs=3, n_steps=11
            )
            self.vae_logger.log_factor_traversal_matrix(
                lightning_module=self, n_dims=8, n_steps=7
            )

        except Exception as e:
            _logger.error(f"Error logging enhanced visualizations: {e}")

    def configure_optimizers(self):
        """Configure optimizer for VAE training."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch: TripletSample, batch_idx, dataloader_idx=0) -> dict:
        """Prediction step for VAE inference."""
        x = batch["anchor"]
        model_output = self(x)

        return {
            "latent": model_output["z"],
            "reconstruction": model_output["recon_x"],
            "index": batch["index"],
        }
