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
from viscy.representation.disentanglement_metrics import DisentanglementMetrics
from viscy.representation.vae import BetaVae25D, BetaVaeMonai
from viscy.representation.vae_logging import BetaVaeLogger
from viscy.utils.log_images import detach_sample, render_images

_logger = logging.getLogger("lightning.pytorch")

_VAE_ARCHITECTURE = {
    "2.5D": BetaVae25D,
    "monai_beta": BetaVaeMonai,
}

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
        log_embeddings: bool = True,
        embedding_log_frequency: int = 20,
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
        self.embedding_log_frequency = embedding_log_frequency

    def on_train_start(self) -> None:
        """Log comprehensive hyperparameters including model architecture details."""
        super().on_train_start()
        
        # Collect comprehensive hyperparameters
        hparams = {
            # Training hyperparameters
            "lr": self.lr,
            "schedule": self.schedule,
            "input_shape": self.example_input_array, 
            "loss_function_class": self.loss_function.__class__.__name__,
        }
        
        # Add loss function specific parameters
        if hasattr(self.loss_function, 'margin'):
            hparams["loss_margin"] = self.loss_function.margin
        if hasattr(self.loss_function, 'temperature'):
            hparams["loss_temperature"] = self.loss_function.temperature
        if hasattr(self.loss_function, 'normalize_embeddings'):
            hparams["loss_normalize_embeddings"] = self.loss_function.normalize_embeddings
            
        # Add encoder details if it's a ContrastiveEncoder
        if hasattr(self.model, 'backbone'):
            hparams["encoder_backbone"] = self.model.backbone
        if hasattr(self.model, 'in_channels'):
            hparams["encoder_in_channels"] = self.model.in_channels
        if hasattr(self.model, 'in_stack_depth'):
            hparams["encoder_in_stack_depth"] = self.model.in_stack_depth
        if hasattr(self.model, 'embedding_dim'):
            hparams["encoder_embedding_dim"] = self.model.embedding_dim
        if hasattr(self.model, 'projection_dim'):
            hparams["encoder_projection_dim"] = self.model.projection_dim
        if hasattr(self.model, 'drop_path_rate'):
            hparams["encoder_drop_path_rate"] = self.model.drop_path_rate
        if hasattr(self.model, 'stem_kernel_size'):
            hparams["encoder_stem_kernel_size"] = str(self.model.stem_kernel_size)
        if hasattr(self.model, 'stem_stride'):
            hparams["encoder_stem_stride"] = str(self.model.stem_stride)
            
        # Log to TensorBoard
        if self.logger is not None:
            self.logger.log_hyperparams(hparams)

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
        
        # Log UMAP embeddings from validation set every N epochs
        if (
            self.log_embeddings 
            and self.current_epoch % self.embedding_log_frequency == 0 
            and self.current_epoch > 0
        ):
            self._collect_and_log_embeddings()

        self.validation_step_outputs = []

    def _collect_and_log_embeddings(self):
        """Collect embeddings from validation dataloader and log UMAP visualization."""
        try:
            # Get validation dataloader
            val_dataloaders = self.trainer.val_dataloaders
            if val_dataloaders is None:
                _logger.warning("No validation dataloader available for embedding logging")
                return
            elif isinstance(val_dataloaders, list):
                val_dataloader = val_dataloaders[0] if val_dataloaders else None
            else:
                val_dataloader = val_dataloaders

            if val_dataloader is None:
                _logger.warning("No validation dataloader available for embedding logging")
                return

            _logger.info(f"Collecting embeddings for visualization at epoch {self.current_epoch}")
            
            # Collect embeddings, images, and metadata from validation set
            embeddings_list = []
            images_list = []
            labels_list = []
            max_samples = 500  # Reduced for memory efficiency with images
            sample_count = 0
            
            self.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    if sample_count >= max_samples:
                        break
                    
                    # Move batch to device
                    anchor = batch["anchor"].to(self.device)
                    batch_size = anchor.size(0)
                    
                    # Get embeddings (features, not projections)
                    features, _ = self(anchor)
                    embeddings_list.append(features.cpu())
                    
                    # Collect images for sprite visualization
                    # Take middle slice for 3D data and first channel if multi-channel
                    if anchor.ndim == 5:  # (B, C, D, H, W)
                        mid_z = anchor.size(2) // 2
                        img_slice = anchor[:, 0, mid_z].cpu()  # (B, H, W)
                    else:  # (B, C, H, W)
                        img_slice = anchor[:, 0].cpu()  # (B, H, W)
                    images_list.append(img_slice)
                    
                    # Collect labels from index information
                    if "index" in batch and batch["index"] is not None:
                        for i, idx_info in enumerate(batch["index"][:batch_size]):
                            if isinstance(idx_info, dict):
                                # Create label from track_id and time info
                                track_id = idx_info.get("track_id", "unknown")
                                t = idx_info.get("t", "unknown")
                                labels_list.append(f"track_{track_id}_t_{t}")
                            else:
                                labels_list.append(f"sample_{sample_count + i}")
                    else:
                        # Fallback labels
                        for i in range(batch_size):
                            labels_list.append(f"sample_{sample_count + i}")
                    
                    sample_count += batch_size
                    
            if embeddings_list:
                embeddings = torch.cat(embeddings_list, dim=0)[:max_samples]
                images = torch.cat(images_list, dim=0)[:max_samples]
                labels = labels_list[:max_samples]
                
                # Normalize images for visualization (0-1 range)
                images = (images - images.min()) / (images.max() - images.min() + 1e-8)
                
                # Log UMAP visualization  
                self.log_embedding_umap(embeddings, tag="validation")
                
                # Log to TensorBoard's embedding projector with images and labels
                self.logger.experiment.add_embedding(
                    embeddings,
                    metadata=labels,
                    label_img=images.unsqueeze(1),  # Add channel dimension
                    global_step=self.current_epoch,
                    tag="validation_embeddings"
                )
                
                _logger.info(f"Logged {len(embeddings)} embeddings with images and labels")
            else:
                _logger.warning("No embeddings collected from validation set")
                
        except Exception as e:
            _logger.error(f"Error collecting embeddings: {e}")

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
        architecture: Literal["monai_beta","2.5D"],
        model_config: dict = {},
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
        compute_disentanglement: bool = True,
        disentanglement_frequency: int = 10,
        log_enhanced_visualizations: bool = False,
        log_enhanced_visualizations_frequency: int = 30,
    ):
        super().__init__()

        net_class= _VAE_ARCHITECTURE.get(architecture)
        if not net_class:
            raise ValueError(
                f"Architecture {architecture} not in {_VAE_ARCHITECTURE.keys()}"
            )

        self.model = net_class(**model_config)
        self.model_config = model_config
        self.architecture = architecture
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
        self.compute_disentanglement = compute_disentanglement
        self.disentanglement_frequency = disentanglement_frequency
        
        self.log_enhanced_visualizations = log_enhanced_visualizations
        self.log_enhanced_visualizations_frequency = (
            log_enhanced_visualizations_frequency
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self._min_beta = 1e-15
        self._logvar_minmax = (-20,20)

        # Handle different parameter names for latent dimensions
        latent_dim = None
        if "latent_dim" in self.model_config:
            latent_dim = self.model_config["latent_dim"]
        elif "latent_size" in self.model_config:
            latent_dim = self.model_config["latent_size"]
            
        if latent_dim is not None:
            self.vae_logger = BetaVaeLogger(latent_dim=latent_dim)
        else:
            _logger.warning("No latent dimension provided for BetaVaeLogger. Using default with 128 dimensions.")
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

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, x: Tensor) -> dict:
        """Forward pass through Beta-VAE."""

        original_shape = x.shape
        is_monai_2d = (self.architecture == "monai_beta" and 
                      self.model_config.get("spatial_dims") == 2)
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
        x_original = x if not (is_monai_2d and len(original_shape) == 5 and original_shape[2] == 1) else x.unsqueeze(2)
        recon_loss = self.loss_function(recon_x, x_original)
        if isinstance(self.loss_function, nn.MSELoss):
            if hasattr(self.loss_function, 'reduction') and self.loss_function.reduction == 'sum':
                recon_loss = recon_loss / batch_size

        kl_loss = -0.5 * torch.sum(1 + torch.clamp(logvar,self._logvar_minmax[0],self._logvar_minmax[1]) - mu.pow(2) - logvar.exp(), dim=1)
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

        # Compute disentanglement metrics periodically
        if (
            self.compute_disentanglement
            and self.current_epoch % self.disentanglement_frequency == 0
            and self.current_epoch > 0
        ):
            self._compute_and_log_disentanglement_metrics()

        if (
            self.log_enhanced_visualizations
            and self.current_epoch % self.log_enhanced_visualizations_frequency == 0
            and self.current_epoch > 0
        ):
            self._log_enhanced_visualizations()

    def _compute_and_log_disentanglement_metrics(self):
        """Compute and log disentanglement metrics."""
        try:
            # Get validation dataloader - handle both single DataLoader and list cases
            val_dataloaders = self.trainer.val_dataloaders
            if val_dataloaders is None:
                val_dataloader = None
            elif isinstance(val_dataloaders, list):
                val_dataloader = val_dataloaders[0] if val_dataloaders else None
            else:
                val_dataloader = val_dataloaders

            if val_dataloader is None:
                _logger.warning(
                    "No validation dataloader available for disentanglement metrics"
                )
                return

            # Use the logger's disentanglement metrics method
            self.vae_logger.log_disentanglement_metrics(
                lightning_module=self,
                dataloader=val_dataloader,
                max_samples=200,
            )

        except Exception as e:
            _logger.error(f"Error computing disentanglement metrics: {e}")

    def _log_enhanced_visualizations(self):
        """Log enhanced β-VAE visualizations."""
        try:
            # Get validation dataloader - handle both single DataLoader and list cases
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

            # Log latent traversals -for how recons change when moving along a latent dim
            self.vae_logger.log_latent_traversal(
                lightning_module=self, n_dims=8, n_steps=11
            )

            # Log latent interpolations - smooth transitions between different data points in the latent space
            self.vae_logger.log_latent_interpolation(
                lightning_module=self, n_pairs=3, n_steps=11
            )

            # Log factor traversal matrix - grid visualization how each dim affects the recon
            self.vae_logger.log_factor_traversal_matrix(
                lightning_module=self, n_dims=8, n_steps=7
            )

            # Log latent space visualization (every 40 epochs to avoid overhead)
            if self.current_epoch % 40 == 0:
                self.vae_logger.log_latent_space_visualization(
                    lightning_module=self,
                    dataloader=val_dataloader,
                    max_samples=500,
                    method="pca",
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
