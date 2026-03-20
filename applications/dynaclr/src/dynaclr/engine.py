"""ContrastiveModule and BetaVaeModule LightningModules for DynaCLR."""

import logging
from typing import Literal, Sequence, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from pytorch_metric_learning.losses import NTXentLoss
from torch import Tensor, nn

from viscy_data._typing import CellIndex, TripletSample
from viscy_models.contrastive import ContrastiveEncoder
from viscy_models.vae import BetaVae25D, BetaVaeMonai
from viscy_utils.log_embeddings import pca_pairplot
from viscy_utils.log_images import detach_sample, log_chw_tensor, log_histogram, log_image_grid

_logger = logging.getLogger("lightning.pytorch")


class ContrastivePrediction(TypedDict):
    """Output type for contrastive prediction step."""

    features: Tensor
    projections: Tensor
    index: list[CellIndex]


class ContrastiveModule(LightningModule):
    """Contrastive Learning Model for self-supervised learning."""

    def __init__(
        self,
        encoder: nn.Module | ContrastiveEncoder,
        loss_function: (nn.Module | nn.CosineEmbeddingLoss | nn.TripletMarginLoss | NTXentLoss) = nn.TripletMarginLoss(
            margin=0.5
        ),
        lr: float = 1e-3,
        schedule: Literal["WarmupCosine", "Constant"] = "Constant",
        log_batches_per_epoch: int = 8,
        log_samples_per_batch: int = 1,
        log_embeddings_every_n_epochs: int | None = 10,
        pca_color_key: str | None = "condition",
        log_negative_metrics_every_n_epochs: int = 2,
        example_input_array_shape: Sequence[int] = (1, 2, 15, 256, 256),
        ckpt_path: str | None = None,
        freeze_backbone: bool = False,
        projection: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = encoder
        if projection is not None:
            self.model.projection = projection
        self.loss_function = loss_function
        self.lr = lr
        self.schedule = schedule
        self.log_batches_per_epoch = log_batches_per_epoch
        self.log_samples_per_batch = log_samples_per_batch
        self.example_input_array = torch.rand(*example_input_array_shape)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.log_embeddings_every_n_epochs = log_embeddings_every_n_epochs
        self.pca_color_key = pca_color_key
        self.log_negative_metrics_every_n_epochs = log_negative_metrics_every_n_epochs
        self._embedding_outputs: list[tuple[Tensor, list]] = []
        self.freeze_backbone = freeze_backbone

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, weights_only=True)["state_dict"])

    def on_fit_start(self) -> None:  # noqa: D102
        if self.freeze_backbone:
            for param in self.model.stem.parameters():
                param.requires_grad = False
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return both features and projections."""
        return self.model(x)

    def log_feature_statistics(self, embeddings: Tensor, prefix: str):
        """Log feature statistics for debugging."""
        mean = torch.mean(embeddings, dim=0).detach().cpu().numpy()
        std = torch.std(embeddings, dim=0).detach().cpu().numpy()
        _logger.debug(f"{prefix}_mean: {mean}")
        _logger.debug(f"{prefix}_std: {std}")

    def print_embedding_norms(self, anchor, positive, negative, phase):
        """Log embedding norms for debugging."""
        anchor_norm = torch.norm(anchor, dim=1).mean().item()
        positive_norm = torch.norm(positive, dim=1).mean().item()
        negative_norm = torch.norm(negative, dim=1).mean().item()
        _logger.debug(f"{phase}/anchor_norm: {anchor_norm}")
        _logger.debug(f"{phase}/positive_norm: {positive_norm}")
        _logger.debug(f"{phase}/negative_norm: {negative_norm}")

    def _log_metrics(self, loss, anchor, positive, stage: Literal["train", "val"], negative=None):
        self.log(
            f"loss/{stage}",
            loss.to(self.device),
            on_step=(stage == "train"),
            on_epoch=(stage == "val"),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=anchor.size(0),
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
            log_metric_dict[f"metrics/cosine_similarity_negative/{stage}"] = cosine_sim_neg
            log_metric_dict[f"metrics/euclidean_distance_negative/{stage}"] = euclidean_dist_neg
        elif isinstance(self.loss_function, NTXentLoss):
            if self.current_epoch % self.log_negative_metrics_every_n_epochs == 0:
                batch_size = anchor.size(0)
                anchor_norm = F.normalize(anchor, dim=1)
                positive_norm = F.normalize(positive, dim=1)
                all_embeddings_norm = torch.cat([anchor_norm, positive_norm], dim=0)
                sim_matrix = torch.mm(anchor_norm, all_embeddings_norm.t())

                mask = torch.ones_like(sim_matrix, dtype=torch.bool)
                mask[range(batch_size), range(batch_size)] = False
                mask[range(batch_size), range(batch_size, 2 * batch_size)] = False

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
        if self.trainer.is_global_zero and self.logger is not None:
            log_image_grid(self.logger, key, imgs, self.current_epoch, cmaps=["gray"] * 3)

    def _log_step_samples(self, batch_idx, samples: tuple, stage: Literal["train", "val"]):
        if batch_idx < self.log_batches_per_epoch:
            output_list = self.training_step_outputs if stage == "train" else self.validation_step_outputs
            output_list.extend(detach_sample(samples, self.log_samples_per_batch))

    def log_embedding_pca(self, embeddings: Tensor, meta: list[dict], tag: str, n_components: int = 8):
        """Compute PCA and log a pairplot colored by condition to WandB."""
        import matplotlib.pyplot as plt
        from lightning.pytorch.loggers import WandbLogger

        if not self.trainer.is_global_zero:
            return
        if not isinstance(self.logger, WandbLogger):
            return

        import wandb

        embeddings_np = embeddings.detach().cpu().numpy()
        _logger.debug(f"Computing PCA for {tag}: {len(embeddings_np)} embeddings, {len(meta)} meta dicts.")
        if len(meta) != len(embeddings_np):
            _logger.warning("PCA meta/embedding count mismatch: %d vs %d.", len(meta), len(embeddings_np))

        fig = pca_pairplot(
            embeddings_np,
            meta,
            color_key=self.pca_color_key,
            n_components=n_components,
            title=f"{tag} PCA pairplot — epoch {self.current_epoch}",
        )
        self.logger.experiment.log({f"{tag}_pca": wandb.Image(fig), "epoch": self.current_epoch})
        plt.close(fig)

    def training_step(self, batch: TripletSample, batch_idx: int) -> Tensor:  # noqa: D102
        anchor_img = batch["anchor"]
        pos_img = batch["positive"]
        _, anchor_projection = self(anchor_img)
        _, positive_projection = self(pos_img)
        negative_projection = None
        if isinstance(self.loss_function, NTXentLoss):
            indices = torch.arange(0, anchor_projection.size(0), device=anchor_projection.device)
            labels = torch.cat((indices, indices))
            embeddings = torch.cat((anchor_projection, positive_projection))
            loss = self.loss_function(embeddings, labels)
            self._log_step_samples(batch_idx, (anchor_img, pos_img), "train")
        else:
            neg_img = batch["negative"]
            _, negative_projection = self(neg_img)
            loss = self.loss_function(anchor_projection, positive_projection, negative_projection)
            self._log_step_samples(batch_idx, (anchor_img, pos_img, neg_img), "train")
        self._log_metrics(
            loss=loss,
            anchor=anchor_projection,
            positive=positive_projection,
            negative=negative_projection,
            stage="train",
        )
        return loss

    def on_train_epoch_end(self) -> None:  # noqa: D102
        super().on_train_epoch_end()
        self._log_samples("train_samples", self.training_step_outputs)
        self.training_step_outputs = []

    def validation_step(self, batch: TripletSample, batch_idx: int) -> Tensor:  # noqa: D102
        anchor = batch["anchor"]
        pos_img = batch["positive"]
        anchor_features, anchor_projection = self(anchor)
        _, positive_projection = self(pos_img)
        negative_projection = None
        if isinstance(self.loss_function, NTXentLoss):
            indices = torch.arange(0, anchor_projection.size(0), device=anchor_projection.device)
            labels = torch.cat((indices, indices))
            embeddings = torch.cat((anchor_projection, positive_projection))
            loss = self.loss_function(embeddings, labels)
            self._log_step_samples(batch_idx, (anchor, pos_img), "val")
        else:
            neg_img = batch["negative"]
            _, negative_projection = self(neg_img)
            loss = self.loss_function(anchor_projection, positive_projection, negative_projection)
            self._log_step_samples(batch_idx, (anchor, pos_img, neg_img), "val")
        self._log_metrics(
            loss=loss,
            anchor=anchor_projection,
            positive=positive_projection,
            negative=negative_projection,
            stage="val",
        )
        n = self.log_embeddings_every_n_epochs
        if n is not None and self.current_epoch % n == 0 and not self.trainer.sanity_checking:
            self._embedding_outputs.append((anchor_features.detach().cpu(), batch.get("anchor_meta", [])))
        return loss

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        super().on_validation_epoch_end()
        self._log_samples("val_samples", self.validation_step_outputs)
        self.validation_step_outputs = []
        if self._embedding_outputs:
            all_embeddings = torch.cat([e for e, _ in self._embedding_outputs])
            all_meta = [m for _, ms in self._embedding_outputs for m in ms]
            self.log_embedding_pca(all_embeddings, all_meta, "val")
            self._embedding_outputs = []

    def configure_optimizers(self):  # noqa: D102
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch: TripletSample, batch_idx, dataloader_idx=0) -> ContrastivePrediction:
        """Extract embeddings from anchor images."""
        features, projections = self.model(batch["anchor"])
        return {
            "features": features,
            "projections": projections,
            "index": batch["index"],
        }


class BetaVaeModule(LightningModule):
    """Beta-VAE LightningModule with KL annealing and scheduled beta."""

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
        self.log_enhanced_visualizations_frequency = log_enhanced_visualizations_frequency
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self._min_beta = 1e-15
        self._logvar_minmax = (-20, 20)

        if hasattr(self.model, "latent_dim"):
            self._latent_dim = self.model.latent_dim
        elif hasattr(self.model, "latent_size"):
            self._latent_dim = self.model.latent_size
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "latent_dim"):
            self._latent_dim = self.model.encoder.latent_dim
        else:
            _logger.warning("Could not infer latent_dim from model; defaulting to 128.")
            self._latent_dim = 128

    def _get_current_beta(self) -> float:
        """Get current beta value based on scheduling."""
        if self.beta_schedule is None:
            return max(self.beta, self._min_beta)

        epoch = self.current_epoch

        if self.beta_schedule == "linear":
            if epoch < self.beta_warmup_epochs:
                beta_val = self.beta_min + (self.beta - self.beta_min) * epoch / self.beta_warmup_epochs
                return max(beta_val, self._min_beta)
            else:
                return max(self.beta, self._min_beta)

        elif self.beta_schedule == "cosine":
            if epoch < self.beta_warmup_epochs:
                import math

                progress = epoch / self.beta_warmup_epochs
                beta_val = self.beta_min + (self.beta - self.beta_min) * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
                return max(beta_val, self._min_beta)
            else:
                return max(self.beta, self._min_beta)

        elif self.beta_schedule == "warmup":
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

        model_output = self.model(x)
        recon_x = model_output.recon_x
        mu = model_output.mean
        logvar = model_output.logvar
        z = model_output.z

        if is_monai_2d and len(original_shape) == 5 and original_shape[2] == 1:
            recon_x = recon_x.unsqueeze(2)

        current_beta = self._get_current_beta()
        batch_size = original_shape[0]

        x_original = x if not (is_monai_2d and len(original_shape) == 5 and original_shape[2] == 1) else x.unsqueeze(2)
        recon_loss = self.loss_function(recon_x, x_original)
        if isinstance(self.loss_function, nn.MSELoss):
            if hasattr(self.loss_function, "reduction") and self.loss_function.reduction == "sum":
                recon_loss = recon_loss / batch_size
            elif hasattr(self.loss_function, "reduction") and self.loss_function.reduction == "mean":
                num_elements_per_image = x_original[0].numel()
                recon_loss = recon_loss * num_elements_per_image

        kl_loss = -0.5 * torch.sum(
            1 + torch.clamp(logvar, self._logvar_minmax[0], self._logvar_minmax[1]) - mu.pow(2) - logvar.exp(),
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

    def _log_metrics(self, model_output: dict, x: Tensor, stage: str) -> None:
        z = model_output["z"]
        variances = torch.var(z, dim=0)
        active_dims = torch.sum(variances > 0.01)
        effective_dim = torch.sum(variances) ** 2 / torch.sum(variances**2)

        encoder_grad_norm = (
            sum(
                p.grad.data.norm(2).item() ** 2
                for n, p in self.named_parameters()
                if "encoder" in n and p.grad is not None
            )
            ** 0.5
        )
        decoder_grad_norm = (
            sum(
                p.grad.data.norm(2).item() ** 2
                for n, p in self.named_parameters()
                if "decoder" in n and p.grad is not None
            )
            ** 0.5
        )

        self.log_dict(
            {
                f"loss/{stage}/total": model_output["total_loss"],
                f"loss/{stage}/reconstruction": model_output["recon_loss"],
                f"loss/{stage}/kl": model_output["kl_loss"],
                f"beta/{stage}": self._get_current_beta(),
                f"latent/active_dims/{stage}": active_dims.float(),
                f"latent/effective_dim/{stage}": effective_dim,
                f"latent/utilization/{stage}": active_dims / self._latent_dim,
                "diagnostics/encoder_grad_norm": encoder_grad_norm,
                "diagnostics/decoder_grad_norm": decoder_grad_norm,
                "diagnostics/recon_has_nan": torch.isnan(model_output["recon_x"]).any().float(),
                "diagnostics/input_has_nan": torch.isnan(x).any().float(),
                "diagnostics/latent_has_nan": torch.isnan(z).any().float(),
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        if stage == "val" and self.current_epoch % 10 == 0 and self.trainer.is_global_zero:
            z_np = z.detach().cpu().numpy()
            for i in range(min(16, z_np.shape[1])):
                log_histogram(self.logger, f"latent_distributions/dim_{i}_{stage}", z_np[:, i], self.current_epoch)

    def training_step(self, batch: TripletSample, batch_idx: int) -> Tensor:  # noqa: D102
        x = batch["anchor"]
        model_output = self(x)
        self._log_metrics(model_output, x, "train")
        self._log_step_samples(batch_idx, x, model_output["recon_x"], "train")
        return model_output["total_loss"]

    def validation_step(self, batch: TripletSample, batch_idx: int) -> Tensor:  # noqa: D102
        x = batch["anchor"]
        model_output = self(x)
        self._log_metrics(model_output, x, "val")
        self._log_step_samples(batch_idx, x, model_output["recon_x"], "val")
        return model_output["total_loss"]

    def _log_step_samples(self, batch_idx, original, reconstruction, stage: Literal["train", "val"]):
        if batch_idx < self.log_batches_per_epoch:
            output_list = self.training_step_outputs if stage == "train" else self.validation_step_outputs
            samples = {
                "original": original.detach().cpu()[: self.log_samples_per_batch],
                "reconstruction": reconstruction.detach().cpu()[: self.log_samples_per_batch],
            }
            output_list.append(samples)

    def _log_samples(self, key: str, samples_list: list):
        if not self.trainer.is_global_zero or len(samples_list) == 0:
            return
        mid_z = samples_list[0]["original"].shape[2] // 2
        originals = []
        reconstructions = []
        for sample in samples_list:
            orig = sample["original"][:, :, mid_z].numpy()
            recon = sample["reconstruction"][:, :, mid_z].numpy()
            originals.extend([orig[i] for i in range(orig.shape[0])])
            reconstructions.extend([recon[i] for i in range(recon.shape[0])])
        combined = [[o, r] for o, r in zip(originals[:4], reconstructions[:4])]
        log_image_grid(self.logger, key, combined, self.current_epoch, cmaps=["gray", "gray"])

    def on_train_epoch_end(self) -> None:  # noqa: D102
        super().on_train_epoch_end()
        self._log_samples("train_reconstructions", self.training_step_outputs)
        self.training_step_outputs = []

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        super().on_validation_epoch_end()
        self._log_samples("val_reconstructions", self.validation_step_outputs)
        self.validation_step_outputs = []

        if (
            self.trainer.is_global_zero
            and self.log_enhanced_visualizations
            and self.current_epoch % self.log_enhanced_visualizations_frequency == 0
            and self.current_epoch > 0
        ):
            self._log_enhanced_visualizations()

    def _get_decoder(self) -> nn.Module | None:
        if hasattr(self.model, "decoder"):
            return self.model.decoder
        _logger.warning("No decoder found in model, skipping visualization.")
        return None

    def _log_enhanced_visualizations(self):
        from torchvision.utils import make_grid

        decoder = self._get_decoder()
        if decoder is None:
            return

        _logger.info(f"Logging enhanced visualizations at epoch {self.current_epoch}")
        self.model.eval()
        with torch.no_grad():
            # Latent traversal: vary each dim independently
            z_base = torch.randn(1, self._latent_dim, device=self.device)
            for dim in range(min(8, self._latent_dim)):
                imgs = []
                for val in np.linspace(-3, 3, 11):
                    z = z_base.clone()
                    z[0, dim] = val
                    recon = decoder(z)
                    mid = recon.shape[2] // 2
                    img = recon[0, 0, mid].cpu()
                    imgs.append((img - img.min()) / (img.max() - img.min() + 1e-8))
                grid = make_grid(torch.stack(imgs).unsqueeze(1), nrow=11, normalize=True)
                log_chw_tensor(self.logger, f"latent_traversal/dim_{dim}", grid, self.current_epoch)

            # Latent interpolation: interpolate between random pairs
            for pair_idx in range(3):
                z1 = torch.randn(1, self._latent_dim, device=self.device)
                z2 = torch.randn(1, self._latent_dim, device=self.device)
                imgs = []
                for alpha in np.linspace(0, 1, 11):
                    recon = decoder(alpha * z1 + (1 - alpha) * z2)
                    mid = recon.shape[2] // 2
                    img = recon[0, 0, mid].cpu()
                    imgs.append((img - img.min()) / (img.max() - img.min() + 1e-8))
                grid = make_grid(torch.stack(imgs).unsqueeze(1), nrow=11, normalize=True)
                log_chw_tensor(self.logger, f"latent_interpolation/pair_{pair_idx}", grid, self.current_epoch)

            # Factor traversal matrix: all dims × steps in one grid
            z_base = torch.randn(1, self._latent_dim, device=self.device)
            rows = []
            for dim in range(min(8, self._latent_dim)):
                row = []
                for step in range(7):
                    val = -3 + 6 * step / 6
                    z = z_base.clone()
                    z[0, dim] = val
                    recon = decoder(z)
                    mid = recon.shape[2] // 2
                    img = recon[0, 0, mid].cpu()
                    row.append((img - img.min()) / (img.max() - img.min() + 1e-8))
                rows.append(torch.stack(row))
            grid = make_grid(torch.cat(rows).unsqueeze(1), nrow=7, normalize=True)
            log_chw_tensor(self.logger, "factor_traversal_matrix", grid, self.current_epoch)

    def configure_optimizers(self):  # noqa: D102
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch: TripletSample, batch_idx, dataloader_idx=0) -> dict:  # noqa: D102
        x = batch["anchor"]
        model_output = self(x)
        return {
            "latent": model_output["z"],
            "reconstruction": model_output["recon_x"],
            "index": batch["index"],
        }
