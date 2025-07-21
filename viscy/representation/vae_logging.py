import io
import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.utils import make_grid

from viscy.representation.disentanglement_metrics import DisentanglementMetrics

_logger = logging.getLogger(__name__)


class BetaVaeLogger:
    """
    Enhanced logging utilities for β-VAE training with TensorBoard.

    Provides comprehensive logging of β-VAE specific metrics, visualizations,
    and latent space analysis for microscopy data.
    """

    def __init__(self, latent_dim: int = 128, device: str = "cuda"):
        self.latent_dim = latent_dim
        self.disentanglement_metrics = DisentanglementMetrics(device=device)

    def log_enhanced_metrics(
        self, lightning_module, model_output: dict, batch: dict, stage: str = "train"
    ):
        """
        Log enhanced β-VAE metrics.

        Args:
            lightning_module: Lightning module instance
            model_output: VAE model output
            batch: Input batch
            stage: Training stage ("train" or "val")
        """
        # Extract components
        x = batch["anchor"]
        # Handle both Pythae dict format and object format
        if isinstance(model_output, dict):
            z = model_output["z"]
            recon_x = model_output["recon_x"]
            recon_loss = model_output["recon_loss"]
            kl_loss = model_output["kl_loss"]
        else:
            z = model_output.z if hasattr(model_output, "z") else model_output.embedding
            recon_x = (
                model_output.recon_x
                if hasattr(model_output, "recon_x")
                else model_output.reconstruction
            )
            recon_loss = model_output.recon_loss
            kl_loss = model_output.kl_loss

        # Get β directly from lightning module
        beta = getattr(lightning_module, "beta", 1.0)

        # Record losses and reconstruction quality metrics
        total_loss = recon_loss + beta * kl_loss
        kl_recon_ratio = kl_loss / (recon_loss + 1e-8)

        mse_loss = F.mse_loss(recon_x, x)
        mae_loss = F.l1_loss(recon_x, x)

        metrics = {
            # All losses in one consolidated group
            f"losses/reconstruction/{stage}": recon_loss,
            f"losses/kl/{stage}": kl_loss,
            f"losses/weighted_kl/{stage}": beta * kl_loss,
            f"losses/total/{stage}": total_loss,
            f"losses/mse/{stage}": mse_loss,
            f"losses/mae/{stage}": mae_loss,
            f"losses/beta_value/{stage}": beta,
            f"losses/kl_recon_ratio/{stage}": kl_recon_ratio,
            f"losses/recon_contribution/{stage}": recon_loss / total_loss,
        }

        # Latent space statistics
        latent_mean = torch.mean(z, dim=0)
        latent_std = torch.std(z, dim=0)

        active_dims = torch.sum(torch.var(z, dim=0) > 0.01)
        variances = torch.var(z, dim=0)
        effective_dim = torch.sum(variances) ** 2 / torch.sum(variances**2)

        metrics.update(
            {
                # Consolidated latent statistics
                f"latent_statistics/mean_avg/{stage}": torch.mean(latent_mean),
                f"latent_statistics/std_avg/{stage}": torch.mean(latent_std),
                f"latent_statistics/mean_max/{stage}": torch.max(latent_mean),
                f"latent_statistics/std_max/{stage}": torch.max(latent_std),
                f"latent_statistics/active_dims/{stage}": active_dims,
                f"latent_statistics/effective_dim/{stage}": effective_dim,
                f"latent_statistics/utilization/{stage}": active_dims / self.latent_dim,
            }
        )

        # Log all metrics
        lightning_module.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # Log latent dimension histograms (periodically)
        if stage == "val" and lightning_module.current_epoch % 10 == 0:
            self._log_latent_histograms(lightning_module, z, stage)

    def _log_latent_histograms(self, lightning_module, z: torch.Tensor, stage: str):
        """Log histograms of latent dimensions."""
        z_np = z.detach().cpu().numpy()

        # Log first 16 dimensions to avoid clutter
        n_dims_to_log = min(16, z_np.shape[1])

        for i in range(n_dims_to_log):
            lightning_module.logger.experiment.add_histogram(
                f"latent_distributions/dim_{i}_{stage}",
                z_np[:, i],
                lightning_module.current_epoch,
            )

    def log_latent_traversal(
        self,
        lightning_module,
        n_dims: int = 8,
        n_steps: int = 11,
        range_vals: Tuple[float, float] = (-3, 3),
    ):
        """
        Log latent space traversal visualizations.

        Args:
            lightning_module: Lightning module instance
            n_dims: Number of latent dimensions to traverse
            n_steps: Number of steps in traversal
            range_vals: Range of values to traverse
        """
        if not hasattr(lightning_module, "model"):
            return

        lightning_module.model.eval()

        with torch.no_grad():
            # Sample a base latent vector
            z_base = torch.randn(1, self.latent_dim, device=lightning_module.device)

            # Traverse each dimension
            for dim in range(min(n_dims, self.latent_dim)):
                traversal_images = []

                for val in np.linspace(range_vals[0], range_vals[1], n_steps):
                    z_modified = z_base.clone()
                    z_modified[0, dim] = val

                    # Generate reconstruction using lightning module's decoder
                    recon = lightning_module.decoder(z_modified)

                    # Take middle z-slice for visualization
                    mid_z = recon.shape[2] // 2
                    img_2d = recon[0, 0, mid_z].cpu()  # First channel, middle z-slice

                    # Normalize for visualization
                    img_2d = (img_2d - img_2d.min()) / (
                        img_2d.max() - img_2d.min() + 1e-8
                    )
                    traversal_images.append(img_2d)

                # Create grid
                grid = make_grid(
                    torch.stack(traversal_images).unsqueeze(1),
                    nrow=n_steps,
                    normalize=True,
                )

                lightning_module.logger.experiment.add_image(
                    f"latent_traversal/dim_{dim}", grid, lightning_module.current_epoch
                )

    def log_latent_interpolation(
        self, lightning_module, n_pairs: int = 3, n_steps: int = 11
    ):
        """
        Log latent space interpolation between random pairs.

        Args:
            lightning_module: Lightning module instance
            n_pairs: Number of interpolation pairs
            n_steps: Number of interpolation steps
        """
        if not hasattr(lightning_module, "model"):
            return

        lightning_module.model.eval()

        with torch.no_grad():
            for pair_idx in range(n_pairs):
                # Sample two random latent vectors
                z1 = torch.randn(1, self.latent_dim, device=lightning_module.device)
                z2 = torch.randn(1, self.latent_dim, device=lightning_module.device)

                interp_images = []

                for alpha in np.linspace(0, 1, n_steps):
                    z_interp = alpha * z1 + (1 - alpha) * z2

                    # Generate reconstruction using lightning module's decoder
                    recon = lightning_module.decoder(z_interp)

                    # Take middle z-slice for visualization
                    mid_z = recon.shape[2] // 2
                    img_2d = recon[0, 0, mid_z].cpu()  # First channel, middle z-slice

                    # Normalize for visualization
                    img_2d = (img_2d - img_2d.min()) / (
                        img_2d.max() - img_2d.min() + 1e-8
                    )
                    interp_images.append(img_2d)

                # Create grid
                grid = make_grid(
                    torch.stack(interp_images).unsqueeze(1),
                    nrow=n_steps,
                    normalize=True,
                )

                lightning_module.logger.experiment.add_image(
                    f"latent_interpolation/pair_{pair_idx}",
                    grid,
                    lightning_module.current_epoch,
                )

    def log_factor_traversal_matrix(
        self, lightning_module, n_dims: int = 8, n_steps: int = 7
    ):
        """
        Log factor traversal matrix showing effect of each latent dimension.

        Args:
            lightning_module: Lightning module instance
            n_dims: Number of latent dimensions to show
            n_steps: Number of steps per dimension
        """
        if not hasattr(lightning_module, "model"):
            return

        lightning_module.model.eval()

        with torch.no_grad():
            # Base latent vector
            z_base = torch.randn(1, self.latent_dim, device=lightning_module.device)

            matrix_rows = []

            for dim in range(min(n_dims, self.latent_dim)):
                row_images = []

                for step in range(n_steps):
                    val = -3 + 6 * step / (n_steps - 1)  # Range [-3, 3]
                    z_mod = z_base.clone()
                    z_mod[0, dim] = val

                    # Generate reconstruction using lightning module's decoder
                    recon = lightning_module.decoder(z_mod)

                    # Take middle z-slice for visualization
                    mid_z = recon.shape[2] // 2
                    img_2d = recon[0, 0, mid_z].cpu()  # First channel, middle z-slice

                    # Normalize for visualization
                    img_2d = (img_2d - img_2d.min()) / (
                        img_2d.max() - img_2d.min() + 1e-8
                    )
                    row_images.append(img_2d)

                matrix_rows.append(torch.stack(row_images))

            # Create matrix grid
            all_images = torch.cat(matrix_rows, dim=0)
            grid = make_grid(all_images.unsqueeze(1), nrow=n_steps, normalize=True)

            lightning_module.logger.experiment.add_image(
                "factor_traversal_matrix", grid, lightning_module.current_epoch
            )

    def log_latent_space_visualization(
        self, lightning_module, dataloader, max_samples: int = 500, method: str = "pca"
    ):
        """
        Log 2D visualization of latent space using PCA or t-SNE.

        Args:
            lightning_module: Lightning module instance
            dataloader: DataLoader for samples
            max_samples: Maximum samples to visualize
            method: Visualization method ("pca" or "tsne")
        """
        if not hasattr(lightning_module, "model"):
            return

        lightning_module.model.eval()

        # Collect latent representations
        latents = []
        samples_collected = 0

        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= max_samples:
                    break

                x = batch["anchor"].to(lightning_module.device)
                model_output = lightning_module(x)  # Use lightning module forward
                # Handle both Pythae dict format and object format
                if isinstance(model_output, dict):
                    z = model_output["z"]
                else:
                    z = (
                        model_output.z
                        if hasattr(model_output, "z")
                        else model_output.embedding
                    )

                latents.append(z.cpu().numpy())
                samples_collected += x.shape[0]

        if not latents:
            return

        latents = np.vstack(latents)[:max_samples]

        # Apply dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(latents)
            title = f"PCA Latent Space (Variance: {reducer.explained_variance_ratio_.sum():.2f})"
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            reduced = reducer.fit_transform(latents)
            title = "t-SNE Latent Space"
        else:
            _logger.warning(f"Unknown method: {method}")
            return

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=20)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, alpha=0.3)

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)

        # Log to TensorBoard
        img = Image.open(buf)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 255.0

        lightning_module.logger.experiment.add_image(
            f"latent_space_{method}", img_tensor, lightning_module.current_epoch
        )

        plt.close()
        buf.close()

    def log_beta_schedule(
        self, lightning_module, beta_schedule: Optional[callable] = None
    ):
        """
        Log β annealing schedule.

        Args:
            lightning_module: Lightning module instance
            beta_schedule: Function that returns β value for given epoch
        """
        if beta_schedule is None:
            # Default β schedule
            max_epochs = lightning_module.trainer.max_epochs
            epoch = lightning_module.current_epoch

            if epoch < max_epochs * 0.1:  # Warm up
                beta = 0.1
            elif epoch < max_epochs * 0.5:  # Gradual increase
                beta = 0.1 + (4.0 - 0.1) * (epoch - max_epochs * 0.1) / (
                    max_epochs * 0.4
                )
            else:  # Final β
                beta = 4.0
        else:
            beta = beta_schedule(lightning_module.current_epoch)

        lightning_module.log("beta_schedule", beta)
        return beta

    def log_disentanglement_metrics(
        self,
        lightning_module,
        dataloader: torch.utils.data.DataLoader,
        max_samples: int = 500,
    ):
        """
        Log disentanglement metrics to TensorBoard every 10 epochs.

        Args:
            lightning_module: Lightning module instance
            dataloader: DataLoader for evaluation
            max_samples: Maximum samples to use for evaluation
        """
        # Only compute every 10 epochs to save compute
        if lightning_module.current_epoch % 10 != 0:
            return

        _logger.info(
            f"Computing disentanglement metrics at epoch {lightning_module.current_epoch}"
        )

        try:
            # Use the lightning module directly (no separate model attribute after refactoring)
            vae_model = lightning_module

            # Compute all disentanglement metrics
            metrics = self.disentanglement_metrics.compute_all_metrics(
                vae_model=vae_model, dataloader=dataloader, max_samples=max_samples
            )

            # Log metrics with organized naming
            tensorboard_metrics = {}
            for metric_name, value in metrics.items():
                tensorboard_metrics[f"disentanglement_metrics/{metric_name}"] = value

            lightning_module.log_dict(
                tensorboard_metrics,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

            _logger.info(f"Logged disentanglement metrics: {metrics}")

        except Exception as e:
            _logger.warning(f"Failed to compute disentanglement metrics: {e}")
            # Log a placeholder to indicate the attempt
            lightning_module.log(
                "disentanglement_metrics/computation_failed",
                1.0,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
