import io
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.utils import make_grid

_logger = logging.getLogger(__name__)


class BetaVaeLogger:
    """
    Enhanced logging utilities for β-VAE training with TensorBoard.

    Provides comprehensive logging of β-VAE specific metrics, visualizations,
    and latent space analysis for microscopy data.
    """

    def __init__(self, latent_dim: int = 128):
        self.latent_dim = latent_dim

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
            kl_loss = model_output["reg_loss"]  # Pythae uses 'reg_loss' for KL
        else:
            z = model_output.z if hasattr(model_output, "z") else model_output.embedding
            recon_x = (
                model_output.recon_x
                if hasattr(model_output, "recon_x")
                else model_output.reconstruction
            )
            recon_loss = model_output.recon_loss
            kl_loss = model_output.kl_loss

        # Get β from model config
        beta = (
            lightning_module.model.model_config.beta
            if hasattr(lightning_module.model, "model_config")
            else 1.0
        )

        # 1. Core VAE Loss Components (organized in one TensorBoard group)
        total_loss = recon_loss + beta * kl_loss
        kl_recon_ratio = kl_loss / (recon_loss + 1e-8)

        metrics = {
            # Core loss components
            f"loss_components/reconstruction_loss/{stage}": recon_loss,
            f"loss_components/kl_loss/{stage}": kl_loss,
            f"loss_components/weighted_kl_loss/{stage}": beta * kl_loss,
            f"loss_components/total_loss/{stage}": total_loss,
            f"loss_components/beta_value/{stage}": beta,
            
            # Loss analysis ratios
            f"loss_analysis/kl_recon_ratio/{stage}": kl_recon_ratio,
            f"loss_analysis/recon_contribution/{stage}": recon_loss / total_loss,
        }

        # 2. Latent space statistics
        latent_mean = torch.mean(z, dim=0)
        latent_std = torch.std(z, dim=0)

        metrics.update(
            {
                f"latent_stats/mean_avg/{stage}": torch.mean(latent_mean),
                f"latent_stats/std_avg/{stage}": torch.mean(latent_std),
                f"latent_stats/mean_max/{stage}": torch.max(latent_mean),
                f"latent_stats/std_max/{stage}": torch.max(latent_std),
            }
        )

        # 3. Reconstruction quality metrics
        mse_loss = F.mse_loss(recon_x, x)
        mae_loss = F.l1_loss(recon_x, x)

        metrics.update(
            {
                f"reconstruction_quality/mse/{stage}": mse_loss,
                f"reconstruction_quality/mae/{stage}": mae_loss,
            }
        )

        # 4. Latent capacity metrics
        active_dims = torch.sum(torch.var(z, dim=0) > 0.01)
        variances = torch.var(z, dim=0)
        effective_dim = torch.sum(variances) ** 2 / torch.sum(variances**2)

        metrics.update(
            {
                f"latent_capacity/active_dims/{stage}": active_dims,
                f"latent_capacity/effective_dim/{stage}": effective_dim,
                f"latent_capacity/utilization/{stage}": active_dims / self.latent_dim,
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

        # 5. Log latent dimension histograms (periodically)
        if stage == "val" and lightning_module.current_epoch % 10 == 0:
            self._log_latent_histograms(lightning_module, z, stage)

    def _log_latent_histograms(self, lightning_module, z: torch.Tensor, stage: str):
        """Log histograms of latent dimensions."""
        z_np = z.detach().cpu().numpy()

        # Log first 16 dimensions to avoid clutter
        n_dims_to_log = min(16, z_np.shape[1])

        for i in range(n_dims_to_log):
            lightning_module.logger.experiment.add_histogram(
                f"latent_dim_{i}_distribution/{stage}",
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

                    # Generate reconstruction
                    decoder_output = lightning_module.model.decoder(z_modified)
                    # Handle both Pythae dict format and object format
                    if isinstance(decoder_output, dict):
                        recon = decoder_output["reconstruction"]
                    else:
                        recon = (
                            decoder_output.reconstruction
                            if hasattr(decoder_output, "reconstruction")
                            else decoder_output
                        )

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

                    # Generate reconstruction
                    decoder_output = lightning_module.model.decoder(z_interp)
                    # Handle both Pythae dict format and object format
                    if isinstance(decoder_output, dict):
                        recon = decoder_output["reconstruction"]
                    else:
                        recon = (
                            decoder_output.reconstruction
                            if hasattr(decoder_output, "reconstruction")
                            else decoder_output
                        )

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

                    # Generate reconstruction
                    decoder_output = lightning_module.model.decoder(z_mod)
                    # Handle both Pythae dict format and object format
                    if isinstance(decoder_output, dict):
                        recon = decoder_output["reconstruction"]
                    else:
                        recon = (
                            decoder_output.reconstruction
                            if hasattr(decoder_output, "reconstruction")
                            else decoder_output
                        )

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
