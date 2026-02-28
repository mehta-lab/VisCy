"""Enhanced logging utilities for Beta-VAE training with TensorBoard."""

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torchvision.utils import make_grid

_logger = logging.getLogger("lightning.pytorch")


class BetaVaeLogger:
    """Enhanced logging utilities for Beta-VAE training.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    """

    def __init__(self, latent_dim: int = 128):
        self.latent_dim = latent_dim
        self.device = None

    def setup(self, device: str):
        """Initialize device-dependent components."""
        self.device = device

    def log_enhanced_metrics(self, lightning_module, model_output: dict, batch: dict, stage: str = "train"):
        """Log enhanced Beta-VAE metrics."""
        x = batch["anchor"]
        z = model_output["z"]
        recon_x = model_output["recon_x"]
        recon_loss = model_output["recon_loss"]
        kl_loss = model_output["kl_loss"]
        total_loss = model_output["total_loss"]

        beta = getattr(
            lightning_module,
            "_get_current_beta",
            lambda: getattr(lightning_module, "beta", 1.0),
        )()

        grad_diagnostics = self._compute_gradient_diagnostics(lightning_module)
        nan_inf_diagnostics = self._check_nan_inf(recon_x, x, z)

        metrics = {
            f"loss/{stage}/total": total_loss,
            f"loss/{stage}/reconstruction": recon_loss,
            f"loss/{stage}/kl": kl_loss,
            f"beta/{stage}": beta,
        }

        metrics.update(grad_diagnostics)
        metrics.update(nan_inf_diagnostics)

        latent_mean = torch.mean(z, dim=0)
        latent_std = torch.std(z, dim=0)

        active_dims = torch.sum(torch.var(z, dim=0) > 0.01)
        variances = torch.var(z, dim=0)
        effective_dim = torch.sum(variances) ** 2 / torch.sum(variances**2)

        metrics.update(
            {
                f"latent_statistics/mean_avg/{stage}": torch.mean(latent_mean),
                f"latent_statistics/std_avg/{stage}": torch.mean(latent_std),
                f"latent_statistics/mean_max/{stage}": torch.max(latent_mean),
                f"latent_statistics/std_max/{stage}": torch.max(latent_std),
                f"latent_statistics/active_dims/{stage}": active_dims.float(),
                f"latent_statistics/effective_dim/{stage}": effective_dim,
                f"latent_statistics/utilization/{stage}": active_dims / self.latent_dim,
            }
        )

        lightning_module.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        if stage == "val" and lightning_module.current_epoch % 10 == 0:
            self._log_latent_histograms(lightning_module, z, stage)

    def _compute_gradient_diagnostics(self, lightning_module):
        grad_diagnostics = {}
        encoder_grad_norm = 0.0
        decoder_grad_norm = 0.0
        encoder_param_norm = 0.0
        decoder_param_norm = 0.0

        for name, param in lightning_module.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                if "encoder" in name:
                    encoder_grad_norm += param_norm.item() ** 2
                elif "decoder" in name:
                    decoder_grad_norm += param_norm.item() ** 2

            if "encoder" in name:
                encoder_param_norm += param.data.norm(2).item() ** 2
            elif "decoder" in name:
                decoder_param_norm += param.data.norm(2).item() ** 2

        grad_diagnostics.update(
            {
                "diagnostics/encoder_grad_norm": encoder_grad_norm**0.5,
                "diagnostics/decoder_grad_norm": decoder_grad_norm**0.5,
                "diagnostics/encoder_param_norm": encoder_param_norm**0.5,
                "diagnostics/decoder_param_norm": decoder_param_norm**0.5,
            }
        )
        return grad_diagnostics

    def _check_nan_inf(self, recon_x, x, z):
        diagnostics = {
            "diagnostics/recon_has_nan": torch.isnan(recon_x).any().float(),
            "diagnostics/recon_has_inf": torch.isinf(recon_x).any().float(),
            "diagnostics/input_has_nan": torch.isnan(x).any().float(),
            "diagnostics/latent_has_nan": torch.isnan(z).any().float(),
            "diagnostics/recon_max_val": torch.max(torch.abs(recon_x)),
            "diagnostics/recon_min_val": torch.min(recon_x),
        }
        return diagnostics

    def _log_latent_histograms(self, lightning_module, z: torch.Tensor, stage: str):
        z_np = z.detach().cpu().numpy()
        n_dims_to_log = min(16, z_np.shape[1])
        for i in range(n_dims_to_log):
            lightning_module.logger.experiment.add_histogram(
                f"latent_distributions/dim_{i}_{stage}",
                z_np[:, i],
                lightning_module.current_epoch,
            )

    def _get_decoder(self, lightning_module):
        """Resolve decoder from model hierarchy."""
        if hasattr(lightning_module.model, "decoder"):
            return lightning_module.model.decoder
        _logger.warning("No decoder found in model, skipping visualization.")
        return None

    def log_latent_traversal(
        self,
        lightning_module,
        n_dims: int = 8,
        n_steps: int = 11,
        range_vals: Tuple[float, float] = (-3, 3),
    ):
        """Log latent space traversal visualizations."""
        if not hasattr(lightning_module, "model"):
            return

        decoder = self._get_decoder(lightning_module)
        if decoder is None:
            return

        lightning_module.model.eval()

        with torch.no_grad():
            z_base = torch.randn(1, self.latent_dim, device=lightning_module.device)

            for dim in range(min(n_dims, self.latent_dim)):
                traversal_images = []
                for val in np.linspace(range_vals[0], range_vals[1], n_steps):
                    z_modified = z_base.clone()
                    z_modified[0, dim] = val
                    recon = decoder(z_modified)
                    mid_z = recon.shape[2] // 2
                    img_2d = recon[0, 0, mid_z].cpu()
                    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
                    traversal_images.append(img_2d)

                grid = make_grid(
                    torch.stack(traversal_images).unsqueeze(1),
                    nrow=n_steps,
                    normalize=True,
                )
                lightning_module.logger.experiment.add_image(
                    f"latent_traversal/dim_{dim}",
                    grid,
                    lightning_module.current_epoch,
                    dataformats="CHW",
                )

    def log_latent_interpolation(self, lightning_module, n_pairs: int = 3, n_steps: int = 11):
        """Log latent space interpolation between random pairs."""
        if not hasattr(lightning_module, "model"):
            return

        decoder = self._get_decoder(lightning_module)
        if decoder is None:
            return

        lightning_module.model.eval()

        with torch.no_grad():
            for pair_idx in range(n_pairs):
                z1 = torch.randn(1, self.latent_dim, device=lightning_module.device)
                z2 = torch.randn(1, self.latent_dim, device=lightning_module.device)

                interp_images = []
                for alpha in np.linspace(0, 1, n_steps):
                    z_interp = alpha * z1 + (1 - alpha) * z2
                    recon = decoder(z_interp)
                    mid_z = recon.shape[2] // 2
                    img_2d = recon[0, 0, mid_z].cpu()
                    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
                    interp_images.append(img_2d)

                grid = make_grid(
                    torch.stack(interp_images).unsqueeze(1),
                    nrow=n_steps,
                    normalize=True,
                )
                lightning_module.logger.experiment.add_image(
                    f"latent_interpolation/pair_{pair_idx}",
                    grid,
                    lightning_module.current_epoch,
                    dataformats="CHW",
                )

    def log_factor_traversal_matrix(self, lightning_module, n_dims: int = 8, n_steps: int = 7):
        """Log factor traversal matrix."""
        if not hasattr(lightning_module, "model"):
            return

        decoder = self._get_decoder(lightning_module)
        if decoder is None:
            return

        lightning_module.model.eval()

        with torch.no_grad():
            z_base = torch.randn(1, self.latent_dim, device=lightning_module.device)
            matrix_rows = []

            for dim in range(min(n_dims, self.latent_dim)):
                row_images = []
                for step in range(n_steps):
                    val = -3 + 6 * step / (n_steps - 1)
                    z_mod = z_base.clone()
                    z_mod[0, dim] = val
                    recon = decoder(z_mod)
                    mid_z = recon.shape[2] // 2
                    img_2d = recon[0, 0, mid_z].cpu()
                    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
                    row_images.append(img_2d)
                matrix_rows.append(torch.stack(row_images))

            all_images = torch.cat(matrix_rows, dim=0)
            grid = make_grid(all_images.unsqueeze(1), nrow=n_steps, normalize=True)
            lightning_module.logger.experiment.add_image(
                "factor_traversal_matrix",
                grid,
                lightning_module.current_epoch,
                dataformats="CHW",
            )

    def log_beta_schedule(self, lightning_module, beta_schedule: Optional[Callable] = None):
        """Log beta annealing schedule."""
        if beta_schedule is None:
            max_epochs = lightning_module.trainer.max_epochs
            epoch = lightning_module.current_epoch
            if epoch < max_epochs * 0.1:
                beta = 0.1
            elif epoch < max_epochs * 0.5:
                beta = 0.1 + (4.0 - 0.1) * (epoch - max_epochs * 0.1) / (max_epochs * 0.4)
            else:
                beta = 4.0
        else:
            beta = beta_schedule(lightning_module.current_epoch)

        lightning_module.log("beta_schedule", beta)
        return beta
