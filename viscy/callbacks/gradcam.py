import logging
from typing import List

import torch
import torchvision
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

logger = logging.getLogger(__name__)


class GradCAMCallback(Callback):
    """Callback for computing and logging GradCAM visualizations.

    Parameters
    ----------
    every_n_epochs : int, default=10
        Generate visualizations every n epochs
    max_samples : int, default=5
        Maximum number of samples to visualize per dataset
    max_height : int, default=720
        Maximum height of output visualization
    mode : str, default="overlay"
        Visualization mode: "separate" for individual images and activations,
        or "overlay" for activation map overlaid on input image
    """

    def __init__(
        self,
        every_n_epochs: int = 10,
        max_samples: int = 5,
        max_height: int = 720,
        mode: str = "overlay",
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.max_samples = max_samples
        self.max_height = max_height
        assert mode in ["separate", "overlay"], "Mode must be 'separate' or 'overlay'"
        self.mode = mode

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Generate and log GradCAM visualizations"""
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        if not hasattr(trainer.datamodule, "visual_dataloader"):
            logger.warning(
                "DataModule does not have visual_dataloader method. Skipping GradCAM visualization."
            )
            return

        pl_module.eval()
        device = pl_module.device

        # Get visual dataloader from the datamodule
        visual_loader = trainer.datamodule.visual_dataloader()

        # Get a few samples
        samples = []
        activations = []

        for batch_idx, (x, _) in enumerate(visual_loader):
            if batch_idx >= self.max_samples:
                break

            try:
                # Move tensor to same device as model
                x = x.to(device)

                # Generate class activation map
                activation_map = pl_module.gradcam(x)

                # Convert to RGB images for visualization
                # Handle 5D tensor [B, T, C, H, W] -> take first batch and timepoint
                x_img = x[0, 0].cpu().numpy()  # Take first batch and timepoint
                if x_img.ndim == 3:  # Handle [C, H, W] case
                    x_img = x_img[0]  # Take first channel to get [H, W]
                x_img = rescale_intensity(x_img, in_range="image", out_range=(0, 1))

                # Create activation map visualization
                activation_norm = self._normalize_cam(torch.from_numpy(activation_map))
                activation_rgb = plt.cm.magma(activation_norm.numpy())[..., :3]

                if self.mode == "separate":
                    # Keep sample as grayscale
                    x_vis = (
                        torch.from_numpy(x_img).unsqueeze(0).float()
                    )  # Add channel dim [1, H, W]
                    activation_vis = (
                        torch.from_numpy(activation_rgb).permute(2, 0, 1).float()
                    )  # [3, H, W]
                else:  # overlay mode
                    # Convert input to RGB
                    x_rgb = np.stack([x_img] * 3, axis=-1)  # [H, W, 3]
                    # Create overlay
                    overlay = self._create_overlay(x_rgb, activation_rgb)
                    x_vis = (
                        torch.from_numpy(x_rgb).permute(2, 0, 1).float()
                    )  # [3, H, W]
                    activation_vis = (
                        torch.from_numpy(overlay).permute(2, 0, 1).float()
                    )  # [3, H, W]

                samples.append(x_vis.cpu())  # Ensure on CPU for visualization
                activations.append(
                    activation_vis.cpu()
                )  # Ensure on CPU for visualization

            except Exception as e:
                logger.error(f"Error processing sample {batch_idx}: {str(e)}")
                continue

        if samples:  # Only proceed if we have samples
            try:
                # Stack images for grid visualization
                samples_grid = torchvision.utils.make_grid(
                    samples, nrow=len(samples), normalize=True, value_range=(0, 1)
                )
                activations_grid = torchvision.utils.make_grid(
                    activations,
                    nrow=len(activations),
                    normalize=True,
                    value_range=(0, 1),
                )

                # Log to tensorboard
                trainer.logger.experiment.add_image(
                    f"gradcam/samples",
                    samples_grid,
                    trainer.current_epoch,
                )
                trainer.logger.experiment.add_image(
                    f"gradcam/{'overlays' if self.mode == 'overlay' else 'activations'}",
                    activations_grid,
                    trainer.current_epoch,
                )
            except Exception as e:
                logger.error(f"Error creating visualization grid: {str(e)}")

    @staticmethod
    def _tensor_to_img(tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to normalized image tensor"""
        img = tensor.cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        return img

    @staticmethod
    def _create_overlay(
        img: torch.Tensor, cam: torch.Tensor, alpha: float = 0.5
    ) -> torch.Tensor:
        """Create overlay of image and CAM"""
        return (1 - alpha) * img + alpha * cam

    @staticmethod
    def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
        """Normalize CAM to [0,1]"""
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
