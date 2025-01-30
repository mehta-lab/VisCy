from typing import List

import torch
import torchvision
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class GradCAMCallback(Callback):
    """Callback for computing and logging GradCAM visualizations.

    Parameters
    ----------
    visual_datasets : list
        List of datasets to generate visualizations from
    every_n_epochs : int, default=10
        Generate visualizations every n epochs
    max_samples : int, default=5
        Maximum number of samples to visualize per dataset
    max_height : int, default=720
        Maximum height of output visualization
    """

    def __init__(
        self,
        visual_datasets: List,
        every_n_epochs: int = 10,
        max_samples: int = 5,
        max_height: int = 720,
    ):
        super().__init__()
        self.visual_datasets = visual_datasets
        self.every_n_epochs = every_n_epochs
        self.max_samples = max_samples
        self.max_height = max_height

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Generate and log GradCAM visualizations"""
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        pl_module.eval()

        for dataset_idx, dataset in enumerate(self.visual_datasets):
            # Get a few samples
            samples = []
            cams = []

            for i, (x, _) in enumerate(dataset):
                if i >= self.max_samples:
                    break

                # Move tensor to same device as model
                x = x.to(pl_module.device)

                # Generate GradCAM - no need to add batch dimension here since gradcam() does it
                cam = pl_module.gradcam(x)

                # Convert to RGB images for visualization
                x_img = self._tensor_to_img(
                    x.cpu()
                )  # removed [0] since x is already unbatched
                cam_img = self._tensor_to_img(torch.from_numpy(cam))
                overlay = self._create_overlay(x_img, cam_img)

                samples.append(x_img)
                cams.append(overlay)

            # Stack images for grid visualization
            samples_grid = torchvision.utils.make_grid(
                [torch.from_numpy(img) for img in samples], nrow=len(samples)
            )
            cams_grid = torchvision.utils.make_grid(
                [torch.from_numpy(img) for img in cams], nrow=len(cams)
            )

            # Log to tensorboard
            trainer.logger.experiment.add_image(
                f"gradcam/dataset_{dataset_idx}/samples",
                samples_grid,
                trainer.current_epoch,
            )
            trainer.logger.experiment.add_image(
                f"gradcam/dataset_{dataset_idx}/cams",
                cams_grid,
                trainer.current_epoch,
            )

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
