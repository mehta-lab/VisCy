"""Logging example images during training."""

from typing import TYPE_CHECKING, Sequence

import numpy as np
from matplotlib.pyplot import get_cmap
from skimage.exposure import rescale_intensity
from torch import Tensor

from viscy_utils.tensor_utils import to_numpy

if TYPE_CHECKING:
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


def detach_sample(imgs: Sequence[Tensor], log_samples_per_batch: int) -> list[list[np.ndarray]]:
    """Extract middle-Z slices from a batch for image grid logging.

    Layout: one row per sample, columns ordered as
    ``anchor_ch0, anchor_ch1, ..., positive_ch0, positive_ch1, ..., negative_ch0, ...``
    Channels expand horizontally within each view, which suits landscape monitors.

    Parameters
    ----------
    imgs : Sequence[Tensor]
        One ``(B, C, Z, Y, X)`` tensor per view (anchor, positive, negative).
    log_samples_per_batch : int
        Number of samples from the batch to include (first N).

    Returns
    -------
    list[list[np.ndarray]]
        Grid of 2-D ``(H, W)`` arrays. Rows are samples, columns are
        ``(view, channel)`` pairs in view-major order.
    """
    num_samples = min(imgs[0].shape[0], log_samples_per_batch)
    n_channels = imgs[0].shape[1]
    rows = []
    for i in range(num_samples):
        row = []
        for img in imgs:
            patch = to_numpy(img[i])
            mid_z = patch.shape[1] // 2
            for c in range(n_channels):
                row.append(patch[c, mid_z])
        rows.append(row)
    return rows


def render_images(imgs: Sequence[Sequence[np.ndarray]], cmaps: list[str] = []) -> np.ndarray:
    """Render images in a grid.

    Parameters
    ----------
    imgs : Sequence[Sequence[np.ndarray]]
        Grid of images to render, output of ``detach_sample``.
    cmaps : list[str], optional
        Colormaps for each column, by default [].

    Returns
    -------
    np.ndarray
        Rendered RGB images grid.
    """
    images_grid = []
    for sample_images in imgs:
        images_row = []
        for i, image in enumerate(sample_images):
            if cmaps:
                cm_name = cmaps[i % len(cmaps)]
            else:
                cm_name = "gray" if i == 0 else "inferno"
            if image.ndim == 2:
                image = image[np.newaxis]
            for channel in image:
                channel = rescale_intensity(channel, out_range=(0, 1))
                render = get_cmap(cm_name)(channel, bytes=True)[..., :3]
                images_row.append(render)
        images_grid.append(np.concatenate(images_row, axis=1))
    return np.concatenate(images_grid, axis=0)


def log_chw_tensor(
    logger: "TensorBoardLogger | WandbLogger",
    key: str,
    grid: "Tensor",
    step: int,
) -> None:
    """Log a CHW tensor image to TensorBoard or WandB.

    Parameters
    ----------
    logger : TensorBoardLogger | WandbLogger
        Lightning logger instance.
    key : str
        Tag name for the image.
    grid : Tensor
        CHW image tensor (e.g., output of ``torchvision.utils.make_grid``).
    step : int
        Global step or epoch number.
    """
    from lightning.pytorch.loggers import WandbLogger

    if isinstance(logger, WandbLogger):
        import wandb

        img_np = to_numpy(grid.permute(1, 2, 0))
        logger.experiment.log({key: wandb.Image(img_np), "epoch": step})
    else:
        logger.experiment.add_image(key, grid, step, dataformats="CHW")


def log_histogram(
    logger: "TensorBoardLogger | WandbLogger",
    key: str,
    values: "np.ndarray",
    step: int,
) -> None:
    """Log a 1-D histogram to TensorBoard or WandB.

    Parameters
    ----------
    logger : TensorBoardLogger | WandbLogger
        Lightning logger instance.
    key : str
        Tag name for the histogram.
    values : np.ndarray
        1-D array of values to histogram.
    step : int
        Global step or epoch number.
    """
    from lightning.pytorch.loggers import WandbLogger

    if isinstance(logger, WandbLogger):
        import wandb

        logger.experiment.log({key: wandb.Histogram(values), "epoch": step})
    else:
        logger.experiment.add_histogram(key, values, step)


def log_image_grid(
    logger: "TensorBoardLogger | WandbLogger",
    key: str,
    imgs: Sequence[Sequence[np.ndarray]],
    step: int,
    cmaps: list[str] = [],
) -> None:
    """Log a grid of images to TensorBoard or WandB.

    Parameters
    ----------
    logger : TensorBoardLogger | WandbLogger
        Lightning logger instance.
    key : str
        Tag name for the image grid.
    imgs : Sequence[Sequence[np.ndarray]]
        Grid of images, output of ``detach_sample``.
        Rows are samples, columns are views/channels.
    step : int
        Global step or epoch number.
    cmaps : list[str], optional
        Colormaps for each column, by default [].
    """
    from lightning.pytorch.loggers import WandbLogger

    grid = render_images(imgs, cmaps=cmaps)
    if isinstance(logger, WandbLogger):
        import wandb

        logger.experiment.log({key: wandb.Image(grid), "epoch": step})
    else:
        logger.experiment.add_image(key, grid, step, dataformats="HWC")
