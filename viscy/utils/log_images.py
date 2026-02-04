"""Logging example images during training."""

from typing import Sequence

import numpy as np
from matplotlib.pyplot import get_cmap
from skimage.exposure import rescale_intensity
from torch import Tensor


def _detect_logger_type(logger) -> str:
    """Detect logger type (TensorBoard, WandB, or unknown).

    Parameters
    ----------
    logger : Logger
        PyTorch Lightning logger instance.

    Returns
    -------
    str
        Logger type: "tensorboard", "wandb", or "unknown".
    """
    if logger is None:
        return "none"

    try:
        from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

        if isinstance(logger, TensorBoardLogger):
            return "tensorboard"
        elif isinstance(logger, WandbLogger):
            return "wandb"
    except ImportError:
        pass

    return "unknown"


def _log_to_tensorboard(
    logger,
    key: str,
    samples: Sequence[Sequence[np.ndarray]],
    step: int,
    cmaps: list[str],
):
    """Log images to TensorBoard.

    Parameters
    ----------
    logger : TensorBoardLogger
        TensorBoard logger instance.
    key : str
        Logging key/tag for the images.
    samples : Sequence[Sequence[np.ndarray]]
        Grid of samples - rows are samples, columns are channels.
    step : int
        Training step/epoch for logging.
    cmaps : list[str]
        Colormaps for each column.
    """
    grid = render_images(samples, cmaps=cmaps)
    logger.experiment.add_image(key, grid, step, dataformats="HWC")


def _log_to_wandb(
    logger,
    key: str,
    samples: Sequence[Sequence[np.ndarray]],
    step: int,
    cmaps: list[str],
):
    """Log images to Weights & Biases.

    Parameters
    ----------
    logger : WandbLogger
        WandB logger instance.
    key : str
        Logging key/tag for the images.
    samples : Sequence[Sequence[np.ndarray]]
        Grid of samples - rows are samples, columns are channels.
    step : int
        Training step/epoch for logging.
    cmaps : list[str]
        Colormaps for each column.
    """
    try:
        import wandb

        # Create grid using existing render_images function
        grid = render_images(samples, cmaps=cmaps)

        # Wrap grid in wandb.Image and log
        logger.log_image(key=key, images=[wandb.Image(grid)], step=step)
    except ImportError:
        # wandb not installed, skip logging
        pass


def log_image_samples(
    logger,
    key: str,
    samples: Sequence[Sequence[np.ndarray]],
    step: int,
    cmaps: list[str] = [],
):
    """Log image samples to both TensorBoard and WandB loggers.

    Unified interface that detects logger type and routes to appropriate backend.
    Supports both single loggers and multiple loggers simultaneously.

    Parameters
    ----------
    logger : Logger or list[Logger]
        PyTorch Lightning logger instance(s).
        Supports TensorBoardLogger, WandbLogger, or list of loggers.
    key : str
        Logging key/tag for the images.
    samples : Sequence[Sequence[np.ndarray]]
        Grid of samples from detach_sample().
        Rows are samples, columns are channels.
    step : int
        Training step/epoch for logging.
    cmaps : list[str], optional
        Colormaps for each column.
        Default uses "gray" for first column, "inferno" for others.

    Notes
    -----
    - For TensorBoard: Creates single concatenated grid (existing behavior)
    - For WandB: Creates grid view for overview compatibility
    - Gracefully handles None logger or unknown logger types
    - For multi-logger setups, logs to all compatible loggers

    Examples
    --------
    >>> # Single logger
    >>> log_image_samples(wandb_logger, "train_samples", samples, epoch, ["gray"]*3)
    >>>
    >>> # Multiple loggers
    >>> loggers = [TensorBoardLogger(...), WandbLogger(...)]
    >>> log_image_samples(loggers, "train_samples", samples, epoch, ["gray"]*3)
    """
    # Handle multiple loggers
    if isinstance(logger, list):
        for single_logger in logger:
            log_image_samples(single_logger, key, samples, step, cmaps)
        return

    # Detect logger type and route appropriately
    logger_type = _detect_logger_type(logger)

    if logger_type == "tensorboard":
        _log_to_tensorboard(logger, key, samples, step, cmaps)
    elif logger_type == "wandb":
        _log_to_wandb(logger, key, samples, step, cmaps)
    elif logger_type == "none":
        # No logger, skip
        pass
    else:
        # Unknown logger type, try TensorBoard approach as fallback
        try:
            _log_to_tensorboard(logger, key, samples, step, cmaps)
        except Exception:
            # If fallback fails, skip logging
            pass


def detach_sample(
    imgs: Sequence[Tensor], log_samples_per_batch: int
) -> list[list[np.ndarray]]:
    """Detach example images from the batch and convert them to numpy arrays.

    Parameters
    ----------
    imgs : Sequence[Tensor]
        Sequence of example images.
    log_samples_per_batch : int
        Number of first N samples in the sequence to detach.

    Returns
    -------
    list[list[np.ndarray]]
        Grid of example images.
        Rows are samples, columns are channels.
    """
    num_samples = min(imgs[0].shape[0], log_samples_per_batch)
    samples = []
    for i in range(num_samples):
        patches = []
        for img in imgs:
            patch = img[i].detach().cpu().numpy()
            patch = np.squeeze(patch[:, patch.shape[1] // 2])
            patches.append(patch)
        samples.append(patches)
    return samples


def render_images(
    imgs: Sequence[Sequence[np.ndarray]], cmaps: list[str] = []
) -> np.ndarray:
    """Render images in a grid.

    Parameters
    ----------
    imgs : Sequence[Sequence[np.ndarray]]
        Grid of images to render, output of `detach_sample`.
    cmaps : list[str], optional
        Colormaps for each column, by default []

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
                cm_name = cmaps[i]
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
