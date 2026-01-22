"""Logging example images during training."""

from typing import Sequence

import numpy as np
from matplotlib.pyplot import get_cmap
from skimage.exposure import rescale_intensity
from torch import Tensor


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
