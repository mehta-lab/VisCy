import numpy as np
from cmap import Colormap
from skimage.exposure import rescale_intensity


def blend_channels(
    image: np.ndarray, cmaps: list[Colormap], rescale: bool
) -> np.ndarray:
    rendered_channels = []
    for channel, cmap in zip(image, cmaps):
        colormap = Colormap(cmap)
        if rescale:
            channel = rescale_intensity(channel, in_range="image", out_range=(0, 1))
        rendered_channels.append(colormap(channel))
    return np.sum(rendered_channels, axis=0).clip(0, 1)
