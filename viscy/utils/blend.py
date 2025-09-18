import numpy as np
from cmap import Colormap
from skimage.exposure import rescale_intensity


def blend_channels(
    image: np.ndarray, cmaps: list[Colormap], rescale: bool
) -> np.ndarray:
    """Blend multi-channel images using specified colormaps.

    Parameters
    ----------
    image : np.ndarray
        Multi-channel image array to blend.
    cmaps : list[Colormap]
        List of colormaps for each channel.
    rescale : bool
        Whether to rescale intensity values to [0, 1] range.

    Returns
    -------
    np.ndarray
        Blended RGB image clipped to [0, 1] range.
    """
    rendered_channels = []
    for channel, cmap in zip(image, cmaps):
        colormap = Colormap(cmap)
        if rescale:
            channel = rescale_intensity(channel, in_range="image", out_range=(0, 1))
        rendered_channels.append(colormap(channel))
    return np.sum(rendered_channels, axis=0).clip(0, 1)
