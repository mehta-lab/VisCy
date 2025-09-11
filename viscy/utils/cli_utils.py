"""Command-line interface utilities for data processing and visualization."""

import collections
import os
import re
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader


def unique_tags(directory: str | Path) -> dict[str, int]:
    """Return list of unique nume tags from data directory.

    Parameters
    ----------
    directory : str | Path
        Directory containing '.tif' files.

    Returns
    -------
    dict[str, int]
        Dictionary of unique tags and their counts.

    Notes
    -----
    TODO: Remove, unused and poorly written.
    """
    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]

    tags = collections.defaultdict(lambda: 0)
    for f in files:
        f_name, f_type = f.split(".")[0], f.split(".")[1]
        if f_type == "tif":
            suffixes = re.split("_", f_name)

            unique_tag = suffixes[2] + "_" + suffixes[3] + "_" + suffixes[4]
            tags[unique_tag + "." + f_type] += 1
    return tags


class MultiProcessProgressBar:
    """Progress bar for multi-processed tasks.

    Provides the ability to create & update a single progress bar for multi-depth
    multi-processed tasks by calling updates on a single object.

    Parameters
    ----------
    total_updates : int
        Total number of updates expected for this progress bar.
    """

    def __init__(self, total_updates: int) -> None:
        self.dataloader = list(range(total_updates))
        self.current = 0

    def tick(self, process: str) -> None:
        """Update progress bar with current process status.

        Parameters
        ----------
        process : str
            Description of the current process being executed.
        """
        self.current += 1
        show_progress_bar(self.dataloader, self.current, process)


def show_progress_bar(
    dataloader: DataLoader, current: int, process: str = "training", interval: int = 1
) -> None:
    """Print TensorFlow-like progress bar for batch processing.

    Written instead of using tqdm to allow for custom progress bar readouts.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader currently being processed.
    current : int
        Current index in dataloader.
    process : str, optional
        Current process being performed, by default "training".
    interval : int, optional
        Interval at which to update progress bar, by default 1.
    """
    current += 1
    bar_length = 50
    fraction_computed = current / dataloader.__len__()

    if current % interval != 0 and fraction_computed < 1:
        return

    # pointer = ">" if fraction_computed < 1 else "="
    loading_string = (
        "=" * int(bar_length * fraction_computed)
        + ">"
        + "_" * int(bar_length * (1 - fraction_computed))
    )
    output_string = (
        f"\t {process} {current}/{dataloader.__len__()} "
        f"[{loading_string}] ({int(fraction_computed * 100)}%)"
    )

    if fraction_computed <= (dataloader.__len__() - interval) / dataloader.__len__():
        print(" " * (bar_length + len(process) + 5), end="\r")
        print(output_string, end="\r")
    else:
        print(output_string)


def save_figure(
    data: NDArray | torch.Tensor,
    save_folder: str | Path,
    name: str,
    title: str | None = None,
    vmax: float = 0,
    ext: str = ".png",
) -> None:
    """Save image data as PNG or JPEG figure.

    Saves .png or .jpeg figure of data to folder save_folder under 'name'.
    'data' must be a 3d tensor or numpy array, in channels_first format.

    Parameters
    ----------
    data : NDArray | torch.Tensor
        Input image/stack data to save in channels_first format.
    save_folder : str | Path
        Global path to folder where data is saved.
    name : str
        Name of data, no extension specified.
    title : str, optional
        Image title, if none specified, defaults used, by default None.
    vmax : float, optional
        Value to normalize figure to, by default 0 (uses data max).
    ext : str, optional
        Image save file extension, by default ".png".

    Raises
    ------
    AttributeError
        If data is not a torch tensor or numpy array.
    """
    assert len(data.shape) == 3, f"'{len(data.shape)}d' data must be 3-dimensional"

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, NDArray):
        raise AttributeError(
            f"'data' of type {type(data)} must be torch tensor or numpy array."
        )
    if vmax == 0:
        vmax = np.max(data)

    # normalize and convert to uint8
    data = np.array(((data - np.min(data)) / float(vmax)) * 255, dtype=np.uint8)

    # save
    if data.shape[-3] > 1:
        data = np.mean(data, 0)
        im = Image.fromarray(data).convert("L")
        im.info["size"] = data.shape
        im.save(os.path.join(save_folder, name + ext))
    else:
        data = data[0]
        im = Image.fromarray(data).convert("L")
        im.info["size"] = data.shape
        im.save(os.path.join(save_folder, name + ext))
