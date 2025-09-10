"""Command-line interface utilities for data processing and visualization."""

import collections
import os
import re

import numpy as np
import torch
from PIL import Image


def unique_tags(directory):
    """Return list of unique nume tags from data directory.

    Parameters
    ----------
    directory : str
        Directory containing '.tif' files.

    Returns
    -------
    dict
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
    """

    def __init__(self, total_updates):
        self.dataloader = list(range(total_updates))
        self.current = 0

    def tick(self, process):
        """Update progress bar with current process status.

        Parameters
        ----------
        process : str
            Description of the current process being executed.
        """
        self.current += 1
        show_progress_bar(self.dataloader, self.current, process)


def show_progress_bar(dataloader, current, process="training", interval=1):
    """Print TensorFlow-like progress bar for batch processing.

    Written instead of using tqdm to allow for custom progress bar readouts.

    Parameters
    ----------
    dataloader : iterable
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


def save_figure(data, save_folder, name, title=None, vmax=0, ext=".png"):
    """Save image data as PNG or JPEG figure.

    Saves .png or .jpeg figure of data to folder save_folder under 'name'.
    'data' must be a 3d tensor or numpy array, in channels_first format.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Input image/stack data to save in channels_first format.
    save_folder : str
        Global path to folder where data is saved.
    name : str
        Name of data, no extension specified.
    title : str, optional
        Image title, if none specified, defaults used, by default None.
    vmax : float, optional
        Value to normalize figure to, by default 0 (uses data max).
    ext : str, optional
        Image save file extension, by default ".png".
    """
    assert len(data.shape) == 3, f"'{len(data.shape)}d' data must be 3-dimensional"

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
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
