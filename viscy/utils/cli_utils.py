import collections
import os
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


def unique_tags(directory):
    """
    Returns list of unique nume tags from data directory

    :param str directory: directory containing '.tif' files
    TODO: Remove, unused and poorly written
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


class MultiProcessProgressBar(object):
    """
    Provides the ability to create & update a single progress bar for multi-depth
    multi-processed tasks by calling updates on a single object
    """

    def __init__(self, total_updates):
        self.dataloader = list(range(total_updates))
        self.current = 0

    def tick(self, process):
        self.current += 1
        show_progress_bar(self.dataloader, self.current, process)


def show_progress_bar(dataloader, current, process="training", interval=1):
    """
    Utility function to print tensorflow-like progress bar.

    Written instead of using tqdm to allow for custom progress bar readouts.

    :param iterable dataloader: dataloader currently being processed
    :param int current: current index in dataloader
    :param str proces: current process being performed
    :param int interval: interval at which to update progress bar
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
    """
    Saves .png or .jpeg figure of data to folder save_folder under 'name'.
    'data' must be a 3d tensor or numpy array, in channels_first format

    :param numpy.ndarray/torch.tensor data: input image/stack data to save
    :param str save_folder: global path to folder where data is saved.
    :param str name: name of data, no extension specified
    :param str/None title: image title, if none specified, defaults used
    :param float vmax: value to normalize figure to, by default uses data max
    :param str ext: image save file extension
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


def yaml_to_model(yaml_path: Path, model):
    """
    Load model settings from a YAML file and create a model instance.

    Borrowing from recOrder==0.4.0

    Parameters
    ----------
    yaml_path : Path
        The path to the YAML file containing the model settings.
    model : class
        The model class used to create an instance with the loaded settings.

    Returns
    -------
    object
        An instance of the model class with the loaded settings.

    Raises
    ------
    TypeError
        If the provided model is not a class or does not have a callable constructor.
    FileNotFoundError
        If the YAML file specified by `yaml_path` does not exist.

    Notes
    -----
    This function loads model settings from a YAML file using `yaml.safe_load()`.
    It then creates an instance of the provided `model` class using the loaded settings.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = yaml_to_model('model.yaml', MyModel)

    """
    yaml_path = Path(yaml_path)

    if not callable(getattr(model, "__init__", None)):
        raise TypeError(
            "The provided model must be a class with a callable constructor."
        )

    try:
        with open(yaml_path, "r") as file:
            raw_settings = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML file '{yaml_path}' does not exist.")

    return model(**raw_settings)
