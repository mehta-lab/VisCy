import datetime
import os
import time
from typing import Any

import torch

from viscy.utils.cli_utils import save_figure
from viscy.utils.normalize import hist_clipping


def log_feature(
    feature_map: torch.Tensor, name: str, log_save_folder: str, debug_mode: bool
) -> None:
    """Create visual feature map logs for debugging deep learning models.

    If debug_mode is enabled, creates a visual of the given feature map and saves it at
    'log_save_folder'. If no log_save_folder specified, saves relative to working
    directory with timestamp.

    Currently only saving in working directory is supported.
    This is meant to be an analysis tool, and results should not be saved permanently.

    Parameters
    ----------
    feature_map : torch.Tensor
        Feature map to create visualization log of.
    name : str
        Name identifier for the feature map visualization.
    log_save_folder : str
        Directory path for saving the visualization output.
    debug_mode : bool
        Whether to enable debug mode visualization logging.
    """
    try:
        if debug_mode:
            now = datetime.datetime.now()
            log_save_folder = (
                f"feature_map_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}/"
            )
            logger = FeatureLogger(
                save_folder=log_save_folder,
                spatial_dims=3,
                grid_width=8,
            )
            logger.log_feature_map(
                feature_map,
                name,
                dim_names=["batch", "channels"],
            )
    except Exception:
        print(
            "Features of one input logged. Results saved at:"
            f"\n\t  {log_save_folder}. Will not log to avoid overwrite. \n"
        )


class FeatureLogger:
    """Logger for visualizing neural network feature maps during training and debugging.

    This utility class provides comprehensive feature map visualization capabilities
    for monitoring convolutional neural network activations. It supports both
    individual channel visualization and grid-based multi-channel displays,
    with flexible normalization and spatial dimension handling.

    The logger is designed for debugging deep learning models by capturing
    intermediate layer activations and saving them as organized image files.
    It handles multi-dimensional tensors commonly found in computer vision
    tasks, including 2D/3D spatial dimensions with batch and channel axes.

    Parameters
    ----------
    save_folder : str
        Output directory for saving visualization files.
    spatial_dims : int, optional
        Number of spatial dimensions in feature tensors, by default 3.
    full_batch : bool, optional
        If true, log all samples in batch (warning: slow!), by default False.
    save_as_grid : bool, optional
        If true, feature maps are saved as a grid containing all channels,
        else saved individually, by default True.
    grid_width : int, optional
        Desired width of grid if save_as_grid. If 0, defaults to 1/4 the
        number of channels, by default 0.
    normalize_by_grid : bool, optional
        If true, images saved in grid are normalized to brightest pixel in
        entire grid, by default False.

    Attributes
    ----------
    save_folder : str
        Directory path for saving visualization outputs.
    spatial_dims : int
        Number of spatial dimensions in feature tensors (2D or 3D).
    full_batch : bool
        Whether to log all samples in batch or just the first.
    save_as_grid : bool
        Whether to arrange channels in a grid layout.
    grid_width : int
        Number of columns in grid visualization.
    normalize_by_grid : bool
        Whether to normalize intensities across entire grid.

    Examples
    --------
    >>> logger = FeatureLogger(
    ...     save_folder="./feature_logs",
    ...     spatial_dims=3,
    ...     save_as_grid=True,
    ...     grid_width=8,
    ... )
    >>> logger.log_feature_map(
    ...     conv_features, "conv1_activations", dim_names=["batch", "channels"]
    ... )
    """

    def __init__(
        self,
        save_folder: str,
        spatial_dims: int = 3,
        full_batch: bool = False,
        save_as_grid: bool = True,
        grid_width: int = 0,
        normalize_by_grid: bool = False,
    ) -> None:
        self.save_folder = save_folder
        self.spatial_dims = spatial_dims
        self.full_batch = full_batch
        self.save_as_grid = save_as_grid
        self.grid_width = grid_width
        self.normalize_by_grid = normalize_by_grid

        print("--- Initializing Logger ---")

    def log_feature_map(
        self,
        feature_map: torch.Tensor,
        feature_name: str,
        dim_names: list[str] | None = None,
        vmax: float = 0,
    ) -> None:
        """Create a log of figures for the given feature map tensor.

        Log is saved as images of feature maps in nested directory tree at save_folder.

        By default assumes that batch dimension is the first dimension, and only logs
        the first sample in the batch for performance reasons. Feature map logs cannot
        overwrite existing files.

        Parameters
        ----------
        feature_map : torch.Tensor
            Feature map to log, typically 5D tensor (BCDHW or BCTHW).
        feature_name : str
            Name of feature, used as directory name for organizing outputs.
        dim_names : list[str] | None, optional
            Names of each non-spatial dimension, by default just numbers.
        vmax : float, optional
            Maximum intensity to normalize figures by. If 0, uses relative
            normalization, by default 0.
        """
        # take tensor off of gpu and detach gradient
        feature_map = feature_map.detach().cpu()

        # handle dim names
        num_dims = len(feature_map.shape)
        if dim_names is None:
            dim_names = ["dim_" + str(i) for i in range(num_dims)]
        else:
            assert len(dim_names) + self.spatial_dims == num_dims, (
                "dim_names must be same length as nonspatial tensor dim length"
            )
        self.dim_names = dim_names

        # handle current feature_name
        feature_name = " " + feature_name if len(feature_name) > 0 else ""
        print(f"Logging{feature_name} feature map...", end="")
        self.feature_save_folder = os.path.join(self.save_folder, feature_name)

        start = time.time()
        self.map_feature_dims(feature_map, self.save_as_grid, vmax=vmax)

        print(f"done. Took {time.time() - start:.2f} seconds")

    def map_feature_dims(
        self,
        feature_map: torch.Tensor,
        save_as_grid: bool,
        vmax: float = 0,
        depth: int = 0,
    ) -> None:
        """Recursively create directory structure for organizing feature map logs.

        If save_as_grid is True, compiles 'channels' (assumed to be last non-spatial
        dimension) into a single large image grid before saving.

        Parameters
        ----------
        feature_map : torch.Tensor
            Feature tensor to process and save.
        save_as_grid : bool
            If true, saves images as channel grid layout.
        vmax : float, optional
            Maximum intensity to normalize figures by, by default 0.
        depth : int, optional
            Recursion counter tracking depth in tensor dimensions, by default 0.

        Raises
        ------
        AttributeError
            If the feature map has an invalid number of dimensions.
        """
        for i in range(feature_map.shape[0]):
            if len(feature_map.shape) == 3:
                # individual saving
                z_slice = feature_map[i]
                save_figure(
                    z_slice.unsqueeze(0),
                    self.feature_save_folder,
                    f"z_slice_{i}",
                    vmax=vmax,
                )

            elif len(feature_map.shape) == 4 and save_as_grid:
                if feature_map.shape[0] == 1:
                    # if a single channel, can't save as grid
                    self.map_feature_dims(
                        feature_map,
                        save_as_grid=False,
                        depth=depth,
                    )
                else:
                    # grid saving
                    for z_depth in range(feature_map.shape[1]):
                        # set grid_width
                        if self.grid_width == 0:
                            if feature_map.shape[0] % 4 != 0:
                                raise AttributeError(
                                    f"number of channels ({feature_map.shape[0]}) "
                                    "must be divisible by 4 if grid_width unspecified"
                                )
                            self.grid_width = feature_map.shape[0] // 4
                        else:
                            if feature_map.shape[0] % self.grid_width != 0:
                                raise AttributeError(
                                    f"Grid width {self.grid_width} must be a divisor "
                                    f"of the number of channels {feature_map.shape[0]}"
                                )
                        # build grid by rows
                        # interleaving bars for ease of visualization
                        feature_map_grid = []
                        current_grid_row = []

                        for channel_num in range(feature_map.shape[0]):
                            # build rows by item in col
                            col_num = channel_num % self.grid_width
                            if col_num == 0 and channel_num != 0:
                                feature_map_grid.append(
                                    torch.cat(
                                        self.interleave_bars(current_grid_row, axis=1),
                                        dim=1,
                                    )
                                )
                                current_grid_row = []

                            # get 2d slice
                            map_slice = feature_map[channel_num, z_depth]

                            # norm slice to (0,1) unless normalize_by_grid
                            # which is done later
                            if not self.normalize_by_grid:
                                map_slice = torch.tensor(
                                    hist_clipping(
                                        map_slice.numpy(),
                                        min_percentile=0,
                                        max_percentile=100,
                                    )
                                )
                                map_slice = (
                                    map_slice - torch.min(map_slice)
                                ) / torch.max(map_slice)

                            current_grid_row.append(map_slice)
                        feature_map_grid.append(
                            torch.cat(
                                self.interleave_bars(current_grid_row, axis=1),
                                dim=1,
                            )
                        )
                        feature_map_grid = torch.cat(
                            self.interleave_bars(feature_map_grid, axis=0), dim=0
                        )
                        save_figure(
                            torch.unsqueeze(feature_map_grid, 0),
                            self.feature_save_folder,
                            f"z_slice_{z_depth}_channels_0-{feature_map.shape[0]}",
                            vmax=vmax,
                        )
                    break
            else:
                # tree recursion
                try:
                    name = os.path.join(
                        self.feature_save_folder, self.dim_names[depth] + f"_{i}"
                    )
                except Exception:
                    raise AttributeError("error in recursion")
                os.makedirs(name, exist_ok=False)
                self.map_feature_dims(
                    feature_map[i],
                    name,
                    save_as_grid,
                    depth=depth + 1,
                )

            if depth == 0 and not self.full_batch:
                break
        return

    def interleave_bars(
        self,
        arrays: list[torch.Tensor],
        axis: int,
        pixel_width: int = 3,
        value: float = 0,
    ) -> list[torch.Tensor]:
        """Interleave separator bars between tensors to improve grid visualization.

        Takes list of 2D torch tensors and interleaves bars to improve grid
        visualization quality. Assumes arrays are all of the same shape.

        Parameters
        ----------
        arrays : list[torch.Tensor]
            List of tensors to place separator bars between.
        axis : int
            Axis on which to interleave bars (0 or 1).
        pixel_width : int, optional
            Width of separator bar in pixels, by default 3.
        value : float, optional
            Pixel value for separator bars, by default 0.

        Returns
        -------
        list[torch.Tensor]
            List of tensors with separator bars interleaved for grid visualization.
        """
        shape_match_axis = abs(axis - 1)
        length = arrays[0].shape[shape_match_axis]

        if axis == 0:
            bar = torch.ones((pixel_width, length)) * value
        elif axis == 1:
            bar = torch.ones((length, pixel_width)) * value
        else:
            raise AttributeError("axis must be 0 or 1")

        for i in range(1, len(arrays) * 2 - 1, 2):
            arrays.insert(i, bar)
        return arrays
