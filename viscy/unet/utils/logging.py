import datetime
import os
import time

import torch

from viscy.utils.cli_utils import save_figure
from viscy.utils.normalize import hist_clipping


def log_feature(feature_map, name, log_save_folder, debug_mode):
    """
    If self.debug_mode, creates a visual of the given feature map, and saves it at
    'log_save_folder'
    If no log_save_folder specified, saves relative to working directory with timestamp.

    Currently only saving in working directory is supported.
    This is meant to be an analysis tool,
    and results should not be saved permanently.

    :param torch.tensor feature_map: feature map to create visualization log of
    :param str name: string
    :param str log_save_folder
    """
    try:
        if debug_mode:
            now = datetime.datetime.now()
            log_save_folder = (
                f"feature_map_{now.year}_{now.month}_"
                f"{now.day}_{now.hour}_{now.minute}/"
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
    def __init__(
        self,
        save_folder,
        spatial_dims=3,
        full_batch=False,
        save_as_grid=True,
        grid_width=0,
        normalize_by_grid=False,
    ):
        """
        Logger object for handling logging feature maps inside network architectures.

        Saves each 2d slice of a feature map in either a single grid per feature map
        stack or a directory tree of labeled slices.

        By default saves images into grid.

        :param str save_folder: output directory
        :param bool full_batch: if true, log all sample in batch (warning slow!),
                                defaults to False
        :param bool save_as_grid: if true feature maps are to be saved as a grid
                                containing all channels, else saved individually,
                                defaults to True
        :param int grid_width: desired width of grid if save_as_grid, by default
                                1/4 the number of channels, defaults to 0
        :param bool normalize_by_grid: if true, images saved in grid are normalized
                                to brightest pixel in entire grid, defaults to False

        """
        self.save_folder = save_folder
        self.spatial_dims = spatial_dims
        self.full_batch = full_batch
        self.save_as_grid = save_as_grid
        self.grid_width = grid_width
        self.normalize_by_grid = normalize_by_grid

        print("--- Initializing Logger ---")

    def log_feature_map(
        self,
        feature_map,
        feature_name,
        dim_names=[],
        vmax=0,
    ):
        """
        Creates a log of figures the given feature map tensor at 'save_folder'.
        Log is saved as images of feature maps in nested directory tree.

        By default _assumes that batch dimension is the first dimension_, and
        only logs the first sample in the batch, for performance reasons.

        Feature map logs cannot overwrite.

        :param torch.Tensor feature_map: feature map to log (typically 5d tensor)
        :parapm str feature_name: name of feature (will be used as dir name)
        :param list dim_names: names of each dimension, by default just numbers
        :param int spatial_dims: number of spatial dims, defaults to 3
        :param float vmax: maximum intensity to normalize figures by, by default
                        (if given 0) does relative normalization
        """
        # take tensor off of gpu and detach gradient
        feature_map = feature_map.detach().cpu()

        # handle dim names
        num_dims = len(feature_map.shape)
        if len(dim_names) == 0:
            dim_names = ["dim_" + str(i) for i in range(len(num_dims))]
        else:
            assert len(dim_names) + self.spatial_dims == num_dims, (
                "dim_names must be " "same length as nonspatial tensor dim length"
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
        feature_map,
        save_as_grid,
        vmax=0,
        depth=0,
    ):
        """
        Recursive directory creation for organizing feature map logs

        If save_as_grid, will compile 'channels' (assumed to be last
        non-spatial dimension) into a single large image grid before saving.

        :param numpy.ndarray feature_map: see name
        :param str save_dir: see name
        :param bool save_as_grid: if true, saves images as channel grid
        :param float vmax: maximum intensity to normalize figures by
        :param int depth: recursion counter. depth in dimensions
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

    def interleave_bars(self, arrays, axis, pixel_width=3, value=0):
        """
        Takes list of 2d torch tensors and interleaves bars to improve
        grid visualization quality.
        Assumes arrays are all of the same shape.

        :param list grid_arrays: list of tensors to place bars between
        :param int axis: axis on which to interleave bars (0 or 1)
        :param int pixel_width: width of bar, defaults to 3
        :param int value: value of bar pixels, defaults to 0
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
