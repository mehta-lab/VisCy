from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import iohub.ngff as ngff
import numpy as np
import scipy.stats
import zarr
from numpy.typing import NDArray

import viscy.utils.image_utils as image_utils
import viscy.utils.masks as mask_utils


def mp_wrapper(
    fn: Callable[..., Any], fn_args: list[tuple[Any, ...]], workers: int
) -> list[Any]:
    """Create and save masks with multiprocessing.

    Parameters
    ----------
    fn : callable
        Function to be applied with multiprocessing.
    fn_args : list of tuple
        List with tuples of function arguments.
    workers : int
        Max number of workers.

    Returns
    -------
    list
        List of returned dicts from create_save_mask.
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(fn, *zip(*fn_args))
    return list(res)


def mp_create_and_write_mask(fn_args: list[tuple[Any, ...]], workers: int) -> list[Any]:
    """Create and save masks with multiprocessing.

    For argument parameters see mp_utils.create_and_write_mask.

    Parameters
    ----------
    fn_args : list of tuple
        List with tuples of function arguments.
    workers : int
        Max number of workers.

    Returns
    -------
    list
        List of returned dicts from create_save_mask.
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(create_and_write_mask, *zip(*fn_args))
    return list(res)


def add_channel(
    position: ngff.Position,
    new_channel_array: NDArray,
    new_channel_name: str,
    overwrite_ok: bool = False,
) -> None:
    """Add a channel to the data array at specified position.

    Note that there is only one 'tracked' data array in current HCS spec at each position.
    Also updates the 'omero' channel-tracking metadata to track the new channel.

    The 'new_channel_array' must match the dimensions of the current array in
    all positions but the channel position (1) and have the same datatype.

    Note: to maintain HCS compatibility of the zarr store, all positions (wells)
    must maintain arrays with congruent channels. That is, if you add a channel
    to one position of an HCS compatible zarr store, an additional channel must
    be added to every position in that store to maintain HCS compatibility.

    Parameters
    ----------
    position : ngff.Position
        NGFF position node object.
    new_channel_array : NDArray
        Array to add as new channel with matching dimensions (except channel dim)
        and dtype.
    new_channel_name : str
        Name of new channel.
    overwrite_ok : bool, optional
        If true, if a channel with the same name as 'new_channel_name' is found,
        will overwrite, by default False.
    """
    assert len(new_channel_array.shape) == len(position.data.shape) - 1, (
        "New channel array must match all dimensions of the position array, "
        "except in the inferred channel dimension: "
        f"array shape: {position.data.shape}, "
        "expected channel shape: "
        f"{(position.data.shape[0],) + position.data.shape[2:]}, "
        f"received channel shape: {new_channel_array.shape}"
    )
    # determine whether to overwrite or append
    if new_channel_name in position.channel_names and overwrite_ok:
        new_channel_index = list(position.channel_names).index(new_channel_name)
    else:
        new_channel_index = len(position.channel_names)
        position.append_channel(new_channel_name, resize_arrays=True)

    # replace or append channel
    position["0"][:, new_channel_index] = new_channel_array


def create_and_write_mask(
    position: ngff.Position,
    time_indices: list[int],
    channel_indices: list[int],
    structure_elem_radius: int,
    mask_type: str,
    mask_name: str,
    verbose: bool = False,
) -> None:
    """Create mask for all depth slices at specified time and channel indices.

    Creates masks at each time and channel index specified in this position,
    and saves them both as an additional channel in the data array of the given
    zarr store and a separate 'untracked' array with specified name.

    Saves custom metadata related to the mask creation in the well-level
    .zattrs in the 'mask' field.

    When >1 channel are used to generate the mask, mask of each channel is
    generated then added together. Foreground fraction is calculated on
    a timepoint-position basis. That is, it will be recorded as an average
    foreground fraction over all slices in any given timepoint.

    Parameters
    ----------
    position : ngff.Position
        NGFF position node object.
    time_indices : list
        List of time indices for mask generation. If an index is skipped over,
        will populate with zeros.
    channel_indices : list
        List of channel indices for mask generation. If more than 1 channel
        specified, masks from all channels are aggregated.
    structure_elem_radius : int
        Size of structuring element used for binary opening. str_elem: disk or ball.
    mask_type : str
        Thresholding type used for masking or str to map to masking function.
    mask_name : str
        Name under which to save untracked copy of mask in position.
    verbose : bool, optional
        Whether this process should send updates to stdout, by default False.
    """
    shape = position.data.shape
    position_masks_shape = tuple([shape[0], len(channel_indices), *shape[2:]])

    # calculate masks over every time index and channel slice
    position_masks = np.zeros(position_masks_shape)
    position_foreground_fractions = {}

    for time_index in range(shape[0]):
        timepoint_foreground_fraction = {}

        for channel_index in channel_indices:
            channel_name = position.channel_names[channel_index]
            mask_array_chan_idx = channel_indices.index(channel_index)

            if "mask" in channel_name:
                print("\n")
                if mask_type in channel_name:
                    print(f"Found existing channel: '{channel_name}'.")
                    print("You are likely creating duplicates, which is bad practice.")
                print(f"Skipping mask channel '{channel_name}' for thresholding")
            else:
                # print progress update
                if verbose:
                    time_progress = f"time {time_index + 1}/{shape[0]}"
                    channel_progress = f"chan {channel_index}/{channel_indices}"
                    position_progress = f"pos {position.zgroup.name}"
                    p = (
                        f"Computing masks slice [{position_progress}, {time_progress},"
                        f" {channel_progress}]\n"
                    )
                    print(p)

                # get mask for image slice or populate with zeros
                if time_index in time_indices:
                    mask = get_mask_slice(
                        position_zarr=position.data,
                        time_index=time_index,
                        channel_index=channel_index,
                        mask_type=mask_type,
                        structure_elem_radius=structure_elem_radius,
                    )
                else:
                    mask = np.zeros(shape[-2:])

                position_masks[time_index, mask_array_chan_idx] = mask

                # compute & record channel-wise foreground fractions
                frame_foreground_fraction = float(
                    np.mean(position_masks[time_index, mask_array_chan_idx]).item()
                )
                timepoint_foreground_fraction[channel_name] = frame_foreground_fraction
        position_foreground_fractions[time_index] = timepoint_foreground_fraction

    # combine masks along channels and compute & record combined foreground fraction
    position_masks = np.sum(position_masks, axis=1)
    position_masks = np.where(position_masks > 0.5, 1, 0)
    for time_index in time_indices:
        frame_foreground_fraction = float(np.mean(position_masks[time_index]).item())
        timepoint_foreground_fraction["combined_fraction"] = frame_foreground_fraction

    # save masks as additional channel
    position_masks = position_masks.astype(position.data.dtype)
    new_channel_name = channel_name + "_mask"
    add_channel(
        position=position,
        new_channel_array=position_masks,
        new_channel_name=new_channel_name,
        overwrite_ok=True,
    )


def get_mask_slice(
    position_zarr: zarr.Array,
    time_index: int,
    channel_index: int,
    mask_type: str,
    structure_elem_radius: int,
) -> NDArray:
    """Compute mask for a single image slice.

    Given a set of indices, mask type, and structuring element,
    pulls an image slice from the given zarr array, computes the
    requested mask and returns.

    Parameters
    ----------
    position_zarr : zarr.Array
        Zarr array of the desired position.
    time_index : int
        Time index for the slice.
    channel_index : int
        Channel index for the slice.
    mask_type : str
        Mask type, options are {otsu, unimodal, mem_detection, borders_weight_loss_map}.
    structure_elem_radius : int
        Creation radius for the structuring element.

    Returns
    -------
    NDArray
        2D mask for this slice.
    """
    # read and correct/preprocess slice
    im = position_zarr[time_index, channel_index]
    im = image_utils.preprocess_image(im, hist_clip_limits=(1, 99))
    # generate mask for slice
    if mask_type == "otsu":
        mask = mask_utils.create_otsu_mask(im.astype("float32"))
    elif mask_type == "unimodal":
        mask = mask_utils.create_unimodal_mask(
            im.astype("float32"), structure_elem_radius
        )
    elif mask_type == "mem_detection":
        mask = mask_utils.create_membrane_mask(
            im.astype("float32"),
            structure_elem_radius,
        )
    elif mask_type == "borders_weight_loss_map":
        mask = mask_utils.get_unet_border_weight_map(im)
        mask = image_utils.im_adjust(mask).astype(position_zarr.dtype)

    return mask


def mp_get_val_stats(fn_args: list[Any], workers: int) -> list[dict[str, float]]:
    """
    Compute statistics of numpy arrays with multiprocessing

    Parameters
    ----------
    fn_args : list of tuple
        List with tuples of function arguments.
    workers : int
        Max number of workers.

    Returns
    -------
    list[dict[str, float]]
        List of returned df from get_im_stats.
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(get_val_stats, fn_args)
    return list(res)


def get_val_stats(sample_values: list[float]) -> dict[str, float]:
    """Compute statistics of a numpy array.

    Computes the statistics of a numpy array and returns a dictionary
    of metadata corresponding to input sample values.

    Parameters
    ----------
    sample_values : list of float
        List of sample values at respective indices.

    Returns
    -------
    dict[str, float]
        Dictionary with intensity data for image.
    """
    meta_row = {
        "mean": float(np.nanmean(sample_values)),
        "std": float(np.nanstd(sample_values)),
        "median": float(np.nanmedian(sample_values)),
        "iqr": float(scipy.stats.iqr(sample_values)),
    }
    return meta_row


def mp_sample_im_pixels(
    fn_args: list[tuple[Any, ...]], workers: int
) -> list[list[Any]]:
    """Read and computes statistics of images with multiprocessing

    Parameters
    ----------
    fn_args : list[tuple[Any, ...]]
        List with tuples of function arguments.
    workers : int
        Max number of workers.

    Returns
    -------
    list[list[Any]]
        List of paths and corresponding returned df from get_im_stats.
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(sample_im_pixels, *zip(*fn_args))
    return list(map(list, zip(*list(res))))


def sample_im_pixels(
    position: ngff.Position,
    grid_spacing: int,
    channel: int,
) -> tuple[ngff.Position, NDArray]:
    # TODO move out of mp utils into normalization utils
    """Read and compute statistics of images for each point in a grid.

    Grid spacing determines distance in pixels between grid points
    for rows and cols. By default, samples from every time position and every z-depth,
    and assumes that the data in the zarr store is stored in [T,C,Z,Y,X] format,
    for time, channel, z, y, x.

    Parameters
    ----------
    position : ngff.Position
        NGFF position node object.
    grid_spacing : int
        Spacing of sampling grid in x and y.
    channel : int
        Channel to sample from.

    Returns
    -------
    tuple[ngff.Position, NDArray]
        Position and array with intensity data for each grid point.
    """
    image_zarr = position.data

    all_sample_values = []
    all_time_indices = list(range(image_zarr.shape[0]))
    all_z_indices = list(range(image_zarr.shape[2]))

    for time_index in all_time_indices:
        for z_index in all_z_indices:
            image_slice = image_zarr[time_index, channel, z_index, :, :]
            _, _, sample_values = image_utils.grid_sample_pixel_values(
                image_slice, grid_spacing
            )
            all_sample_values.append(sample_values)
    sample_values = np.stack(all_sample_values, 0).flatten()

    return position, sample_values
