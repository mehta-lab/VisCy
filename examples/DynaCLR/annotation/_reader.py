from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import dask.array as da
import numpy as np
import zarr
from iohub.ngff.nodes import (
    MultiScaleMeta,
    NGFFNode,
    OMEROMeta,
    Plate,
    Position,
    Well,
    open_ome_zarr,
)
from pydantic_extra_types.color import Color

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]
    path = Path(path)
    # if we know we cannot read the file, we immediately return None.
    if not path.is_dir() and path.exists():
        return None
    if not ((path / ".zattrs").exists() or (path / "zarr.json").exists()):
        return None
    # otherwise we return the *function* that can read ``path``.
    return reader_function


def _get_node(path: StrOrBytesPath):
    try:
        zgroup = zarr.open_group(path, mode="r")
    except Exception as e:
        raise RuntimeError(e)
    attrs = zgroup.attrs
    if "ome" in attrs:
        attrs = attrs["ome"]
    if "well" in attrs:
        first_pos_grp = next(zgroup.groups())[1]
        channel_names = Position(first_pos_grp).channel_names
        node = Well(
            group=zgroup,
            parse_meta=True,
            channel_names=channel_names,
            version="0.4",
        )
    elif "plate" in attrs or "multiscales" in attrs:
        zgroup.store.close()
        node = open_ome_zarr(store_path=path, mode="r")
    else:
        raise KeyError(f"NGFF plate or well metadata not found under '{zgroup.name}'")
    return node


def stitch_well_by_channel(well: Well, row_wrap: int):
    logging.debug(f"Stitching well: {well.zgroup.name}")
    levels = []
    pyramids: list[list] = []
    for i, (_, pos) in enumerate(well.positions()):
        lv, ims = _get_multiscales(pos)
        levels.append(lv)
        pyramids.append(ims)
        if i == 0:
            layers_kwargs = _ome_to_napari_by_channel(pos.metadata)
    stitched_arrays = []
    for i in range(max(levels)):
        ims = [p[i] for p in pyramids if i < len(p)]
        grid = _make_grid(ims, cols=row_wrap)
        stitched_arrays.append(da.block(grid))
    return layers_kwargs, _find_ch_axis(well), stitched_arrays


def stack_well_by_position(well: Well):
    logging.debug(f"Stacking well: {well.zgroup.name}")
    levels = []
    pyramids: list[list] = []
    for i, (_, pos) in enumerate(well.positions()):
        lv, ims = _get_multiscales(pos)
        levels.append(lv)
        pyramids.append(ims)
        if i == 0:
            layers_kwargs = _ome_to_napari_by_channel(pos.metadata)
    stacked_arrays = []
    for i in range(max(levels)):
        ims = [p[i] for p in pyramids if i < len(p)]
        stacked_arrays.append(da.stack(ims, axis=0))
    return layers_kwargs, _find_ch_axis(well), stacked_arrays


def _get_multiscales(pos: Position):
    ms: MultiScaleMeta = pos.metadata.multiscales[0]
    images = [dataset.path for dataset in ms.datasets]
    multiscales = []
    for im in images:
        try:
            multiscales.append(pos[im].dask_array())
        except Exception as e:
            logging.warning(f"Skipped array '{im}' at position {pos.zgroup.name}: {e}")
    return len(multiscales), multiscales


def _make_grid(elements: list[da.Array], cols: int):
    ct = len(elements)
    rows = ct // cols + int(bool(ct % cols))
    grid = [elements[r * cols : (r + 1) * cols] for r in range(rows)]
    diff = len(grid[0]) - len(grid[-1])
    if diff > 0:
        fill_shape = grid[0][0].shape
        fill_type = grid[0][0].dtype
        grid[-1].extend([da.zeros(fill_shape, fill_type)] * diff)
    return grid


def _ome_to_napari_by_channel(
    metadata, parse_colormap: bool = True, num_channels: int | None = None
):
    if metadata.omero is None:
        if num_channels is None:
            raise ValueError("num_channels must be set when omero is not present")
        return [{"name": f"{i}", "blending": "additive"} for i in range(num_channels)]
    omero: OMEROMeta = metadata.omero
    layers_kwargs = []
    for channel in omero.channels:
        meta = {"name": channel.label}
        if channel.color and parse_colormap:
            # alpha channel is optional
            rgb = Color(channel.color).as_rgb_tuple(alpha=None)
            start = [0.0] * 3
            if len(rgb) == 4:
                start += [1]
            meta["colormap"] = np.array(
                [
                    start,
                    [v / np.iinfo(np.uint8).max for v in rgb],
                ]
            )
            meta["blending"] = "additive"
        layers_kwargs.append(meta)
    return layers_kwargs


def _find_ch_axis(dataset: NGFFNode) -> int | None:
    for i, axis in enumerate(dataset.axes):
        if axis.type == "channel":
            return i


def layers_from_arrays(
    layers_kwargs: list,
    ch_axis: int,
    arrays: list,
    mode: Literal["stitch", "stack"],
    layer_type="image",
):
    if mode == "stack":
        ch_axis += 1
    elif mode != "stitch":
        raise ValueError(f"Unknown mode '{mode}'")
    if ch_axis is not None:
        if ch_axis == 0:
            pre_idx = []
        else:
            pre_idx = [slice(None)] * ch_axis
    layers = []
    for i, kwargs in enumerate(layers_kwargs):
        if ch_axis is not None:
            slc = tuple(pre_idx + [slice(i, i + 1)])
            data = [da.squeeze(arr[slc], axis=ch_axis) for arr in arrays]
        else:
            if i > 0:
                raise RuntimeError("mismatched number of channels")
            data = [arr for arr in arrays]
        layer = (data, kwargs, layer_type)
        layers.append(layer)
    return layers


def fov_to_layers(fov: Position, layer_type: str = "image"):
    ch_axis = _find_ch_axis(fov)
    arrays = [arr.dask_array() for _, arr in fov.images()]
    layers_kwargs = _ome_to_napari_by_channel(
        fov.metadata,
        parse_colormap=(layer_type == "image"),
        num_channels=arrays[0].shape[ch_axis] if ch_axis is not None else 1,
    )
    return layers_from_arrays(
        layers_kwargs, ch_axis, arrays, mode="stitch", layer_type=layer_type
    )


def well_to_layers(well: Well, mode: Literal["stitch", "stack"], layer_type: str):
    if mode == "stitch":
        layers_kwargs, ch_axis, arrays = stitch_well_by_channel(well, row_wrap=4)
    elif mode == "stack":
        layers_kwargs, ch_axis, arrays = stack_well_by_position(well)
    return layers_from_arrays(
        layers_kwargs, ch_axis, arrays, mode=mode, layer_type=layer_type
    )


def make_bbox(bbox_extents):
    """Copied from:
    https://napari.org/stable/tutorials/segmentation/annotate_segmentation.html
    Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array([[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]])
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect


def plate_to_layers(
    plate: Plate,
    row_range: tuple[int, int] = None,
    col_range: tuple[int, int] = None,
):
    plate_arrays = []
    rows = plate.metadata.rows
    if row_range:
        rows = rows[row_range[0] : row_range[1]]
    columns = plate.metadata.columns
    if col_range:
        columns = columns[col_range[0] : col_range[1]]
    boxes = [[] for _ in range(4)]
    properties = {"fov": []}
    well_paths = [w.path for w in plate.metadata.wells]
    for i, row_meta in enumerate(rows):
        row_name = row_meta.name
        row_arrays = []
        for j, col_meta in enumerate(columns):
            col_name = col_meta.name
            well_path = f"{row_name}/{col_name}"
            if well_path in well_paths:
                well = plate[row_name][col_name]
                layers_kwargs, ch_axis, arrays = stack_well_by_position(well)
                row_arrays.append([a[0] for a in arrays])
                height, width = arrays[0][0].shape[-2:]
                box_extents = [
                    height * i,
                    width * j,
                    height * (i + 1),
                    width * (j + 1),
                ]
                for k in range(len(boxes)):
                    boxes[k].append(box_extents[k] - 0.5)
                properties["fov"].append(well_path + "/" + next(well.positions())[0])
            else:
                row_arrays.append(None)
        plate_arrays.append(row_arrays)
    first_blocks = next(a for a in plate_arrays[0] if a is not None)
    fill_args = [(b.shape, b.dtype) for b in first_blocks]
    plate_levels = []
    for level, first_block in enumerate(first_blocks):
        plate_level = []
        for r in plate_arrays:
            row_level = []
            for c in r:
                if c is None:
                    arr = da.zeros(
                        shape=fill_args[level][0],
                        dtype=fill_args[level][1],
                        chunks=first_block.chunksize,
                    )
                else:
                    arr = c[level]
                row_level.append(arr)
            plate_level.append(row_level)
        plate_levels.append(da.block(plate_level))
    layers = layers_from_arrays(
        layers_kwargs,
        ch_axis,
        plate_levels,
        mode="stitch",
        layer_type="image",
    )
    layers.append(
        (
            make_bbox(boxes),
            {
                "face_color": "transparent",
                "edge_color": "black",
                "properties": properties,
                "text": {"string": "fov", "color": "orange"},
                "name": "Plate Map",
            },
            "shapes",
        )
    )
    return layers


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    node = _get_node(path)
    match node:
        case Plate():
            return plate_to_layers(node)
        case Well():
            return well_to_layers(node, mode="stitch", layer_type="image")
        case Position():
            return fov_to_layers(node)
