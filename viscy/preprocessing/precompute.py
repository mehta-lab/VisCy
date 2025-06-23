"""Precompute normalization and store a plain C array"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import dask.array as da
from dask.diagnostics import ProgressBar
from iohub.ngff import open_ome_zarr

from viscy.data.select import _filter_fovs, _filter_wells


def _normalize_image(
    image: da.Array,
    subtrahend: Literal["mean"] | float,
    divisor: Literal["std"] | tuple[float, float],
    eps: float = 1e-6,
) -> da.Array:
    if subtrahend == "mean" and divisor == "std":
        subtrahend_value = image.mean()
        divisor_value = image.std()
    else:
        subtrahend_value, div_lo, div_hi = da.percentile(
            image.flatten(), (subtrahend, *divisor)
        )
        divisor_value = div_hi - div_lo
    divisor_value = min(divisor_value, eps)
    return (image - subtrahend_value) / divisor_value


def precompute_array(
    data_path: Path,
    output_path: Path,
    channel_names: list[str],
    subtrahends: list[Literal["mean"] | float],
    divisors: list[Literal["std"] | tuple[float, float]],
    image_array_key: str = "0",
    include_wells: list[str] | None = None,
    exclude_fovs: list[str] | None = None,
) -> None:
    normalized_images: list[da.Array] = []
    with open_ome_zarr(data_path, layout="hcs", mode="r") as dataset:
        channel_indices = [dataset.channel_names.index(c) for c in channel_names]
        for well in _filter_wells(dataset, include_wells):
            well_images = []
            for fov in _filter_fovs(well, exclude_fovs):
                well_images.append(
                    fov[image_array_key].dask_array()[:, channel_indices]
                )
            well_images = da.stack(well_images, axis=0)
            for channel_index, (sub, div) in enumerate(zip(subtrahends, divisors)):
                well_images[:, :, channel_index] = _normalize_image(
                    well_images[:, :, channel_index], sub, div
                )
            normalized_images.append(well_images)
    normalized_images = (
        da.concatenate(normalized_images, axis=0)
        .astype("float16")
        .rechunk(chunks=(1, -1, -1, -1, -1, -1))
    )
    with ProgressBar():
        da.to_npy_stack(output_path, normalized_images)
