"""Shared utility functions extracted from hcs.py and triplet.py.

This module centralizes helper functions that are used by multiple data modules:
- From ``hcs.py``: ``_ensure_channel_list``, ``_search_int_in_str``,
  ``_collate_samples``, ``_read_norm_meta``
- From ``triplet.py``: ``_scatter_channels``, ``_gather_channels``,
  ``_transform_channel_wise``
"""

import copy
import re
from typing import Sequence

import torch
from iohub.ngff import Position
from monai.data.utils import collate_meta_tensor
from monai.transforms import CenterSpatialCrop, Cropd
from torch import Tensor

from viscy_data._typing import DictTransform, NormMeta, Sample

__all__ = [
    "BatchedCenterSpatialCropd",
    "_collate_samples",
    "_ensure_channel_list",
    "_gather_channels",
    "_read_norm_meta",
    "_scatter_channels",
    "_search_int_in_str",
    "_transform_channel_wise",
]


class _BatchedCenterSpatialCrop(CenterSpatialCrop):
    """CenterSpatialCrop that operates on (B, C, *spatial) tensors.

    Standard MONAI CenterSpatialCrop expects (C, *spatial) and crops
    spatial dims = img.shape[1:]. This variant skips both batch and
    channel dimensions, cropping spatial dims = img.shape[2:].
    """

    def __init__(self, roi_size: Sequence[int] | int) -> None:
        super().__init__(roi_size, lazy=False)

    def __call__(
        self,
        img: torch.Tensor,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        spatial_size = img.shape[2:]
        crop_slices = self.compute_slices(spatial_size)
        slices = (slice(None), slice(None)) + crop_slices
        return img[slices]


class BatchedCenterSpatialCropd(Cropd):
    """CenterSpatialCropd for (B, C, *spatial) batched tensors.

    Parameters
    ----------
    keys : Sequence[str]
        Keys to pick data for transformation.
    roi_size : Sequence[int] | int
        Expected ROI size to crop.
    allow_missing_keys : bool, optional
        Don't raise exception if key is missing. Default is False.
    """

    def __init__(
        self,
        keys: Sequence[str],
        roi_size: Sequence[int] | int,
        allow_missing_keys: bool = False,
    ) -> None:
        cropper = _BatchedCenterSpatialCrop(roi_size)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys)


def _ensure_channel_list(str_or_seq: str | Sequence[str]) -> list[str]:
    """Ensure channel argument is a list of strings.

    Parameters
    ----------
    str_or_seq : str | Sequence[str]
        Channel name or list of channel names.

    Returns
    -------
    list[str]
        List of channel names.
    """
    if isinstance(str_or_seq, str):
        return [str_or_seq]
    try:
        return list(str_or_seq)
    except TypeError:
        raise TypeError(f"Channel argument must be a string or sequence of strings. Got {str_or_seq}.")


def _search_int_in_str(pattern: str, file_name: str) -> str:
    """Search image indices in a file name with regex patterns.

    E.g. ``'001'`` -> ``1``.
    """
    match = re.search(pattern, file_name)
    if match:
        return match.group()
    else:
        raise ValueError(f"Cannot find pattern {pattern} in {file_name}.")


def _collate_samples(batch: Sequence[Sample]) -> Sample:
    """Collate samples into a batch sample.

    Parameters
    ----------
    batch : Sequence[Sample]
        A sequence of dictionaries, where each key may point to a value of a
        single tensor or a list of tensors, as is the case with
        ``train_patches_per_stack > 1``.

    Returns
    -------
    Sample
        Batch sample (dictionary of tensors).
    """
    collated: Sample = {}
    for key in batch[0].keys():
        data = []
        for sample in batch:
            if isinstance(sample[key], Sequence):
                data.extend(sample[key])
            else:
                data.append(sample[key])
        collated[key] = collate_meta_tensor(data)
    return collated


def _read_norm_meta(fov: Position) -> NormMeta | None:
    """Read normalization metadata from the FOV.

    Convert to float32 tensors to avoid automatic casting to float64.
    """
    raw = fov.zattrs.get("normalization", None)
    if raw is None:
        return None
    norm_meta = copy.deepcopy(raw)
    for channel, channel_values in norm_meta.items():
        for level, level_values in channel_values.items():
            if level == "timepoint_statistics":
                for tp_idx, tp_values in level_values.items():
                    for stat, value in tp_values.items():
                        if isinstance(value, Tensor):
                            value = value.clone().float()
                        else:
                            value = torch.tensor(value, dtype=torch.float32)
                        norm_meta[channel][level][tp_idx][stat] = value
            else:
                for stat, value in level_values.items():
                    if isinstance(value, Tensor):
                        value = value.clone().float()
                    else:
                        value = torch.tensor(value, dtype=torch.float32)
                    norm_meta[channel][level][stat] = value
    return norm_meta


def _collate_norm_meta(norm_metas: list[NormMeta]) -> NormMeta:
    """Stack per-sample norm_meta dicts into batched tensors.

    Each input dict has structure
    ``{channel: {level: {stat: scalar_tensor, ...}, ...}, ...}``.
    Returns the same structure but with ``(B,)`` tensors so that
    ``_match_image`` broadcasts them against ``(B, 1, Z, Y, X)`` patches.
    """
    ref = norm_metas[0]
    result: NormMeta = {}
    for ch, ch_stats in ref.items():
        result[ch] = {}
        for level, level_stats in ch_stats.items():
            if level_stats is None:
                result[ch][level] = None
                continue
            if level == "timepoint_statistics":
                # timepoint_statistics is {tp_idx: {stat: tensor}} — resolved per-sample
                # upstream in NormalizeSampled, not collatable across samples here
                result[ch][level] = level_stats
                continue
            result[ch][level] = {stat: torch.stack([m[ch][level][stat] for m in norm_metas]) for stat in level_stats}
    return result


def _scatter_channels(
    channel_names: list[str],
    patch: Tensor,
    norm_meta: list[NormMeta] | None,
    extra: dict | None = None,
) -> dict[str, Tensor | NormMeta] | dict[str, Tensor]:
    channels = {name: patch[:, c : c + 1] for name, c in zip(channel_names, range(patch.shape[1]))}
    if norm_meta is not None:
        channels["norm_meta"] = _collate_norm_meta(norm_meta)
    if extra is not None:
        channels.update(extra)
    return channels


def _gather_channels(
    patch_channels: dict[str, Tensor | NormMeta],
    extra_keys: tuple[str, ...] = ("norm_meta",),
) -> list[Tensor]:
    for k in extra_keys:
        patch_channels.pop(k, None)
    return torch.cat(list(patch_channels.values()), dim=1)


def _transform_channel_wise(
    transform: DictTransform,
    channel_names: list[str],
    patch: Tensor,
    norm_meta: list[NormMeta] | None,
    extra: dict | None = None,
) -> list[Tensor]:
    scattered_channels = _scatter_channels(channel_names, patch, norm_meta, extra)
    transformed_channels = transform(scattered_channels)
    return _gather_channels(transformed_channels)
