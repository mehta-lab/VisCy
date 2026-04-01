"""Sliding window datasets for HCS NGFF stores."""

import bisect
import logging
import random
from pathlib import Path

import numpy as np
import torch
from imageio import imread
from iohub.ngff import ImageArray, Position
from torch import Tensor
from torch.utils.data import Dataset

from viscy_data._typing import ChannelMap, DictTransform, HCSStackIndex, NormMeta, Sample
from viscy_data._utils import _ensure_channel_list, _read_norm_meta, _search_int_in_str
from viscy_data.foreground_masks import ForegroundMaskSupport

_logger = logging.getLogger("lightning.pytorch")


class SlidingWindowDataset(Dataset):
    """Sliding window dataset over HCS NGFF positions.

    Each element is a window of (C, Z, Y, X) where C=2 (source and target)
    and Z is ``z_window_size``.

    Parameters
    ----------
    positions : list[Position]
        FOVs to include in dataset.
    channels : ChannelMap
        Source and target channel names,
        e.g. ``{'source': 'Phase', 'target': ['Nuclei', 'Membrane']}``.
    z_window_size : int
        Z window size of the 2.5D U-Net, 1 for 2D.
    array_key : str
        Name of the image arrays (multiscales level), by default "0".
    transform : DictTransform | None
        A callable that transforms data, defaults to None.
    load_normalization_metadata : bool
        Whether to load normalization metadata, defaults to True.
    min_nonzero_fraction : float
        Minimum fraction of voxels above ``nonzero_threshold`` for a patch
        to be used. Patches below this fraction are retried up to
        ``max_nonzero_retries`` times. Default 0.0 disables filtering.
    nonzero_threshold : float
        Intensity threshold for the nonzero fraction check.
        Default 0.0 means any nonzero voxel counts.
    nonzero_channel : str or None
        Channel name to check for nonzero fraction. ``None`` defaults
        to the first target channel.
    max_nonzero_retries : int
        Maximum number of random re-samples when a patch fails the
        nonzero fraction check. Default 100.
    fg_mask_key : str or None
        Zarr array key for precomputed foreground masks. When set,
        masks are loaded alongside images and included in the sample
        as ``"fg_mask"``. Default None disables mask loading.
    """

    def __init__(
        self,
        positions: list[Position],
        channels: ChannelMap,
        z_window_size: int,
        array_key: str = "0",
        transform: DictTransform | None = None,
        load_normalization_metadata: bool = True,
        min_nonzero_fraction: float = 0.0,
        nonzero_threshold: float = 0.0,
        nonzero_channel: str | None = None,
        max_nonzero_retries: int = 100,
        fg_mask_key: str | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= min_nonzero_fraction <= 1.0:
            raise ValueError(f"min_nonzero_fraction must be in [0, 1], got {min_nonzero_fraction}")
        if max_nonzero_retries < 0:
            raise ValueError(f"max_nonzero_retries must be >= 0, got {max_nonzero_retries}")
        self.positions = positions
        self.channels = {k: _ensure_channel_list(v) for k, v in channels.items()}
        self.source_ch_idx = [positions[0].get_channel_index(c) for c in channels["source"]]
        self.target_ch_idx = (
            [positions[0].get_channel_index(c) for c in channels["target"]] if "target" in channels else None
        )
        self.z_window_size = z_window_size
        self.transform = transform
        self.array_key = array_key
        self.load_normalization_metadata = load_normalization_metadata
        self.min_nonzero_fraction = min_nonzero_fraction
        self.nonzero_threshold = nonzero_threshold
        self.nonzero_channel = nonzero_channel
        self.max_nonzero_retries = max_nonzero_retries
        target_channels = list(channels.get("target", []))
        if fg_mask_key is not None and target_channels:
            self.fg_mask_support = ForegroundMaskSupport(fg_mask_key, target_channels)
        else:
            self.fg_mask_support = None
        # Cache combined channel names and indices (used every __getitem__)
        self._all_ch_names = self.channels["source"].copy()
        self._all_ch_idx = self.source_ch_idx.copy()
        if self.target_ch_idx is not None:
            self._all_ch_names.extend(self.channels["target"])
            self._all_ch_idx.extend(self.target_ch_idx)
        self._get_windows()
        if nonzero_channel is not None:
            all_channels = list(self.channels.get("source", [])) + list(self.channels.get("target", []))
            if nonzero_channel not in all_channels:
                raise ValueError(f"nonzero_channel '{nonzero_channel}' not found in channels: {all_channels}")

    def _get_windows(self) -> None:
        """Count the sliding windows along T and Z, and build an index-to-window LUT."""
        w = 0
        self.window_keys = []
        self.window_arrays = []
        self.window_norm_meta: list[NormMeta | None] = []
        for fov in self.positions:
            img_arr: ImageArray = fov[str(self.array_key)]
            ts = img_arr.frames
            zs = img_arr.slices - self.z_window_size + 1
            if zs < 1:
                raise IndexError(
                    f"Z window size {self.z_window_size} "
                    f"is larger than the number of Z slices ({img_arr.slices}) "
                    f"for FOV {fov.name}."
                )
            w += ts * zs
            self.window_keys.append(w)
            self.window_arrays.append(img_arr)
            self.window_norm_meta.append(_read_norm_meta(fov))
            if self.fg_mask_support is not None:
                self.fg_mask_support.validate_and_store(fov, img_arr, self.target_ch_idx)
        self._max_window = w

    def _find_window(self, index: int) -> tuple[ImageArray, int, NormMeta | None, int]:
        """Look up window given index."""
        arr_idx = bisect.bisect_right(self.window_keys, index)
        tz = index - self.window_keys[arr_idx - 1] if arr_idx > 0 else index
        return (self.window_arrays[arr_idx], tz, self.window_norm_meta[arr_idx], arr_idx)

    def _read_img_window(self, img: ImageArray, ch_idx: list[int], tz: int) -> tuple[list[Tensor], HCSStackIndex]:
        """Read image window as tensor.

        Parameters
        ----------
        img : ImageArray
            NGFF image array.
        ch_idx : list[int]
            List of channel indices to read,
            output channel ordering will reflect the sequence.
        tz : int
            Window index within the FOV, counted Z-first.

        Returns
        -------
        list[Tensor], HCSStackIndex
            List of (C=1, Z, Y, X) image tensors,
            tuple of image name, time index, and Z index.
        """
        zs = img.shape[-3] - self.z_window_size + 1
        t = (tz + zs) // zs - 1
        z = tz - t * zs
        data = img.oindex[
            slice(t, t + 1),
            [int(i) for i in ch_idx],
            slice(z, z + self.z_window_size),
        ].astype(np.float32)
        return torch.from_numpy(data).unbind(dim=1), (img.name, t, z)

    def __len__(self) -> int:
        """Return total number of windows."""
        return self._max_window

    def _stack_channels(
        self,
        sample_images: list[dict[str, Tensor]] | dict[str, Tensor],
        key: str | None = None,
        keys: list[str] | None = None,
    ) -> Tensor | list[Tensor]:
        """Stack single-channel images into a multi-channel tensor."""
        ch_keys = keys if keys is not None else self.channels[key]
        if not isinstance(sample_images, list):
            return torch.stack([sample_images[ch][0] for ch in ch_keys])
        return [torch.stack([im[ch][0] for ch in ch_keys]) for im in sample_images]

    def __getitem__(self, index: int) -> Sample:
        """Return a sample for the given index."""
        check_key = (
            (self.nonzero_channel or self.channels.get("target", [None])[0]) if self.min_nonzero_fraction > 0 else None
        )
        idx = index
        for attempt in range(self.max_nonzero_retries + 1):
            img, tz, norm_meta, arr_idx = self._find_window(idx)
            images, sample_index = self._read_img_window(img, self._all_ch_idx, tz)
            sample_images = dict(zip(self._all_ch_names, images))
            # Read mask once — reused for both nonzero check and sample output
            mask_images = None
            if self.fg_mask_support is not None and self.target_ch_idx is not None:
                mask_images = self.fg_mask_support.read_window(arr_idx, tz, self._read_img_window)
            if check_key is not None:
                if mask_images is not None and check_key in self.channels.get("target", []):
                    check_ch = self.channels["target"].index(check_key)
                    frac = mask_images[check_ch].sum().item() / mask_images[check_ch].numel()
                elif check_key in sample_images:
                    patch = sample_images[check_key]
                    frac = (patch >= self.nonzero_threshold).sum().item() / patch.numel()
                else:
                    break
                if frac < self.min_nonzero_fraction:
                    if attempt < self.max_nonzero_retries:
                        idx = random.randint(0, len(self) - 1)
                        continue
                    _logger.warning(
                        f"Exhausted {self.max_nonzero_retries} retries for nonzero fraction "
                        f">= {self.min_nonzero_fraction} on channel '{check_key}' "
                        f"(index {index}). Returning last sample."
                    )
            break
        # Inject mask as temp keys so spatial transforms in the pipeline co-align them
        has_masks = self.fg_mask_support is not None and mask_images is not None
        if has_masks:
            self.fg_mask_support.inject_into_sample(sample_images, mask_images)
        if self.target_ch_idx is not None:
            # NOTE: uses only the first target channel as weight for MONAI
            # spatial transform co-alignment. This does not copy the tensor.
            sample_images["weight"] = sample_images[self.channels["target"][0]]
        if norm_meta is not None:
            sample_images["norm_meta"] = norm_meta
        if self.transform:
            sample_images = self.transform(sample_images)
        if "weight" in sample_images:
            del sample_images["weight"]
        sample = {
            "index": sample_index,
            "source": self._stack_channels(sample_images, "source"),
        }
        if self.target_ch_idx is not None:
            sample["target"] = self._stack_channels(sample_images, "target")
        if has_masks:
            sample["fg_mask"] = self._stack_channels(sample_images, keys=list(self.fg_mask_support.mask_keys))
        if self.load_normalization_metadata and norm_meta is not None:
            sample["norm_meta"] = norm_meta
        return sample


class MaskTestDataset(SlidingWindowDataset):
    """Test dataset with optional ground truth masks.

    Each element is a window of (C, Z, Y, X) where C=2 (source and target)
    and Z is ``z_window_size``.

    This a testing stage version of
    :py:class:`viscy_data.sliding_window.SlidingWindowDataset`,
    and can only be used with batch size 1 for efficiency (no padding for collation),
    since the mask is not available for each stack.

    Parameters
    ----------
    positions : list[Position]
        FOVs to include in dataset.
    channels : ChannelMap
        Source and target channel names,
        e.g. ``{'source': 'Phase', 'target': ['Nuclei', 'Membrane']}``.
    z_window_size : int
        Z window size of the 2.5D U-Net, 1 for 2D.
    transform : DictTransform
        A callable that transforms data, defaults to None.
    ground_truth_masks : str | None
        Path to the ground truth masks.
    array_key : str, optional
        Name of the image arrays (multiscales level), by default "0".
    """

    def __init__(
        self,
        positions: list[Position],
        channels: ChannelMap,
        z_window_size: int,
        transform: DictTransform | None = None,
        ground_truth_masks: str | None = None,
        array_key: str = "0",
        **kwargs,
    ) -> None:
        super().__init__(positions, channels, z_window_size, array_key=array_key, transform=transform, **kwargs)
        self.masks = {}
        if ground_truth_masks is None:
            return
        for img_path in Path(ground_truth_masks).glob("*cp_masks.png"):
            img_name = img_path.name
            position_name = _search_int_in_str(r"(?<=_p)\d{3}", img_name)
            # TODO: specify time index in the file name
            t_idx = 0
            # TODO: record channel name
            # channel_name = re.search(r"^.+(?=_p\d{3})", img_name).group()
            z_idx = _search_int_in_str(r"(?<=_z)\d+", img_name)
            self.masks[(int(position_name), int(t_idx), int(z_idx))] = img_path
        _logger.info(str(self.masks))

    def __getitem__(self, index: int) -> Sample:
        """Return a sample with optional ground truth mask."""
        sample = super().__getitem__(index)
        img_name, t_idx, z_idx = sample["index"]
        position_name = int(img_name.split("/")[-2])
        key = (position_name, int(t_idx), int(z_idx) + self.z_window_size // 2)
        if img_path := self.masks.get(key):
            sample["labels"] = torch.from_numpy(imread(img_path).astype(np.int16))
        return sample
