"""
Image loading and caching with multi-dataset support.

This module provides classes for loading and caching microscopy images from
zarr stores, with support for multiple datasets and FOV name pattern matching.

Classes
-------
ImageCache : Single dataset image cache with FOV pattern matching
MultiDatasetImageCache : Multi-dataset image cache wrapper
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from iohub import open_ome_zarr
from PIL import Image

from .config import DatasetConfig

logger = logging.getLogger(__name__)


class ImageCache:
    """Cache for loading and storing microscopy images."""

    def __init__(
        self,
        data_path: Path,
        channels: list[str],
        z_range: tuple[int, int],
        yx_patch_size: tuple[int, int],
        fov_filter: list[str] | None = None,
    ):
        """
        Initialize image cache.

        Parameters
        ----------
        data_path : Path
            Path to microscopy data zarr store.
        channels : list[str]
            Channel names to load.
        z_range : tuple[int, int]
            Z slice range (start, end).
        yx_patch_size : tuple[int, int]
            Patch size in (Y, X) dimensions.
        fov_filter : list[str], optional
            FOV filter patterns used during data loading (e.g., ["A/1", "A/2"]).
            Used to reconstruct full FOV paths when needed.
        """
        self.data_path = data_path
        self.channels = channels
        self.z_range = z_range
        self.yx_patch_size = yx_patch_size
        self.fov_filter = fov_filter
        self.cache = {}

        logger.info(f"Initializing ImageCache with data from {data_path}")
        try:
            self.data_store = open_ome_zarr(str(data_path), mode="r")
            logger.info("Successfully opened data store")
        except Exception as e:
            logger.error(f"Failed to open data store: {e}")
            self.data_store = None

    @staticmethod
    def _normalize_image(img_array: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 255] range.

        Parameters
        ----------
        img_array : np.ndarray
            Input image array.

        Returns
        -------
        np.ndarray
            Normalized image as uint8.
        """
        min_val = img_array.min()
        max_val = img_array.max()
        if min_val == max_val:
            return np.zeros_like(img_array, dtype=np.uint8)
        return ((img_array - min_val) * 255 / (max_val - min_val)).astype(np.uint8)

    @staticmethod
    def _numpy_to_base64(img_array: np.ndarray) -> str:
        """
        Convert numpy array to base64-encoded JPEG string.

        Parameters
        ----------
        img_array : np.ndarray
            Input image array (uint8).

        Returns
        -------
        str
            Base64-encoded JPEG string with data URI prefix.
        """
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def load_image(
        self, fov_name: str, track_id: int, t: int, channel: str, y: float, x: float
    ) -> Optional[str]:
        """
        Load image for specific observation.

        Parameters
        ----------
        fov_name : str
            Field of view name.
        track_id : int
            Track identifier.
        t : int
            Timepoint.
        channel : str
            Channel name.
        y : float
            Y coordinate (centroid).
        x : float
            X coordinate (centroid).

        Returns
        -------
        str or None
            Base64-encoded image string, or None if loading fails.
        """
        cache_key = (fov_name, track_id, t, channel)

        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.data_store is None:
            logger.error("Data store not initialized")
            return None

        try:
            try:
                position = self.data_store[fov_name]
            except KeyError:
                matching_positions = [
                    pos_name
                    for pos_name, _ in self.data_store.positions()
                    if pos_name.endswith(fov_name)
                ]

                if self.fov_filter and len(matching_positions) > 1:
                    filtered_matches = [
                        pos
                        for pos in matching_positions
                        if any(pattern in pos for pattern in self.fov_filter)
                    ]
                    if filtered_matches:
                        matching_positions = filtered_matches

                if len(matching_positions) == 1:
                    position = self.data_store[matching_positions[0]]
                elif len(matching_positions) > 1:
                    logger.warning(
                        f"Multiple positions match FOV '{fov_name}' (filter: {self.fov_filter}): "
                        f"{matching_positions[:5]}. Using first match."
                    )
                    position = self.data_store[matching_positions[0]]
                else:
                    available_positions = list(self.data_store.positions())
                    logger.error(
                        f"FOV '{fov_name}' not found in data store. "
                        f"Available positions (first 10): {available_positions[:10]}"
                    )
                    return None

            channel_idx = position.get_channel_index(channel)

            y_int, x_int = int(round(y)), int(round(x))
            y_half, x_half = self.yx_patch_size[0] // 2, self.yx_patch_size[1] // 2

            image = position["0"].tensorstore()
            patch = (
                image.oindex[
                    t,
                    [channel_idx],
                    slice(self.z_range[0], self.z_range[1]),
                    slice(y_int - y_half, y_int + y_half),
                    slice(x_int - x_half, x_int + x_half),
                ]
                .read()
                .result()
            )

            patch_2d = patch[0].max(axis=0)

            patch_normalized = self._normalize_image(patch_2d)
            img_base64 = self._numpy_to_base64(patch_normalized)

            self.cache[cache_key] = img_base64

            return img_base64

        except Exception as e:
            logger.error(f"Failed to load image for {fov_name}/{track_id}/t={t}: {e}")
            return None


class MultiDatasetImageCache:
    """Cache for loading images from multiple dataset sources."""

    def __init__(
        self,
        dataset_configs: list[DatasetConfig],
        yx_patch_size: tuple[int, int],
    ):
        """
        Initialize multi-dataset image cache.

        Parameters
        ----------
        dataset_configs : list[DatasetConfig]
            List of dataset configurations. Each dataset must specify its own
            channels and z_range.
        yx_patch_size : tuple[int, int]
            Patch size in (Y, X) dimensions.
        """
        self.caches = {}
        self.dataset_channels = {}

        for config in dataset_configs:
            dataset_id = config.dataset_id
            if dataset_id is None or dataset_id == "":
                dataset_id = config.data_path.name.replace(".zarr", "")

            channels = list(config.channels)
            z_range = config.z_range

            self.caches[dataset_id] = ImageCache(
                config.data_path, channels, z_range, yx_patch_size, config.fov_filter
            )
            self.dataset_channels[dataset_id] = channels

        logger.info(
            f"Initialized MultiDatasetImageCache for {len(self.caches)} datasets"
        )

    def get_channels(self, dataset_id: str) -> list[str]:
        """
        Get available channels for a specific dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.

        Returns
        -------
        list[str]
            List of channel names for this dataset.
        """
        return self.dataset_channels.get(dataset_id, [])

    def load_image(
        self,
        dataset_id: str,
        fov_name: str,
        track_id: int,
        t: int,
        channel: str,
        y: float,
        x: float,
    ) -> Optional[str]:
        """
        Load image from appropriate dataset source.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        fov_name : str
            Field of view name.
        track_id : int
            Track identifier.
        t : int
            Timepoint.
        channel : str
            Channel name.
        y : float
            Y coordinate (centroid).
        x : float
            X coordinate (centroid).

        Returns
        -------
        str or None
            Base64-encoded image string, or None if loading fails.
        """
        if dataset_id not in self.caches:
            logger.error(
                f"Dataset ID '{dataset_id}' not found in caches. "
                f"Available datasets: {list(self.caches.keys())}"
            )
            return None

        return self.caches[dataset_id].load_image(fov_name, track_id, t, channel, y, x)
