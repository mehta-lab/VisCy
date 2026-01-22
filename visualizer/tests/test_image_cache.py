"""Tests for image loading and caching."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from visualizer.config import DatasetConfig
from visualizer.image_cache import ImageCache, MultiDatasetImageCache


class TestImageCache:
    """Tests for ImageCache class."""

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_init_success(self, mock_open_zarr):
        """Test ImageCache initialization."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        cache = ImageCache(
            data_path=Path("/path/to/data.zarr"),
            channels=["Phase3D"],
            z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        assert cache.data_path == Path("/path/to/data.zarr")
        assert cache.channels == ["Phase3D"]
        assert cache.z_range == (0, 1)
        assert cache.yx_patch_size == (160, 160)
        assert cache.data_store is not None

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_init_failure(self, mock_open_zarr):
        """Test ImageCache initialization failure."""
        mock_open_zarr.side_effect = Exception("Failed to open")

        cache = ImageCache(
            data_path=Path("/path/to/data.zarr"),
            channels=["Phase3D"],
            z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        assert cache.data_store is None

    def test_normalize_image(self):
        """Test image normalization."""
        img = np.array([[0, 50], [100, 200]], dtype=np.float32)

        normalized = ImageCache._normalize_image(img)

        assert normalized.dtype == np.uint8
        assert normalized.min() == 0
        assert normalized.max() == 255

    def test_normalize_image_constant(self):
        """Test normalization of constant image."""
        img = np.ones((10, 10), dtype=np.float32) * 100

        normalized = ImageCache._normalize_image(img)

        assert normalized.dtype == np.uint8
        assert np.all(normalized == 0)

    def test_numpy_to_base64(self):
        """Test conversion of numpy array to base64."""
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

        base64_str = ImageCache._numpy_to_base64(img)

        assert isinstance(base64_str, str)
        assert base64_str.startswith("data:image/jpeg;base64,")

    def test_numpy_to_base64_auto_convert(self):
        """Test base64 conversion with auto-conversion to uint8."""
        img = np.random.rand(64, 64) * 255

        base64_str = ImageCache._numpy_to_base64(img)

        assert isinstance(base64_str, str)
        assert base64_str.startswith("data:image/jpeg;base64,")

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_load_image_cached(self, mock_open_zarr):
        """Test loading cached image."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        cache = ImageCache(
            data_path=Path("/path/to/data.zarr"),
            channels=["Phase3D"],
            z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        cache.cache[("A/1/0", 1, 0, "Phase3D")] = "cached_image_data"

        result = cache.load_image("A/1/0", 1, 0, "Phase3D", 50.0, 100.0)

        assert result == "cached_image_data"

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_load_image_no_store(self, mock_open_zarr):
        """Test loading image when data store is not initialized."""
        mock_open_zarr.side_effect = Exception("Failed")

        cache = ImageCache(
            data_path=Path("/path/to/data.zarr"),
            channels=["Phase3D"],
            z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        result = cache.load_image("A/1/0", 1, 0, "Phase3D", 50.0, 100.0)

        assert result is None

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_load_image_fov_not_found(self, mock_open_zarr):
        """Test loading image when FOV is not found."""
        mock_store = MagicMock()
        mock_store.__getitem__.side_effect = KeyError("FOV not found")
        mock_store.positions.return_value = []
        mock_open_zarr.return_value = mock_store

        cache = ImageCache(
            data_path=Path("/path/to/data.zarr"),
            channels=["Phase3D"],
            z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        result = cache.load_image("A/1/0", 1, 0, "Phase3D", 50.0, 100.0)

        assert result is None

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_load_image_matching_position(self, mock_open_zarr):
        """Test loading image with position name matching."""
        mock_position = MagicMock()
        mock_position.get_channel_index.return_value = 0

        mock_image_data = np.random.randint(0, 255, (1, 1, 160, 160), dtype=np.uint8)
        mock_tensor = MagicMock()
        mock_tensor.read.return_value.result.return_value = mock_image_data
        mock_position.__getitem__.return_value.tensorstore.return_value.oindex.__getitem__.return_value = mock_tensor

        mock_store = MagicMock()
        mock_store.__getitem__.side_effect = KeyError("Direct access failed")
        mock_store.positions.return_value = [("full/path/A/1/0", None)]
        mock_store.__getitem__ = MagicMock(
            side_effect=lambda x: mock_position
            if x == "full/path/A/1/0"
            else KeyError()
        )

        mock_open_zarr.return_value = mock_store

        cache = ImageCache(
            data_path=Path("/path/to/data.zarr"),
            channels=["Phase3D"],
            z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        cache.load_image("A/1/0", 1, 0, "Phase3D", 80.0, 80.0)

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_init_with_fov_filter(self, mock_open_zarr):
        """Test ImageCache initialization with FOV filter."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        cache = ImageCache(
            data_path=Path("/path/to/data.zarr"),
            channels=["Phase3D"],
            z_range=(0, 1),
            yx_patch_size=(160, 160),
            fov_filter=["A/1", "A/2"],
        )

        assert cache.fov_filter == ["A/1", "A/2"]


class TestMultiDatasetImageCache:
    """Tests for MultiDatasetImageCache class."""

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_init_single_dataset(self, mock_open_zarr):
        """Test MultiDatasetImageCache with single dataset."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        config = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"),
            data_path=Path("/path/to/data.zarr"),
            dataset_id="dataset1",
        )

        cache = MultiDatasetImageCache(
            dataset_configs=[config],
            global_channels=["Phase3D"],
            global_z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        assert "dataset1" in cache.caches
        assert len(cache.caches) == 1

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_init_multiple_datasets(self, mock_open_zarr):
        """Test MultiDatasetImageCache with multiple datasets."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        config1 = DatasetConfig(
            adata_path=Path("/path/to/adata1.zarr"),
            data_path=Path("/path/to/data1.zarr"),
            dataset_id="dataset1",
        )
        config2 = DatasetConfig(
            adata_path=Path("/path/to/adata2.zarr"),
            data_path=Path("/path/to/data2.zarr"),
            dataset_id="dataset2",
        )

        cache = MultiDatasetImageCache(
            dataset_configs=[config1, config2],
            global_channels=["Phase3D"],
            global_z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        assert "dataset1" in cache.caches
        assert "dataset2" in cache.caches
        assert len(cache.caches) == 2

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_dataset_specific_channels(self, mock_open_zarr):
        """Test dataset-specific channel configuration."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        config = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"),
            data_path=Path("/path/to/data.zarr"),
            dataset_id="dataset1",
            channels=("Phase3D", "Nuclei"),
        )

        cache = MultiDatasetImageCache(
            dataset_configs=[config],
            global_channels=["Phase3D"],
            global_z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        channels = cache.get_channels("dataset1")
        assert channels == ["Phase3D", "Nuclei"]

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_get_channels_missing_dataset(self, mock_open_zarr):
        """Test getting channels for non-existent dataset."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        config = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"),
            data_path=Path("/path/to/data.zarr"),
            dataset_id="dataset1",
        )

        cache = MultiDatasetImageCache(
            dataset_configs=[config],
            global_channels=["Phase3D"],
            global_z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        channels = cache.get_channels("nonexistent")
        assert channels == []

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_load_image_from_dataset(self, mock_open_zarr):
        """Test loading image from specific dataset."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        config = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"),
            data_path=Path("/path/to/data.zarr"),
            dataset_id="dataset1",
        )

        cache = MultiDatasetImageCache(
            dataset_configs=[config],
            global_channels=["Phase3D"],
            global_z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        cache.caches["dataset1"].cache[("A/1/0", 1, 0, "Phase3D")] = "test_image"

        result = cache.load_image("dataset1", "A/1/0", 1, 0, "Phase3D", 50.0, 100.0)

        assert result == "test_image"

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_load_image_missing_dataset(self, mock_open_zarr):
        """Test loading image from non-existent dataset."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        config = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"),
            data_path=Path("/path/to/data.zarr"),
            dataset_id="dataset1",
        )

        cache = MultiDatasetImageCache(
            dataset_configs=[config],
            global_channels=["Phase3D"],
            global_z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        result = cache.load_image("nonexistent", "A/1/0", 1, 0, "Phase3D", 50.0, 100.0)

        assert result is None

    @patch("visualizer.image_cache.open_ome_zarr")
    def test_auto_dataset_id(self, mock_open_zarr):
        """Test automatic dataset_id generation from path."""
        mock_store = MagicMock()
        mock_open_zarr.return_value = mock_store

        config = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"),
            data_path=Path("/path/to/my_dataset.zarr"),
            dataset_id="",
        )

        cache = MultiDatasetImageCache(
            dataset_configs=[config],
            global_channels=["Phase3D"],
            global_z_range=(0, 1),
            yx_patch_size=(160, 160),
        )

        assert "my_dataset" in cache.caches
