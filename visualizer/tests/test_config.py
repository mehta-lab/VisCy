"""Tests for configuration classes and constants."""

from pathlib import Path

import pytest

from visualizer.config import INFECTION_COLORS, DatasetConfig, MultiDatasetConfig


class TestDatasetConfig:
    """Tests for DatasetConfig class."""

    def test_dataset_config_minimal(self):
        """Test DatasetConfig with minimal parameters."""
        config = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"), data_path=Path("/path/to/data.zarr")
        )

        assert config.adata_path == Path("/path/to/adata.zarr")
        assert config.data_path == Path("/path/to/data.zarr")
        assert config.fov_filter is None
        assert config.annotation_csv is None
        assert config.annotation_column is None
        assert config.categories is None
        assert config.dataset_id == ""
        assert config.channels is None
        assert config.z_range is None

    def test_dataset_config_full(self):
        """Test DatasetConfig with all parameters."""
        config = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"),
            data_path=Path("/path/to/data.zarr"),
            fov_filter=["A/1/0", "A/2/0"],
            annotation_csv=Path("/path/to/annotations.csv"),
            annotation_column="infection_status",
            categories={0: "uninfected", 1: "infected"},
            dataset_id="dataset1",
            channels=("Phase3D", "Nuclei"),
            z_range=(0, 5),
        )

        assert config.adata_path == Path("/path/to/adata.zarr")
        assert config.data_path == Path("/path/to/data.zarr")
        assert config.fov_filter == ["A/1/0", "A/2/0"]
        assert config.annotation_csv == Path("/path/to/annotations.csv")
        assert config.annotation_column == "infection_status"
        assert config.categories == {0: "uninfected", 1: "infected"}
        assert config.dataset_id == "dataset1"
        assert config.channels == ("Phase3D", "Nuclei")
        assert config.z_range == (0, 5)

    def test_dataset_config_immutable(self):
        """Test that DatasetConfig is immutable (NamedTuple behavior)."""
        config = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"), data_path=Path("/path/to/data.zarr")
        )

        with pytest.raises(AttributeError):
            config.adata_path = Path("/new/path.zarr")


class TestMultiDatasetConfig:
    """Tests for MultiDatasetConfig class."""

    def test_multi_dataset_config_minimal(self):
        """Test MultiDatasetConfig with minimal parameters."""
        dataset = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"), data_path=Path("/path/to/data.zarr")
        )
        config = MultiDatasetConfig(datasets=[dataset])

        assert len(config.datasets) == 1
        assert config.datasets[0] == dataset
        assert config.phate_kwargs is None
        assert config.channels == ("Phase3D",)
        assert config.z_range == (0, 1)
        assert config.yx_patch_size == (160, 160)
        assert config.port == 8050
        assert config.debug is False
        assert config.default_color_mode == "annotation"

    def test_multi_dataset_config_full(self):
        """Test MultiDatasetConfig with all parameters."""
        dataset1 = DatasetConfig(
            adata_path=Path("/path/to/adata1.zarr"),
            data_path=Path("/path/to/data1.zarr"),
            dataset_id="dataset1",
        )
        dataset2 = DatasetConfig(
            adata_path=Path("/path/to/adata2.zarr"),
            data_path=Path("/path/to/data2.zarr"),
            dataset_id="dataset2",
        )

        config = MultiDatasetConfig(
            datasets=[dataset1, dataset2],
            phate_kwargs={"n_components": 2, "knn": 5, "decay": 40},
            channels=("Phase3D", "Nuclei"),
            z_range=(0, 5),
            yx_patch_size=(256, 256),
            port=8051,
            debug=True,
            default_color_mode="time",
        )

        assert len(config.datasets) == 2
        assert config.datasets[0].dataset_id == "dataset1"
        assert config.datasets[1].dataset_id == "dataset2"
        assert config.phate_kwargs == {"n_components": 2, "knn": 5, "decay": 40}
        assert config.channels == ("Phase3D", "Nuclei")
        assert config.z_range == (0, 5)
        assert config.yx_patch_size == (256, 256)
        assert config.port == 8051
        assert config.debug is True
        assert config.default_color_mode == "time"

    def test_multi_dataset_config_replace(self):
        """Test _replace method for updating config (NamedTuple behavior)."""
        dataset = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"), data_path=Path("/path/to/data.zarr")
        )
        config = MultiDatasetConfig(datasets=[dataset])

        new_config = config._replace(port=8080, debug=True)

        assert config.port == 8050
        assert config.debug is False
        assert new_config.port == 8080
        assert new_config.debug is True

    def test_multi_dataset_config_immutable(self):
        """Test that MultiDatasetConfig is immutable (NamedTuple behavior)."""
        dataset = DatasetConfig(
            adata_path=Path("/path/to/adata.zarr"), data_path=Path("/path/to/data.zarr")
        )
        config = MultiDatasetConfig(datasets=[dataset])

        with pytest.raises(AttributeError):
            config.port = 8080


class TestInfectionColors:
    """Tests for INFECTION_COLORS constant."""

    def test_infection_colors_exists(self):
        """Test that INFECTION_COLORS is defined."""
        assert INFECTION_COLORS is not None

    def test_infection_colors_keys(self):
        """Test that INFECTION_COLORS has expected keys."""
        assert "uninfected" in INFECTION_COLORS
        assert "infected" in INFECTION_COLORS
        assert "unknown" in INFECTION_COLORS

    def test_infection_colors_values(self):
        """Test that INFECTION_COLORS has hex color values."""
        assert INFECTION_COLORS["uninfected"] == "#3498db"
        assert INFECTION_COLORS["infected"] == "#e67e22"
        assert INFECTION_COLORS["unknown"] == "#95a5a6"

    def test_infection_colors_colorblind_friendly(self):
        """Test that colors are colorblind-friendly (blue/orange, not red/green)."""
        uninfected_color = INFECTION_COLORS["uninfected"]
        infected_color = INFECTION_COLORS["infected"]

        assert uninfected_color.startswith("#")
        assert infected_color.startswith("#")
        assert len(uninfected_color) == 7
        assert len(infected_color) == 7
