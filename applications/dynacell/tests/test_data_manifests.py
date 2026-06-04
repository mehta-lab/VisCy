"""Tests for dynacell.data schemas and loaders."""

import pytest
import yaml

from dynacell.data.collections import (
    ChannelEntry,
    CollectionExperiment,
    Provenance,
    load_collection,
)
from dynacell.data.manifests import (
    DatasetManifest,
    SplitDefinition,
    VoxelSpacing,
    get_target,
    load_manifest,
    load_splits,
)
from dynacell.data.specs import BenchmarkSpec, load_benchmark_spec


def _make_manifest_dict(**overrides):
    """Build a minimal valid manifest dict for testing."""
    base = {
        "name": "test-dataset",
        "version": "1",
        "description": "Test dataset",
        "cell_type": "HeLa",
        "imaging_modality": "confocal",
        "spacing": {"z": 0.3, "y": 0.1, "x": 0.1},
        "channels": {"source": "Phase3D"},
        "targets": {
            "sec61b": {
                "gene": "SEC61B",
                "organelle": "er",
                "display_name": "ER",
                "target_channel": "Structure",
                "stores": {
                    "train": "/tmp/train.zarr",
                    "test": "/tmp/test.zarr",
                },
                "splits": "splits/sec61b.yaml",
            }
        },
    }
    base.update(overrides)
    return base


class TestDatasetManifest:
    """Tests for DatasetManifest pydantic model."""

    def test_parses_valid_dict(self):
        """Round-trip from dict to DatasetManifest preserves fields."""
        data = _make_manifest_dict()
        manifest = DatasetManifest.model_validate(data)
        assert manifest.name == "test-dataset"
        assert manifest.version == "1"
        assert manifest.spacing.z == 0.3
        assert "sec61b" in manifest.targets
        assert manifest.targets["sec61b"].organelle == "er"

    def test_rejects_empty_targets(self):
        """Manifest with empty targets dict fails validation."""
        data = _make_manifest_dict(targets={})
        with pytest.raises(ValueError, match="at least one target"):
            DatasetManifest.model_validate(data)


class TestVoxelSpacing:
    """Tests for VoxelSpacing model."""

    def test_as_list(self):
        """as_list returns [z, y, x] order."""
        spacing = VoxelSpacing(z=0.29, y=0.108, x=0.108)
        assert spacing.as_list() == [0.29, 0.108, 0.108]


class TestSplitDefinition:
    """Tests for SplitDefinition validation."""

    def test_validates_count_mismatch(self):
        """Raises when count does not match non-empty fovs list."""
        data = {
            "split_version": "1.0",
            "random_seed": 42,
            "train": {"count": 3, "fovs": ["a", "b"]},
            "test": {"count": 1, "fovs": ["c"]},
        }
        with pytest.raises(ValueError, match="count=3 but has 2 FOVs"):
            SplitDefinition.model_validate(data)

    def test_empty_fovs_with_count_is_valid(self):
        """Empty fovs with a count is a valid placeholder."""
        data = {
            "split_version": "1.0",
            "random_seed": 42,
            "train": {"count": 500, "fovs": []},
            "test": {"count": 100, "fovs": []},
        }
        split = SplitDefinition.model_validate(data)
        assert split.train["count"] == 500

    def test_allows_missing_val(self):
        """val: None is acceptable."""
        data = {
            "split_version": "1.0",
            "random_seed": 42,
            "train": {"count": 10, "fovs": []},
            "test": {"count": 5, "fovs": []},
        }
        split = SplitDefinition.model_validate(data)
        assert split.val is None

    def test_validates_val_count_mismatch(self):
        """Raises when val count does not match non-empty fovs list."""
        data = {
            "split_version": "1.0",
            "random_seed": 42,
            "train": {"count": 1, "fovs": ["a"]},
            "test": {"count": 1, "fovs": ["b"]},
            "val": {"count": 5, "fovs": ["c"]},
        }
        with pytest.raises(ValueError, match="val declares count=5 but has 1"):
            SplitDefinition.model_validate(data)


class TestLoaders:
    """Tests for path-based YAML loaders."""

    def test_load_manifest_roundtrip(self, tmp_path):
        """Load a manifest from a temp YAML file."""
        manifest_data = _make_manifest_dict()
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.dump(manifest_data))
        manifest = load_manifest(path)
        assert manifest.name == "test-dataset"
        assert manifest.targets["sec61b"].gene == "SEC61B"

    def test_get_target_from_loaded_manifest(self, tmp_path):
        """get_target extracts a specific target by name."""
        manifest_data = _make_manifest_dict()
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.dump(manifest_data))
        manifest = load_manifest(path)
        target = get_target(manifest, "sec61b")
        assert target.organelle == "er"

    def test_get_target_raises_on_unknown(self, tmp_path):
        """Unknown target name raises KeyError."""
        manifest_data = _make_manifest_dict()
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.dump(manifest_data))
        manifest = load_manifest(path)
        with pytest.raises(KeyError):
            get_target(manifest, "nonexistent")

    def test_load_splits_roundtrip(self, tmp_path):
        """Load a split definition from a temp YAML file."""
        split_data = {
            "split_version": "1.0",
            "random_seed": 42,
            "train": {"count": 10, "fovs": []},
            "test": {"count": 5, "fovs": []},
        }
        path = tmp_path / "splits.yaml"
        path.write_text(yaml.dump(split_data))
        split = load_splits(path)
        assert split.split_version == "1.0"
        assert split.random_seed == 42


class TestCollectionSchemas:
    """Tests for BenchmarkCollection schemas."""

    def test_provenance_minimal(self):
        """Provenance with required fields only."""
        p = Provenance(created_at="2026-04-14", created_by="test")
        assert p.airtable_base_id is None
        assert p.record_ids == []

    def test_channel_entry(self):
        """ChannelEntry parses name + marker."""
        ch = ChannelEntry(name="Phase3D", marker="phase")
        assert ch.name == "Phase3D"

    def test_collection_experiment(self):
        """CollectionExperiment validates required fields."""
        exp = CollectionExperiment(
            name="exp1",
            data_path="/tmp/data.zarr",
            channels=[{"name": "Phase3D", "marker": "phase"}],
            pixel_size_xy_um=0.108,
        )
        assert exp.pixel_size_z_um is None
        assert len(exp.channels) == 1

    def test_load_collection_roundtrip(self, tmp_path):
        """Load a collection from a temp YAML file."""
        collection_data = {
            "name": "test-collection",
            "description": "Test",
            "provenance": {"created_at": "2026-04-14", "created_by": "test"},
            "experiments": [
                {
                    "name": "exp1",
                    "data_path": "/tmp/data.zarr",
                    "channels": [{"name": "Phase3D", "marker": "phase"}],
                    "pixel_size_xy_um": 0.108,
                }
            ],
        }
        path = tmp_path / "collection.yaml"
        path.write_text(yaml.dump(collection_data))
        coll = load_collection(path)
        assert coll.name == "test-collection"
        assert len(coll.experiments) == 1


class TestBenchmarkSpec:
    """Tests for BenchmarkSpec schema."""

    def test_spec_minimal(self):
        """BenchmarkSpec with required fields only."""
        spec = BenchmarkSpec(
            name="nuclei-mix-v1",
            version="1",
            description="Mixed nuclei benchmark",
            collection_path="/tmp/collection.yaml",
            output_root="/tmp/output",
        )
        assert spec.train_preset is None
        assert spec.preprocess_configs == []

    def test_load_benchmark_spec_roundtrip(self, tmp_path):
        """Load a spec from a temp YAML file."""
        spec_data = {
            "name": "nuclei-mix-v1",
            "version": "1",
            "description": "Mixed nuclei benchmark",
            "collection_path": "/tmp/collection.yaml",
            "output_root": "/tmp/output",
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.dump(spec_data))
        spec = load_benchmark_spec(path)
        assert spec.name == "nuclei-mix-v1"
