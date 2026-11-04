"""Tests for the name-based dataset registry (dynacell.data.registry).

Exercises get_manifest / list_datasets / get_splits against the bundled
``dynacell._manifests`` registry discovered via the ``dynacell.manifest_roots``
entry point. Loads manifest/split YAML only (no zarr access), so these run
without HPC data stores.
"""

import os
from pathlib import Path

import pytest
import yaml

from dynacell.data import get_manifest, get_splits, list_datasets
from dynacell.data.manifests import DatasetManifest, SplitDefinition


def _write_registry(root: Path, name: str, *, splits_exists: bool = True, marker: str = "m") -> None:
    """Write a minimal valid ``<root>/<name>/manifest.yaml`` (+ splits) for tests."""
    ds = root / name
    (ds / "splits").mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "version": "1",
        "description": marker,
        "cell_type": "A549",
        "imaging_modality": "test",
        "spacing": {"z": 0.3, "y": 0.1, "x": 0.1},
        "channels": {"source": "Phase3D"},
        "targets": {
            "t0": {
                "gene": "G",
                "organelle": "er",
                "display_name": "T0",
                "target_channel": "Structure",
                "stores": {"train": "/tmp/train.zarr", "test": "/tmp/test.zarr"},
                "splits": "splits/t0.yaml",
            }
        },
    }
    (ds / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    if splits_exists:
        splits = {
            "split_version": "1.0",
            "random_seed": 0,
            "train": {"count": 0, "fovs": []},
            "test": {"count": 0, "fovs": []},
        }
        (ds / "splits" / "t0.yaml").write_text(yaml.safe_dump(splits))


class TestListDatasets:
    """list_datasets enumerates the bundled registry by name."""

    def test_includes_aics_and_a549_pools(self):
        """The 12 a549-mantis pools + aics-hipsc are all registered."""
        names = set(list_datasets())
        assert "aics-hipsc" in names
        expected_a549 = {
            f"a549-mantis-{gene}-{cond}"
            for gene in ("caax", "h2b", "sec61b", "tomm20")
            for cond in ("mock", "zikv", "denv")
        }
        assert expected_a549 <= names


class TestGetManifest:
    """get_manifest resolves a bare name to a validated manifest."""

    def test_aics_hipsc_targets(self):
        """aics-hipsc carries the four organelle targets."""
        manifest = get_manifest("aics-hipsc")
        assert isinstance(manifest, DatasetManifest)
        assert set(manifest.targets) == {"membrane", "nucleus", "sec61b", "tomm20"}

    def test_raises_on_unknown(self):
        """An unregistered name raises KeyError, not LookupError silently."""
        with pytest.raises(KeyError):
            get_manifest("does-not-exist")

    def test_name_based_target_access(self):
        """Name->target is get_manifest(name).targets[t]; migrated a549
        manifests carry the additive cell_segmentation + gt_cache_dir stores."""
        target = get_manifest("a549-mantis-sec61b-mock").targets["sec61b"]
        assert target.organelle == "er"
        assert target.stores.cell_segmentation is not None
        assert target.stores.gt_cache_dir is not None


class TestGetSplits:
    """get_splits loads a target's split definition by name."""

    def test_aics_hipsc_nucleus(self):
        """Splits resolve relative to the manifest directory."""
        splits = get_splits("aics-hipsc", "nucleus")
        assert isinstance(splits, SplitDefinition)
        assert splits.split_version

    def test_raises_on_unknown_target(self):
        """An absent target name raises KeyError naming the target."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_splits("aics-hipsc", "nonexistent")

    def test_missing_split_file_raises(self, tmp_path, monkeypatch):
        """A present target whose split file is absent raises FileNotFoundError
        naming the dataset/target, not a bare uncontextualized error."""
        _write_registry(tmp_path, "zz-missing-split", splits_exists=False)
        monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="zz-missing-split"):
            get_splits("zz-missing-split", "t0")


class TestRegistryValidatesAll:
    """Every committed manifest + its first target's splits load and validate."""

    @pytest.mark.parametrize("name", list_datasets())
    def test_manifest_and_first_split_load(self, name):
        """get_manifest + get_splits(first target) succeed for each dataset."""
        manifest = get_manifest(name)
        assert isinstance(manifest, DatasetManifest)
        first_target = sorted(manifest.targets)[0]
        assert isinstance(get_splits(name, first_target), SplitDefinition)


class TestListDatasetsPrecedence:
    """Multi-root enumeration dedups by name and honors precedence order."""

    def test_first_root_wins_and_dedups(self, tmp_path, monkeypatch):
        """A name present in two roots appears once and resolves to the first."""
        root1 = tmp_path / "r1"
        root2 = tmp_path / "r2"
        _write_registry(root1, "zz-shared", marker="root1")
        _write_registry(root2, "zz-shared", marker="root2")
        monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", f"{root1}{os.pathsep}{root2}")
        assert list_datasets().count("zz-shared") == 1
        assert get_manifest("zz-shared").description == "root1"
