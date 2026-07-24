"""Tests for the name-based dataset registry (dynacell.data.registry).

Exercises get_manifest / list_datasets / get_splits against the bundled
``dynacell._manifests`` registry discovered via the ``dynacell.manifest_roots``
entry point. Loads manifest/split YAML only (no zarr access), so these run
without HPC data stores.
"""

import pytest

from dynacell.data import get_manifest, get_splits, list_datasets
from dynacell.data.manifests import DatasetManifest, SplitDefinition


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
