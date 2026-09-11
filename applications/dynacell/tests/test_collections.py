"""Tests for dynacell.collections registry + freezer."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from dynacell.collections import (
    ExperimentSelector,
    freeze_collection,
    get_collection,
    list_collections,
)
from dynacell.collections import freezer as freezer_mod
from dynacell.collections import registry as registry_mod


class TestRegistry:
    """Pure-YAML tests — no HPC access, CI-safe."""

    def test_list_collections(self):
        """Registry includes sec61b_ipsc_v1."""
        assert "sec61b_ipsc_v1" in list_collections()

    def test_get_collection_loads_sec61b_ipsc_v1(self):
        """Loads committed collection YAML with expected shape."""
        coll = get_collection("sec61b_ipsc_v1")
        assert coll.name == "sec61b_ipsc_v1"
        # Train and test modeled as separate experiments.
        assert [exp.name for exp in coll.experiments] == [
            "sec61b_hipsc_train",
            "sec61b_hipsc_test",
        ]
        # Channels reflect the on-disk zarr (all five, not just source+target).
        assert {c.name for c in coll.experiments[0].channels} == {
            "Brightfield",
            "Nuclei",
            "Membrane",
            "Structure",
            "Phase3D",
        }
        # Only the target channel's marker differs from its name.
        markers = {c.name: c.marker for c in coll.experiments[0].channels}
        assert markers["Structure"] == "SEC61B"
        assert markers["Phase3D"] == "Phase3D"
        # 423 + 100 from zarr probe at plan time.
        assert len(coll.train_fovs) == 423
        assert len(coll.test_fovs) == 100
        # FOV path encoding: <experiment>/<row>/<col>/<pos>.
        assert coll.train_fovs[0].startswith("sec61b_hipsc_train/")
        assert coll.test_fovs[0].startswith("sec61b_hipsc_test/")

    def test_get_collection_raises_on_unknown(self):
        """Unknown collection name raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_collection("nonexistent")

    def test_get_collection_raises_on_missing_yaml(self, monkeypatch, tmp_path):
        """A registered name whose YAML is absent raises at call time, not import time."""
        monkeypatch.setitem(registry_mod._REGISTRY, "sec61b_ipsc_v1", tmp_path / "gone.yaml")
        with pytest.raises(FileNotFoundError, match="reinstall dynacell"):
            get_collection("sec61b_ipsc_v1")

    def test_list_collections_survives_missing_yamls(self, monkeypatch, tmp_path):
        """Enumeration is pure dict access, so a broken install can still list names."""
        monkeypatch.setitem(registry_mod._REGISTRY, "sec61b_ipsc_v1", tmp_path / "gone.yaml")
        assert "sec61b_ipsc_v1" in list_collections()


_A2_1_COLLECTIONS = [
    # (name, organelle target, experiment name suffixes, expected train, test)
    # ipsc-only and joint variants for nucleus + membrane targets, mirroring
    # the sec61b/tomm20 pattern.
    (
        "nucleus_ipsc_v1",
        "nucleus",
        ["nucleus_hipsc_train", "nucleus_hipsc_test"],
        500,
        100,
    ),
    (
        "nucleus_joint_all_v1",
        "nucleus",
        [
            "nucleus_hipsc_train",
            "nucleus_hipsc_test",
            "h2b_a549_mantis_2026_03_26_train",
            "h2b_a549_mantis_2026_03_26_test",
        ],
        706,
        136,
    ),
    (
        "nucleus_joint_mock_v1",
        "nucleus",
        [
            "nucleus_hipsc_train",
            "nucleus_hipsc_test",
            "h2b_a549_mantis_2026_03_26_train",
            "h2b_a549_mantis_2026_03_26_test",
        ],
        568,
        112,
    ),
    (
        "membrane_ipsc_v1",
        "membrane",
        ["membrane_hipsc_train", "membrane_hipsc_test"],
        500,
        100,
    ),
    (
        "membrane_joint_all_v1",
        "membrane",
        [
            "membrane_hipsc_train",
            "membrane_hipsc_test",
            "caax_a549_mantis_2026_03_26_train",
            "caax_a549_mantis_2026_03_26_test",
        ],
        706,
        136,
    ),
    (
        "membrane_joint_mock_v1",
        "membrane",
        [
            "membrane_hipsc_train",
            "membrane_hipsc_test",
            "caax_a549_mantis_2026_03_26_train",
            "caax_a549_mantis_2026_03_26_test",
        ],
        568,
        112,
    ),
]


class TestA21Collections:
    """Frozen collections shipped in PR A2.1 (nucleus + membrane variants)."""

    @pytest.mark.parametrize("name,organelle,exp_names,n_train,n_test", _A2_1_COLLECTIONS)
    def test_collection_loads(self, name, organelle, exp_names, n_train, n_test):
        """A2.1 collection is registered, loads, and matches expected shape."""
        assert name in list_collections()
        coll = get_collection(name)
        assert coll.name == name
        assert [exp.name for exp in coll.experiments] == exp_names
        assert all(exp.organelle == organelle for exp in coll.experiments)
        assert len(coll.train_fovs) == n_train
        assert len(coll.test_fovs) == n_test


class TestExperimentSelector:
    """Unit tests for the ExperimentSelector dataclass."""

    def test_default_flags_are_true(self):
        """include_train and include_test default to True."""
        sel = ExperimentSelector(dataset="ds", target="t")
        assert sel.include_train is True
        assert sel.include_test is True

    def test_hashable_and_usable_in_set(self):
        """frozen=True makes selectors hashable; equal selectors dedupe."""
        a = ExperimentSelector("ds", "t")
        b = ExperimentSelector("ds", "t")
        c = ExperimentSelector("ds", "t", include_test=False)
        assert {a, b, c} == {a, c}

    def test_frozen_prevents_mutation(self):
        """frozen=True rejects attribute assignment."""
        sel = ExperimentSelector("ds", "t")
        with pytest.raises(FrozenInstanceError):
            sel.dataset = "other"  # type: ignore[misc]

    def test_condition_defaults_to_none_and_participates_in_hash(self):
        """Condition defaults to None; distinct values produce distinct selectors."""
        a = ExperimentSelector("ds", "t")
        assert a.condition is None
        b = ExperimentSelector("ds", "t", condition="mock")
        c = ExperimentSelector("ds", "t", condition="mock")
        d = ExperimentSelector("ds", "t", condition="DENV")
        assert b == c
        assert b != d
        assert {a, b, c, d} == {a, b, d}


class _FakeManifest:
    """Minimal manifest stand-in for multi-selector freeze tests."""

    def __init__(self, targets, spacing):
        self.targets = targets
        self.spacing = spacing


class _FakeStores:
    def __init__(self, train, test):
        self.train = train
        self.test = test


class _FakeTargetCfg:
    def __init__(self, gene, organelle, target_channel, stores):
        self.gene = gene
        self.organelle = organelle
        self.target_channel = target_channel
        self.stores = stores


def _install_fake_manifest(monkeypatch, datasets: dict):
    """Patch ``get_manifest`` so multi-selector tests don't read real YAML."""

    def fake_get_manifest(name):
        if name not in datasets:
            raise KeyError(name)
        return datasets[name]

    monkeypatch.setattr(freezer_mod, "get_manifest", fake_get_manifest)


def _make_snapshot_stub(monkeypatch):
    """Record every ``_snapshot_store`` call and synthesize outputs.

    Each call returns a CollectionExperiment with the requested name plus
    a single prefixed FOV "<name>/A/1/0", so the test can assert on
    ordering and selector aggregation without real zarrs.
    """
    calls: list[dict] = []

    def fake_snapshot(
        *,
        store_path,
        experiment_name,
        target_cfg,
        manifest_spacing,
        condition_filter=None,
    ):
        calls.append(
            {
                "store_path": store_path,
                "experiment_name": experiment_name,
                "target_cfg": target_cfg,
                "manifest_spacing": manifest_spacing,
                "condition_filter": condition_filter,
            }
        )
        from dynacell.data.collections import CollectionExperiment
        from viscy_data.collection import ChannelEntry

        experiment = CollectionExperiment(
            name=experiment_name,
            data_path=store_path,
            channels=[ChannelEntry(name="Phase3D", marker="Phase3D")],
            organelle=target_cfg.organelle,
            marker=target_cfg.gene,
            pixel_size_xy_um=manifest_spacing.x,
            pixel_size_z_um=manifest_spacing.z,
        )
        return experiment, [f"{experiment_name}/A/1/0"]

    monkeypatch.setattr(freezer_mod, "_snapshot_store", fake_snapshot)
    return calls


def _make_spacing():
    from dynacell.data.manifests import VoxelSpacing

    return VoxelSpacing(x=0.1, y=0.1, z=0.2)


class TestFreezerMultiSelector:
    """CI-safe tests for the multi-selector freeze path (mocks zarr I/O)."""

    def _build_fake_registry(self, dataset_names, *, target="sec61b"):
        spacing = _make_spacing()
        datasets = {}
        for ds in dataset_names:
            stores = _FakeStores(
                train=Path(f"/fake/{ds}/train.zarr"),
                test=Path(f"/fake/{ds}/test.zarr"),
            )
            target_cfg = _FakeTargetCfg(
                gene="SEC61B",
                organelle="er",
                target_channel="Structure",
                stores=stores,
            )
            datasets[ds] = _FakeManifest(targets={target: target_cfg}, spacing=spacing)
        return datasets

    def test_freeze_aggregates_across_selectors(self, monkeypatch, tmp_path):
        """Two selectors, both roles → four experiments in selector-then-role order."""
        datasets = self._build_fake_registry(["ds-a", "ds-b"])
        _install_fake_manifest(monkeypatch, datasets)
        calls = _make_snapshot_stub(monkeypatch)

        out = tmp_path / "joint.yaml"
        selectors = [
            ExperimentSelector("ds-a", "sec61b"),
            ExperimentSelector("ds-b", "sec61b"),
        ]
        result = freeze_collection(
            selectors,
            out,
            created_by="tester",
            created_at="2026-04-24T00:00:00Z",
        )

        names = [exp.name for exp in result.experiments]
        assert names == [
            "sec61b_ds_a_train",
            "sec61b_ds_a_test",
            "sec61b_ds_b_train",
            "sec61b_ds_b_test",
        ]
        assert result.train_fovs == [
            "sec61b_ds_a_train/A/1/0",
            "sec61b_ds_b_train/A/1/0",
        ]
        assert result.test_fovs == [
            "sec61b_ds_a_test/A/1/0",
            "sec61b_ds_b_test/A/1/0",
        ]
        assert result.provenance.created_by == "tester"
        assert result.provenance.created_at == "2026-04-24T00:00:00Z"
        assert [c["experiment_name"] for c in calls] == names

    def test_include_flags_skip_stores(self, monkeypatch, tmp_path):
        """train-only + test-only selectors on the same dataset don't collide."""
        datasets = self._build_fake_registry(["ds-a"])
        _install_fake_manifest(monkeypatch, datasets)
        _make_snapshot_stub(monkeypatch)

        out = tmp_path / "split.yaml"
        result = freeze_collection(
            [
                ExperimentSelector("ds-a", "sec61b", include_test=False),
                ExperimentSelector("ds-a", "sec61b", include_train=False),
            ],
            out,
            created_by="tester",
            created_at="2026-04-24T00:00:00Z",
        )
        names = [exp.name for exp in result.experiments]
        assert names == ["sec61b_ds_a_train", "sec61b_ds_a_test"]

    def test_description_default_single_selector_matches_legacy(self, monkeypatch, tmp_path):
        """Single selector reuses the pre-A2 single-pair description template."""
        datasets = self._build_fake_registry(["ds-a"])
        _install_fake_manifest(monkeypatch, datasets)
        _make_snapshot_stub(monkeypatch)

        out = tmp_path / "single.yaml"
        result = freeze_collection(
            [ExperimentSelector("ds-a", "sec61b")],
            out,
            created_by="tester",
            created_at="2026-04-24T00:00:00Z",
        )
        assert result.description == freezer_mod._default_description("ds-a", "sec61b")

    def test_description_default_joint_single_target(self, monkeypatch, tmp_path):
        """Multi-dataset same-target gets a joint-single-target template."""
        datasets = self._build_fake_registry(["ds-a", "ds-b"])
        _install_fake_manifest(monkeypatch, datasets)
        _make_snapshot_stub(monkeypatch)

        out = tmp_path / "joint.yaml"
        result = freeze_collection(
            [
                ExperimentSelector("ds-a", "sec61b"),
                ExperimentSelector("ds-b", "sec61b"),
            ],
            out,
            created_by="tester",
            created_at="2026-04-24T00:00:00Z",
        )
        assert "target 'sec61b'" in result.description
        assert "2 source datasets" in result.description

    def test_description_default_mixed_target(self, monkeypatch, tmp_path):
        """Mixed-target selectors get a generic joint template listing targets."""
        spacing = _make_spacing()
        datasets = {
            "ds-a": _FakeManifest(
                targets={
                    "sec61b": _FakeTargetCfg(
                        gene="SEC61B",
                        organelle="er",
                        target_channel="Structure",
                        stores=_FakeStores(Path("/fake/a/train.zarr"), Path("/fake/a/test.zarr")),
                    ),
                    "tomm20": _FakeTargetCfg(
                        gene="TOMM20",
                        organelle="mitochondria",
                        target_channel="Structure",
                        stores=_FakeStores(
                            Path("/fake/a/t2_train.zarr"),
                            Path("/fake/a/t2_test.zarr"),
                        ),
                    ),
                },
                spacing=spacing,
            ),
        }
        _install_fake_manifest(monkeypatch, datasets)
        _make_snapshot_stub(monkeypatch)

        out = tmp_path / "mixed.yaml"
        result = freeze_collection(
            [
                ExperimentSelector("ds-a", "sec61b"),
                ExperimentSelector("ds-a", "tomm20"),
            ],
            out,
            created_by="tester",
            created_at="2026-04-24T00:00:00Z",
        )
        assert "'sec61b'" in result.description
        assert "'tomm20'" in result.description

    def test_raises_on_empty_list(self):
        """Empty experiments list is rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            freeze_collection([], Path("/tmp/out.yaml"), created_by="tester")

    def test_raises_on_exact_duplicate(self, monkeypatch, tmp_path):
        """Two identical selectors are rejected by the set() check."""
        datasets = self._build_fake_registry(["ds-a"])
        _install_fake_manifest(monkeypatch, datasets)
        _make_snapshot_stub(monkeypatch)

        with pytest.raises(ValueError, match="duplicate ExperimentSelector"):
            freeze_collection(
                [
                    ExperimentSelector("ds-a", "sec61b"),
                    ExperimentSelector("ds-a", "sec61b"),
                ],
                tmp_path / "dup.yaml",
                created_by="tester",
            )

    def test_raises_on_overlapping_role_contribution(self, monkeypatch, tmp_path):
        """A full selector + a train-only selector on the same pair collide on train."""
        datasets = self._build_fake_registry(["ds-a"])
        _install_fake_manifest(monkeypatch, datasets)
        _make_snapshot_stub(monkeypatch)

        with pytest.raises(ValueError, match="same"):
            freeze_collection(
                [
                    ExperimentSelector("ds-a", "sec61b"),
                    ExperimentSelector("ds-a", "sec61b", include_test=False),
                ],
                tmp_path / "overlap.yaml",
                created_by="tester",
            )

    def test_raises_on_no_op_selector(self, tmp_path):
        """Selector with both include flags disabled is rejected."""
        with pytest.raises(ValueError, match="both include_train=False"):
            freeze_collection(
                [
                    ExperimentSelector(
                        "ds-a",
                        "sec61b",
                        include_train=False,
                        include_test=False,
                    )
                ],
                tmp_path / "noop.yaml",
                created_by="tester",
            )

    def test_raises_when_every_requested_role_has_no_store(self, monkeypatch, tmp_path):
        """Eval-only dataset (stores.train=None) + include_test=False writes nothing.

        Reproduces the real registered case: both ``hek-mantis-*`` manifests
        ship a test store and no train store, so this selector skipped train
        for a missing store and test for the include flag, then wrote a YAML
        with ``experiments: []``.
        """
        spacing = _make_spacing()
        stores = _FakeStores(train=None, test=Path("/fake/eval-only/test.zarr"))
        target_cfg = _FakeTargetCfg(
            gene="KRAS",
            organelle="membrane",
            target_channel="Structure",
            stores=stores,
        )
        _install_fake_manifest(
            monkeypatch,
            {"eval-only": _FakeManifest(targets={"kras": target_cfg}, spacing=spacing)},
        )
        calls = _make_snapshot_stub(monkeypatch)

        out = tmp_path / "empty.yaml"
        with pytest.raises(ValueError, match="contributed no experiments"):
            freeze_collection(
                [ExperimentSelector("eval-only", "kras", include_test=False)],
                out,
                created_by="tester",
            )
        assert calls == []
        assert not out.exists()

    def test_condition_reaches_snapshot_store(self, monkeypatch, tmp_path):
        """Per-selector condition is forwarded to ``_snapshot_store``."""
        datasets = self._build_fake_registry(["ds-ipsc", "ds-a549"])
        _install_fake_manifest(monkeypatch, datasets)
        calls = _make_snapshot_stub(monkeypatch)

        freeze_collection(
            [
                ExperimentSelector("ds-ipsc", "sec61b"),
                ExperimentSelector("ds-a549", "sec61b", condition="mock"),
            ],
            tmp_path / "mockjoint.yaml",
            created_by="tester",
            created_at="2026-04-24T00:00:00Z",
        )
        by_name = {c["experiment_name"]: c["condition_filter"] for c in calls}
        assert by_name == {
            "sec61b_ds_ipsc_train": None,
            "sec61b_ds_ipsc_test": None,
            "sec61b_ds_a549_train": "mock",
            "sec61b_ds_a549_test": "mock",
        }

    def test_description_default_names_uniform_condition(self, monkeypatch, tmp_path):
        """When every filtered selector agrees on one value, description names it."""
        datasets = self._build_fake_registry(["ds-ipsc", "ds-a549"])
        _install_fake_manifest(monkeypatch, datasets)
        _make_snapshot_stub(monkeypatch)

        result = freeze_collection(
            [
                ExperimentSelector("ds-ipsc", "sec61b"),
                ExperimentSelector("ds-a549", "sec61b", condition="mock"),
            ],
            tmp_path / "mockjoint.yaml",
            created_by="tester",
            created_at="2026-04-24T00:00:00Z",
        )
        assert "with condition filter 'mock'" in result.description

    def test_same_role_different_conditions_rejected(self, monkeypatch, tmp_path):
        """Same (dataset, target, role) is rejected even across conditions.

        Condition is intentionally *not* part of the contribution key: a
        single frozen collection describes one population per (dataset,
        target, role). Splitting by condition belongs in separate
        collections (e.g. ``sec61b_joint_mock_v1`` vs a hypothetical
        ``sec61b_joint_denv_v1``).
        """
        datasets = self._build_fake_registry(["ds-a"])
        _install_fake_manifest(monkeypatch, datasets)
        _make_snapshot_stub(monkeypatch)

        with pytest.raises(ValueError, match="same"):
            freeze_collection(
                [
                    ExperimentSelector("ds-a", "sec61b", condition="mock"),
                    ExperimentSelector("ds-a", "sec61b", condition="DENV"),
                ],
                tmp_path / "splitcond.yaml",
                created_by="tester",
            )


def _write_condition_zarr(path, *, positions):
    """Write a tiny HCS zarr with per-position ``condition`` zattr.

    ``positions`` is a list of ``(well_row, well_col, fov, condition)``.
    Each position gets a 5D array of shape (1, 1, 1, 2, 2) so iohub opens
    it as a valid OME-zarr. ``condition=None`` omits the zattr entirely,
    exercising the "no condition attr" error path.
    """
    import numpy as np
    from iohub.ngff import open_ome_zarr

    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["Phase3D"], version="0.4") as plate:
        for row, col, fov, condition in positions:
            pos = plate.create_position(row, col, fov)
            pos.create_image("0", data=np.zeros((1, 1, 1, 2, 2), dtype=np.float32))
            if condition is not None:
                pos.zattrs["condition"] = condition


class TestSnapshotConditionFilter:
    """Integration tests for `_snapshot_store` condition filtering."""

    def _target_cfg(self):
        return _FakeTargetCfg(
            gene="SEC61B",
            organelle="er",
            target_channel="Phase3D",
            stores=_FakeStores(train=None, test=None),
        )

    def test_filter_keeps_matching_positions(self, tmp_path):
        """Only positions whose condition zattr matches are retained."""
        store = tmp_path / "mixed.zarr"
        _write_condition_zarr(
            store,
            positions=[
                ("A", "1", "0", "mock"),
                ("A", "2", "0", "DENV"),
                ("B", "1", "0", "mock"),
            ],
        )
        experiment, fovs = freezer_mod._snapshot_store(
            store_path=store,
            experiment_name="sec61b_ds_train",
            target_cfg=self._target_cfg(),
            manifest_spacing=_make_spacing(),
            condition_filter="mock",
        )
        assert fovs == ["sec61b_ds_train/A/1/0", "sec61b_ds_train/B/1/0"]
        assert experiment.name == "sec61b_ds_train"

    def test_filter_none_preserves_every_position(self, tmp_path):
        """condition_filter=None keeps pre-A2-condition behavior — all FOVs kept."""
        store = tmp_path / "mixed.zarr"
        _write_condition_zarr(
            store,
            positions=[("A", "1", "0", "mock"), ("A", "2", "0", "DENV")],
        )
        _, fovs = freezer_mod._snapshot_store(
            store_path=store,
            experiment_name="sec61b_ds_train",
            target_cfg=self._target_cfg(),
            manifest_spacing=_make_spacing(),
            condition_filter=None,
        )
        assert len(fovs) == 2

    def test_filter_matches_zero_positions_raises(self, tmp_path):
        """Condition filter that matches no position is a loud failure."""
        store = tmp_path / "no_mock.zarr"
        _write_condition_zarr(
            store,
            positions=[("A", "1", "0", "DENV"), ("A", "2", "0", "ZIKV")],
        )
        with pytest.raises(ValueError, match="matched 0"):
            freezer_mod._snapshot_store(
                store_path=store,
                experiment_name="sec61b_ds_train",
                target_cfg=self._target_cfg(),
                manifest_spacing=_make_spacing(),
                condition_filter="mock",
            )

    def test_filter_on_store_without_condition_attr_raises(self, tmp_path):
        """iPSC-shaped store (no condition attr on any position) is rejected."""
        store = tmp_path / "no_attr.zarr"
        _write_condition_zarr(
            store,
            positions=[("A", "1", "0", None), ("A", "2", "0", None)],
        )
        with pytest.raises(ValueError, match="'condition' zattr"):
            freezer_mod._snapshot_store(
                store_path=store,
                experiment_name="sec61b_ds_train",
                target_cfg=self._target_cfg(),
                manifest_spacing=_make_spacing(),
                condition_filter="mock",
            )


@pytest.mark.slow
class TestFreezer:
    """HPC-only: opens live zarr stores."""

    def test_freeze_matches_committed(self, tmp_path):
        """Fresh freeze reproduces the committed YAML byte-for-byte."""
        repo_root = Path(__file__).resolve().parents[1]
        committed = repo_root / "configs/collections/virtual_staining/sec61b_ipsc_v1.yaml"
        prov = yaml.safe_load(committed.read_text())["provenance"]
        out = tmp_path / "sec61b_ipsc_v1.yaml"
        freeze_collection(
            [ExperimentSelector("aics-hipsc", "sec61b")],
            out,
            created_by=prov["created_by"],
            created_at=prov["created_at"],
        )
        assert out.read_text() == committed.read_text()
