"""Tests for dynacell.data.resolver and the _dynacell_ref_resolver hook."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dynacell._compose_hook import _dynacell_ref_resolver
from dynacell.data import DatasetRef
from dynacell.data.resolver import (
    ManifestNotFoundError,
    NoManifestRootsError,
    TargetNotFoundError,
    discover_manifest_roots,
    resolve_dataset_ref,
)


def _make_manifest_dict(name: str = "toy", target: str = "sec61b") -> dict:
    """Build a minimal valid manifest dict for on-disk tests."""
    return {
        "name": name,
        "version": "1",
        "description": "toy",
        "cell_type": "HeLa",
        "imaging_modality": "confocal",
        "spacing": {"z": 0.3, "y": 0.1, "x": 0.1},
        "channels": {"source": "Phase3D"},
        "targets": {
            target: {
                "gene": "GENE",
                "organelle": "er",
                "display_name": "Target",
                "target_channel": "Structure",
                "stores": {
                    "train": "/tmp/train.zarr",
                    "test": "/tmp/test.zarr",
                },
                "splits": "splits/foo.yaml",
            }
        },
    }


def _write_manifest(root: Path, dataset: str, content: dict) -> Path:
    dir_ = root / dataset
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / "manifest.yaml"
    path.write_text(yaml.dump(content))
    return path


def test_resolve_happy_path_against_fixture():
    """Happy path: fixture manifest resolves to real zarr paths."""
    resolved = resolve_dataset_ref(DatasetRef(dataset="aics-hipsc", target="sec61b"))
    assert resolved.source_channel == "Phase3D"
    assert resolved.target_channel == "Structure"
    assert resolved.spacing.as_list() == [0.29, 0.108, 0.108]
    assert str(resolved.data_path_train).endswith("train/SEC61B.zarr")
    assert str(resolved.data_path_test).endswith("test_cropped/SEC61B.zarr")
    assert str(resolved.cell_segmentation_path).endswith("SEC61B_segmented_cleaned.zarr")
    assert str(resolved.gt_cache_dir).endswith("eval_cache/SEC61B")


_FIXTURE_TARGET_EXPECTATIONS = [
    ("sec61b", "SEC61B.zarr", "SEC61B_segmented_cleaned.zarr", "eval_cache/SEC61B", "Structure"),
    ("tomm20", "TOMM20.zarr", "TOMM20_segmented_cleaned.zarr", "eval_cache/TOMM20", "Structure"),
    ("nucleus", "cell.zarr", "cell_segmented_cleaned.zarr", "eval_cache/nucleus", "Nuclei"),
    ("membrane", "cell.zarr", "cell_segmented_cleaned.zarr", "eval_cache/membrane", "Membrane"),
]


@pytest.mark.parametrize(
    "target,test_store,seg_store,cache_suffix,target_channel",
    _FIXTURE_TARGET_EXPECTATIONS,
)
def test_resolve_each_fixture_target(
    target: str,
    test_store: str,
    seg_store: str,
    cache_suffix: str,
    target_channel: str,
):
    """All four fixture targets resolve to distinct test store / cache / channel tuples."""
    resolved = resolve_dataset_ref(DatasetRef(dataset="aics-hipsc", target=target))
    assert str(resolved.data_path_test).endswith(f"test_cropped/{test_store}")
    assert str(resolved.cell_segmentation_path).endswith(seg_store)
    assert str(resolved.gt_cache_dir).endswith(cache_suffix)
    assert resolved.target_channel == target_channel


def test_gt_cache_dir_none_when_missing_from_manifest(monkeypatch, tmp_path):
    """Manifest target without gt_cache_dir → ResolvedDataset.gt_cache_dir is None."""
    _write_manifest(tmp_path, "my-dataset", _make_manifest_dict())
    monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(tmp_path))
    resolved = resolve_dataset_ref(DatasetRef(dataset="my-dataset", target="sec61b"))
    assert resolved.gt_cache_dir is None
    assert resolved.cell_segmentation_path is None


def test_unknown_dataset_raises_manifest_not_found(monkeypatch, tmp_path):
    """Unknown dataset slug → ManifestNotFoundError listing searched roots."""
    monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(tmp_path))
    with pytest.raises(ManifestNotFoundError) as exc:
        resolve_dataset_ref(DatasetRef(dataset="does-not-exist", target="sec61b"))
    assert "does-not-exist" in str(exc.value)
    assert str(tmp_path) in str(exc.value)


def test_unknown_target_raises_target_not_found(monkeypatch, tmp_path):
    """Unknown target slug in known dataset → TargetNotFoundError with available."""
    _write_manifest(tmp_path, "my-dataset", _make_manifest_dict(target="sec61b"))
    monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(tmp_path))
    with pytest.raises(TargetNotFoundError) as exc:
        resolve_dataset_ref(DatasetRef(dataset="my-dataset", target="bogus"))
    msg = str(exc.value)
    assert "bogus" in msg
    assert "sec61b" in msg  # available targets listed


def test_no_roots_raises_with_helpful_error(monkeypatch):
    """Unset env var + no cli + no entry points → NoManifestRootsError.

    Patches ``_entry_point_roots`` to return ``[]`` to simulate a
    consumer that uninstalled or excluded VisCy's bundled registry.
    """
    monkeypatch.delenv("DYNACELL_MANIFEST_ROOTS", raising=False)
    monkeypatch.setattr("dynacell.data.resolver._entry_point_roots", lambda: [])
    with pytest.raises(NoManifestRootsError) as exc:
        discover_manifest_roots()
    msg = str(exc.value)
    assert "DYNACELL_MANIFEST_ROOTS" in msg
    assert "dynacell._manifests" in msg


def test_cli_roots_take_precedence_over_env(monkeypatch, tmp_path):
    """cli_roots wins: dataset found in CLI path even when env points elsewhere."""
    cli_root = tmp_path / "cli"
    env_root = tmp_path / "env"
    _write_manifest(cli_root, "data", _make_manifest_dict())
    monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(env_root))
    roots = discover_manifest_roots(cli_roots=[cli_root])
    assert roots[0] == cli_root
    assert roots[1] == env_root


def test_env_var_precedes_entry_points(monkeypatch, tmp_path):
    """Env-var root appears before entry-point roots in the precedence list."""
    monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(tmp_path))
    roots = discover_manifest_roots()
    assert tmp_path in roots


def test_invalid_manifest_yaml_raises_validation_error(monkeypatch, tmp_path):
    """Manifest missing required field (name) → pydantic ValidationError."""
    bad = _make_manifest_dict()
    del bad["name"]
    _write_manifest(tmp_path, "my-dataset", bad)
    monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(tmp_path))
    with pytest.raises(ValidationError):
        resolve_dataset_ref(DatasetRef(dataset="my-dataset", target="sec61b"))


def test_resolver_hook_noop_on_partial_ref_missing_target():
    """_dynacell_ref_resolver: partial ref (only dataset) = no-op, no lookup."""
    composed = {"benchmark": {"dataset_ref": {"dataset": "aics-hipsc"}}}
    result = _dynacell_ref_resolver(composed)
    assert result == composed


def test_resolver_hook_noop_on_partial_ref_missing_dataset():
    """_dynacell_ref_resolver: partial ref (only target) = no-op, no lookup."""
    composed = {"benchmark": {"dataset_ref": {"target": "sec61b"}}}
    result = _dynacell_ref_resolver(composed)
    assert result == composed


def test_resolver_hook_noop_when_no_dataset_ref():
    """_dynacell_ref_resolver: leaf without dataset_ref passes through."""
    composed = {"benchmark": {"target": "er"}, "data": {"init_args": {}}}
    result = _dynacell_ref_resolver(composed)
    assert result == composed
