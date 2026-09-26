"""Integration tests for ``build_cp_reference``.

Synthetic datasets are registered as real manifests (``DYNACELL_MANIFEST_ROOTS``),
each with a real HCS GT store, a real GT CP feature cache with its manifest, and
(for the lite one) a split recording ``selection_criteria.lite_of``. A synthetic
grouped-leaf tree names which datasets the target is evaluated on. The tool
resolves everything through ``apply_dataset_ref`` + ``init_cache_context`` exactly
as an eval would.

Run::

    uv run --no-sync pytest applications/dynacell/tools/build_cp_reference_test.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from build_cp_reference import eval_config_for, main
from iohub.ngff import open_ome_zarr

from dynacell.evaluation.cache import (
    StaleCacheError,
    cache_paths,
    open_features_group,
    save_manifest,
    write_features_to_group,
)
from dynacell.evaluation.cp_reference import MASK_FIT_DATASETS, cp_space, load_cp_reference
from dynacell.evaluation.metrics import active_cp_feature_names

_N_FEATURES = len(active_cp_feature_names(True))  # eval.yaml ships GLCM on
_POSITIONS = ("A/1/0", "A/1/1")
_T = 2
_BUILT_AT = "2026-09-01T00:00:00+00:00"


def _dataset(
    root: Path,
    name: str,
    seed: int,
    *,
    z: int = 3,
    skip: tuple[str, int] | None = None,
    lite_of: str | None = None,
    positions: tuple[str, ...] = _POSITIONS,
) -> None:
    """Register one dataset: manifest, split, GT store, and a GT CP cache holding every (pos, t)."""
    store = root / "stores" / f"{name}.zarr"
    cache_dir = root / "caches" / name
    with open_ome_zarr(store, mode="w", layout="hcs", channel_names=["Structure"], version="0.5") as plate:
        for pos_name in positions:
            row, col, fov = pos_name.split("/")
            plate.create_position(row, col, fov).create_image("0", np.zeros((_T, 1, z, 4, 4), dtype=np.float32))
    manifest = {
        "name": name,
        "version": "1",
        "description": "synthetic",
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
                "stores": {"test": str(store), "gt_cache_dir": str(cache_dir)},
                "splits": "splits/sec61b.yaml",
            }
        },
    }
    split = {
        "split_version": "1",
        "random_seed": 0,
        "selection_criteria": {"lite_of": lite_of} if lite_of is not None else {"positions": "all"},
        "train": {"count": 0, "fovs": []},
        "test": {"count": len(positions), "fovs": list(positions)},
    }
    (root / "manifests" / name / "splits").mkdir(parents=True)
    (root / "manifests" / name / "manifest.yaml").write_text(yaml.dump(manifest))
    (root / "manifests" / name / "splits" / "sec61b.yaml").write_text(yaml.dump(split))
    rng = np.random.default_rng(seed)
    with open_features_group(cache_paths(cache_dir), "cp", mode="a") as group:
        for pos_name in positions:
            for t in range(_T):
                if (pos_name, t) == skip:
                    continue
                feats = rng.standard_normal((5, _N_FEATURES)) * (1 + seed) + 3 * seed
                if (pos_name, t) == ("A/1/0", 0):
                    feats[0, 3] = np.nan  # the pipeline drops non-finite cells
                write_features_to_group(group, pos_name, t, feats)
    identity, _ = cp_space(eval_config_for("er", (name, "sec61b")))
    save_manifest(
        cache_paths(cache_dir),
        {
            "artifacts": {
                "cp_features": {
                    "path": "features/cp.zarr",
                    "built_at": _BUILT_AT,
                    "spacing": [0.3, 0.1, 0.1],
                    **identity,
                }
            }
        },
    )


def _leaves(root: Path, datasets: list[str]) -> Path:
    """Write one grouped leaf for target ``er`` evaluating ``datasets``; return the leaves root."""
    leaves = root / "leaves"
    (leaves / "er_bucket").mkdir(parents=True)
    doc = {
        "target_name": "er",
        "conditions": [{"name": d, "benchmark": {"dataset_ref": {"dataset": d, "target": "sec61b"}}} for d in datasets],
    }
    (leaves / "er_bucket" / "eval_grouped.yaml").write_text(yaml.dump(doc))
    (leaves / "other_bucket").mkdir()
    (leaves / "other_bucket" / "eval_grouped.yaml").write_text(yaml.dump({"target_name": "nucleus", "conditions": []}))
    return leaves


@pytest.fixture
def registry(tmp_path: Path, monkeypatch) -> Path:
    """Mask-fit sets ds-a + ds-b for ``er``, a HEK-like ds-h, and a lite ds-a-lite; returns the tmp root."""
    monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(tmp_path / "manifests"))
    monkeypatch.setitem(MASK_FIT_DATASETS, "er", (("ds-a", "sec61b"), ("ds-b", "sec61b")))
    return tmp_path


def _build_all(root: Path) -> Path:
    """Register ds-a, ds-b, ds-h and ds-a-lite and a leaf evaluating all four; return the leaves root."""
    _dataset(root, "ds-a", seed=0)
    _dataset(root, "ds-b", seed=1)
    _dataset(root, "ds-h", seed=2)
    # Same seed + first position: the lite cache holds exactly ds-a's A/1/0 cells, as a real lite subset does.
    _dataset(root, "ds-a-lite", seed=0, lite_of="ds-a", positions=_POSITIONS[:1])
    return _leaves(root, ["ds-a", "ds-b", "ds-h", "ds-a-lite"])


def test_build_writes_a_loadable_reference(registry: Path) -> None:
    """A real build: one mask over ds-a + ds-b, a scaler per non-lite set, the lite set mapped to its parent."""
    leaves = _build_all(registry)
    out = registry / "er.json"

    main(["--target", "er", "--leaves-root", str(leaves), "--out", str(out)])

    payload = json.loads(out.read_text())
    fit = payload["fit"]
    assert fit["mask_fit"]["datasets"] == ["ds-a", "ds-b"]
    assert sorted(payload["scalers"]) == ["ds-a", "ds-b", "ds-h"]
    assert fit["datasets"]["ds-h"]["in_mask_fit"] is False
    assert payload["lite"] == {"ds-a-lite": {"parent": "ds-a"}}
    lite = fit["lite"]["ds-a-lite"]
    assert lite["gt_cache_dir"] == str(registry / "caches" / "ds-a-lite")
    assert lite["positions"] == ["A/1/0"] and lite["n_cells"] == _T * 5 - 1  # lite GT recorded from its own cache
    assert lite["gt_matrix_sha256"] != fit["datasets"]["ds-a"]["gt_matrix_sha256"]
    a = {**payload["scalers"]["ds-a"], **fit["datasets"]["ds-a"]}
    assert a["n_cells"] == 2 * _T * 5 - 1  # one non-finite cell dropped
    assert a["n_cells_dropped_nonfinite"] == 1
    assert a["positions"] == list(_POSITIONS)
    assert a["cp_cache_built_at"] == _BUILT_AT
    assert fit["mask_fit"]["n_cells"] == 38
    ref = load_cp_reference(out, target_name="er")
    assert ref.for_dataset("ds-a-lite").scaler_dataset == "ds-a"
    # Scalers differ per set: ds-b was drawn with a different offset and scale.
    assert np.abs(np.array(a["mean"]) - np.array(payload["scalers"]["ds-b"]["mean"])).max() > 1


def test_rebuild_is_a_no_op_and_a_changed_reference_needs_force(registry: Path, capsys) -> None:
    """An identical rebuild writes nothing; after a GT re-cache the new reference needs ``--force``."""
    leaves = _build_all(registry)
    out = registry / "er.json"
    args = ["--target", "er", "--leaves-root", str(leaves), "--out", str(out)]
    main(args)
    main(args)
    assert "unchanged (identical reference)" in capsys.readouterr().out
    with open_features_group(cache_paths(registry / "caches" / "ds-b"), "cp", mode="a") as group:
        write_features_to_group(group, "A/1/0", 0, np.ones((5, _N_FEATURES)) * 9.0)
    with pytest.raises(FileExistsError, match="--force"):
        main(args)
    main([*args, "--force"])


def test_dry_run_writes_nothing(registry: Path, capsys) -> None:
    """``--dry-run`` reads and fits everything but writes no file."""
    leaves = _build_all(registry)
    out = registry / "er.json"
    main(["--target", "er", "--leaves-root", str(leaves), "--out", str(out), "--dry-run"])
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "mask fit: 38 cells" in printed and "ds-a-lite -> scaler of ds-a: 1 positions, 9 cells" in printed


def test_missing_timepoint_raises(registry: Path) -> None:
    """A GT (position, timepoint) absent from the CP cache fails instead of being skipped."""
    _dataset(registry, "ds-a", seed=0)
    _dataset(registry, "ds-b", seed=1, skip=("A/1/1", 1))
    leaves = _leaves(registry, ["ds-a", "ds-b"])
    with pytest.raises(StaleCacheError, match="CP cache miss at A/1/1/t1"):
        main(["--target", "er", "--leaves-root", str(leaves), "--out", str(registry / "x.json")])


def test_cache_of_another_recipe_is_refused(registry: Path) -> None:
    """A GT cache stamped with another CP recipe is refused by the pipeline's own identity check."""
    _dataset(registry, "ds-a", seed=0)
    _dataset(registry, "ds-b", seed=1)
    save_manifest(
        cache_paths(registry / "caches" / "ds-a"),
        {"artifacts": {"cp_features": {"path": "features/cp.zarr", "built_at": _BUILT_AT, "cp_feature_version": "v1"}}},
    )
    leaves = _leaves(registry, ["ds-a", "ds-b"])
    with pytest.raises(StaleCacheError, match="cp_feature_version"):
        main(["--target", "er", "--leaves-root", str(leaves), "--out", str(registry / "x.json")])


def test_lite_outside_its_parent_is_refused(registry: Path) -> None:
    """A lite store holding a position its parent's scaler was not fit on is refused at build time."""
    _dataset(registry, "ds-a", seed=0)
    _dataset(registry, "ds-b", seed=1)
    _dataset(registry, "ds-a-lite", seed=0, lite_of="ds-a", positions=("B/1/0",))
    leaves = _leaves(registry, ["ds-a", "ds-b", "ds-a-lite"])
    with pytest.raises(ValueError, match="outside its parent"):
        main(["--target", "er", "--leaves-root", str(leaves), "--out", str(registry / "x.json")])


def test_2d_store_is_refused(registry: Path) -> None:
    """A single-plane GT store cannot feed the 3-D reference."""
    _dataset(registry, "ds-a", seed=0, z=1)
    _dataset(registry, "ds-b", seed=1)
    leaves = _leaves(registry, ["ds-a", "ds-b"])
    with pytest.raises(ValueError, match="CP regionprops are 3-D"):
        main(["--target", "er", "--leaves-root", str(leaves), "--out", str(registry / "x.json")])


def test_verify_passes_on_unchanged_caches_and_fails_on_a_value_only_recache(registry: Path, capsys) -> None:
    """``--verify`` exits 0 on the caches the reference was built from, 1 after a same-count value change.

    The value-only re-cache keeps every cell count and the manifest's built_at, so the
    eval's checks cannot see it; only the recomputed GT-matrix sha256 does.
    """
    leaves = _build_all(registry)
    out = registry / "er.json"
    main(["--target", "er", "--leaves-root", str(leaves), "--out", str(out)])
    assert main(["--target", "er", "--out", str(out), "--verify"]) == 0

    with open_features_group(cache_paths(registry / "caches" / "ds-h"), "cp", mode="a") as group:
        write_features_to_group(group, "A/1/1", 1, np.full((5, _N_FEATURES), 2.0))  # same 5 cells, new values
    assert main(["--target", "er", "--out", str(out), "--verify"]) == 1
    printed = capsys.readouterr().out
    assert "MISMATCH ds-h" in printed and "verify" in printed and "FAILED" in printed
