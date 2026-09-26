"""Integration tests for ``build_cp_reference``.

Two synthetic datasets are registered as real manifests (``DYNACELL_MANIFEST_ROOTS``),
each with a real HCS GT store and a real GT CP feature cache. The tool resolves them
through ``apply_dataset_ref`` + ``init_cache_context`` exactly as an eval would.

Run::

    uv run --no-sync pytest applications/dynacell/tools/build_cp_reference_test.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from build_cp_reference import main
from iohub.ngff import open_ome_zarr

from dynacell.evaluation.cache import (
    StaleCacheError,
    cache_paths,
    open_features_group,
    save_manifest,
    write_features_to_group,
)
from dynacell.evaluation.cp_reference import CP_REFERENCE_DIMENSION, load_cp_reference
from dynacell.evaluation.metrics import active_cp_feature_names

_N_FEATURES = len(active_cp_feature_names(True))  # eval.yaml ships GLCM on
_POSITIONS = ("A/1/0", "A/1/1")
_T = 2


def _dataset(root: Path, name: str, seed: int, *, z: int = 3, skip: tuple[str, int] | None = None) -> None:
    """Register one dataset: manifest, GT store, and a GT CP cache holding every (pos, t)."""
    store = root / "stores" / f"{name}.zarr"
    cache_dir = root / "caches" / name
    with open_ome_zarr(store, mode="w", layout="hcs", channel_names=["Structure"], version="0.5") as plate:
        for pos_name in _POSITIONS:
            row, col, fov = pos_name.split("/")
            plate.create_position(row, col, fov).create_image("0", np.zeros((_T, 1, z, 4, 4), dtype=np.float32))
    rng = np.random.default_rng(seed)
    with open_features_group(cache_paths(cache_dir), "cp", mode="a") as group:
        for pos_name in _POSITIONS:
            for t in range(_T):
                if (pos_name, t) == skip:
                    continue
                feats = rng.standard_normal((5, _N_FEATURES)) + seed
                if (pos_name, t) == ("A/1/0", 0):
                    feats[0, 3] = np.nan  # the pipeline drops non-finite cells
                write_features_to_group(group, pos_name, t, feats)
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
                "stores": {"train": str(store), "test": str(store), "gt_cache_dir": str(cache_dir)},
                "splits": "splits/none.yaml",
            }
        },
    }
    (root / "manifests" / name).mkdir(parents=True)
    (root / "manifests" / name / "manifest.yaml").write_text(yaml.dump(manifest))


@pytest.fixture
def registry(tmp_path: Path, monkeypatch) -> Path:
    """Two registered synthetic datasets; returns the tmp root."""
    monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(tmp_path / "manifests"))
    return tmp_path


_DATASETS = ["--datasets", "ds-a:sec61b", "ds-b:sec61b"]


def test_build_writes_a_loadable_reference(registry: Path) -> None:
    """A real build records per-dataset cell counts and loads back as a verified reference."""
    _dataset(registry, "ds-a", seed=0)
    _dataset(registry, "ds-b", seed=1)
    out = registry / "er__3d.json"

    main(["--target", "er", *_DATASETS, "--out", str(out)])

    payload = json.loads(out.read_text())
    fit = {d["dataset"]: d for d in payload["fit"]["datasets"]}
    assert fit["ds-a"]["n_cells"] == 2 * _T * 5 - 1  # one non-finite cell dropped
    assert fit["ds-a"]["n_cells_dropped_nonfinite"] == 1
    assert fit["ds-b"]["n_cells"] == 2 * _T * 5 - 1
    assert payload["fit"]["n_cells"] == 38
    assert fit["ds-a"]["cp_cache_path"].endswith("caches/ds-a/features/cp.zarr")
    assert payload["cp_identity"]["cp_glcm_enabled"] is True
    ref = load_cp_reference(out, target_name="er", dimension=CP_REFERENCE_DIMENSION)
    assert ref.keep_mask.shape == (_N_FEATURES,)


def test_dry_run_writes_nothing(registry: Path, capsys) -> None:
    """``--dry-run`` reads and counts every cell but writes no file."""
    _dataset(registry, "ds-a", seed=0)
    _dataset(registry, "ds-b", seed=1)
    out = registry / "er__3d.json"
    main(["--target", "er", *_DATASETS, "--out", str(out), "--dry-run"])
    assert not out.exists()
    assert "total 38 cells" in capsys.readouterr().out


def test_missing_timepoint_raises(registry: Path) -> None:
    """A GT (position, timepoint) absent from the CP cache fails instead of being skipped."""
    _dataset(registry, "ds-a", seed=0)
    _dataset(registry, "ds-b", seed=1, skip=("A/1/1", 1))
    with pytest.raises(StaleCacheError, match="CP cache miss at A/1/1/t1"):
        main(["--target", "er", *_DATASETS, "--out", str(registry / "x.json")])


def test_cache_of_another_recipe_is_refused(registry: Path) -> None:
    """A GT cache stamped with another CP recipe is refused by the pipeline's own identity check."""
    _dataset(registry, "ds-a", seed=0)
    save_manifest(
        cache_paths(registry / "caches" / "ds-a"),
        {"artifacts": {"cp_features": {"path": "features/cp.zarr", "cp_feature_version": "v1_old"}}},
    )
    with pytest.raises(StaleCacheError, match="cp_feature_version"):
        main(["--target", "er", "--datasets", "ds-a:sec61b", "--out", str(registry / "x.json")])


def test_2d_store_is_refused(registry: Path) -> None:
    """A single-plane GT store cannot feed the 3-D reference."""
    _dataset(registry, "ds-a", seed=0, z=1)
    with pytest.raises(ValueError, match="the CP reference is 3-D"):
        main(["--target", "er", "--datasets", "ds-a:sec61b", "--out", str(registry / "x.json")])
