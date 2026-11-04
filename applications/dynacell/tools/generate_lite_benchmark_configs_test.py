r"""Integration tests for ``generate_lite_benchmark_configs.py``.

Production stores are tiny synthetic HCS zarrs; the lite stores are built from them
by the real ``build_temporal_subset_zarr`` (no mocks), so the manifest derivation is
checked against exactly the provenance the real builder writes.

Run::

    uv run --no-sync pytest applications/dynacell/tools/generate_lite_benchmark_configs_test.py -q
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from build_temporal_subset_zarr import build_temporal_subset_zarr  # noqa: E402
from generate_lite_benchmark_configs import (  # noqa: E402
    ROSTER_MODELS,
    ROSTER_ORGANELLES,
    ROSTER_TRAIN_DIRS,
    ipsc_lite_positions,
    lite_grouped_leaf,
    lite_manifest,
    lite_predict_leaf,
    lite_predict_set,
    plan_outputs,
    rebase,
)
from iohub.ngff import TransformationMeta, open_ome_zarr

from dynacell.data.manifests import DatasetManifest, SplitDefinition
from dynacell.evaluation import paths


def _store(path: Path, frames: dict[str, int], channels: list[str]) -> None:
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=channels, version="0.5") as plate:
        for name, n_frames in frames.items():
            row, col, fov = name.split("/")
            pos = plate.create_position(row, col, fov)
            pos.create_image(
                "0",
                np.zeros((n_frames, len(channels), 2, 4, 4), dtype=np.float32),
                transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, 0.174, 0.1494, 0.1494])],
            )


def _production_manifest(test: Path, seg: Path, cache: Path) -> dict:
    return {
        "name": "a549-mantis-h2b-mock",
        "version": "1",
        "description": "production",
        "cell_type": "A549",
        "imaging_modality": "mantis-lightsheet",
        "spacing": {"z": 0.174, "y": 0.1494, "x": 0.1494},
        "channels": {"source": "Phase3D"},
        "targets": {
            "h2b": {
                "gene": "H2B",
                "organelle": "nuclei",
                "display_name": "Nuclei",
                "target_channel": "Nuclei",
                "stores": {
                    "train": "/unused/train.zarr",
                    "test": str(test),
                    "cell_segmentation": str(seg),
                    "gt_cache_dir": str(cache),
                },
                "splits": "splits/h2b_train_test.yaml",
            }
        },
    }


def _build_pair(tmp_path: Path, frames: dict[str, int], **kwargs) -> tuple[Path, Path, dict]:
    prod, lite = tmp_path / "prod", tmp_path / "lite"
    test, seg = prod / "test/gt.zarr", prod / "test/gt_seg.zarr"
    _store(test, frames, ["Phase3D", "Nuclei"])
    _store(seg, frames, ["segmentation"])
    for src, chans in ((test, ["Phase3D", "Nuclei"]), (seg, ["segmentation"])):
        build_temporal_subset_zarr(src, rebase(src, prod, lite), channels=chans, **kwargs)
    return prod, lite, _production_manifest(test, seg, prod / "eval_cache/h2b_mock")


def test_rebase_refuses_paths_outside_the_production_root() -> None:
    """A path not under DATA_ROOT has no lite counterpart."""
    assert rebase(paths.DATA_ROOT / "a549/x.zarr") == paths.LITE_DATA_ROOT / "a549/x.zarr"
    with pytest.raises(ValueError):
        rebase("/elsewhere/x.zarr")


def test_ipsc_lite_positions_is_a_fixed_seeded_draw() -> None:
    """The draw is deterministic, sorted, distinct and ignores input order."""
    names = [f"4/{i}/{i}" for i in range(100)]
    pick = ipsc_lite_positions(names)
    assert pick == ipsc_lite_positions(list(reversed(names)))
    assert pick == sorted(set(pick)) and len(pick) == 50
    assert pick != ipsc_lite_positions(names, seed=1)


def test_lite_manifest_rebases_stores_and_lists_lite_fovs(tmp_path: Path) -> None:
    """A549: every store and the GT cache move to the lite root; the split lists the lite FOVs."""
    frames = {f"0/0/fov{i:04d}": 10 for i in range(3)}
    prod, lite_root, production = _build_pair(tmp_path, frames, mode="spread", n_timepoints=5)
    manifest, splits = lite_manifest(production, data_root=prod, lite_root=lite_root)
    DatasetManifest.model_validate(manifest)
    stores = manifest["targets"]["h2b"]["stores"]
    assert manifest["name"] == "a549-mantis-h2b-mock-lite"
    assert "train" not in stores
    assert stores == {
        "test": str(lite_root / "test/gt.zarr"),
        "cell_segmentation": str(lite_root / "test/gt_seg.zarr"),
        "gt_cache_dir": str(lite_root / "eval_cache/h2b_mock"),
    }
    split = SplitDefinition.model_validate(splits["splits/h2b_train_test.yaml"])
    assert split.test == {"count": 3, "fovs": sorted(frames)}
    assert split.train == {"count": 0, "fovs": []}


def test_lite_manifest_rejects_a_store_not_derived_from_production(tmp_path: Path) -> None:
    """A lite path holding an unrelated store (no temporal_subset provenance) is refused."""
    frames = {"0/0/fov0000": 2}
    prod, lite_root, production = _build_pair(tmp_path, frames, mode="early", n_timepoints=1)
    other = tmp_path / "other_seg.zarr"
    _store(other, frames, ["segmentation"])
    production["targets"]["h2b"]["stores"]["cell_segmentation"] = str(prod / "test/other.zarr")
    (lite_root / "test/other.zarr").symlink_to(other)
    with pytest.raises(ValueError, match="not a temporal subset"):
        lite_manifest(production, data_root=prod, lite_root=lite_root)


def test_lite_manifest_rejects_gt_seg_frame_disagreement(tmp_path: Path) -> None:
    """GT and segmentation lite stores with different T per position are refused."""
    frames = {"0/0/fov0000": 10}
    prod, lite_root, production = _build_pair(tmp_path, frames, mode="spread", n_timepoints=5)
    seg = prod / "test/gt_seg.zarr"
    lite_seg = rebase(seg, prod, lite_root)
    shutil.rmtree(lite_seg)
    build_temporal_subset_zarr(seg, lite_seg, channels=["segmentation"], mode="spread", n_timepoints=3)
    with pytest.raises(ValueError, match="disagree"):
        lite_manifest(production, data_root=prod, lite_root=lite_root)


def test_lite_manifest_checks_the_ipsc_draw(tmp_path: Path) -> None:
    """A position-subset store must hold exactly the seeded lite draw of its source."""
    frames = {f"4/{i}/{i}": 1 for i in range(60)}
    wrong = sorted(frames)[:50]
    prod, lite_root, production = _build_pair(tmp_path, frames, mode="early", n_timepoints=1, positions=wrong)
    with pytest.raises(ValueError, match="seeded lite draw"):
        lite_manifest(production, data_root=prod, lite_root=lite_root)

    right = ipsc_lite_positions(list(frames))
    prod2, lite2, production2 = _build_pair(tmp_path / "ok", frames, mode="early", n_timepoints=1, positions=right)
    _, splits = lite_manifest(production2, data_root=prod2, lite_root=lite2)
    assert splits["splits/h2b_train_test.yaml"]["test"]["fovs"] == right


def _predict_leaf(ckpt: Path) -> dict:
    root = paths.DATA_ROOT / "nucleus/fnet3d_paper/a549/a549__mock"
    return {
        "base": [
            "../../../_internal/shared/model/predict_sets/a549_mantis_h2b_mock.yml",
            "../../../_internal/shared/model/targets/nucleus.yml",
        ],
        "benchmark": {"predict_set": "a549_mantis_h2b_mock", "experiment_id": "n__a549__fnet3d"},
        "model": {"init_args": {"ckpt_path": str(ckpt)}},
        "trainer": {
            "callbacks": [
                {
                    "class_path": "viscy_utils.callbacks.prediction_writer.HCSPredictionWriter",
                    "init_args": {"output_store": str(root / "prediction.zarr")},
                }
            ]
        },
        "launcher": {"job_name": "FNET_MOCK", "run_root": str(root)},
    }


def test_lite_predict_leaf_keeps_the_checkpoint_and_rebases_outputs(tmp_path: Path) -> None:
    """Same checkpoint and overlays; lite predict set; outputs under the lite root."""
    ckpt = tmp_path / "best.ckpt"
    ckpt.touch()
    prod = _predict_leaf(ckpt)
    lite = lite_predict_leaf(prod, {"a549_mantis_h2b_mock"})
    assert lite["base"][0].endswith("/predict_sets/a549_mantis_h2b_mock_lite.yml")
    assert lite["base"][1:] == prod["base"][1:]
    assert lite["model"] == prod["model"]
    assert lite["benchmark"]["predict_set"] == "a549_mantis_h2b_mock_lite"
    out = lite["trainer"]["callbacks"][0]["init_args"]["output_store"]
    assert out == str(paths.LITE_DATA_ROOT / "nucleus/fnet3d_paper/a549/a549__mock/prediction.zarr")
    assert lite["launcher"]["job_name"] == "FNET_MOCK_LITE"
    assert prod["launcher"]["job_name"] == "FNET_MOCK"


def test_lite_predict_leaf_refuses_a_missing_checkpoint(tmp_path: Path) -> None:
    """A lite prediction from a different checkpoint would not reproduce the full benchmark."""
    with pytest.raises(ValueError, match="is gone"):
        lite_predict_leaf(_predict_leaf(tmp_path / "gone.ckpt"), {"a549_mantis_h2b_mock"})


def test_lite_predict_set_points_at_the_lite_dataset() -> None:
    """Only the predict-set name and dataset slug change."""
    prod = {"benchmark": {"predict_set": "ipsc_confocal", "dataset_ref": {"dataset": "aics-hipsc"}}, "data": {}}
    lite = lite_predict_set(prod)
    assert lite["benchmark"] == {"predict_set": "ipsc_confocal_lite", "dataset_ref": {"dataset": "aics-hipsc-lite"}}


def test_lite_grouped_leaf_keeps_only_generated_predictions() -> None:
    """Conditions without a lite prediction are dropped; paths rebase; nuclei store is unset."""
    root = paths.DATA_ROOT

    def cond(model: str) -> dict:
        leaf = root / f"membrane/{model}/a549/a549__mock"
        return {
            "name": f"{model}__mock",
            "benchmark": {"dataset_ref": {"dataset": "a549-mantis-caax-mock", "target": "caax"}},
            "io": {
                "pred_path": str(leaf / "prediction.zarr"),
                "pred_cache_dir": str(root / f"a549/eval_cache_pred/membrane/{model}/a549/a549__mock"),
                "nuclei_gt_path": str(root / "a549/mantis/test/dual_nucl_memb_mock.zarr"),
            },
            "save": {"save_dir": str(leaf)},
        }

    prod = {"target_name": "membrane", "feature_metrics": {"patch_size": 256}, "conditions": [cond("a"), cond("b")]}
    keep = {paths.LITE_DATA_ROOT / "membrane/a/a549/a549__mock/prediction.zarr"}
    lite = lite_grouped_leaf(prod, keep)
    assert [c["name"] for c in lite["conditions"]] == ["a__mock"]
    (only,) = lite["conditions"]
    assert only["benchmark"]["dataset_ref"]["dataset"] == "a549-mantis-caax-mock-lite"
    assert "nuclei_gt_path" not in only["io"]
    assert only["save"]["save_dir"].startswith(str(paths.LITE_DATA_ROOT))
    assert lite["compute_microssim"] is False
    assert lite["feature_metrics"] == {
        "patch_size": 256,
        "compute_fid": False,
        "compute_prc": False,
        "compute_mind": False,
    }
    with pytest.raises(ValueError, match="no condition"):
        lite_grouped_leaf(prod, set())


def test_committed_lite_configs_are_current() -> None:
    """The committed lite files equal what the generator produces from today's production files."""
    if not paths.LITE_DATA_ROOT.exists():
        pytest.skip(f"{paths.LITE_DATA_ROOT} is not mounted")
    outputs = plan_outputs(ROSTER_ORGANELLES, ROSTER_MODELS, ROSTER_TRAIN_DIRS)
    stale = [str(p) for p, text in outputs.items() if not p.exists() or p.read_text() != text]
    assert not stale, f"regenerate with generate_lite_benchmark_configs.py: {stale}"
