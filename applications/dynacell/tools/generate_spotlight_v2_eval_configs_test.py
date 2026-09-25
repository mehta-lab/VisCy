"""Tests for ``generate_spotlight_v2_eval_configs.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from generate_grouped_eval_configs import _BASE_OVERLAY
from generate_spotlight_v2_eval_configs import (
    DEFAULT_ROSTER,
    PRED_FORCE,
    V2_ROOT,
    build_buckets,
    load_roster,
    main,
    store_problems,
)
from iohub.ngff import open_ome_zarr

from dynacell.evaluation import paths


def _write_roster(tmp_path: Path, waves: dict, leaves: list[str] | None = None) -> Path:
    roster = tmp_path / "roster.yaml"
    roster.write_text(yaml.safe_dump({"leaves": leaves or ["ipsc", "a549__mock"], "waves": waves}))
    return roster


def _write_plate(path: Path, positions: list[tuple[str, str, str]], shape: tuple[int, ...]) -> None:
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["ch"], version="0.5") as plate:
        for row, col, fov in positions:
            pos = plate.create_position(row, col, fov)
            pos.create_image("0", np.ones(shape, dtype=np.float32), chunks=(1, 1, 1, *shape[3:]))


def test_default_roster_rows_and_bucket_layout() -> None:
    """The stage-0 wave expands to 96 rows in 8 buckets of 12 conditions (later waves are separate)."""
    rows = [row for row in load_roster(DEFAULT_ROSTER) if row[0] == "stage0"]
    # 12 models x 2 organelles x 4 test legs, one bucket per (wave, organelle, leaf).
    assert len(rows) == 96
    buckets = build_buckets(rows)
    assert sorted(buckets) == sorted(
        f"spotlight_v2_{org}_{leaf}__stage0"
        for org in ("nucleus", "membrane")
        for leaf in ("ipsc", "a549_mock", "a549_denv", "a549_zikv")
    )
    assert {n for _, n in buckets.values()} == {12}


def test_outputs_go_to_the_v2_root_and_gt_cache_is_left_to_the_manifest() -> None:
    """save_dir and pred_cache_dir are rebased onto V2_ROOT; the GT cache is not overridden."""
    buckets = build_buckets(load_roster(DEFAULT_ROSTER))
    body, _ = buckets["spotlight_v2_membrane_a549_mock__stage0"]
    cond = next(c for c in body["conditions"] if c["name"] == "fnet2d_spotlight__ipsc_trained__a549_mock")
    assert cond["io"]["pred_path"] == str(
        paths.prediction_store("membrane", "fnet2d_spotlight", "ipsc", "a549", "mock")
    )
    assert cond["save"]["save_dir"] == str(V2_ROOT / "membrane/fnet2d_spotlight/ipsc/a549__mock")
    cache = V2_ROOT / "a549/eval_cache_pred/membrane/fnet2d_spotlight/ipsc/a549__mock"
    assert cond["io"]["pred_cache_dir"] == str(cache)
    assert "gt_cache_dir" not in cond["io"]
    # The whole-cell watershed still seeds from the separate A549 H2B store.
    assert cond["io"]["nuclei_gt_path"].endswith("dual_nucl_memb_mock.zarr")
    for body, _ in buckets.values():
        for c in body["conditions"]:
            assert Path(c["save"]["save_dir"]).is_relative_to(V2_ROOT)
            assert Path(c["io"]["pred_cache_dir"]).is_relative_to(V2_ROOT)


def test_every_pred_artifact_is_forced_without_touching_the_shared_overlay() -> None:
    """Every pred_* flag is forced, no gt_* flag is, and the generator overlay is untouched."""
    buckets = build_buckets(load_roster(DEFAULT_ROSTER))
    for body, _ in buckets.values():
        assert body["force_recompute"] == PRED_FORCE
        assert not any(k.startswith("gt_") or k == "all" for k in body["force_recompute"])
        assert body["compute_instance_ap"] is True
        assert body["segmentation"]["backend"] == "cpdino"
    assert _BASE_OVERLAY["force_recompute"] == {"final_metrics": True}


def test_pred_path_override_and_leaf_narrowing(tmp_path: Path) -> None:
    """A roster entry can narrow its test legs and point at a non-canonical store."""
    store = tmp_path / "clipped.zarr"
    roster = _write_roster(
        tmp_path,
        {"w1": {"nucleus": [{"model": "fnet2d", "leaves": ["ipsc"], "pred_paths": {"ipsc": str(store)}}]}},
    )
    assert load_roster(roster) == [("w1", "nucleus", "fnet2d", "ipsc", store)]


@pytest.mark.parametrize(
    ("waves", "match"),
    [
        ({"a": {"nucleus": ["fnet2d"]}, "b": {"nucleus": ["fnet2d"]}}, "is in wave 'a' and wave 'b'"),
        ({"a": {"er": ["fnet2d"]}}, "organelle 'er'"),
        ({"a": {"nucleus": ["not_a_model"]}}, "PAPER_KEY"),
    ],
)
def test_roster_rejects(tmp_path: Path, waves: dict, match: str) -> None:
    """Duplicate conditions across waves, out-of-scope organelles and unknown models raise."""
    with pytest.raises(ValueError, match=match):
        load_roster(_write_roster(tmp_path, waves))


def test_store_problems_catches_missing_positions_chunks_and_shape(tmp_path: Path) -> None:
    """The gate passes a complete store and names each kind of incompleteness."""
    gt = tmp_path / "gt.zarr"
    positions = [("A", "1", "0"), ("A", "1", "1")]
    _write_plate(gt, positions, (2, 1, 3, 8, 8))

    complete = tmp_path / "complete.zarr"
    _write_plate(complete, positions, (2, 1, 3, 8, 8))
    assert store_problems(complete, gt) == []

    missing_fov = tmp_path / "missing_fov.zarr"
    _write_plate(missing_fov, positions[:1], (2, 1, 3, 8, 8))
    assert any("position set differs" in p for p in store_problems(missing_fov, gt))

    wrong_z = tmp_path / "wrong_z.zarr"
    _write_plate(wrong_z, positions, (2, 1, 2, 8, 8))
    assert any("TZYX" in p for p in store_problems(wrong_z, gt))

    truncated = tmp_path / "truncated.zarr"
    _write_plate(truncated, positions, (2, 1, 3, 8, 8))
    chunk = sorted(p for p in (truncated / "A/1/1/0/c").rglob("*") if p.is_file())[-1]
    chunk.unlink()
    assert store_problems(truncated, gt) == ["A/1/1/0: 5/6 chunk files"]

    assert store_problems(tmp_path / "absent.zarr", gt) == [f"missing store {tmp_path / 'absent.zarr'}"]


def test_main_refuses_to_write_when_a_store_is_incomplete(tmp_path: Path) -> None:
    """An incomplete store fails the run before any leaf is written."""
    entry = {"model": "fnet2d", "leaves": ["ipsc"], "pred_paths": {"ipsc": str(tmp_path / "x.zarr")}}
    roster = _write_roster(tmp_path, {"w1": {"nucleus": [entry]}})
    out = tmp_path / "leaves"
    assert main(["--roster", str(roster), "--out-root", str(out)]) == 1
    assert not out.exists()
