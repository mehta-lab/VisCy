"""Tests for ``generate_spotlight_v2_eval_configs.py``."""

from __future__ import annotations

from pathlib import Path

import generate_spotlight_v2_eval_configs
import numpy as np
import pytest
import yaml
from generate_grouped_eval_configs import _BASE_OVERLAY, _LEAF_OUT_ROOT
from generate_spotlight_v2_eval_configs import (
    DEFAULT_ROSTER,
    PRED_FORCE,
    V2_ROOT,
    build_buckets,
    gt_test_store,
    load_roster,
    main,
    parsed_row,
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
        if body["target_name"] in {"nucleus", "membrane"}:
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
        # Roster keys are the on-disk tokens: ``mito``, never the eval target ``mitochondria``.
        ({"a": {"mitochondria": ["fnet2d"]}}, "organelle 'mitochondria'"),
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


def test_wave_filter_gates_and_writes_only_the_named_wave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--wave skips an unfinished wave's stores instead of failing on them."""
    gt = tmp_path / "gt.zarr"
    _write_plate(gt, [("A", "1", "0")], (1, 1, 3, 8, 8))
    monkeypatch.setattr(generate_spotlight_v2_eval_configs, "gt_test_store", lambda parsed: gt)
    done = {"model": "fnet2d", "leaves": ["ipsc"], "pred_paths": {"ipsc": str(gt)}}
    pending = {"model": "fnet3d_paper", "leaves": ["ipsc"], "pred_paths": {"ipsc": str(tmp_path / "x.zarr")}}
    roster = _write_roster(tmp_path, {"done": {"nucleus": [done]}, "pending": {"nucleus": [pending]}})
    out = tmp_path / "leaves"
    assert main(["--roster", str(roster), "--out-root", str(out), "--dry-run"]) == 1
    assert main(["--roster", str(roster), "--out-root", str(out), "--wave", "done"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["spotlight_v2_nucleus_ipsc__done"]
    with pytest.raises(ValueError, match="not in roster"):
        main(["--roster", str(roster), "--out-root", str(out), "--wave", "nope"])


# Canonical paper buckets the ER/mito Spotlight-v2 buckets must score like.
_CANONICAL_BUCKET = {"er": "er_ipsc_trained", "mito": "mitochondria_ipsc_trained"}
_LEAF_SUFFIX = {"ipsc": "ipsc", "a549__mock": "a549_mock", "a549__denv": "a549_denv", "a549__zikv": "a549_zikv"}


@pytest.mark.parametrize("organelle", ["er", "mito"])
def test_er_mito_buckets_score_like_the_canonical_grouped_leaf(tmp_path: Path, organelle: str) -> None:
    """Every scoring field of an ER/mito bucket equals the canonical ipsc_trained grouped leaf's.

    Only the outputs (save_dir, pred_cache_dir under V2_ROOT), the store and the
    condition name may differ, plus force_recompute (every pred_* forced).
    """
    leaves = list(_LEAF_SUFFIX)
    rows = load_roster(_write_roster(tmp_path, {"w": {organelle: ["fnet3d_vscyto3daug_v2"]}}, leaves))
    buckets = build_buckets(rows)
    with (_LEAF_OUT_ROOT / _CANONICAL_BUCKET[organelle] / "eval_grouped.yaml").open() as f:
        canonical = yaml.safe_load(f)
    for leaf, suffix in _LEAF_SUFFIX.items():
        body, n = buckets[f"spotlight_v2_{organelle}_{suffix}__w"]
        assert n == 1
        # Top level: same target, overlay and (absent) segmentation / instance-AP settings.
        assert {k: v for k, v in body.items() if k not in {"conditions", "force_recompute"}} == {
            k: v for k, v in canonical.items() if k not in {"conditions", "force_recompute"}
        }
        assert body["target_name"] == {"er": "er", "mito": "mitochondria"}[organelle]
        assert "segmentation" not in body and "compute_instance_ap" not in body
        assert body["force_recompute"] == PRED_FORCE
        (cond,) = body["conditions"]
        ref = next(c for c in canonical["conditions"] if c["name"] == f"fnet3d__ipsc_trained__{suffix}")
        assert cond["name"] == f"fnet3d_vscyto3daug_v2__ipsc_trained__{suffix}"
        assert cond["benchmark"] == ref["benchmark"]
        assert set(cond["io"]) == set(ref["io"]) == {"pred_path", "pred_cache_dir"}
        test_set, _, condition = leaf.partition("__")
        assert cond["io"]["pred_path"] == str(
            paths.prediction_store(organelle, "fnet3d_vscyto3daug_v2", "ipsc", test_set, condition or None)
        )
        assert cond["save"]["save_dir"] == str(V2_ROOT / organelle / "fnet3d_vscyto3daug_v2/ipsc" / leaf)
        assert cond["io"]["pred_cache_dir"] == str(
            V2_ROOT / test_set / "eval_cache_pred" / organelle / "fnet3d_vscyto3daug_v2/ipsc" / leaf
        )


@pytest.mark.parametrize(
    ("organelle", "leaf", "store"),
    [
        ("er", "ipsc", "ipsc/dataset_v4/test_cropped/SEC61B.zarr"),
        ("mito", "ipsc", "ipsc/dataset_v4/test_cropped/TOMM20.zarr"),
        ("er", "a549__denv", "a549/mantis/test/SEC61B_DENV.zarr"),
        ("mito", "a549__zikv", "a549/mantis/test/TOMM20_ZIKV.zarr"),
    ],
)
def test_er_mito_gate_compares_against_the_per_organelle_gt(organelle: str, leaf: str, store: str) -> None:
    """The store gate reads the per-organelle iPSC GT and the A549 sec61b/tomm20 test sets."""
    parsed = parsed_row(organelle, "fnet3d_vscyto3daug_v2", leaf, Path("unused"))
    assert gt_test_store(parsed) == paths.DATA_ROOT / store
