"""Integration tests for ``retokenize_deconv_predictions.py``.

Each test builds a fake mini prediction tree + predict-leaf YAMLs under
``tmp_path`` and drives the REAL planning / apply / rollback functions (no mocks)
via ``--data-root`` / ``--config-root`` overrides.

Run::

    uv run pytest applications/dynacell/tools/retokenize_deconv_predictions_test.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from retokenize_deconv_predictions import (
    RETOKENIZE,
    ConfigEdit,
    DirMove,
    main,
    plan_config_edits,
    plan_dir_moves,
    preflight_moves,
    rollback,
)

_ORGANELLES = ("er", "mito")


def _make_store(train_set_dir: Path, leaf: str) -> Path:
    """Create a fake ``<train_set>/<leaf>/prediction.zarr`` with a sentinel file."""
    store = train_set_dir / leaf / "prediction.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text('{"zarr_format": 3}')
    return store


def _build_tree(data_root: Path) -> None:
    """Build a fake DATA_ROOT with ER/mito deconv + canonical dirs and distractors.

    Layout per organelle:
    - ``celldiff_r2/{a549__deconv,joint__legacy_deconvgt}``  -> re-tokenizable
    - ``fnet3d_paper/{a549__deconv,joint__legacy_deconvgt}`` -> re-tokenizable
    - ``celldiff_r2/ipsc``                                   -> already canonical (untouched)
    - ``celldiff_r2_iterative/ipsc``                         -> distinct model (untouched)
    Plus a nucleus/ tree that must never be touched.
    """
    for organelle in _ORGANELLES:
        for model in ("celldiff_r2", "fnet3d_paper"):
            for token in RETOKENIZE:
                for leaf in ("a549__mock", "a549__denv", "a549__zikv", "ipsc"):
                    _make_store(data_root / organelle / model / token, leaf)
            _make_store(data_root / organelle / model / "ipsc", "ipsc")  # already canonical
        _make_store(data_root / organelle / "celldiff_r2_iterative" / "ipsc", "ipsc")
    # nucleus is already canonical and deconv-invalid: must be ignored entirely.
    _make_store(data_root / "nucleus" / "fcmae_vscyto3d_pretrained" / "a549", "a549__mock")


def _predict_leaf(path: Path, output_store: str) -> None:
    """Write a minimal predict leaf carrying an HCSPredictionWriter output_store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "trainer:\n"
        "  callbacks:\n"
        "    - class_path: viscy_utils.callbacks.prediction_writer.HCSPredictionWriter\n"
        "      init_args:\n"
        f"        output_store: {output_store}\n"
        "# trailing comment preserved\n"
    )


# --------------------------------------------------------------------------- plan


def test_plan_dir_moves_enumerates_only_deconv_ermito(tmp_path: Path) -> None:
    """Only ER/mito deconv train_set dirs are enumerated; canonical/distinct dirs and nucleus are skipped."""
    data_root = tmp_path / "dynacell"
    _build_tree(data_root)

    moves = plan_dir_moves(data_root, _ORGANELLES, skips=[])

    # 2 organelles x 2 models x 2 deconv tokens = 8 moves.
    assert len(moves) == 8
    srcs = {m.src for m in moves}
    # canonical + distinct-model dirs are excluded.
    assert data_root / "er" / "celldiff_r2" / "ipsc" not in srcs
    assert data_root / "er" / "celldiff_r2_iterative" / "ipsc" not in srcs
    # nucleus never appears.
    assert all("nucleus" not in str(m.src) for m in moves)
    # correct token mapping.
    for m in moves:
        assert m.src.name in RETOKENIZE
        assert m.dst.name == RETOKENIZE[m.src.name]
        assert m.dst.parent == m.src.parent


def test_plan_dir_moves_skip_excludes(tmp_path: Path) -> None:
    """A --skip entry removes the matching dir from the move plan."""
    data_root = tmp_path / "dynacell"
    _build_tree(data_root)

    moves = plan_dir_moves(data_root, _ORGANELLES, skips=["mito/celldiff_r2/a549__deconv"])
    excluded = data_root / "mito" / "celldiff_r2" / "a549__deconv"
    assert all(m.src != excluded for m in moves)
    assert len(moves) == 7


def test_plan_dir_moves_rejects_an_unmatched_skip(tmp_path: Path) -> None:
    """A --skip entry that matches nothing raises instead of planning the move.

    The module docstring makes this list the live-writer contract, so an
    unmatched entry is the worst outcome: identical move count to passing no
    skip at all, while the operator believes a dir with an active SLURM writer
    is protected. ``celldiff_r2`` is the natural model-level phrasing and is
    exactly what _is_excluded does not accept — it matches a path tail, not a
    single component.
    """
    data_root = tmp_path / "dynacell"
    _build_tree(data_root)

    with pytest.raises(ValueError, match="matched no scanned directory"):
        plan_dir_moves(data_root, _ORGANELLES, skips=["celldiff_r2"])

    with pytest.raises(ValueError, match=r"\['mito/celldiff_r2/a549__typo'\]"):
        plan_dir_moves(data_root, _ORGANELLES, skips=["mito/celldiff_r2/a549__typo"])


def test_plan_dir_moves_organelle_filter(tmp_path: Path) -> None:
    """Restricting to one organelle limits the moves to that organelle."""
    data_root = tmp_path / "dynacell"
    _build_tree(data_root)

    moves = plan_dir_moves(data_root, ("er",), skips=[])
    assert len(moves) == 4
    assert all(m.organelle == "er" for m in moves)


def test_plan_config_edits_rewrites_deconv_output_store(tmp_path: Path) -> None:
    """Predict-leaf output_store deconv paths are rewritten to raw; canonical paths are left alone."""
    data_root = tmp_path / "dynacell"
    _build_tree(data_root)
    cfg = tmp_path / "configs"

    deconv = f"{data_root}/er/celldiff_r2/a549__deconv/a549__denv/prediction.zarr"
    joint = f"{data_root}/mito/fnet3d_paper/joint__legacy_deconvgt/ipsc/prediction.zarr"
    raw = f"{data_root}/nucleus/fcmae_vscyto3d_pretrained/a549/a549__mock/prediction.zarr"
    _predict_leaf(cfg / "predict__er_denv.yml", deconv)
    _predict_leaf(cfg / "predict__mito_joint.yml", joint)
    _predict_leaf(cfg / "predict__nucleus.yml", raw)  # canonical -> untouched

    edits = plan_config_edits([cfg], data_root, _ORGANELLES, skips=[])

    assert len(edits) == 2
    by_old = {e.old_value: e.new_value for e in edits}
    assert by_old[deconv] == f"{data_root}/er/celldiff_r2/a549/a549__denv/prediction.zarr"
    assert by_old[joint] == f"{data_root}/mito/fnet3d_paper/joint/ipsc/prediction.zarr"
    assert raw not in by_old


def test_plan_config_edits_skip_matches_dir_move(tmp_path: Path) -> None:
    """A skipped dir's config edit is dropped so configs and disk stay consistent."""
    data_root = tmp_path / "dynacell"
    _build_tree(data_root)
    cfg = tmp_path / "configs"
    deconv = f"{data_root}/mito/celldiff_r2/a549__deconv/a549__denv/prediction.zarr"
    _predict_leaf(cfg / "predict__mito.yml", deconv)

    edits = plan_config_edits([cfg], data_root, _ORGANELLES, skips=["mito/celldiff_r2/a549__deconv"])
    assert edits == []


# ----------------------------------------------------------------------- preflight


def test_preflight_collision_guard(tmp_path: Path) -> None:
    """A pre-existing destination is reported as a collision and kept out of pending."""
    data_root = tmp_path / "dynacell"
    _build_tree(data_root)
    # Pre-create a destination -> collision.
    (data_root / "er" / "celldiff_r2" / "a549").mkdir(parents=True)

    moves = plan_dir_moves(data_root, _ORGANELLES, skips=[])
    pending, already, errors = preflight_moves(moves)
    assert any("DEST EXISTS" in e and "er/celldiff_r2/a549" in e for e in errors)
    # the clashing move is not in pending.
    assert all(not (m.dst == data_root / "er" / "celldiff_r2" / "a549") for m in pending)


# --------------------------------------------------------------------------- apply


def _run(argv: list[str]) -> int:
    return main(argv)


def test_apply_moves_and_edits_then_rollback(tmp_path: Path) -> None:
    """Apply renames dirs + rewrites configs, is idempotent, and rolls back cleanly."""
    data_root = tmp_path / "dynacell"
    _build_tree(data_root)
    cfg = tmp_path / "configs"
    deconv = f"{data_root}/er/celldiff_r2/a549__deconv/a549__denv/prediction.zarr"
    leaf = cfg / "predict__er.yml"
    _predict_leaf(leaf, deconv)
    journal = tmp_path / "journal.jsonl"

    common = ["--data-root", str(data_root), "--config-root", str(cfg), "--journal", str(journal)]

    # Dry-run changes nothing.
    assert _run(common) == 0
    assert (data_root / "er" / "celldiff_r2" / "a549__deconv").is_dir()
    assert not (data_root / "er" / "celldiff_r2" / "a549").exists()
    assert deconv in leaf.read_text()
    assert not journal.exists()

    # Apply.
    assert _run(common + ["--apply"]) == 0
    assert not (data_root / "er" / "celldiff_r2" / "a549__deconv").exists()
    moved = data_root / "er" / "celldiff_r2" / "a549" / "a549__denv" / "prediction.zarr"
    assert (moved / "zarr.json").is_file()
    # sibling leaves travelled.
    assert (data_root / "er" / "celldiff_r2" / "a549" / "ipsc" / "prediction.zarr").is_dir()
    new_value = f"{data_root}/er/celldiff_r2/a549/a549__denv/prediction.zarr"
    assert new_value in leaf.read_text()
    assert deconv not in leaf.read_text()
    assert "trailing comment preserved" in leaf.read_text()  # comment survived
    assert journal.exists()

    # Idempotent re-apply: nothing left to do, no error.
    assert _run(common + ["--apply"]) == 0
    assert new_value in leaf.read_text()

    # Rollback restores everything.
    assert _run(["--journal", str(journal), "--rollback", "--apply"]) == 0
    assert (data_root / "er" / "celldiff_r2" / "a549__deconv" / "a549__denv" / "prediction.zarr").is_dir()
    assert not (data_root / "er" / "celldiff_r2" / "a549").exists()
    assert deconv in leaf.read_text()
    assert new_value not in leaf.read_text()


def test_apply_aborts_on_collision_without_mutation(tmp_path: Path) -> None:
    """A collision aborts the whole run with no mutation and no journal."""
    data_root = tmp_path / "dynacell"
    _build_tree(data_root)
    journal = tmp_path / "journal.jsonl"
    (data_root / "mito" / "fnet3d_paper" / "joint").mkdir(parents=True)  # collision

    rc = _run(["--data-root", str(data_root), "--no-configs", "--journal", str(journal), "--apply"])
    assert rc == 1
    # No move happened: the deconv src is still present, journal untouched.
    assert (data_root / "mito" / "fnet3d_paper" / "joint__legacy_deconvgt").is_dir()
    assert not journal.exists()


def test_rollback_missing_journal_raises(tmp_path: Path) -> None:
    """Rollback on a missing journal raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        rollback(tmp_path / "nope.jsonl", dry_run=True)


def test_dataclasses_are_frozen() -> None:
    """DirMove and ConfigEdit are immutable."""
    m = DirMove("er", "celldiff_r2", Path("/a/src"), Path("/a/dst"))
    e = ConfigEdit(Path("/x.yml"), "old", "new")
    with pytest.raises(Exception):
        m.src = Path("/other")  # type: ignore[misc]
    with pytest.raises(Exception):
        e.old_value = "x"  # type: ignore[misc]
