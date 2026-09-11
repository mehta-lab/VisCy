"""Integration tests for the Phase-7 migration manifest builder + apply tool.

Builds a self-contained fixture tree mirroring the on-disk source layout (both
checkpoint roots, prediction dual-homes, the eval families) and exercises the real
collectors + the ``run_migration`` apply / idempotent-resume / rollback cycle.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import build_migration_manifest as bmm
import run_migration as rm


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _make_ckpt_run(models_root: Path, root_name: str, term: str, org: str, run: str, payload: str = "ck") -> Path:
    """Create ``<root>/<term>/<org>/<run>/{checkpoints/last.ckpt,resolved,wandb}``."""
    run_dir = (
        models_root.parent / root_name / term / org / run if root_name != "dynacell" else models_root / term / org / run
    )
    _write(run_dir / "checkpoints" / "last.ckpt", payload)
    _write(run_dir / "checkpoints" / "epoch=9.ckpt", payload)
    (run_dir / "resolved").mkdir(parents=True, exist_ok=True)
    _write(run_dir / "resolved" / "fit.yml", "cfg")
    (run_dir / "wandb").mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_zarr(store: Path, meta: str = "meta") -> Path:
    store.mkdir(parents=True, exist_ok=True)
    _write(store / "zarr.json", meta)
    _write(store / "0" / "0", "chunk")
    return store


def _hardlink_tree(src: Path, dst: Path) -> None:
    """Recreate ``src`` at ``dst`` with every file hardlinked (``cp -al``)."""
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            os.link(f, dst / rel)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Return (models_root, data_root) populated with a representative slice."""
    models_root = tmp_path / "models" / "dynacell"
    data_root = tmp_path / "data" / "dynacell"
    models_root.mkdir(parents=True)
    data_root.mkdir(parents=True)

    # --- checkpoints ---
    # ER (sec61b) a549: single dynacell-root run, run-dir name carries a recipe suffix.
    _make_ckpt_run(models_root, "dynacell", "a549_mantis", "sec61b", "fcmae_vscyto3d_pretrained_ws8500")
    # Membrane joint: hardlink-duplicated across BOTH roots -> one canonical dest (dedup).
    # The whole run dir (checkpoints/ + resolved/ ...) is hardlinked, mirroring cp -al.
    a = _make_ckpt_run(models_root, "dynacell", "joint_ipsc_confocal_a549_mantis", "memb", "pix2pix3d_unetvit")
    b_run = "pix2pix3d_unetvit_modernized_lambdaL1_10_lecam_40ep"
    b = models_root.parent / "cell_diff_vs_viscy" / "joint_ipsc_confocal_a549_mantis" / "memb" / b_run
    _hardlink_tree(a, b)
    # A smoke run must be skipped (would collide with the real pix2pix run).
    _make_ckpt_run(models_root, "cell_diff_vs_viscy", "joint_ipsc_confocal_a549_mantis", "nucl", b_run + "_smoke")

    # --- predictions ---
    # Joint membrane prediction dual-homed (hardlinked) in predictions/ + joint_predictions/.
    p_name = "memb_fnet3d_paper_jointtrained_mock.zarr"
    p1 = _make_zarr(data_root / "a549" / "predictions" / p_name)
    _hardlink_tree(p1, data_root / "a549" / "joint_predictions" / p_name)
    # A plain single-home a549-trained nucleus prediction.
    _make_zarr(data_root / "a549" / "predictions" / "nucl_fnet3d_paper_a549trained_mock.zarr")
    # An ablation prediction -> skipped (own track).
    _make_zarr(data_root / "a549" / "predictions" / "nucl_vscyto3d_randinit_mock.zarr")

    # --- evals ---
    # Triad (default track), iPSC-trained on iPSC.
    (data_root / "ipsc" / "evaluations_with_embeddings" / "eval_vscyto3d_nucleus").mkdir(parents=True)
    # instance_ap subtrack, a549.
    (data_root / "a549" / "evaluations_instance_ap" / "eval_vscyto3d_nucleus_mock").mkdir(parents=True)
    # Own-track ablation eval (default-excluded).
    (data_root / "a549" / "evaluations_randinit" / "eval_vscyto3d_randinit_nucleus_mock").mkdir(parents=True)
    # Stale pre-2D parent -> normalize_legacy None -> skip.
    (data_root / "a549" / "evaluations" / "eval_vscyto3d_nucleus_mock").mkdir(parents=True)
    return models_root, data_root


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def test_checkpoints_dedup_and_noise_skip(tmp_path):
    models_root, _ = _fixture(tmp_path)
    rows, gaps = bmm.collect_checkpoints(models_root, full_hardlink_check=True)
    assert gaps == []
    moves = {Path(r.src): Path(r.dest) for r in rows if r.status == "move"}
    dedup = [r for r in rows if r.status == "dedup_legacy"]
    skips = [r for r in rows if r.status == "skip"]

    # Recipe-suffixed run canonicalizes to the bare model code key.
    er = models_root / "a549_mantis" / "sec61b" / "fcmae_vscyto3d_pretrained_ws8500"
    assert moves[er] == models_root / "a549" / "er" / "fcmae_vscyto3d_pretrained"

    # Hardlinked cross-root dup: dynacell-root copy wins the move; cell_diff is dedup_legacy.
    win = models_root / "joint_ipsc_confocal_a549_mantis" / "memb" / "pix2pix3d_unetvit"
    dest = models_root / "joint" / "membrane" / "pix2pix3d_unetvit"
    assert moves[win] == dest
    assert len(dedup) == 1
    assert dedup[0].dest == str(dest)
    assert "cell_diff_vs_viscy" in dedup[0].src

    # Smoke run skipped.
    assert any("_smoke" in s.src for s in skips)


def test_checkpoint_curation_prunes_experiments(tmp_path):
    """iPSC graveyard: noise/pre-D pix2pix/celldiff-R1 skipped; distinct-run collisions
    resolved by config reference (paper's final recipe wins)."""
    models_root = tmp_path / "models" / "dynacell"
    cd = models_root.parent / "cell_diff_vs_viscy"

    def mk(root, org, run):
        d = root / "ipsc" / org / run / "checkpoints"
        d.mkdir(parents=True)
        (d / "last.ckpt").write_text(run)  # distinct content -> non-hardlinked
        return d.parent

    # sec61b (er): a config-referenced final fcmae + an unreferenced bare variant + noise + R1 celldiff.
    mk(models_root, "sec61b", "fcmae_vscyto3d_pretrained_ws8500")  # paper final (referenced)
    mk(models_root, "sec61b", "fcmae_vscyto3d_pretrained")  # bare, superseded (unreferenced)
    mk(models_root, "sec61b", "fcmae_vscyto3d_pretrained_H100_sanity")  # noise
    mk(cd, "sec61b", "celldiff")  # pre-R2, dropped
    mk(cd, "sec61b", "celldiff_r2")  # paper final
    mk(cd, "sec61b", "pix2pix3d_unetvit_modernized_lambdaL1_10_lecam_40ep")  # Run D (kept)
    mk(cd, "sec61b", "pix2pix3d_unetvit_modernized_lambdaL1_1")  # pre-D (dropped)

    referenced = {str(models_root / "ipsc" / "sec61b" / "fcmae_vscyto3d_pretrained_ws8500")}
    rows, gaps = bmm.collect_checkpoints(models_root, full_hardlink_check=True, referenced_dirs=referenced)
    assert gaps == []
    moved = {r.src for r in rows if r.status == "move"}
    skipped = {r.src: r.reason for r in rows if r.status == "skip"}

    # Config-referenced final fcmae wins its dest; bare variant skipped as unreferenced.
    assert str(models_root / "ipsc" / "sec61b" / "fcmae_vscyto3d_pretrained_ws8500") in moved
    assert "unreferenced" in skipped[str(models_root / "ipsc" / "sec61b" / "fcmae_vscyto3d_pretrained")]
    # celldiff_r2 kept, pre-R2 celldiff dropped; Run D kept, pre-D pruned.
    assert str(cd / "ipsc" / "sec61b" / "celldiff_r2") in moved
    assert "superseded" in skipped[str(cd / "ipsc" / "sec61b" / "celldiff")]
    assert str(cd / "ipsc" / "sec61b" / "pix2pix3d_unetvit_modernized_lambdaL1_10_lecam_40ep") in moved
    assert "pre-D" in skipped[str(cd / "ipsc" / "sec61b" / "pix2pix3d_unetvit_modernized_lambdaL1_1")]
    assert "noise" in skipped[str(models_root / "ipsc" / "sec61b" / "fcmae_vscyto3d_pretrained_H100_sanity")]
    # Every canonical dest is unique (no collision survived).
    dests = [r.dest for r in rows if r.status == "move"]
    assert len(dests) == len(set(dests))


def test_predictions_dedup_and_ablation_skip(tmp_path):
    _, data_root = _fixture(tmp_path)
    rows, skips, gaps = bmm.collect_predictions(data_root, full_hardlink_check=True)
    assert gaps == []
    moves = {Path(r.src): Path(r.dest) for r in rows if r.status == "move"}
    dedup = [r for r in rows if r.status == "dedup_legacy"]

    dest = data_root / "membrane" / "fnet3d_paper" / "joint" / "a549__mock" / "prediction.zarr"
    # Dual-home dedups to one move; winner is lexicographically first (joint_predictions).
    assert dest in moves.values()
    assert len(dedup) == 1
    assert "joint_predictions" in [r.src for r in rows if r.status == "move" and r.dest == str(dest)][0]

    # Plain single-home prediction present.
    plain = data_root / "a549" / "predictions" / "nucl_fnet3d_paper_a549trained_mock.zarr"
    assert moves[plain] == data_root / "nucleus" / "fnet3d_paper" / "a549" / "a549__mock" / "prediction.zarr"

    # Ablation prediction skipped, not a gap.
    assert any("randinit" in s for s in skips)


def test_evals_scope_default_excludes_own_track(tmp_path):
    _, data_root = _fixture(tmp_path)
    rows, skips, gaps = bmm.collect_evals(data_root, include_own_track=False)
    assert gaps == []
    dests = {r.dest for r in rows if r.status == "move"}

    # Triad default-track leaf.
    assert str(data_root / "nucleus" / "fcmae_vscyto3d_pretrained" / "ipsc" / "ipsc") in dests
    # instance_ap subtrack leaf.
    assert str(data_root / "nucleus" / "fcmae_vscyto3d_pretrained" / "ipsc" / "a549__mock" / "instance_ap") in dests
    # Own-track ablation + stale pre-2D are skipped (not moved).
    assert not any("randinit" in r.dest for r in rows if r.status == "move")
    assert any("own-track" in s for s in skips)
    assert any("normalize->None" in s for s in skips)


def test_evals_scope_include_own_track(tmp_path):
    _, data_root = _fixture(tmp_path)
    rows, _, gaps = bmm.collect_evals(data_root, include_own_track=True)
    assert gaps == []
    assert any(r.status == "move" and "fcmae_vscyto3d_pretrained_randinit" in r.dest for r in rows)


def test_builder_main_writes_manifest(tmp_path, monkeypatch):
    models_root, data_root = _fixture(tmp_path)
    monkeypatch.setattr(bmm, "MODELS_ROOT", models_root)
    monkeypatch.setattr(bmm, "DATA_ROOT", data_root)
    out = tmp_path / "manifest.csv"
    rc = bmm.main(["--out", str(out), "--full-hardlink-check"])
    assert rc == 0
    assert out.exists()
    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["status"] for r in rows} <= {"move", "dedup_legacy", "skip"}
    assert any(r["kind"] == "checkpoint" and r["status"] == "move" for r in rows)
    assert (tmp_path / "manifest.report.txt").exists()


# ---------------------------------------------------------------------------
# apply tool
# ---------------------------------------------------------------------------


def test_apply_idempotent_and_rollback(tmp_path, monkeypatch):
    models_root, data_root = _fixture(tmp_path)
    monkeypatch.setattr(bmm, "MODELS_ROOT", models_root)
    monkeypatch.setattr(bmm, "DATA_ROOT", data_root)
    manifest = tmp_path / "manifest.csv"
    assert bmm.main(["--out", str(manifest), "--full-hardlink-check"]) == 0

    moves = rm.load_moves(manifest)
    pending, merges, already, errors = rm.preflight(moves)
    assert errors == []
    assert already == []
    assert merges == []  # fixture evals map to fresh leaves -> plain renames
    assert len(pending) == len(moves)

    # Apply for real.
    journal = tmp_path / "manifest.journal.csv"
    applied = rm.apply_moves(pending, merges, journal, dry_run=False)
    assert applied == len(pending)
    # Every winner src is gone; every dest exists.
    for m in pending:
        assert not m.src.exists()
        assert m.dest.exists()
    # A moved checkpoint carried its siblings (resolved/) along.
    er_dest = models_root / "a549" / "er" / "fcmae_vscyto3d_pretrained"
    assert (er_dest / "resolved" / "fit.yml").is_file()
    assert (er_dest / "checkpoints" / "last.ckpt").is_file()

    # Idempotent re-run: everything is already-applied, nothing pending.
    pending2, merges2, already2, errors2 = rm.preflight(rm.load_moves(manifest))
    assert errors2 == []
    assert pending2 == []
    assert merges2 == []
    assert len(already2) == len(moves)

    # Rollback reverses every applied row.
    reversed_n = rm.rollback(journal, dry_run=False)
    assert reversed_n == applied
    for m in pending:
        assert m.src.exists()
        assert not m.dest.exists()


def test_preflight_blocks_preexisting_dest(tmp_path, monkeypatch):
    models_root, data_root = _fixture(tmp_path)
    monkeypatch.setattr(bmm, "MODELS_ROOT", models_root)
    monkeypatch.setattr(bmm, "DATA_ROOT", data_root)
    manifest = tmp_path / "manifest.csv"
    assert bmm.main(["--out", str(manifest), "--full-hardlink-check"]) == 0
    moves = rm.load_moves(manifest)
    # Pre-create one checkpoint dest so BOTH src and dest exist -> unexpected, must
    # error (the merge path is eval-only; checkpoints/predictions never merge).
    ckpt_move = next(m for m in moves if m.kind == "checkpoint")
    ckpt_move.dest.mkdir(parents=True, exist_ok=True)
    _, _, _, errors = rm.preflight(moves)
    assert any("DEST ALREADY EXISTS" in e for e in errors)


def test_eval_merge_is_planned_when_the_prediction_move_will_create_the_leaf(tmp_path):
    """The eval dest need not exist YET for the eval to be a merge, not a rename.

    apply_moves walks pending in manifest order (checkpoints, predictions,
    evals) and mkdir(parents=True)s each dest's parent, and prediction_store()
    returns ``<leaf>/prediction.zarr`` — a child of the eval's dest. So the
    prediction move materializes the eval's dest, non-empty, before the eval
    move is reached. Classifying on the pre-move tree called that eval a
    whole-dir rename, and src.rename(dest) then raised ENOTEMPTY *after* the
    checkpoint and prediction renames had already committed — the partial,
    half-journaled mutation this module promises cannot happen.
    """
    leaf = tmp_path / "data" / "er" / "celldiff_r2" / "a549__deconv" / "a549__denv"
    # Nothing at the leaf yet: the prediction move has not run.
    pred_src = tmp_path / "data" / "a549" / "predictions" / "sec61b_celldiff_r2_denv.zarr"
    _make_zarr(pred_src)
    eval_src = (
        tmp_path / "data" / "a549" / "evaluations_a549trained_with_embeddings" / "eval_celldiff_r2_a549trained_er_denv"
    )
    _write(eval_src / "pixel_metrics.csv", "p")
    _write(eval_src / "feature_metrics.csv", "f")

    pred_move = rm.Move("prediction", pred_src, leaf / "prediction.zarr", "pred->leaf")
    eval_move = rm.Move("eval", eval_src, leaf, "eval->leaf")

    pending, merges, already, errors = rm.preflight([pred_move, eval_move])
    assert errors == []
    assert pending == [pred_move]
    assert merges == [eval_move], "eval must be a merge, not a whole-dir rename"

    # And the whole plan applies cleanly in one pass, with nothing left behind.
    journal = tmp_path / "journal.csv"
    assert rm.apply_moves(pending, merges, journal, dry_run=False) == 2
    assert (leaf / "prediction.zarr" / "zarr.json").is_file()
    assert (leaf / "pixel_metrics.csv").is_file()
    assert (leaf / "feature_metrics.csv").is_file()
    assert not eval_src.exists()
    assert not pred_src.exists()


def test_eval_merges_into_shared_prediction_leaf(tmp_path):
    """An eval whose canonical dest leaf already holds prediction.zarr is merged,
    not whole-dir renamed onto it; rollback restores the eval without touching
    prediction.zarr."""
    leaf = tmp_path / "data" / "er" / "celldiff_r2" / "a549__deconv" / "a549__denv"
    # The prediction move ran first: the leaf exists and holds prediction.zarr.
    _make_zarr(leaf / "prediction.zarr")
    # Legacy eval src with typical outputs (no prediction.zarr, no instance_ap).
    src = (
        tmp_path / "data" / "a549" / "evaluations_a549trained_with_embeddings" / "eval_celldiff_r2_a549trained_er_denv"
    )
    _write(src / "pixel_metrics.csv", "p")
    _write(src / "feature_metrics.csv", "f")
    _write(src / "embeddings" / "gt_cp.npz", "e")
    _make_zarr(src / "segmentation_results.zarr")

    move = rm.Move("eval", src, leaf, "eval->leaf")
    pending, merges, already, errors = rm.preflight([move])
    assert errors == []
    assert pending == []
    assert merges == [move]

    journal = tmp_path / "journal.csv"
    assert rm.apply_moves(pending, merges, journal, dry_run=False) == 1
    # Merge result: prediction.zarr survives + every eval child landed alongside it.
    assert (leaf / "prediction.zarr" / "zarr.json").is_file()
    assert (leaf / "pixel_metrics.csv").is_file()
    assert (leaf / "embeddings" / "gt_cp.npz").is_file()
    assert (leaf / "segmentation_results.zarr" / "zarr.json").is_file()
    assert not src.exists()  # emptied src dir removed

    # Rollback: eval children return to src; prediction.zarr stays in the leaf.
    rm.rollback(journal, dry_run=False)
    assert (src / "pixel_metrics.csv").is_file()
    assert (src / "embeddings" / "gt_cp.npz").is_file()
    assert not (leaf / "pixel_metrics.csv").exists()
    assert (leaf / "prediction.zarr" / "zarr.json").is_file()
