"""Tests for submit_benchmark_job.py: sbatch rendering, byte-equivalence, flags."""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch
from iohub.ngff import open_ome_zarr
from lightning.pytorch import LightningModule, Trainer

from viscy_data import HCSDataModule
from viscy_utils.callbacks.prediction_writer import PREDICTION_COMPLETE_KEY, HCSPredictionWriter

yaml = pytest.importorskip("yaml")

# submit_benchmark_job is importable because the root pyproject.toml's
# [tool.pytest.ini_options].pythonpath adds applications/dynacell/tools to sys.path.
import submit_benchmark_job as sbj  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = REPO_ROOT / "applications" / "dynacell" / "configs" / "benchmarks" / "virtual_staining"


@pytest.fixture(scope="module")
def rendered_celldiff_sbatch():
    """Render the celldiff leaf once per module; tests below share the output.

    Resolver auto-discovers the bundled manifest registry via the
    ``dynacell.manifest_roots`` entry point — no ``DYNACELL_MANIFEST_ROOTS``
    setup needed. ``capsys`` is function-scoped, so use
    ``contextlib.redirect_stdout`` to capture the print-script output.
    """
    leaf = BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sbj.submit([str(leaf), "--print-script"])
    assert rc == 0
    return buf.getvalue()


def test_parse_override_scalar_and_nested():
    path, val = sbj._parse_override("trainer.max_epochs=50")
    assert path == ["trainer", "max_epochs"]
    assert val == 50


def test_parse_override_rejects_interpolation():
    with pytest.raises(SystemExit, match=r"\$\{\.\.\.\} interpolation"):
        sbj._parse_override("trainer.devices=${oc.env:NGPUS}")


def test_parse_override_missing_equals():
    with pytest.raises(SystemExit, match="missing '='"):
        sbj._parse_override("trainer.max_epochs")


def test_apply_override_deep_merges():
    composed = {"trainer": {"max_epochs": 20, "precision": "bf16"}}
    result = sbj._apply_override(composed, ["trainer", "max_epochs"], 50)
    assert result == {"trainer": {"max_epochs": 50, "precision": "bf16"}}


def test_render_sbatch_directives_matches_dihan_order():
    sbatch = {
        "partition": "gpu",
        "nodes": 1,
        "ntasks_per_node": 1,
        "cpus_per_task": 32,
        "gpus": 1,
        "mem": "256G",
        "constraint": "h200",
        "time": "4-00:00:00",
    }
    rendered = sbj._render_sbatch_directives("CELLDiff_SEC61B", "/foo/bar", sbatch)
    lines = rendered.splitlines()
    # First line is job-name, last two are output/error.
    assert lines[0] == "#SBATCH --job-name=CELLDiff_SEC61B"
    assert lines[1] == "#SBATCH --time=4-00:00:00"
    assert '--constraint="h200"' in rendered
    assert lines[-2] == "#SBATCH --output=/foo/bar/slurm/%j.out"
    assert lines[-1] == "#SBATCH --error=/foo/bar/slurm/%j.err"


def test_render_env_block_preserves_order():
    env = {"PYTHONUNBUFFERED": "1", "NCCL_DEBUG": "INFO", "PYTHONFAULTHANDLER": "1"}
    rendered = sbj._render_env_block(env)
    assert rendered.splitlines() == [
        "export PYTHONUNBUFFERED=1",
        "export NCCL_DEBUG=INFO",
        "export PYTHONFAULTHANDLER=1",
    ]


@pytest.mark.parametrize(
    "leaf_subpath,expected_resolved_prefix",
    [
        ("er/celldiff/ipsc_confocal/train.yml", "/resolved/fit_CELLDiff_SEC61B_"),
        ("er/unetvit3d/ipsc_confocal/train.yml", "/resolved/fit_UNetViT3D_SEC61B_"),
    ],
)
def test_rendered_sbatch_has_srun_at_expected_resolved_path(capsys, leaf_subpath, expected_resolved_prefix):
    """Rendered sbatch ends with an srun line pointing at the frozen resolved config."""
    leaf = BENCHMARKS / leaf_subpath

    # --print-script is preview-only (no disk writes), so this is safe to run
    # against a leaf whose launcher.run_root we may not have permission to write.
    rc = sbj.submit([str(leaf), "--print-script"])
    assert rc == 0
    rendered = capsys.readouterr().out

    srun_line = rendered.splitlines()[-1]
    assert srun_line.startswith("srun --cpu-bind=none uv run python -m dynacell fit --config")
    assert expected_resolved_prefix in srun_line


def test_resume_from_renders_ckpt_path(capsys, tmp_path):
    """--resume-from appends --ckpt_path=<explicit path> to the fit srun line."""
    ckpt = tmp_path / "resume.ckpt"
    ckpt.write_bytes(b"stub")
    leaf = BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"
    rc = sbj.submit([str(leaf), "--resume-from", str(ckpt), "--print-script"])
    assert rc == 0
    srun_line = capsys.readouterr().out.splitlines()[-1]
    assert srun_line.startswith("srun --cpu-bind=none uv run python -m dynacell fit --config")
    assert f"--ckpt_path={ckpt}" in srun_line


def test_resume_missing_checkpoint_raises(tmp_path):
    """--resume-from a nonexistent checkpoint fails fast before submission."""
    leaf = BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"
    with pytest.raises(SystemExit, match="resume checkpoint not found"):
        sbj.submit([str(leaf), "--resume-from", str(tmp_path / "missing.ckpt"), "--print-script"])


def test_resume_prefers_newest_last_v_ckpt(capsys, tmp_path):
    """--resume anchors on the NEWEST last*.ckpt by mtime, not the fixed last.ckpt.

    Lightning writes last-v1.ckpt (then -v2, ...) whenever last.ckpt already exists,
    so on a resumed run last.ckpt is the FIRST segment's state -- measured 14-91
    epochs stale across seven joint FCMAE arms at completion. Baking it into the
    sbatch script rewinds the run (job 35154718 lost 83 epochs on 2026-08-07, and
    Slurm replays the stored script verbatim on every requeue)."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    stale = ckpt_dir / "last.ckpt"
    stale.write_bytes(b"stub")
    newest = ckpt_dir / "last-v1.ckpt"
    newest.write_bytes(b"stub")
    older = newest.stat().st_mtime - 100
    os.utime(stale, (older, older))

    leaf = BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"
    rc = sbj.submit([str(leaf), "--override", f"launcher.run_root={tmp_path}", "--resume", "--print-script"])
    assert rc == 0
    srun_line = capsys.readouterr().out.splitlines()[-1]
    assert f"--ckpt_path={newest}" in srun_line
    assert f"--ckpt_path={stale}" not in srun_line


def test_resume_from_wins_over_newest_last_ckpt(capsys, tmp_path):
    """An explicit --resume-from is used verbatim, even when a newer last*.ckpt exists."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "last-v9.ckpt").write_bytes(b"stub")
    explicit = tmp_path / "explicit.ckpt"
    explicit.write_bytes(b"stub")

    leaf = BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"
    rc = sbj.submit(
        [
            str(leaf),
            "--override",
            f"launcher.run_root={tmp_path}",
            "--resume-from",
            str(explicit),
            "--print-script",
        ]
    )
    assert rc == 0
    assert f"--ckpt_path={explicit}" in capsys.readouterr().out.splitlines()[-1]


def test_resume_without_any_last_ckpt_raises(tmp_path):
    """--resume against a checkpoint dir holding no last*.ckpt fails before submission."""
    (tmp_path / "checkpoints").mkdir()
    leaf = BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"
    with pytest.raises(SystemExit, match=r"resume checkpoint not found: no last\*\.ckpt in"):
        sbj.submit([str(leaf), "--override", f"launcher.run_root={tmp_path}", "--resume", "--print-script"])


def test_resume_rejects_predict_mode(tmp_path):
    """--resume/--resume-from is fit-only; a predict leaf must error before rendering."""
    ckpt = tmp_path / "resume.ckpt"
    ckpt.write_bytes(b"stub")
    leaf = BENCHMARKS / "mito/fcmae_vscyto3d_scratch/a549_mantis/predict__a549_mantis_denv.yml"
    with pytest.raises(SystemExit, match="only valid for fit mode"):
        sbj.submit([str(leaf), "--resume-from", str(ckpt), "--print-script"])


def test_ckpt_explicit_path_overrides(capsys, tmp_path):
    """--ckpt PATH replaces the predict leaf's hardcoded model.init_args.ckpt_path."""
    ckpt = tmp_path / "custom.ckpt"
    ckpt.write_bytes(b"stub")
    leaf = BENCHMARKS / "mito/fcmae_vscyto3d_scratch/a549_mantis/predict__a549_mantis_denv.yml"
    rc = sbj.submit([str(leaf), "--ckpt", str(ckpt), "--print-resolved-config"])
    assert rc == 0
    assert f"ckpt_path: {ckpt}" in capsys.readouterr().out


def test_ckpt_last_prefers_newest_last_v_ckpt(capsys, tmp_path):
    """--ckpt last resolves the NEWEST last*.ckpt by mtime, not the fixed name.

    Same invariant as --resume (resolve_newest_last_ckpt): Lightning renames to
    last-vN.ckpt whenever last.ckpt exists, so on any resumed fit the file literally
    named last.ckpt is the first segment's state. Measured on live dirs, ipsc/er
    fnet3d_paper holds last.ckpt at epoch 2 beside last-v5.ckpt at epoch 228 -- so
    keying on the name predicted from a near-init checkpoint, silently.
    """
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    stale = ckpt_dir / "last.ckpt"
    stale.write_bytes(b"stub")
    newest = ckpt_dir / "last-v1.ckpt"
    newest.write_bytes(b"stub")
    older = newest.stat().st_mtime - 100
    os.utime(stale, (older, older))

    leaf = BENCHMARKS / "mito/fcmae_vscyto3d_scratch/a549_mantis/predict__a549_mantis_denv.yml"
    rc = sbj.submit(
        [
            str(leaf),
            "--override",
            f"model.init_args.ckpt_path={stale}",
            "--ckpt",
            "last",
            "--print-resolved-config",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert f"ckpt_path: {newest}" in out
    assert f"ckpt_path: {stale}" not in out


def test_ckpt_missing_raises(tmp_path):
    """--ckpt pointing at a nonexistent checkpoint fails fast before submission."""
    leaf = BENCHMARKS / "mito/fcmae_vscyto3d_scratch/a549_mantis/predict__a549_mantis_denv.yml"
    with pytest.raises(SystemExit, match="missing checkpoint"):
        sbj.submit([str(leaf), "--ckpt", str(tmp_path / "nope.ckpt"), "--print-resolved-config"])


def test_ckpt_rejects_fit_mode():
    """--ckpt is predict-only; a fit leaf must error (fit uses --resume)."""
    leaf = BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"
    with pytest.raises(SystemExit, match="only valid for predict"):
        sbj.submit([str(leaf), "--ckpt", "best", "--print-script"])


def test_resolve_best_ckpt_reads_best_model_path(tmp_path):
    """_resolve_best_ckpt returns ModelCheckpoint.best_model_path from last.ckpt."""
    best = tmp_path / "epoch=7-step=100.ckpt"
    best.write_bytes(b"x")
    (tmp_path / "epoch=3-step=40.ckpt").write_bytes(b"x")
    torch.save(
        {"callbacks": {"ModelCheckpoint{'monitor': 'loss/validate'}": {"best_model_path": str(best)}}},
        tmp_path / "last.ckpt",
    )
    assert sbj._resolve_best_ckpt(tmp_path) == best


def test_resolve_best_ckpt_fallback_highest_epoch(tmp_path):
    """Without a usable last.ckpt state, fall back to the highest-epoch ckpt."""
    for name in ("epoch=2-step=20.ckpt", "epoch=13-step=130.ckpt", "epoch=5-step=50.ckpt"):
        (tmp_path / name).write_bytes(b"x")
    assert sbj._resolve_best_ckpt(tmp_path).name == "epoch=13-step=130.ckpt"


def test_resolve_best_ckpt_skips_nonconforming_epoch_files(tmp_path):
    """A nonconforming ``epoch=*.ckpt`` (no digits) is skipped, not a crash, in the
    highest-epoch fallback."""
    (tmp_path / "epoch=final.ckpt").write_bytes(b"x")  # would break int(re.match(...).group(1))
    (tmp_path / "epoch=4-step=40.ckpt").write_bytes(b"x")
    (tmp_path / "epoch=11-step=110.ckpt").write_bytes(b"x")
    assert sbj._resolve_best_ckpt(tmp_path).name == "epoch=11-step=110.ckpt"


def test_resolve_best_ckpt_rebases_moved_dir(tmp_path):
    """best_model_path stored as a stale absolute path (moved/renamed ckpt dir) is
    re-based onto ckpt_dir, NOT silently degraded to the highest-epoch fallback."""
    # the true best (ep7) exists in the *current* dir; a later, more-overfit ckpt
    # (ep13) also exists, which the highest-epoch fallback would wrongly pick.
    best = tmp_path / "epoch=7-step=100.ckpt"
    best.write_bytes(b"x")
    (tmp_path / "epoch=13-step=130.ckpt").write_bytes(b"x")
    stale_abs = f"/some/old/moved/tree/checkpoints/{best.name}"  # does not exist
    torch.save(
        {"callbacks": {"ModelCheckpoint{'monitor': 'loss/validate'}": {"best_model_path": stale_abs}}},
        tmp_path / "last.ckpt",
    )
    assert sbj._resolve_best_ckpt(tmp_path) == best


def test_resolve_best_ckpt_prefers_newest_last_v(tmp_path):
    """A resumed run leaves last.ckpt + last-vN.ckpt; the newest (by mtime) is
    authoritative. Keying on last.ckpt alone mis-resolves to the first segment's best
    (the messy legacy iPSC dirs: last.ckpt is epoch 1, last-v5.ckpt is epoch 183)."""
    stale_best = tmp_path / "epoch=1-step=10.ckpt"
    stale_best.write_bytes(b"x")
    true_best = tmp_path / "epoch=183-step=1830.ckpt"
    true_best.write_bytes(b"x")
    torch.save(
        {"callbacks": {"ModelCheckpoint{'monitor': 'loss/validate'}": {"best_model_path": str(stale_best)}}},
        tmp_path / "last.ckpt",
    )
    torch.save(
        {"callbacks": {"ModelCheckpoint{'monitor': 'loss/validate'}": {"best_model_path": str(true_best)}}},
        tmp_path / "last-v5.ckpt",
    )
    # make last.ckpt older so last-v5.ckpt is the newest by mtime
    older = (tmp_path / "last-v5.ckpt").stat().st_mtime - 100
    os.utime(tmp_path / "last.ckpt", (older, older))
    assert sbj.resolve_best_ckpt(tmp_path) == true_best


def test_rendered_sbatch_has_preflight_srun_absolute_path(rendered_celldiff_sbatch):
    """Preflight srun invokes nccl_smoke_test.py by absolute path (no bare ``applications/...``)."""
    preflight_line = next(
        line for line in rendered_celldiff_sbatch.splitlines() if "nccl_smoke_test.py" in line and "srun" in line
    )
    script_token = preflight_line.split()[-1]
    assert script_token.startswith("/"), f"preflight srun used relative path: {preflight_line!r}"
    assert script_token.endswith("/applications/dynacell/tools/nccl_smoke_test.py")


def test_repo_root_substituted_in_preflight_path(rendered_celldiff_sbatch):
    """``@@repo_root`` resolves to the actual VisCy repo root (not left unsubstituted)."""
    assert "@@repo_root" not in rendered_celldiff_sbatch
    expected_path = str(REPO_ROOT / "applications" / "dynacell" / "tools" / "nccl_smoke_test.py")
    assert expected_path in rendered_celldiff_sbatch


def test_preflight_failure_exits_before_main_srun(rendered_celldiff_sbatch):
    """``exit $SMOKE_RC`` appears ahead of the main dynacell srun line."""
    exit_idx = rendered_celldiff_sbatch.index("exit $SMOKE_RC")
    main_srun_idx = rendered_celldiff_sbatch.index("srun --cpu-bind=none uv run python -m dynacell")
    assert exit_idx < main_srun_idx


def test_submit_raises_on_missing_launcher(tmp_path):
    leaf = tmp_path / "leaf.yml"
    leaf.write_text(yaml.safe_dump({"model": {}, "data": {}}))
    with pytest.raises(SystemExit, match="missing required 'launcher:'"):
        sbj.submit([str(leaf), "--dry-run"])


def test_submit_rejects_non_absolute_run_root(tmp_path):
    leaf = tmp_path / "leaf.yml"
    leaf.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "mode": "fit",
                    "job_name": "JOB",
                    "run_root": "relative/path",
                    "sbatch": {"gpus": 1},
                },
                "trainer": {"devices": 1},
            }
        )
    )
    with pytest.raises(SystemExit, match="must be an absolute path"):
        sbj.submit([str(leaf), "--dry-run"])


def test_exclude_directive_rendered_when_set():
    """When launcher.sbatch.exclude is set, render a bare ``#SBATCH --exclude=<hostlist>`` line."""
    sbatch = {
        "partition": "gpu",
        "nodes": 1,
        "ntasks_per_node": 1,
        "cpus_per_task": 8,
        "gpus": 1,
        "mem": "64G",
        "constraint": "h200",
        "time": "1:00:00",
        "exclude": "gpu-d-1",
    }
    rendered = sbj._render_sbatch_directives("JOB", "/run", sbatch)
    assert "#SBATCH --exclude=gpu-d-1" in rendered
    exclude_idx = rendered.index("#SBATCH --exclude=gpu-d-1")
    constraint_idx = rendered.index("#SBATCH --constraint=")
    output_idx = rendered.index("#SBATCH --output=")
    assert constraint_idx < exclude_idx < output_idx


def test_exclude_directive_skipped_when_absent():
    """Absent or None ``exclude`` renders no ``--exclude`` line."""
    sbatch = {
        "partition": "gpu",
        "nodes": 1,
        "ntasks_per_node": 1,
        "cpus_per_task": 8,
        "gpus": 1,
        "mem": "64G",
        "constraint": "h200",
        "time": "1:00:00",
    }
    rendered_absent = sbj._render_sbatch_directives("JOB", "/run", sbatch)
    assert "--exclude" not in rendered_absent

    rendered_none = sbj._render_sbatch_directives("JOB", "/run", {**sbatch, "exclude": None})
    assert "--exclude" not in rendered_none


def test_submit_rejects_devices_gpus_mismatch(tmp_path):
    leaf = tmp_path / "leaf.yml"
    leaf.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "mode": "fit",
                    "job_name": "JOB",
                    "run_root": "/abs/path",
                    "sbatch": {
                        "partition": "gpu",
                        "nodes": 1,
                        "ntasks_per_node": 1,
                        "cpus_per_task": 1,
                        "gpus": 1,
                        "mem": "1G",
                        "constraint": "h200",
                        "time": "1:00:00",
                    },
                },
                "trainer": {"devices": 4},
            }
        )
    )
    with pytest.raises(SystemExit, match="topology mismatch"):
        sbj.submit([str(leaf), "--dry-run"])


def _write_minimal_valid_leaf(tmp_path: Path) -> Path:
    """Synthetic leaf with consistent topology so submit() reaches sbatch."""
    leaf = tmp_path / "leaf.yml"
    leaf.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "mode": "fit",
                    "job_name": "JOB",
                    "run_root": str(tmp_path / "run_root"),
                    "sbatch": {
                        "partition": "gpu",
                        "nodes": 1,
                        "ntasks_per_node": 1,
                        "cpus_per_task": 1,
                        "gpus": 1,
                        "mem": "1G",
                        "constraint": "h200",
                        "time": "1:00:00",
                    },
                },
                "trainer": {"devices": 1},
            }
        )
    )
    return leaf


def test_sbatch_cmd_default_no_flags(monkeypatch, tmp_path):
    """No flags → ``sbatch <script>`` with stdout untouched (existing shape)."""
    leaf = _write_minimal_valid_leaf(tmp_path)
    captured: dict = {}

    def _fake_run(cmd, check=True, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(sbj.subprocess, "run", _fake_run)
    rc = sbj.submit([str(leaf)])
    assert rc == 0
    assert captured["cmd"][0] == "sbatch"
    assert captured["cmd"][-1].endswith(".sbatch")
    assert "--parsable" not in captured["cmd"]
    assert not any(a.startswith("--dependency") for a in captured["cmd"])
    # Backward compat: no capture_output, so sbatch prose flows to stdout.
    assert "capture_output" not in captured["kwargs"]


def test_sbatch_cmd_with_dependency(monkeypatch, tmp_path):
    """--dependency afterok:<id> appends ``--dependency=afterok:<id>`` to sbatch."""
    leaf = _write_minimal_valid_leaf(tmp_path)
    captured: dict = {}

    def _fake_run(cmd, check=True, **kwargs):
        captured["cmd"] = cmd

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(sbj.subprocess, "run", _fake_run)
    sbj.submit([str(leaf), "--dependency", "afterok:12345"])
    assert "--dependency=afterok:12345" in captured["cmd"]


def test_sbatch_cmd_with_parsable(monkeypatch, capsys, tmp_path):
    """--parsable adds ``--parsable``, captures sbatch stdout, forwards job ID."""
    leaf = _write_minimal_valid_leaf(tmp_path)
    captured: dict = {}

    def _fake_run(cmd, check=True, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

        class _Result:
            returncode = 0
            stdout = "67890\n"

        return _Result()

    monkeypatch.setattr(sbj.subprocess, "run", _fake_run)
    sbj.submit([str(leaf), "--parsable"])
    assert "--parsable" in captured["cmd"]
    # stdout captured for forwarding; stderr left attached so sbatch
    # warnings/diagnostics remain visible to the operator.
    assert captured["kwargs"]["stdout"] is sbj.subprocess.PIPE
    assert "stderr" not in captured["kwargs"]
    assert "capture_output" not in captured["kwargs"]
    assert captured["kwargs"]["text"] is True
    out = capsys.readouterr().out
    assert "67890" in out


def test_sbatch_cmd_dependency_and_parsable(monkeypatch, tmp_path):
    """Both flags compose; --parsable, then --dependency, then script path."""
    leaf = _write_minimal_valid_leaf(tmp_path)
    captured: dict = {}

    def _fake_run(cmd, check=True, **kwargs):
        captured["cmd"] = cmd

        class _Result:
            returncode = 0
            stdout = "11111\n"

        return _Result()

    monkeypatch.setattr(sbj.subprocess, "run", _fake_run)
    sbj.submit([str(leaf), "--parsable", "--dependency", "afterok:42"])
    cmd = captured["cmd"]
    assert cmd[0] == "sbatch"
    assert "--parsable" in cmd
    assert "--dependency=afterok:42" in cmd
    assert cmd[-1].endswith(".sbatch")


# --- predict resume (--resume-predict) ---------------------------------------


def _write_hcs_store(
    path: Path, channels: list[str], fov_t: dict[str, int], *, completed: set[str] | None = None
) -> None:
    """Write a minimal HCS OME-Zarr with one array per FOV at the given T length."""
    with open_ome_zarr(path, layout="hcs", mode="a", channel_names=channels) as plate:
        for fov_name, t in fov_t.items():
            row, col, pos = fov_name.split("/")
            position = plate.create_position(row, col, pos)
            position.create_zeros("0", shape=(t, len(channels), 4, 8, 8), dtype=np.float32, chunks=(1, 1, 4, 8, 8))
            if completed and fov_name in completed:
                position.zattrs[PREDICTION_COMPLETE_KEY] = {channel: [t, 4, 8, 8] for channel in channels}


def test_completed_prediction_fovs_detects_partial(tmp_path):
    """Only marked FOVs whose output T matches the input count as complete."""
    inp = tmp_path / "input.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 10, "0/0/fov0001": 10, "0/0/fov0002": 10})
    out = tmp_path / "pred.zarr"
    # fov0000, fov0001 complete (T=10); fov0002 killed mid-run (T=5).
    _write_hcs_store(
        out,
        ["Structure_prediction"],
        {"0/0/fov0000": 10, "0/0/fov0001": 10, "0/0/fov0002": 5},
        completed={"0/0/fov0000", "0/0/fov0001", "0/0/fov0002"},
    )
    completed, total = sbj._completed_prediction_fovs(str(out), str(inp), ["Structure_prediction"])
    assert total == 3
    assert completed == {"0/0/fov0000", "0/0/fov0001"}


def test_completed_prediction_fovs_requires_explicit_completion(tmp_path):
    """Legacy output shapes cannot establish whether every Z window was written."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 2})
    _write_hcs_store(out, ["Structure_prediction"], {"0/0/fov0000": 2})
    assert sbj._completed_prediction_fovs(str(out), str(inp), ["Structure_prediction"]) == (set(), 1)


class _ConstantPrediction(LightningModule):
    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        return torch.ones_like(batch["source"])


@pytest.mark.parametrize("z_window_size", [1, 3])
def test_resume_prediction_requires_all_z_windows_and_invalidates_overwrites(tmp_path, z_window_size):
    """2D and overlapping 3D windows reach full T before the final writes."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 2})

    def predict(limit_batches: int, *, overwrite: bool = False) -> None:
        data = HCSDataModule(
            data_path=str(inp),
            source_channel=["Phase3D"],
            target_channel=["Structure"],
            z_window_size=z_window_size,
            batch_size=1,
            num_workers=0,
            yx_patch_size=[8, 8],
            normalizations=[],
            augmentations=[],
        )
        Trainer(
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            limit_predict_batches=limit_batches,
            callbacks=[HCSPredictionWriter(str(out), overwrite=overwrite)],
        ).predict(_ConstantPrediction(), datamodule=data, return_predictions=False)

    windows_per_t = 4 - z_window_size + 1
    predict(windows_per_t + 1)
    with open_ome_zarr(out, mode="r") as plate:
        image = plate["0/0/fov0000/0"]
        assert image.shape == (2, 1, 4, 8, 8)
        np.testing.assert_array_equal(image[1, 0, :, 0, 0], [1] * z_window_size + [0] * (4 - z_window_size))
    assert sbj._completed_prediction_fovs(str(out), str(inp), ["Structure_prediction"]) == (set(), 1)

    predict(2 * windows_per_t, overwrite=True)
    assert sbj._completed_prediction_fovs(str(out), str(inp), ["Structure_prediction"]) == ({"0/0/fov0000"}, 1)
    with open_ome_zarr(out, mode="r") as plate:
        np.testing.assert_array_equal(plate["0/0/fov0000/0"][:], 1)

    predict(1, overwrite=True)
    assert sbj._completed_prediction_fovs(str(out), str(inp), ["Structure_prediction"]) == (set(), 1)


def test_resume_prediction_rejects_stale_extra_timepoints_after_overwrite(tmp_path):
    """A complete T=2 overwrite cannot certify a T=3 array's stale final frame."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 2})
    _write_hcs_store(out, ["Structure_prediction"], {"0/0/fov0000": 3})
    with open_ome_zarr(out, mode="r+") as plate:
        plate["0/0/fov0000/0"][:] = 999
    data = HCSDataModule(
        data_path=str(inp),
        source_channel=["Phase3D"],
        target_channel=["Structure"],
        z_window_size=1,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[8, 8],
        normalizations=[],
        augmentations=[],
    )
    Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        callbacks=[HCSPredictionWriter(str(out), overwrite=True)],
    ).predict(_ConstantPrediction(), datamodule=data, return_predictions=False)

    with open_ome_zarr(out, mode="r") as plate:
        image = plate["0/0/fov0000/0"]
        assert image.shape[0] == 3
        np.testing.assert_array_equal(image[:2], 1)
        np.testing.assert_array_equal(image[2], 999)
        assert plate["0/0/fov0000"].zattrs[PREDICTION_COMPLETE_KEY] == {"Structure_prediction": [2, 4, 8, 8]}
    assert sbj._completed_prediction_fovs(str(out), str(inp), ["Structure_prediction"]) == (set(), 1)


def test_completed_prediction_fovs_no_store(tmp_path):
    """A missing output store yields no completed FOVs but the correct input total."""
    inp = tmp_path / "input.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 10, "0/0/fov0001": 10})
    completed, total = sbj._completed_prediction_fovs(str(tmp_path / "absent.zarr"), str(inp), ["Structure_prediction"])
    assert total == 2
    assert completed == set()


def test_completed_prediction_fovs_missing_channel_not_complete(tmp_path):
    """A FOV at full T but lacking the prediction channel is not complete."""
    inp = tmp_path / "input.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 10})
    out = tmp_path / "pred.zarr"
    _write_hcs_store(out, ["Other_prediction"], {"0/0/fov0000": 10})
    completed, total = sbj._completed_prediction_fovs(str(out), str(inp), ["Structure_prediction"])
    assert total == 1
    assert completed == set()


def test_resume_predict_rejects_fit_mode():
    """--resume-predict is predict-only; a fit leaf must error before rendering."""
    leaf = BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"
    with pytest.raises(SystemExit, match="only valid for predict"):
        sbj.submit([str(leaf), "--resume-predict", "--print-script"])


def test_resume_predict_rejects_ckpt_combo(tmp_path):
    """--resume-predict and --ckpt are mutually exclusive (would mix two models)."""
    ckpt = tmp_path / "custom.ckpt"
    ckpt.write_bytes(b"stub")
    leaf = BENCHMARKS / "mito/fcmae_vscyto3d_scratch/a549_mantis/predict__a549_mantis_denv.yml"
    with pytest.raises(SystemExit, match="cannot be combined with --ckpt"):
        sbj.submit([str(leaf), "--resume-predict", "--ckpt", str(ckpt), "--print-resolved-config"])
