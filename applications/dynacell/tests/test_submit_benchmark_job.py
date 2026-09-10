"""Tests for submit_benchmark_job.py: sbatch rendering, byte-equivalence, flags."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch
from iohub.ngff import open_ome_zarr
from lightning.pytorch import LightningModule, Trainer

from viscy_data import HCSDataModule
from viscy_utils.callbacks.prediction_writer import HCSPredictionWriter
from viscy_utils.prediction_metadata import (
    PREDICTION_COMPLETE_KEY,
    completion_marker,
    mark_complete,
    mark_started,
    prediction_run,
    started_marker,
    tzyx_shape,
)

yaml = pytest.importorskip("yaml")

# submit_benchmark_job is importable because the root pyproject.toml's
# [tool.pytest.ini_options].pythonpath adds applications/dynacell/tools to sys.path.
import submit_benchmark_job as sbj  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = REPO_ROOT / "applications" / "dynacell" / "configs" / "benchmarks" / "virtual_staining"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--help"], "usage:"),
        ([str(BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"), "--print-script"], "#SBATCH"),
    ],
)
def test_submitter_runs_without_optional_evaluation_dependencies(args, expected):
    """Submission and config rendering must work without scikit-learn installed."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; sys.modules['sklearn'] = None; "
            "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')",
            str(REPO_ROOT / "applications/dynacell/tools/submit_benchmark_job.py"),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


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


def _run(z_window_size: int, checkpoint: Path | None = None) -> dict:
    """Run identity of the test predicts: array ``0``, blended windows, an optional stub checkpoint."""
    return prediction_run(array_key="0", z_window_size=z_window_size, z_reduction="blend", checkpoint_path=checkpoint)


def _write_hcs_store(
    path: Path,
    channels: list[str],
    fov_t: dict[str, int],
    *,
    completed: set[str] | None = None,
    run: dict | None = None,
    unmarked: bool = False,
) -> None:
    """Write a minimal HCS OME-Zarr with one array per FOV at the given T length.

    Passing ``completed`` makes it an output store: every FOV gets the started
    marker the writer stamps at creation and the named FOVs are marked as fully
    predicted by ``run``. ``unmarked`` mimics an output written before markers
    existed.
    """
    with open_ome_zarr(path, layout="hcs", mode="a", channel_names=channels) as plate:
        for fov_name, t in fov_t.items():
            row, col, pos = fov_name.split("/")
            position = plate.create_position(row, col, pos)
            array = position.create_zeros(
                "0", shape=(t, len(channels), 4, 8, 8), dtype=np.float32, chunks=(1, 1, 4, 8, 8)
            )
            if completed is not None and not unmarked:
                mark_started(position, channels, run)
            if completed and fov_name in completed:
                mark_complete(position, channels, completion_marker(tzyx_shape(array), run))


def _completed(out: Path, inp: Path, run: dict) -> set[str]:
    """FOVs the launcher would skip on ``--resume-predict`` for this run."""
    return sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], run).completed


def test_survey_detects_partial(tmp_path):
    """Only marked FOVs whose output T matches the input count as complete."""
    inp = tmp_path / "input.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 10, "0/0/fov0001": 10, "0/0/fov0002": 10})
    out = tmp_path / "pred.zarr"
    run = _run(1)
    # fov0000, fov0001 complete (T=10); fov0002 killed mid-run (T=5, never marked).
    _write_hcs_store(
        out,
        ["Structure_prediction"],
        {"0/0/fov0000": 10, "0/0/fov0001": 10, "0/0/fov0002": 5},
        completed={"0/0/fov0000", "0/0/fov0001"},
        run=run,
    )
    survey = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], run)
    assert survey.total == 3
    assert survey.completed == {"0/0/fov0000", "0/0/fov0001"}
    assert (survey.conflicting, survey.unverifiable) == (set(), set())


def test_survey_reports_output_without_markers_as_unverifiable(tmp_path):
    """A full-T output written before markers existed proves nothing about its Z windows."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 2})
    _write_hcs_store(out, ["Structure_prediction"], {"0/0/fov0000": 2}, completed=set(), unmarked=True)
    survey = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], _run(1))
    assert (survey.total, survey.completed, survey.conflicting) == (1, set(), set())
    assert survey.unverifiable == {"0/0/fov0000"}


def test_survey_treats_a_channel_without_its_own_marker_as_unverifiable(tmp_path):
    """Another channel's marker vouches for nothing: a legacy channel stays unverifiable, never merely incomplete."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 2})
    run = _run(1)
    _write_hcs_store(
        out, ["Other_prediction", "Structure_prediction"], {"0/0/fov0000": 2}, completed=set(), unmarked=True
    )
    with open_ome_zarr(out, mode="r+") as plate:
        mark_complete(plate["0/0/fov0000"], ["Other_prediction"], completion_marker([2, 4, 8, 8], run))

    survey = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], run)
    assert (survey.completed, survey.conflicting, survey.unverifiable) == (set(), set(), {"0/0/fov0000"})


def test_survey_reads_the_configured_array_level(tmp_path):
    """Both shapes come from ``array_key``; a level the output never wrote is incomplete, not an error."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    with open_ome_zarr(inp, layout="hcs", mode="a", channel_names=["Phase3D"]) as plate:
        position = plate.create_position("0", "0", "fov0000")
        position.create_zeros("0", shape=(2, 1, 4, 8, 8), dtype=np.float32)
        position.create_zeros("1", shape=(2, 1, 4, 4, 4), dtype=np.float32)
    level_one = prediction_run(array_key="1", z_window_size=1, z_reduction="blend", checkpoint_path=None)
    with open_ome_zarr(out, layout="hcs", mode="a", channel_names=["Structure_prediction"]) as plate:
        position = plate.create_position("0", "0", "fov0000")
        array = position.create_zeros("1", shape=(2, 1, 4, 4, 4), dtype=np.float32)
        mark_complete(position, ["Structure_prediction"], completion_marker(tzyx_shape(array), level_one))

    survey = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], level_one)
    assert (survey.completed, survey.conflicting) == ({"0/0/fov0000"}, set())
    # Completed at another level: that is a different run, so it conflicts rather than resumes.
    survey = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], _run(1))
    assert (survey.completed, survey.conflicting) == (set(), {"0/0/fov0000"})
    with pytest.raises(KeyError):
        sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], dict(level_one, array_key="2"))


def test_survey_flags_fovs_predicted_with_another_checkpoint(tmp_path):
    """A FOV marked complete by other weights is conflicting, not merely incomplete.

    The comparison is on checkpoint content, so the same file at another path
    is still the same run.
    """
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 2})
    ckpt_a = tmp_path / "a.ckpt"
    ckpt_a.write_bytes(b"weights-a")
    ckpt_b = tmp_path / "b.ckpt"
    ckpt_b.write_bytes(b"weights-b")
    moved_a = tmp_path / "moved" / "a.ckpt"
    moved_a.parent.mkdir()
    moved_a.write_bytes(b"weights-a")
    _write_hcs_store(out, ["Structure_prediction"], {"0/0/fov0000": 2}, completed={"0/0/fov0000"}, run=_run(1, ckpt_a))

    same = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], _run(1, ckpt_a))
    assert (same.completed, same.conflicting) == ({"0/0/fov0000"}, set())
    moved = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], _run(1, moved_a))
    assert (moved.completed, moved.conflicting) == ({"0/0/fov0000"}, set())
    other = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], _run(1, ckpt_b))
    assert (other.completed, other.conflicting) == (set(), {"0/0/fov0000"})


class _ConstantPrediction(LightningModule):
    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        return torch.ones_like(batch["source"])


def _predict(
    inp: Path,
    out: Path,
    *,
    z_window_size: int,
    limit_batches: int | None = None,
    overwrite: bool = False,
    checkpoint: Path | None = None,
    settings: str | None = None,
    target: str = "Structure",
    exclude: list[str] | None = None,
) -> None:
    """Write all-ones predictions for the first ``limit_batches`` windows of ``inp`` into ``out``."""
    data = HCSDataModule(
        data_path=str(inp),
        source_channel=["Phase3D"],
        target_channel=[target],
        z_window_size=z_window_size,
        batch_size=1,
        num_workers=0,
        yx_patch_size=[8, 8],
        normalizations=[],
        augmentations=[],
        exclude_fov_names=exclude,
    )
    writer = HCSPredictionWriter(
        str(out),
        overwrite=overwrite,
        checkpoint_path=None if checkpoint is None else str(checkpoint),
        settings_sha256_12=settings,
    )
    Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        limit_predict_batches=limit_batches,
        callbacks=[writer],
    ).predict(_ConstantPrediction(), datamodule=data, return_predictions=False)


@pytest.mark.parametrize("z_window_size", [1, 3])
def test_resume_prediction_requires_all_z_windows_and_invalidates_overwrites(tmp_path, z_window_size):
    """2D and overlapping 3D windows reach full T before the final writes."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 2})
    run = _run(z_window_size)

    windows_per_t = 4 - z_window_size + 1
    _predict(inp, out, z_window_size=z_window_size, limit_batches=windows_per_t + 1)
    with open_ome_zarr(out, mode="r") as plate:
        image = plate["0/0/fov0000/0"]
        assert image.shape == (2, 1, 4, 8, 8)
        np.testing.assert_array_equal(image[1, 0, :, 0, 0], [1] * z_window_size + [0] * (4 - z_window_size))
    # Interrupted, not legacy: the writer stamped a started marker when it created the FOV.
    survey = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], run)
    assert (survey.completed, survey.unverifiable) == (set(), set())

    _predict(inp, out, z_window_size=z_window_size, limit_batches=2 * windows_per_t, overwrite=True)
    assert _completed(out, inp, run) == {"0/0/fov0000"}
    with open_ome_zarr(out, mode="r") as plate:
        np.testing.assert_array_equal(plate["0/0/fov0000/0"][:], 1)

    _predict(inp, out, z_window_size=z_window_size, limit_batches=1, overwrite=True)
    assert _completed(out, inp, run) == set()


@pytest.mark.parametrize("output_shape", [(3, 1, 4, 8, 8), (2, 1, 5, 8, 8)], ids=["extra_T", "extra_Z"])
def test_predict_refuses_an_output_that_outruns_its_source(tmp_path, output_shape):
    """Arrays only grow, so an output larger than its (2, 4, 8, 8) source in T or Z would keep stale
    planes under a fresh completion marker; refuse before writing anything."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 2})
    _write_hcs_store(out, ["Structure_prediction"], {"0/0/fov0000": 2}, completed=set(), run=_run(1))
    with open_ome_zarr(out, mode="r+") as plate:
        plate["0/0/fov0000/0"].resize(output_shape)
        plate["0/0/fov0000/0"][:] = 999

    with pytest.raises(ValueError, match="more timepoints or depth slices"):
        _predict(inp, out, z_window_size=1, overwrite=True)

    with open_ome_zarr(out, mode="r") as plate:
        np.testing.assert_array_equal(plate["0/0/fov0000/0"][:], 999)
        assert plate["0/0/fov0000"].zattrs[PREDICTION_COMPLETE_KEY] == {"Structure_prediction": started_marker(_run(1))}


@pytest.mark.parametrize("output_shape", [(3, 1, 4, 8, 8), (2, 1, 5, 8, 8)], ids=["extra_T", "extra_Z"])
def test_survey_flags_a_completed_output_that_outruns_its_source(tmp_path, output_shape):
    """A valid marker does not certify an array that another channel's run has since grown."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 2})
    run = _run(1)
    _write_hcs_store(out, ["Structure_prediction"], {"0/0/fov0000": 2}, completed={"0/0/fov0000"}, run=run)
    assert _completed(out, inp, run) == {"0/0/fov0000"}
    with open_ome_zarr(out, mode="r+") as plate:
        plate["0/0/fov0000/0"].resize(output_shape)

    survey = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], run)
    assert (survey.completed, survey.conflicting, survey.oversized) == (set(), set(), {"0/0/fov0000"})


def test_survey_no_store(tmp_path):
    """A missing output store yields no completed FOVs but the correct input total."""
    inp = tmp_path / "input.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 10, "0/0/fov0001": 10})
    survey = sbj._survey_prediction_store(str(tmp_path / "absent.zarr"), str(inp), ["Structure_prediction"], _run(1))
    assert survey.total == 2
    assert survey.completed == set()


def test_survey_missing_channel_not_complete(tmp_path):
    """A FOV at full T but lacking the prediction channel is not complete."""
    inp = tmp_path / "input.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 10})
    out = tmp_path / "pred.zarr"
    _write_hcs_store(out, ["Other_prediction"], {"0/0/fov0000": 10})
    survey = sbj._survey_prediction_store(str(out), str(inp), ["Structure_prediction"], _run(1))
    assert survey.total == 1
    assert survey.completed == set()


def _write_predict_leaf(tmp_path: Path, *, data_path: Path, output_store: Path, ckpt: Path, z_window_size: int) -> Path:
    """Synthetic predict leaf: a writer on ``output_store`` predicting ``data_path`` with ``ckpt``."""
    leaf = tmp_path / "predict.yml"
    leaf.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "mode": "predict",
                    "job_name": "PRED",
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
                "trainer": {
                    "devices": 1,
                    "callbacks": [
                        {
                            "class_path": "viscy_utils.callbacks.prediction_writer.HCSPredictionWriter",
                            "init_args": {"output_store": str(output_store)},
                        }
                    ],
                },
                "model": {"class_path": "dynacell.engine.DynacellUNet", "init_args": {"ckpt_path": str(ckpt)}},
                "data": {
                    "class_path": "viscy_data.HCSDataModule",
                    "init_args": {
                        "data_path": str(data_path),
                        "source_channel": ["Phase3D"],
                        "target_channel": ["Structure"],
                        "z_window_size": z_window_size,
                    },
                },
            }
        )
    )
    return leaf


def _resolved_config(capsys) -> dict:
    """Parse the YAML that ``--print-resolved-config`` wrote, skipping resume status lines."""
    lines = [line for line in capsys.readouterr().out.splitlines() if not line.startswith("--resume-predict")]
    return yaml.safe_load("\n".join(lines))


def _leaf_settings(leaf: Path) -> str:
    """The settings hash the launcher binds to a leaf's writer."""
    return sbj.prediction_settings_sha256_12(sbj.load_composed_config(leaf, resolver=sbj._dynacell_ref_resolver))


def test_predict_submit_records_the_run_identity_in_writer(capsys, tmp_path):
    """Every predict submission names its checkpoint and settings hash to the writer, so markers carry them."""
    inp = tmp_path / "input.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 1})
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    leaf = _write_predict_leaf(tmp_path, data_path=inp, output_store=tmp_path / "pred.zarr", ckpt=ckpt, z_window_size=4)

    assert sbj.submit([str(leaf), "--print-resolved-config"]) == 0
    config = _resolved_config(capsys)
    writer_init = config["trainer"]["callbacks"][0]["init_args"]
    assert writer_init["checkpoint_path"] == str(ckpt)
    assert writer_init["settings_sha256_12"] == _leaf_settings(leaf) == sbj.prediction_settings_sha256_12(config)


def test_prediction_settings_hash_tracks_only_what_shapes_the_voxels():
    """Inference arguments, normalization and precision change the hash; loading and FOV selection do not."""
    base = {
        "model": {"class_path": "m.Model", "init_args": {"ckpt_path": "/a.ckpt", "num_generate_steps": 100}},
        "data": {
            "class_path": "viscy_data.HCSDataModule",
            "init_args": {"data_path": "/in.zarr", "normalizations": [{"class_path": "t.Norm"}], "batch_size": 1},
        },
        "trainer": {"precision": "32-true"},
    }
    reference = sbj.prediction_settings_sha256_12(base)
    assert len(reference) == 12

    def variant(section: str, **changes) -> str:
        composed = {k: dict(v) for k, v in base.items()}
        composed[section]["init_args"] = {**base[section]["init_args"], **changes}
        return sbj.prediction_settings_sha256_12(composed)

    assert variant("model", ckpt_path="/b.ckpt") == reference
    assert (
        variant("data", data_path="/moved.zarr", batch_size=8, num_workers=4, exclude_fov_names=["0/0/1"]) == reference
    )
    assert variant("model", num_generate_steps=50) != reference
    assert variant("data", normalizations=[]) != reference
    assert sbj.prediction_settings_sha256_12({**base, "trainer": {"precision": "bf16-mixed"}}) != reference


def test_resume_predict_excludes_complete_fovs_and_keeps_the_checkpoint(capsys, tmp_path):
    """A resume skips FOVs the same checkpoint completed and rewrites the rest."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 1, "0/0/fov0001": 1})
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    leaf = _write_predict_leaf(tmp_path, data_path=inp, output_store=out, ckpt=ckpt, z_window_size=4)
    # One full-depth window per FOV: the first batch completes fov0000 only.
    _predict(inp, out, z_window_size=4, limit_batches=1, checkpoint=ckpt, settings=_leaf_settings(leaf))

    assert sbj.submit([str(leaf), "--resume-predict", "--print-resolved-config"]) == 0
    config = _resolved_config(capsys)
    assert config["data"]["init_args"]["exclude_fov_names"] == ["0/0/fov0000"]
    writer_init = config["trainer"]["callbacks"][0]["init_args"]
    assert writer_init["overwrite"] is True
    assert writer_init["checkpoint_path"] == str(ckpt)
    assert writer_init["settings_sha256_12"] == _leaf_settings(leaf)


def test_resume_predict_preview_writes_no_sidecar_but_a_dry_run_does(tmp_path):
    """--print-* stays a pure preview even though the survey hashes the checkpoint; --dry-run may cache it."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 1, "0/0/fov0001": 1})
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    leaf = _write_predict_leaf(tmp_path, data_path=inp, output_store=out, ckpt=ckpt, z_window_size=4)
    _predict(inp, out, z_window_size=4, limit_batches=1, checkpoint=ckpt, settings=_leaf_settings(leaf))
    sidecar = tmp_path / "a.ckpt.sha256"
    sidecar.unlink()

    assert sbj.submit([str(leaf), "--resume-predict", "--print-resolved-config"]) == 0
    assert not sidecar.exists()
    assert sbj.submit([str(leaf), "--resume-predict", "--dry-run"]) == 0
    assert sidecar.exists()


def test_resume_predict_refuses_a_store_predicted_with_other_settings(tmp_path):
    """Same weights, other inference settings: different voxels, so the store must not be completed."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 1, "0/0/fov0001": 1})
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    leaf = _write_predict_leaf(tmp_path, data_path=inp, output_store=out, ckpt=ckpt, z_window_size=4)
    _predict(inp, out, z_window_size=4, limit_batches=1, checkpoint=ckpt, settings=_leaf_settings(leaf))

    with pytest.raises(SystemExit, match="another checkpoint or settings"):
        sbj.submit(
            [str(leaf), "--resume-predict", "--override", "model.init_args.num_generate_steps=5", "--print-script"]
        )


def test_resume_predict_refuses_a_store_without_markers(tmp_path):
    """Outputs written before completion markers existed cannot be resumed; they would be overwritten."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 1, "0/0/fov0001": 1})
    _write_hcs_store(out, ["Structure_prediction"], {"0/0/fov0000": 1}, completed=set(), unmarked=True)
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    leaf = _write_predict_leaf(tmp_path, data_path=inp, output_store=out, ckpt=ckpt, z_window_size=4)

    with pytest.raises(SystemExit, match="cannot be verified"):
        sbj.submit([str(leaf), "--resume-predict", "--print-resolved-config"])


def test_resume_predict_finishes_a_channel_appended_to_excluded_fovs(capsys, tmp_path):
    """Predicting a new channel into part of a store allocates it everywhere; the rest resumes, unrefused."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 1, "0/0/fov0001": 1})
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    leaf = _write_predict_leaf(tmp_path, data_path=inp, output_store=out, ckpt=ckpt, z_window_size=4)
    settings = _leaf_settings(leaf)
    _predict(inp, out, z_window_size=4, checkpoint=ckpt, settings=settings, target="Other")
    _predict(inp, out, z_window_size=4, checkpoint=ckpt, settings=settings, exclude=["0/0/fov0001"])
    with open_ome_zarr(out, mode="r") as plate:
        assert plate["0/0/fov0001"].channel_names == ["Other_prediction", "Structure_prediction"]

    assert sbj.submit([str(leaf), "--resume-predict", "--print-resolved-config"]) == 0
    config = _resolved_config(capsys)
    assert config["data"]["init_args"]["exclude_fov_names"] == ["0/0/fov0000"]


def test_resume_predict_refuses_an_output_that_outruns_its_source(tmp_path):
    """The writer would refuse such a store at job start; the launcher refuses before submitting."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 1, "0/0/fov0001": 1})
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    leaf = _write_predict_leaf(tmp_path, data_path=inp, output_store=out, ckpt=ckpt, z_window_size=4)
    _predict(inp, out, z_window_size=4, limit_batches=1, checkpoint=ckpt, settings=_leaf_settings(leaf))
    with open_ome_zarr(out, mode="r+") as plate:
        plate["0/0/fov0000/0"].resize((2, 1, 4, 8, 8))

    with pytest.raises(SystemExit, match="more timepoints or depth slices"):
        sbj.submit([str(leaf), "--resume-predict", "--print-resolved-config"])


def test_resume_predict_refuses_a_store_from_another_checkpoint(tmp_path):
    """Re-baking the leaf's checkpoint must not finish a store begun with other weights."""
    inp = tmp_path / "input.zarr"
    out = tmp_path / "pred.zarr"
    _write_hcs_store(inp, ["Phase3D"], {"0/0/fov0000": 1, "0/0/fov0001": 1})
    ckpt_a = tmp_path / "a.ckpt"
    ckpt_a.write_bytes(b"weights-a")
    _predict(inp, out, z_window_size=4, limit_batches=1, checkpoint=ckpt_a)
    ckpt_b = tmp_path / "b.ckpt"
    ckpt_b.write_bytes(b"weights-b")
    leaf = _write_predict_leaf(tmp_path, data_path=inp, output_store=out, ckpt=ckpt_b, z_window_size=4)

    with pytest.raises(SystemExit, match="another checkpoint"):
        sbj.submit([str(leaf), "--resume-predict", "--print-resolved-config"])


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
