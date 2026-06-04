"""Tests for submit_benchmark_job.py: sbatch rendering, byte-equivalence, flags."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

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
    assert srun_line.startswith("srun uv run python -m dynacell fit --config")
    assert expected_resolved_prefix in srun_line


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
    main_srun_idx = rendered_celldiff_sbatch.index("srun uv run python -m dynacell")
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
