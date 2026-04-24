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


FIXTURE_MANIFEST_ROOT = Path(__file__).resolve().parent / "fixtures" / "manifests"


@pytest.fixture(scope="module")
def rendered_celldiff_sbatch():
    """Render the celldiff leaf once per module; tests below share the output.

    Module-scoped fixtures run before conftest's function-scoped autouse
    monkeypatch, so ``DYNACELL_MANIFEST_ROOTS`` must be set here via a
    module-scoped ``MonkeyPatch``. ``capsys`` is also function-scoped —
    use ``contextlib.redirect_stdout`` instead.
    """
    mp = pytest.MonkeyPatch()
    mp.setenv("DYNACELL_MANIFEST_ROOTS", str(FIXTURE_MANIFEST_ROOT))
    try:
        leaf = BENCHMARKS / "er/celldiff/ipsc_confocal/train.yml"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sbj.submit([str(leaf), "--print-script"])
        assert rc == 0
        yield buf.getvalue()
    finally:
        mp.undo()


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
