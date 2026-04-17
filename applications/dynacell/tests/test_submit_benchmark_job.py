"""Tests for submit_benchmark_job.py: sbatch rendering, byte-equivalence, flags."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# submit_benchmark_job is importable because the root pyproject.toml's
# [tool.pytest.ini_options].pythonpath adds applications/dynacell/tools to sys.path.
import submit_benchmark_job as sbj  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "applications" / "dynacell" / "tools" / "LEGACY" / "examples_configs"
BENCHMARKS = REPO_ROOT / "applications" / "dynacell" / "configs" / "benchmarks" / "virtual_staining"


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
        "ntasks": 1,
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
    "leaf_subpath,legacy_slurm,expected_resolved_prefix",
    [
        (
            "train/er/ipsc_confocal/celldiff.yml",
            "sec61b/run_celldiff.slurm",
            "/resolved/fit_CELLDiff_SEC61B_",
        ),
        (
            "train/er/ipsc_confocal/unetvit3d.yml",
            "sec61b/run_unetvit3d.slurm",
            "/resolved/fit_UNetViT3D_SEC61B_",
        ),
    ],
)
def test_byte_equivalence_sec61b_train_leaf(capsys, leaf_subpath, legacy_slurm, expected_resolved_prefix):
    """Rendered sbatch differs from Dihan's legacy .slurm only on the srun line."""
    legacy = (EXAMPLES / legacy_slurm).read_text()
    leaf = BENCHMARKS / leaf_subpath

    # --print-script is preview-only (no disk writes), so this is safe to run
    # against a leaf whose launcher.run_root we may not have permission to write.
    rc = sbj.submit([str(leaf), "--print-script"])
    assert rc == 0
    rendered = capsys.readouterr().out

    legacy_lines = legacy.splitlines()
    rendered_lines = rendered.splitlines()

    # Every line identical except the final srun line.
    assert len(legacy_lines) == len(rendered_lines), (
        f"line count differs: legacy={len(legacy_lines)} rendered={len(rendered_lines)}"
    )
    srun_idx = len(legacy_lines) - 1
    for i, (a, b) in enumerate(zip(legacy_lines, rendered_lines)):
        if i == srun_idx:
            continue
        assert a == b, f"line {i} differs:\n  legacy:   {a!r}\n  rendered: {b!r}"
    # srun line — both start with the same prefix, differ on --config path
    legacy_srun = legacy_lines[srun_idx]
    rendered_srun = rendered_lines[srun_idx]
    assert legacy_srun.startswith("srun uv run python -m dynacell fit --config")
    assert rendered_srun.startswith("srun uv run python -m dynacell fit --config")
    assert expected_resolved_prefix in rendered_srun


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
                        "ntasks": 1,
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
    with pytest.raises(SystemExit, match="does not match"):
        sbj.submit([str(leaf), "--dry-run"])
