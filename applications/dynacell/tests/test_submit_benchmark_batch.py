"""Tests for ``submit_benchmark_batch.py``: serial + array submission shapes."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# submit_benchmark_batch is on sys.path via root pyproject.toml's
# [tool.pytest.ini_options].pythonpath.
import submit_benchmark_batch as sbb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = REPO_ROOT / "applications" / "dynacell" / "configs" / "benchmarks" / "virtual_staining"

# Three same-bucket A549-test predict leaves (rand-init VSCyto3D, ER, mock/denv/zikv).
#
# These must genuinely share one ``launcher.run_root``, which is what serial and
# array mode batch on. The CellDiff leaves this fixture used to name no longer
# qualify: the canonical artifact-path migration (569bf51b) re-tokenized their
# run_roots to per-condition ``.../a549/a549__{cond}``, so all four tests below
# died on the shared-run_root guard rather than on anything they assert.
ER_A549_LEAVES = [
    BENCHMARKS / f"er/fcmae_vscyto3d_pretrained/_no_train_randinit/predict__a549_mantis_{cond}.yml"
    for cond in ("denv", "mock", "zikv")
]

# 4 ER leaves spanning two run_roots (3 a549-test + 1 ipsc-test).
ER_MIXED_RUN_ROOT_LEAVES = ER_A549_LEAVES + [
    BENCHMARKS / "er/fcmae_vscyto3d_pretrained/_no_train_randinit/predict__ipsc_confocal.yml"
]


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


def test_max_array_concurrency_without_array_rejected():
    with pytest.raises(SystemExit, match="--max-array-concurrency requires --array"):
        sbb.submit([str(ER_A549_LEAVES[0]), "--max-array-concurrency", "2", "--print-script"])


def test_allow_mixed_directives_without_array_rejected():
    with pytest.raises(SystemExit, match="--allow-mixed-directives requires --array"):
        sbb.submit([str(ER_A549_LEAVES[0]), "--allow-mixed-directives", "--print-script"])


def test_negative_max_array_concurrency_rejected():
    with pytest.raises(SystemExit, match="--max-array-concurrency must be >=1"):
        sbb.submit(
            [
                str(ER_A549_LEAVES[0]),
                "--array",
                "--max-array-concurrency",
                "0",
                "--print-script",
            ]
        )


def test_print_script_writes_nothing(monkeypatch, tmp_path):
    """``--print-script`` must not create or write anything under run_root.

    The flag's own help says "no writes", and the leaves it renders point at
    /hpc project shares the caller often cannot write -- on a CI runner /hpc
    does not exist, so a stray mkdir is a hard PermissionError rather than a
    stray directory. Assert against the syscalls, not against a path, so this
    holds wherever the suite runs.
    """
    import os

    real_mkdir, real_write = os.mkdir, Path.write_text
    calls: list[str] = []

    def guard_mkdir(path, *a, **kw):
        calls.append(f"mkdir {path}")
        return real_mkdir(path, *a, **kw)

    def guard_write(self, *a, **kw):
        calls.append(f"write {self}")
        return real_write(self, *a, **kw)

    monkeypatch.setattr(os, "mkdir", guard_mkdir)
    monkeypatch.setattr(Path, "write_text", guard_write)

    buf = io.StringIO()
    args = [str(p) for p in ER_A549_LEAVES] + ["--job-name", "TEST", "--print-script"]
    with redirect_stdout(buf):
        rc = sbb.submit(args)
    assert rc == 0
    assert buf.getvalue().startswith("#!/bin/bash")
    assert calls == [], f"--print-script touched the filesystem: {calls}"


def _write_predict_leaf(tmp_path: Path, ckpt: Path) -> Path:
    """Synthetic predict leaf: one writer, a model with ``ckpt_path``, an experiment id."""
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
                "benchmark": {"experiment_id": "pred_test"},
                "trainer": {
                    "devices": 1,
                    "callbacks": [
                        {
                            "class_path": "viscy_utils.callbacks.prediction_writer.HCSPredictionWriter",
                            "init_args": {"output_store": str(tmp_path / "pred.zarr")},
                        }
                    ],
                },
                "model": {"class_path": "dynacell.engine.DynacellUNet", "init_args": {"ckpt_path": str(ckpt)}},
                "data": {
                    "class_path": "viscy_data.HCSDataModule",
                    "init_args": {
                        "data_path": str(tmp_path / "input.zarr"),
                        "source_channel": ["Phase3D"],
                        "target_channel": ["Structure"],
                        "z_window_size": 4,
                    },
                },
            }
        )
    )
    return leaf


def test_batch_composition_binds_the_checkpoint_to_the_writer(tmp_path):
    """Batch submissions record the checkpoint in the writer exactly as single-job ones do."""
    ckpt = tmp_path / "a.ckpt"
    ckpt.write_bytes(b"weights-a")
    composed_list, _, _ = sbb._compose_leaves([_write_predict_leaf(tmp_path, ckpt)], [], False)
    writer_init = composed_list[0]["trainer"]["callbacks"][0]["init_args"]
    assert writer_init["checkpoint_path"] == str(ckpt)


def test_mixed_run_roots_rejected_without_allow_mixed():
    """Without ``--allow-mixed-directives``, leaves spanning two ``run_root``s must raise."""
    args = [str(p) for p in ER_MIXED_RUN_ROOT_LEAVES] + ["--array", "--print-script"]
    with pytest.raises(SystemExit, match="all leaves must share launcher.run_root"):
        sbb.submit(args)


# ---------------------------------------------------------------------------
# Serial mode unchanged
# ---------------------------------------------------------------------------


def test_serial_mode_renders_one_sbatch_with_chained_sruns():
    buf = io.StringIO()
    args = [str(p) for p in ER_A549_LEAVES] + ["--job-name", "TEST", "--print-script"]
    with redirect_stdout(buf):
        rc = sbb.submit(args)
    assert rc == 0
    out = buf.getvalue()
    # Three echo-step lines, three srun lines.
    assert out.count("[batch] step ") == 3
    assert out.count("srun --cpu-bind=none uv run python -m dynacell predict --config") == 3
    # No array directive, no CONFIGS array.
    assert "#SBATCH --array=" not in out
    assert "CONFIGS=(" not in out


# ---------------------------------------------------------------------------
# Array mode
# ---------------------------------------------------------------------------


def test_array_mode_emits_array_directive_and_configs_block():
    buf = io.StringIO()
    args = [str(p) for p in ER_A549_LEAVES] + [
        "--array",
        "--job-name",
        "TEST_ARR",
        "--print-script",
    ]
    with redirect_stdout(buf):
        rc = sbb.submit(args)
    assert rc == 0
    out = buf.getvalue()
    # Array directive present, no %N suffix when --max-array-concurrency omitted.
    assert "#SBATCH --array=0-2\n" in out
    # Per-task logs.
    assert "#SBATCH --output=" in out and "%A_%a.out" in out
    # CONFIGS block has exactly 3 entries.
    cfg_start = out.index("CONFIGS=(")
    cfg_end = out.index(")", cfg_start)
    assert out[cfg_start:cfg_end].count('  "') == 3
    # Dispatch line.
    assert 'CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"' in out


def test_max_array_concurrency_appends_pct_suffix():
    buf = io.StringIO()
    args = [str(p) for p in ER_A549_LEAVES] + [
        "--array",
        "--max-array-concurrency",
        "2",
        "--job-name",
        "TEST_ARR_CC",
        "--print-script",
    ]
    with redirect_stdout(buf):
        rc = sbb.submit(args)
    assert rc == 0
    assert "#SBATCH --array=0-2%2\n" in buf.getvalue()


def test_allow_mixed_directives_buckets_by_run_root():
    """4 ER leaves (3 a549 + 1 ipsc) should render as TWO sbatch arrays."""
    buf = io.StringIO()
    args = [str(p) for p in ER_MIXED_RUN_ROOT_LEAVES] + [
        "--array",
        "--allow-mixed-directives",
        "--max-array-concurrency",
        "2",
        "--job-name",
        "TEST_MIXED",
        "--print-script",
    ]
    with redirect_stdout(buf):
        rc = sbb.submit(args)
    assert rc == 0
    out = buf.getvalue()
    # Two array directives in the concatenated print-script output: one
    # 3-leaf bucket and one 1-leaf bucket.
    assert out.count("#SBATCH --array=") == 2
    assert "#SBATCH --array=0-2%2\n" in out
    assert "#SBATCH --array=0-0%2\n" in out
    # Two distinct job_name suffixes (g0 / g1).
    assert "#SBATCH --job-name=TEST_MIXED_g0\n" in out
    assert "#SBATCH --job-name=TEST_MIXED_g1\n" in out


def test_directive_bucket_key_distinguishes_constraint():
    a = {
        "nodes": 1,
        "ntasks_per_node": 1,
        "partition": "gpu",
        "cpus_per_task": 32,
        "gpus": 1,
        "mem": "256G",
        "constraint": "h200",
    }
    b = {**a, "constraint": "h100"}
    assert sbb._directive_bucket_key(a) != sbb._directive_bucket_key(b)


def test_directive_bucket_key_ignores_time():
    a = {"nodes": 1, "ntasks_per_node": 1, "partition": "gpu", "cpus_per_task": 32, "gpus": 1, "mem": "256G"}
    b = {**a}
    # Both buckets identical even though they don't carry `time`.
    assert sbb._directive_bucket_key(a) == sbb._directive_bucket_key(b)
