"""Unit tests for ``dynacell.evaluation.runtime``."""

from __future__ import annotations

import csv
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dynacell.evaluation.runtime import (
    ResolvedRuntime,
    apply_thread_budget,
    dump_timings_csv,
    early_apply_env_caps,
    extend_timings,
    get_timings,
    gpu_lock_path_for_job,
    gpu_serialization_lock,
    make_fov_executor,
    maybe_empty_cuda_cache,
    maybe_gc_collect,
    region_timer,
    reset_timings,
    resolve_runtime,
)


def test_apply_thread_budget_runs_without_error():
    """apply_thread_budget should not raise even when called from Hydra's main.

    set_num_interop_threads raises after any parallel op has run; the function
    catches that and continues. Test covers the import-after-parallel-op path.
    """
    # Force-import torch (parallel op) before applying budget.
    import torch

    _ = torch.zeros(4) + torch.ones(4)
    apply_thread_budget(2)  # should not raise


def test_apply_thread_budget_rejects_zero():
    with pytest.raises(ValueError, match="threads must be >= 1"):
        apply_thread_budget(0)


def test_resolve_runtime_no_runtime_block_falls_back_to_serial():
    cfg = OmegaConf.create({"io": {}})
    r = resolve_runtime(cfg)
    assert r.fov_workers == 1
    assert r.executor == "serial"
    assert r.cuda_empty_cache_every_n_timepoints == 0
    assert r.gc_collect_every_n_fovs == 0


def test_resolve_runtime_auto_serial_clamps_workers_to_one():
    cfg = OmegaConf.create({"runtime": {"fov_workers": "auto", "threads_per_worker": "auto", "executor": "serial"}})
    r = resolve_runtime(cfg)
    assert r.fov_workers == 1
    assert r.executor == "serial"
    assert r.threads_per_worker >= 1


def test_resolve_runtime_auto_process_clamps_to_n_positions():
    cfg = OmegaConf.create({"runtime": {"fov_workers": "auto", "threads_per_worker": "auto", "executor": "process"}})
    r = resolve_runtime(cfg, n_positions=2)
    assert r.fov_workers <= 2
    # Auto-demotes to serial when only 1 position is left after clamping.
    if r.fov_workers == 1:
        assert r.executor == "serial"


def test_resolve_runtime_freeze_threads_per_worker():
    """Phase-2 contract: ``freeze_threads_per_worker`` returns the frozen value."""
    cfg = OmegaConf.create({"runtime": {"fov_workers": "auto", "threads_per_worker": "auto", "executor": "process"}})
    r = resolve_runtime(cfg, n_positions=3, freeze_threads_per_worker=4)
    assert r.threads_per_worker == 4


def test_resolve_runtime_literal_fov_workers_in_serial_raises():
    cfg = OmegaConf.create({"runtime": {"fov_workers": 4, "executor": "serial"}})
    with pytest.raises(ValueError, match=r"fov_workers=4 requires runtime.executor='process'"):
        resolve_runtime(cfg)


def test_resolve_runtime_auto_demote_process_to_serial_at_one_worker():
    """fov_workers literally set to 1 with executor=process should auto-demote to serial."""
    cfg = OmegaConf.create({"runtime": {"fov_workers": 1, "executor": "process"}})
    r = resolve_runtime(cfg)
    assert r.fov_workers == 1
    assert r.executor == "serial"


def test_resolve_runtime_invalid_executor_raises():
    cfg = OmegaConf.create({"runtime": {"executor": "ray"}})
    with pytest.raises(ValueError, match="executor must be 'serial' or 'process'"):
        resolve_runtime(cfg)


def test_resolve_runtime_force_per_t_hygiene_env(monkeypatch):
    """DYNACELL_FORCE_PER_T_HYGIENE=1 turns on hygiene knobs regardless of YAML."""
    monkeypatch.setenv("DYNACELL_FORCE_PER_T_HYGIENE", "1")
    cfg = OmegaConf.create(
        {
            "runtime": {
                "cuda_empty_cache_every_n_timepoints": 0,
                "gc_collect_every_n_fovs": 0,
            }
        }
    )
    r = resolve_runtime(cfg)
    assert r.cuda_empty_cache_every_n_timepoints >= 1
    assert r.gc_collect_every_n_fovs >= 1


def test_early_apply_env_caps_respects_existing(monkeypatch):
    """Should only export env vars when DYNACELL_THREADS_PER_WORKER is set."""
    monkeypatch.delenv("DYNACELL_THREADS_PER_WORKER", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    early_apply_env_caps()
    assert os.environ.get("OMP_NUM_THREADS") is None  # noop without env set

    monkeypatch.setenv("DYNACELL_THREADS_PER_WORKER", "3")
    early_apply_env_caps()
    assert os.environ.get("OMP_NUM_THREADS") == "3"


def test_resolved_runtime_dataclass_is_pickleable():
    # Import inside test body so we pick up the *currently cached*
    # dynacell.evaluation.runtime module. test_lazy_init.py clears
    # `dynacell.*` modules from sys.modules — a module-level import
    # at collection time would bind to a stale class object.
    import importlib

    runtime_mod = importlib.import_module("dynacell.evaluation.runtime")
    r = runtime_mod.ResolvedRuntime(
        fov_workers=2,
        threads_per_worker=4,
        executor="process",
        cuda_empty_cache_every_n_timepoints=0,
        gc_collect_every_n_fovs=0,
    )
    restored = pickle.loads(pickle.dumps(r))
    assert restored == r


def test_region_timer_records_to_collector():
    reset_timings()
    with region_timer("test_region", "A/1/0"):
        pass
    with region_timer("test_per_t", "A/1/0", t=5):
        pass
    rows = get_timings()
    assert len(rows) == 2
    assert rows[0][0] == "A/1/0"
    assert rows[0][1] is None
    assert rows[0][2] == "test_region"
    assert rows[0][3] >= 0.0
    assert rows[1][1] == 5


def test_dump_timings_csv_writes_correct_schema(tmp_path: Path):
    reset_timings()
    with region_timer("test_region", "A/1/0"):
        pass
    with region_timer("test_per_t", "A/1/0", t=2):
        pass
    out_path = dump_timings_csv(tmp_path)
    assert out_path is not None
    with out_path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["pos_name", "t", "region", "seconds"]
    assert rows[1][:3] == ["A/1/0", "", "test_region"]
    assert rows[2][:3] == ["A/1/0", "2", "test_per_t"]


def test_dump_timings_csv_returns_none_when_empty(tmp_path: Path):
    reset_timings()
    assert dump_timings_csv(tmp_path) is None


def test_extend_timings_appends_batch():
    reset_timings()
    extend_timings([("X/1/0", None, "foo", 0.1), ("X/1/0", 0, "bar", 0.2)])
    assert len(get_timings()) == 2


def test_maybe_empty_cuda_cache_is_noop_when_every_n_zero():
    # No-op should not raise even without CUDA.
    maybe_empty_cuda_cache(t=3, every_n=0)


def test_maybe_gc_collect_is_noop_when_every_n_zero():
    maybe_gc_collect(fov_idx=0, every_n=0)


def test_maybe_gc_collect_runs_at_interval():
    """fov_idx+1 % every_n == 0 triggers the collect."""
    # Just verify it doesn't raise for valid intervals; gc.collect side-effect
    # is opaque but must not crash.
    maybe_gc_collect(fov_idx=0, every_n=1)
    maybe_gc_collect(fov_idx=4, every_n=5)


def test_gpu_lock_path_for_job_uses_tmp():
    path = gpu_lock_path_for_job()
    assert path.startswith("/tmp/") or path.startswith("/var/")  # platform-dependent
    assert path.endswith(".lock")


def test_gpu_serialization_lock_is_noop_in_parent():
    """No worker initializer has run, so the lock should yield immediately."""
    with gpu_serialization_lock():
        pass  # Must not block or raise.


def test_make_fov_executor_serial_yields_none():
    runtime = ResolvedRuntime(
        fov_workers=1,
        threads_per_worker=4,
        executor="serial",
        cuda_empty_cache_every_n_timepoints=0,
        gc_collect_every_n_fovs=0,
    )
    with make_fov_executor(runtime) as pool:
        assert pool is None


def test_make_fov_executor_process_yields_real_executor():
    """Spawn smoke: ProcessPoolExecutor with worker initializer comes up cleanly.

    Submits ``os.getpid`` (stdlib, no test-tree imports needed in spawn workers
    — tests/ is not on the python path so test-module-level helpers can't be
    pickled into the child) and verifies the worker returns its own PID,
    distinct from the parent's.
    """
    import importlib

    runtime_mod = importlib.import_module("dynacell.evaluation.runtime")
    runtime = runtime_mod.ResolvedRuntime(
        fov_workers=2,
        threads_per_worker=2,
        executor="process",
        cuda_empty_cache_every_n_timepoints=0,
        gc_collect_every_n_fovs=0,
    )
    parent_pid = os.getpid()
    with runtime_mod.make_fov_executor(runtime) as pool:
        assert pool is not None
        assert isinstance(pool, ProcessPoolExecutor)
        worker_pid = pool.submit(os.getpid).result(timeout=60)
        assert worker_pid != parent_pid
        assert isinstance(worker_pid, int)
