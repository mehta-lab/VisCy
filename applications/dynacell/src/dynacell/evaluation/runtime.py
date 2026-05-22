"""Eval runtime: thread budgeting + (in later commits) process-pool primitives.

Three layers of thread-cap discipline, in order of when they bite:

1. ``early_apply_env_caps()`` reads ``DYNACELL_THREADS_PER_WORKER`` from the
   environment and sets BLAS/OMP env vars before any C extension loads. Called
   from ``dynacell.__main__:main_cli()`` as the first statement.
2. ``apply_thread_budget(threads)`` is the in-process safety net: sets env
   (respecting caller-set values), calls ``torch.set_num_threads``, and
   activates a module-level ``threadpoolctl.threadpool_limits`` cap.
3. (C3) per-worker initializer re-applies the cap in each spawned child.

Module-level imports are stdlib + ``threadpoolctl`` only. ``torch`` is imported
lazily inside function bodies so spawn-context workers can set
``CUDA_VISIBLE_DEVICES`` before any torch-driven CUDA context creation.
"""

from __future__ import annotations

import csv
import gc
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits

logger = logging.getLogger(__name__)

# Module-level handle for the active threadpool_limits context. We enter the
# context once at startup and keep the handle live for the rest of the process
# lifetime (threadpool_limits is a per-process thread-local API; the cap stays
# armed as long as the context object is not exited).
_ACTIVE_THREADPOOL_HANDLE: Any = None

# Env vars that affect BLAS / OMP / framework thread counts. Listed in
# priority order; the first three actually matter for this codebase (numpy is
# scipy-openblas, torch + cellpose honor OMP), the rest are defense-in-depth.
_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TBB_NUM_THREADS",
)

# Env var that triggers the per-T memory hygiene knobs at runtime even when
# the YAML defaults are off. Useful for operator mitigation post-ship.
_FORCE_PER_T_HYGIENE_ENV: str = "DYNACELL_FORCE_PER_T_HYGIENE"


def _cpu_count() -> int:
    """Return the number of CPUs visible to this process.

    Prefers ``os.sched_getaffinity(0)`` when available (Linux): it respects
    SLURM cgroups + cpuset limits, so we don't oversize ``fov_workers`` or
    ``threads_per_worker`` past the cgroup-visible budget. Falls back to
    ``os.cpu_count()`` (which reports the host's logical CPU count, ignoring
    cgroups) on platforms without ``sched_getaffinity``.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class ResolvedRuntime:
    """Materialized runtime config with all ``"auto"`` values resolved.

    Parameters
    ----------
    fov_workers : int
        Outer FOV-level parallelism. 1 = sequential.
    threads_per_worker : int
        BLAS/OMP/torch intra-op threads per worker process.
    executor : {"serial", "process"}
        FOV loop executor.
    cuda_empty_cache_every_n_timepoints : int
        Call ``torch.cuda.empty_cache()`` every N timepoints (0 = off).
    gc_collect_every_n_fovs : int
        Call ``gc.collect()`` every N FOVs (0 = off).
    """

    fov_workers: int
    threads_per_worker: int
    executor: Literal["serial", "process"]
    cuda_empty_cache_every_n_timepoints: int
    gc_collect_every_n_fovs: int


def early_apply_env_caps() -> None:
    """Set BLAS/OMP env vars before any C-extension load.

    Reads ``DYNACELL_THREADS_PER_WORKER`` from env; if set, exports the BLAS
    env vars to that value. This is the only layer that bites BLAS at
    extension-load time. Idempotent: existing caller-set env vars are
    respected (we only write if the var is unset).

    Called from ``dynacell.__main__:main_cli()`` as the first statement,
    before any heavy import triggers numpy/torch BLAS initialization.
    """
    threads = os.environ.get("DYNACELL_THREADS_PER_WORKER")
    if threads is None:
        return
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, threads)


def apply_thread_budget(threads: int) -> None:
    """Apply a thread budget to the current process.

    Performs three layered actions:

    1. Sets BLAS/OMP env vars (only when not already set by the caller);
       logs at WARNING if a caller-set value differs from ``threads``.
    2. Calls ``torch.set_num_threads(threads)``.
    3. Calls ``torch.set_num_interop_threads(max(1, threads // 2))`` —
       wrapped in try/except since this raises after any parallel op has
       run (the common case from Hydra's main).
    4. Activates a module-level ``threadpool_limits(limits=threads,
       user_api="blas")`` context for the rest of the process lifetime.

    Parameters
    ----------
    threads : int
        Number of threads to cap BLAS/OMP/torch to. Must be >= 1.
    """
    global _ACTIVE_THREADPOOL_HANDLE
    if threads < 1:
        raise ValueError(f"apply_thread_budget: threads must be >= 1, got {threads}")

    # 1. Env vars: respect caller-set values; WARN on mismatch.
    threads_str = str(threads)
    for var in _THREAD_ENV_VARS:
        existing = os.environ.get(var)
        if existing is None:
            os.environ[var] = threads_str
        elif existing != threads_str:
            logger.log(
                logging.WARNING,
                "%s already set to %r by environment; not overriding with %r",
                var,
                existing,
                threads_str,
            )

    # 2 + 3. Torch thread caps (deferred import).
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, threads // 2))
    except RuntimeError as e:
        logger.info("torch.set_num_interop_threads: %s; continuing with current value", e)

    # 4. Activate threadpool_limits for the rest of the process.
    # On re-arm, the old handle becomes unreachable and is GC'd. We don't
    # call its __exit__ because that would *restore* the prior thread cap
    # — but we want the new cap to be the active one. threadpoolctl's
    # native thread-limit state is set when the new limiter is entered
    # (via __enter__ inside threadpool_limits.__init__), so the new limit
    # is active immediately; dropping the old handle is intentional.
    _ACTIVE_THREADPOOL_HANDLE = threadpool_limits(limits=threads, user_api="blas")


def _get_int(config: DictConfig, key: str, default: int) -> int:
    """Read an int field from a runtime config with a default fallback."""
    val = OmegaConf.select(config, key, default=default)
    return int(val)


def resolve_runtime(
    config: DictConfig,
    n_positions: int | None = None,
    freeze_threads_per_worker: int | None = None,
) -> ResolvedRuntime:
    """Materialize the runtime config block with all ``"auto"`` values resolved.

    Two-phase use from ``evaluate_predictions``:

    * **Phase 1** (top of function): ``resolve_runtime(config)``. Provisional
      ``fov_workers`` from ``cpu_count // 4``; ``threads_per_worker`` from
      ``cpu_count // provisional_fov_workers``. Parent applies BLAS cap with
      this value.
    * **Phase 2** (after position list built):
      ``resolve_runtime(config, n_positions=N, freeze_threads_per_worker=T)``.
      Clamps ``fov_workers`` to ``min(provisional, N)`` and returns
      ``threads_per_worker == T`` unchanged so the worker initializer matches
      what the parent already capped to.

    The ``DYNACELL_FORCE_PER_T_HYGIENE=1`` env var flips both hygiene knobs
    on at runtime regardless of YAML defaults — operator escape hatch.

    Parameters
    ----------
    config : DictConfig
        The eval config (with ``config.runtime`` block).
    n_positions : int or None
        Number of positions in the eval; used to clamp ``"auto"`` workers in
        Phase 2. ``None`` in Phase 1.
    freeze_threads_per_worker : int or None
        If set, ``threads_per_worker`` returns this value unconditionally
        (Phase 2 uses this to stay consistent with the parent's BLAS cap).

    Returns
    -------
    ResolvedRuntime
        Materialized config; safe to pass across pickle boundaries.

    Raises
    ------
    ValueError
        If literal ``fov_workers > 1`` is set with ``executor == "serial"``.
    """
    runtime = OmegaConf.select(config, "runtime", default=None)
    if runtime is None:
        # No runtime block configured — fall back to sequential defaults
        # matching today's behavior.
        return ResolvedRuntime(
            fov_workers=1,
            threads_per_worker=_cpu_count(),
            executor="serial",
            cuda_empty_cache_every_n_timepoints=0,
            gc_collect_every_n_fovs=0,
        )

    executor = str(OmegaConf.select(runtime, "executor", default="serial"))
    if executor not in ("serial", "process"):
        raise ValueError(f"runtime.executor must be 'serial' or 'process', got {executor!r}")

    cpu_count = _cpu_count()
    raw_fov_workers = OmegaConf.select(runtime, "fov_workers", default=1)
    raw_threads = OmegaConf.select(runtime, "threads_per_worker", default="auto")

    # Resolve fov_workers.
    if isinstance(raw_fov_workers, int):
        resolved_fov_workers = int(raw_fov_workers)
        if resolved_fov_workers < 1:
            raise ValueError(f"runtime.fov_workers must be >= 1, got {resolved_fov_workers}")
        if resolved_fov_workers > 1 and executor == "serial":
            raise ValueError(
                f"runtime.fov_workers={resolved_fov_workers} requires runtime.executor='process' (got 'serial')"
            )
    elif raw_fov_workers == "auto":
        if executor == "serial":
            resolved_fov_workers = 1
        else:
            # Provisional divisor of 4 when threads_per_worker is also "auto".
            divisor = int(raw_threads) if isinstance(raw_threads, int) else 4
            provisional = max(1, cpu_count // divisor)
            clamp = n_positions if n_positions is not None else cpu_count
            resolved_fov_workers = max(1, min(provisional, clamp))
    else:
        raise ValueError(f"runtime.fov_workers must be int or 'auto', got {raw_fov_workers!r}")

    # Auto-demote process→serial when only 1 worker resolves (avoids 5s spawn cost).
    if executor == "process" and resolved_fov_workers == 1:
        logger.info("runtime.fov_workers resolved to 1; auto-demoting executor 'process' -> 'serial'")
        executor = "serial"

    # Resolve threads_per_worker.
    if freeze_threads_per_worker is not None:
        resolved_threads = int(freeze_threads_per_worker)
    elif isinstance(raw_threads, int):
        resolved_threads = int(raw_threads)
        if resolved_threads < 1:
            raise ValueError(f"runtime.threads_per_worker must be >= 1, got {resolved_threads}")
    elif raw_threads == "auto":
        resolved_threads = max(1, cpu_count // resolved_fov_workers)
    else:
        raise ValueError(f"runtime.threads_per_worker must be int or 'auto', got {raw_threads!r}")

    # Per-T memory hygiene knobs; env-var escape hatch flips both on.
    force_hygiene = os.environ.get(_FORCE_PER_T_HYGIENE_ENV, "0") == "1"
    if force_hygiene:
        cuda_empty_n = max(1, _get_int(runtime, "cuda_empty_cache_every_n_timepoints", 0))
        gc_collect_n = max(1, _get_int(runtime, "gc_collect_every_n_fovs", 0))
        logger.warning(
            "%s=1 — forcing cuda_empty_cache_every_n_timepoints=%d, gc_collect_every_n_fovs=%d",
            _FORCE_PER_T_HYGIENE_ENV,
            cuda_empty_n,
            gc_collect_n,
        )
    else:
        cuda_empty_n = _get_int(runtime, "cuda_empty_cache_every_n_timepoints", 0)
        gc_collect_n = _get_int(runtime, "gc_collect_every_n_fovs", 0)

    return ResolvedRuntime(
        fov_workers=resolved_fov_workers,
        threads_per_worker=resolved_threads,
        executor=executor,  # type: ignore[arg-type]
        cuda_empty_cache_every_n_timepoints=cuda_empty_n,
        gc_collect_every_n_fovs=gc_collect_n,
    )


# ---------------------------------------------------------------------------
# Per-T memory hygiene + region timing instrumentation.
# ---------------------------------------------------------------------------

# Per-process timing collector. Each entry is (pos_name, t_or_None, region,
# seconds). Under executor=process (C3+), workers return their slice in
# FovResult.timings; the parent's aggregator concatenates.
_TIMINGS: list[tuple[str, int | None, str, float]] = []


def reset_timings() -> None:
    """Clear the per-process timing collector."""
    _TIMINGS.clear()


def get_timings() -> list[tuple[str, int | None, str, float]]:
    """Return a copy of the per-process timing collector."""
    return list(_TIMINGS)


def extend_timings(rows: list[tuple[str, int | None, str, float]]) -> None:
    """Append a batch of timing rows (used by parent aggregator under process mode)."""
    _TIMINGS.extend(rows)


@contextmanager
def region_timer(region: str, pos_name: str, t: int | None = None) -> Iterator[None]:
    """Record wall time of the wrapped block to the per-process timing collector.

    Parameters
    ----------
    region : str
        Region tag (e.g. "pixel_metrics", "mask_gt", "features_pred_per_t").
    pos_name : str
        FOV identifier (typically "<row>/<col>/<fov>").
    t : int or None
        Timepoint index when the region is per-T; ``None`` for FOV-level regions.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        _TIMINGS.append((pos_name, t, region, time.perf_counter() - start))


def dump_timings_csv(save_dir: Path) -> Path | None:
    """Write the collected timings to ``<save_dir>/eval_timing.csv`` and return the path.

    Returns ``None`` if no timings were recorded.
    """
    if not _TIMINGS:
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "eval_timing.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pos_name", "t", "region", "seconds"])
        for pos_name, t, region, seconds in _TIMINGS:
            writer.writerow([pos_name, "" if t is None else t, region, f"{seconds:.6f}"])
    return out_path


def maybe_empty_cuda_cache(t: int, every_n: int) -> None:
    """Call ``torch.cuda.empty_cache()`` every ``every_n`` timepoints.

    No-op when ``every_n <= 0`` or CUDA is unavailable.
    """
    if every_n <= 0:
        return
    if (t + 1) % every_n != 0:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_gc_collect(fov_idx: int, every_n: int) -> None:
    """Call ``gc.collect()`` every ``every_n`` FOVs.

    ``fov_idx`` is 0-based; collects when ``(fov_idx + 1) % every_n == 0``.
    No-op when ``every_n <= 0``.
    """
    if every_n <= 0:
        return
    if (fov_idx + 1) % every_n != 0:
        return
    gc.collect()


# ---------------------------------------------------------------------------
# Process-pool primitives for FOV-level parallelism (C4).
# Spawn context so each worker gets its own CUDA context and BLAS load.
# ---------------------------------------------------------------------------

# Worker-global GPU lock path, set by _worker_initializer. ``None`` in the
# parent / serial mode -> gpu_serialization_lock is a no-op.
_GPU_LOCK_PATH: str | None = None


def _worker_initializer(threads: int, gpu_lock_path: str | None) -> None:
    """Initialize a freshly-spawned worker process.

    Order matters: env caps are set BEFORE ``apply_thread_budget`` is called,
    which lazy-imports torch. If torch were imported first, ``CUDA_VISIBLE_DEVICES``
    overrides (if any) would come too late.
    """
    global _GPU_LOCK_PATH
    # Env caps first so BLAS C-extensions load with the right thread count.
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(threads))
    apply_thread_budget(threads)
    _GPU_LOCK_PATH = gpu_lock_path


def is_worker() -> bool:
    """Return True if called from a spawn worker (post-``_worker_initializer``).

    Used to suppress inner per-T tqdm bars in workers — under
    ``fov_workers>1`` the N concurrent inner bars interleave on the parent's
    stderr unreadably. The outer per-FOV tqdm (in the parent) stays visible.
    """
    return _GPU_LOCK_PATH is not None


@contextmanager
def gpu_serialization_lock(gate: bool = True) -> Iterator[None]:
    """Serialize GPU operations across workers via an ``fcntl.flock``.

    Parameters
    ----------
    gate : bool
        When False (e.g., the protected region runs CPU-only under
        ``use_gpu=false``), this is a no-op even in process mode.
        Callers pass ``gate=config.use_gpu`` for regions whose GPU
        usage tracks the ``use_gpu`` flag, so CPU-only runs don't
        serialize across workers needlessly.

    Other no-op condition: ``_GPU_LOCK_PATH`` unset (parent / serial
    mode). Used to wrap model loads and per-FOV GPU operations under
    ``executor=process``. CPU-only execution is opted into at the call
    site by passing ``gate=False`` (typically ``gate=config.use_gpu``);
    the lock itself does NOT probe ``torch.cuda.is_available()``.
    """
    if not gate or _GPU_LOCK_PATH is None:
        yield
        return
    import fcntl

    with open(_GPU_LOCK_PATH, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def gpu_lock_path_for_job() -> str:
    """Return a node-local lock path tagged by SLURM_JOB_ID (or PID as fallback).

    Stays on local /tmp so fcntl serialization is fast and doesn't go through
    NFS lockd. One lock per job, shared across all workers in that job.
    """
    import tempfile

    tag = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    return str(Path(tempfile.gettempdir()) / f"dynacell_gpu_{tag}.lock")


@contextmanager
def make_fov_executor(runtime: ResolvedRuntime) -> Iterator[Any]:
    """Yield a ``ProcessPoolExecutor`` for ``executor=process`` or ``None`` for serial.

    Callers branch on the yielded value: ``None`` means run inline; an
    ``Executor`` means submit ``_process_one_fov`` invocations to workers.

    Cleanup: on exit (normal or exception), the pool is shut down with
    ``cancel_futures=True``. In-flight workers continue to completion
    (Python's ``ProcessPoolExecutor`` does not signal them); user may need
    ``scancel`` to fully release GPU memory on Ctrl-C.
    """
    if runtime.executor == "serial":
        yield None
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    lock_path = gpu_lock_path_for_job()
    # Touch the lock file so workers can flock-open it before any FOV runs.
    Path(lock_path).touch()
    ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(
        max_workers=runtime.fov_workers,
        mp_context=ctx,
        initializer=_worker_initializer,
        initargs=(runtime.threads_per_worker, lock_path),
    )
    try:
        yield pool
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
