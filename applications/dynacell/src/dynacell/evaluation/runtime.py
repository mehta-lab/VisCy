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

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

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
    if _ACTIVE_THREADPOOL_HANDLE is None:
        _ACTIVE_THREADPOOL_HANDLE = threadpool_limits(limits=threads, user_api="blas")
    else:
        # Re-arming: enter a fresh context with the new limit. The old handle
        # stays alive (its __exit__ is never called); threadpoolctl honors
        # the innermost active limit. Acceptable for re-call from a worker
        # initializer that inherited a parent's handle.
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
            threads_per_worker=max(1, os.cpu_count() or 1),
            executor="serial",
            cuda_empty_cache_every_n_timepoints=0,
            gc_collect_every_n_fovs=0,
        )

    executor = str(OmegaConf.select(runtime, "executor", default="serial"))
    if executor not in ("serial", "process"):
        raise ValueError(f"runtime.executor must be 'serial' or 'process', got {executor!r}")

    cpu_count = max(1, os.cpu_count() or 1)
    raw_fov_workers = OmegaConf.select(runtime, "fov_workers", default=1)
    raw_threads = OmegaConf.select(runtime, "threads_per_worker", default="auto")

    # Resolve fov_workers.
    if isinstance(raw_fov_workers, int):
        resolved_fov_workers = int(raw_fov_workers)
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
