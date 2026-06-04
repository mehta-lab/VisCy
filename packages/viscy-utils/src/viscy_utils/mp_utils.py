"""Multiprocessing utilities for dataset processing."""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def available_cpus(default: int = 1) -> int:
    """Return the number of CPUs the current process is allowed to use.

    Prefers ``SLURM_CPUS_PER_TASK`` (the cluster-scheduler-allocated count)
    over ``os.cpu_count()`` (the node's total physical cores). On a 48-core
    node where SLURM allocated 16 cores, this returns 16 — preventing
    oversubscription and respecting the cgroup pinning that SLURM sets.

    Use this anywhere you'd otherwise reach for ``os.cpu_count()`` to size
    a thread pool, ``n_jobs``, ``num_workers``, ``data_copy_concurrency``,
    etc. Library defaults like sklearn's ``n_jobs=-1`` and BLAS env autodetect
    are NOT SLURM-aware and will spawn one thread per physical core.

    Parameters
    ----------
    default : int, optional
        Fallback when neither ``SLURM_CPUS_PER_TASK`` nor ``os.cpu_count()``
        is informative. Defaults to 1 (safe single-threaded).

    Returns
    -------
    int
        Number of CPUs available to this process, ``>= 1``.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return int(slurm_cpus)
    return os.cpu_count() or default


def mp_wrapper(fn, fn_args, workers):
    """Execute function with multiprocessing.

    Parameters
    ----------
    fn : callable
        Function to execute.
    fn_args : list of tuple
        List of tuples of function arguments.
    workers : int
        Max number of workers.

    Returns
    -------
    list
        List of returned values.
    """
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(fn, *zip(*fn_args))
    return list(res)


def mp_get_val_stats(fn_args, workers):
    """Compute statistics of numpy arrays with multiprocessing.

    Parameters
    ----------
    fn_args : list of tuple
        List of tuples of function arguments.
    workers : int
        Max number of workers.

    Returns
    -------
    list
        List of statistics dictionaries.
    """
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(get_val_stats, fn_args)
    return list(res)


def get_val_stats(sample_values):
    """Compute statistics of a numpy array.

    Parameters
    ----------
    sample_values : array_like
        Values to compute statistics for.

    Returns
    -------
    dict
        Dictionary with intensity statistics (mean, std, median, iqr,
        percentiles).
    """
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    percentile_values = {k: float(v) for k, v in zip(percentiles, np.nanpercentile(sample_values, percentiles))}
    meta_row = {
        "min": float(np.nanmin(sample_values)),
        "max": float(np.nanmax(sample_values)),
        "mean": float(np.nanmean(sample_values)),
        "std": float(np.nanstd(sample_values)),
        "median": percentile_values[50],
        "iqr": percentile_values[75] - percentile_values[25],
        "p5": percentile_values[5],
        "p95": percentile_values[95],
        "p95_p5": percentile_values[95] - percentile_values[5],
        "p1": percentile_values[1],
        "p99": percentile_values[99],
        "p99_p1": percentile_values[99] - percentile_values[1],
    }
    return meta_row
