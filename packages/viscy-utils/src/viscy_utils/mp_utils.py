"""Multiprocessing utilities for dataset processing."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np


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
    percentile_values = {
        k: float(v)
        for k, v in zip(percentiles, np.nanpercentile(sample_values, percentiles))
    }
    meta_row = {
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
