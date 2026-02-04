import numpy as np

from viscy.utils.mp_utils import get_val_stats


def test_get_val_stats():
    sample_values = np.arange(0, 101)
    stats = get_val_stats(sample_values)
    expected_stats = {
        "mean": 50.0,
        "std": np.std(sample_values),
        "median": 50.0,
        "iqr": 50.0,
        "p1": 1.0,
        "p5": 5.0,
        "p95": 95.0,
        "p99": 99.0,
        "p99_p1": 98.0,
        "p95_p5": 90.0,
    }
    assert stats == expected_stats
