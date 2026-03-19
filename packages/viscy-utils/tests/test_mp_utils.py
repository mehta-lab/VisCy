import numpy as np

from viscy_utils.mp_utils import get_val_stats


def test_get_val_stats():
    values = np.random.randn(1000)
    stats = get_val_stats(values)
    assert "mean" in stats
    assert "std" in stats
    assert "median" in stats
    assert "iqr" in stats
    assert "p5" in stats
    assert "p95" in stats
    assert stats["iqr"] >= 0
    assert abs(stats["mean"] - float(np.nanmean(values))) < 1e-6
