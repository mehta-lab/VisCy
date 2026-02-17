import numpy as np

from viscy_utils.normalize import hist_clipping, unzscore, zscore


def test_zscore():
    img = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = zscore(img)
    assert np.abs(np.mean(result)) < 1e-6
    assert np.abs(np.std(result) - 1.0) < 0.01


def test_zscore_with_params():
    img = np.array([10.0, 20.0, 30.0])
    result = zscore(img, im_mean=20.0, im_std=10.0)
    np.testing.assert_allclose(result, [-1.0, 0.0, 1.0], atol=1e-6)


def test_unzscore_roundtrip():
    img = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    median = np.median(img)
    iqr = np.percentile(img, 75) - np.percentile(img, 25)
    normed = (img - median) / iqr
    result = unzscore(normed, median, iqr)
    np.testing.assert_allclose(result, img, atol=1e-6)


def test_hist_clipping():
    img = np.arange(100, dtype=float)
    clipped = hist_clipping(img, min_percentile=10, max_percentile=90)
    assert clipped.min() >= np.percentile(img, 10) - 1
    assert clipped.max() <= np.percentile(img, 90) + 1
