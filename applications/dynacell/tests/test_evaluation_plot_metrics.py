"""Integration tests for ``evaluation.utils.plot_metrics``.

``test_evaluation_pipeline.py`` stubs this function out, so the real grouping /
sorting logic had no coverage. These tests call it directly and assert on the
values it hands to matplotlib, which is what pins the per-FOV mean and the
per-timepoint ordering.
"""

import pytest

pd = pytest.importorskip("pandas")
mpl_axes = pytest.importorskip("matplotlib.axes")

from dynacell.evaluation.utils import plot_metrics  # noqa: E402


@pytest.fixture
def bar_spy(monkeypatch):
    """Record the bar heights of every ``Axes.bar`` call, in call order."""
    calls: list[list[float]] = []
    real_bar = mpl_axes.Axes.bar

    def spy(self, x, height, *args, **kwargs):
        calls.append([float(h) for h in height])
        return real_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(mpl_axes.Axes, "bar", spy)
    return calls


@pytest.fixture
def line_spy(monkeypatch):
    """Record ``(x, y, label)`` of every ``Axes.plot`` call, in call order."""
    calls: list[tuple[list, list, object]] = []
    real_plot = mpl_axes.Axes.plot

    def spy(self, *args, **kwargs):
        if len(args) >= 2:
            calls.append(([*args[0]], [*args[1]], kwargs.get("label")))
        return real_plot(self, *args, **kwargs)

    monkeypatch.setattr(mpl_axes.Axes, "plot", spy)
    return calls


def _multi_timepoint_frame() -> pd.DataFrame:
    """Two FOVs x three timepoints, rows deliberately out of Timepoint order."""
    rows = []
    for fov, base in (("A/0/0", 0.1), ("A/0/1", 0.5)):
        for t in (2, 0, 1):  # unsorted on purpose: the sort must be load-bearing
            rows.append({"FOV": fov, "Timepoint": t, "PCC": base + t, "SSIM": base + 10 * t})
    return pd.DataFrame(rows)


def test_plot_metrics_writes_both_plots_per_metric(tmp_path):
    """Multi-timepoint input produces a mean-per-FOV and a per-timepoint plot per metric."""
    plot_metrics(_multi_timepoint_frame(), tmp_path, "pixel_metrics")

    plot_dir = tmp_path / "pixel_metrics"
    assert {p.name for p in plot_dir.glob("*.png")} == {
        "PCC_fov_mean.png",
        "PCC_timepoints.png",
        "SSIM_fov_mean.png",
        "SSIM_timepoints.png",
    }


def test_plot_metrics_bars_are_per_fov_means(tmp_path, bar_spy):
    """Bar heights are the mean over each FOV's timepoints, in sorted FOV order."""
    plot_metrics(_multi_timepoint_frame(), tmp_path, "pixel_metrics")

    # PCC: A/0/0 -> mean(2.1, 0.1, 1.1) = 1.1 ; A/0/1 -> mean(2.5, 0.5, 1.5) = 1.5
    # SSIM: A/0/0 -> mean(20.1, 0.1, 10.1) = 10.1 ; A/0/1 -> mean(20.5, 0.5, 10.5) = 10.5
    assert len(bar_spy) == 2
    assert bar_spy[0] == pytest.approx([1.1, 1.5])
    assert bar_spy[1] == pytest.approx([10.1, 10.5])


def test_plot_metrics_timepoint_series_are_sorted(tmp_path, line_spy):
    """Each FOV's timepoint series is sorted by Timepoint, not left in row order."""
    plot_metrics(_multi_timepoint_frame(), tmp_path, "pixel_metrics")

    # 2 metrics x 2 multi-timepoint FOVs.
    assert len(line_spy) == 4
    for x, y, label in line_spy:
        assert x == [0, 1, 2], f"unsorted timepoints for {label}"
        assert y == sorted(y), f"values not tracking sorted timepoints for {label}"

    pcc_by_label = {label: y for x, y, label in line_spy[:2]}
    assert pcc_by_label["A/0/0"] == pytest.approx([0.1, 1.1, 2.1])
    assert pcc_by_label["A/0/1"] == pytest.approx([0.5, 1.5, 2.5])


def test_plot_metrics_single_timepoint_skips_timepoint_plot(tmp_path, bar_spy, line_spy):
    """With one timepoint per FOV there is nothing to plot over time."""
    df = pd.DataFrame(
        [
            {"FOV": "A/0/0", "Timepoint": 0, "PCC": 0.9},
            {"FOV": "A/0/1", "Timepoint": 0, "PCC": 0.7},
        ]
    )
    plot_metrics(df, tmp_path, "pixel_metrics")

    plot_dir = tmp_path / "pixel_metrics"
    assert {p.name for p in plot_dir.glob("*.png")} == {"PCC_fov_mean.png"}
    assert bar_spy == [pytest.approx([0.9, 0.7])]
    assert line_spy == []
