"""Tests for dynacell.reporting.figures."""

import pytest

pd = pytest.importorskip("pandas")
plt = pytest.importorskip("matplotlib.pyplot")

from dynacell.reporting.figures import metric_comparison_barplot  # noqa: E402


def _write_pixel_csv(path, rows=None):
    """Write a minimal pixel_metrics.csv fixture."""
    if rows is None:
        rows = [
            {"FOV": "A/0/0", "Timepoint": 0, "PCC": 0.9, "SSIM": 0.85},
            {"FOV": "A/0/1", "Timepoint": 0, "PCC": 0.8, "SSIM": 0.80},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


class TestMetricComparisonBarplot:
    """Tests for metric_comparison_barplot."""

    def test_returns_figure(self, tmp_path):
        """Direct call returns a matplotlib Figure."""
        dir_a = tmp_path / "model_a"
        dir_a.mkdir()
        _write_pixel_csv(dir_a / "pixel_metrics.csv")
        fig = metric_comparison_barplot({"ModelA": dir_a}, metrics=["PCC", "SSIM"])
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_empty_model_results(self):
        """Empty model_results dict returns 'No data' figure."""
        fig = metric_comparison_barplot({}, metrics=["PCC"])
        try:
            texts = [t.get_text() for t in fig.axes[0].texts]
            assert "No data" in texts
        finally:
            plt.close(fig)

    def test_saves_to_disk(self, tmp_path):
        """save_path writes a nonzero-size file."""
        dir_a = tmp_path / "model_a"
        dir_a.mkdir()
        _write_pixel_csv(dir_a / "pixel_metrics.csv")
        out = tmp_path / "plot.pdf"
        fig = metric_comparison_barplot({"ModelA": dir_a}, metrics=["PCC"], save_path=out)
        plt.close(fig)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_multiple_models(self, tmp_path):
        """Barplot with two models has correct legend entries."""
        dir_a = tmp_path / "model_a"
        dir_b = tmp_path / "model_b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_pixel_csv(dir_a / "pixel_metrics.csv")
        _write_pixel_csv(
            dir_b / "pixel_metrics.csv",
            [{"FOV": "B/0/0", "Timepoint": 0, "PCC": 0.95, "SSIM": 0.90}],
        )
        fig = metric_comparison_barplot({"ModelA": dir_a, "ModelB": dir_b}, metrics=["PCC", "SSIM"])
        try:
            legend_texts = [t.get_text() for t in fig.axes[0].get_legend().texts]
            assert "ModelA" in legend_texts
            assert "ModelB" in legend_texts
        finally:
            plt.close(fig)

    def test_empty_results_dir(self, tmp_path):
        """Model with empty results dir produces 'No data' figure."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        fig = metric_comparison_barplot({"EmptyModel": empty_dir}, metrics=["PCC"])
        plt.close(fig)
