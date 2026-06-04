"""Tests for dynacell.reporting.tables."""

import pytest

pd = pytest.importorskip("pandas")

from dynacell.reporting.tables import (  # noqa: E402
    aggregate_metrics,
    comparison_table,
    load_eval_results,
    to_latex,
)


def _write_pixel_csv(path, rows=None):
    """Write a minimal pixel_metrics.csv fixture."""
    if rows is None:
        rows = [
            {"FOV": "A/0/0", "Timepoint": 0, "PCC": 0.9, "SSIM": 0.85, "NRMSE": 0.1, "PSNR": 30.0},
            {"FOV": "A/0/1", "Timepoint": 0, "PCC": 0.8, "SSIM": 0.80, "NRMSE": 0.2, "PSNR": 25.0},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_mask_csv(path, rows=None):
    """Write a minimal mask_metrics.csv fixture."""
    if rows is None:
        rows = [
            {"FOV": "A/0/0", "Timepoint": 0, "Dice": 0.7, "IoU": 0.6},
            {"FOV": "A/0/1", "Timepoint": 0, "Dice": 0.8, "IoU": 0.7},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


class TestLoadEvalResults:
    """Tests for load_eval_results."""

    def test_reads_csvs(self, tmp_path):
        """Reads pixel and mask CSVs into DataFrames."""
        _write_pixel_csv(tmp_path / "pixel_metrics.csv")
        _write_mask_csv(tmp_path / "mask_metrics.csv")
        results = load_eval_results(tmp_path)
        assert "pixel" in results
        assert "mask" in results
        assert len(results["pixel"]) == 2

    def test_missing_feature_csv(self, tmp_path):
        """No crash when feature CSV is absent."""
        _write_pixel_csv(tmp_path / "pixel_metrics.csv")
        results = load_eval_results(tmp_path)
        assert "pixel" in results
        assert "feature" not in results


class TestAggregateMetrics:
    """Tests for aggregate_metrics."""

    def test_mean_std(self):
        """Computes correct mean and std."""
        df = pd.DataFrame({"PCC": [0.9, 0.8], "SSIM": [0.85, 0.80]})
        agg = aggregate_metrics(df, metrics=["PCC", "SSIM"])
        assert abs(agg.loc["mean", "PCC"] - 0.85) < 1e-9
        assert agg.loc["std", "PCC"] > 0


class TestComparisonTable:
    """Tests for comparison_table."""

    def test_shape(self, tmp_path):
        """Two model dirs produce correct rows and cols."""
        dir_a = tmp_path / "model_a"
        dir_b = tmp_path / "model_b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_pixel_csv(dir_a / "pixel_metrics.csv")
        _write_mask_csv(dir_a / "mask_metrics.csv")
        _write_pixel_csv(
            dir_b / "pixel_metrics.csv",
            [{"FOV": "B/0/0", "Timepoint": 0, "PCC": 0.95, "SSIM": 0.90, "NRMSE": 0.05, "PSNR": 35.0}],
        )
        _write_mask_csv(
            dir_b / "mask_metrics.csv",
            [{"FOV": "B/0/0", "Timepoint": 0, "Dice": 0.9, "IoU": 0.8}],
        )
        table = comparison_table({"ModelA": dir_a, "ModelB": dir_b}, metrics=["PCC", "SSIM", "Dice"])
        assert table.shape[0] == 2
        assert "PCC" in table.columns
        assert "ModelA" in table.index


class TestToLatex:
    """Tests for to_latex."""

    def test_bolds_best(self, tmp_path):
        """Best value in each column is wrapped in textbf."""
        dir_a = tmp_path / "model_a"
        dir_b = tmp_path / "model_b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_pixel_csv(dir_a / "pixel_metrics.csv")
        _write_pixel_csv(
            dir_b / "pixel_metrics.csv",
            [{"FOV": "B/0/0", "Timepoint": 0, "PCC": 0.95, "SSIM": 0.90, "NRMSE": 0.05, "PSNR": 35.0}],
        )
        table = comparison_table({"ModelA": dir_a, "ModelB": dir_b}, metrics=["PCC"])
        latex = to_latex(table, bold_best=True)
        assert "\\textbf{" in latex
