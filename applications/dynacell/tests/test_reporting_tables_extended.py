"""Extended tests for dynacell.reporting.tables — lower-is-better, caption, edge cases."""

import pytest

pd = pytest.importorskip("pandas")

from dynacell.reporting.tables import comparison_table, load_and_aggregate, to_latex  # noqa: E402


def _write_csv(path, rows):
    """Write rows to a CSV file."""
    pd.DataFrame(rows).to_csv(path, index=False)


class TestToLatexLowerIsBetter:
    """Tests for to_latex bolding of lower-is-better metrics like NRMSE."""

    def test_bolds_lowest_nrmse(self, tmp_path):
        """For NRMSE (lower is better), the lowest value gets bold."""
        dir_a = tmp_path / "model_a"
        dir_b = tmp_path / "model_b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_csv(dir_a / "pixel_metrics.csv", [{"FOV": "A/0/0", "Timepoint": 0, "NRMSE": 0.20}])
        _write_csv(dir_b / "pixel_metrics.csv", [{"FOV": "B/0/0", "Timepoint": 0, "NRMSE": 0.05}])
        table = comparison_table({"ModelA": dir_a, "ModelB": dir_b}, metrics=["NRMSE"])
        latex = to_latex(table, bold_best=True)
        assert "\\textbf{" in latex
        for line in latex.split("\n"):
            if "ModelB" in line:
                assert "\\textbf{" in line


class TestToLatexForegroundColumns:
    """The FG_* columns bold in the direction of their whole-image counterparts."""

    def test_fg_columns_bold_in_their_direction(self, tmp_path):
        """FG_SI_SSIM/FG_PCC/FG_SI_PSNR bold the highest; FG_SI_NRMSE bolds the lowest."""
        dir_a = tmp_path / "model_a"
        dir_b = tmp_path / "model_b"
        dir_a.mkdir()
        dir_b.mkdir()
        good = {"FG_PCC": 0.9, "FG_SI_SSIM": 0.8, "FG_SI_PSNR": 30.0, "FG_SI_NRMSE": 0.05}
        bad = {"FG_PCC": 0.5, "FG_SI_SSIM": 0.4, "FG_SI_PSNR": 20.0, "FG_SI_NRMSE": 0.20}
        _write_csv(dir_a / "pixel_metrics.csv", [{"FOV": "A/0/0", "Timepoint": 0, **bad}])
        _write_csv(dir_b / "pixel_metrics.csv", [{"FOV": "B/0/0", "Timepoint": 0, **good}])
        table = comparison_table({"ModelA": dir_a, "ModelB": dir_b}, metrics=list(good))
        lines = to_latex(table, bold_best=True).split("\n")
        assert sum("\\textbf{" in cell for line in lines if "ModelB" in line for cell in line.split("&")) == 4
        assert not any("\\textbf{" in line for line in lines if "ModelA" in line)

    def test_fg_frac_is_never_bolded(self, tmp_path):
        """FG_frac describes the GT region, not a model: no row of it is bolded as best."""
        dirs = {}
        for name, frac in (("ModelA", 0.20), ("ModelB", 0.25)):
            dirs[name] = tmp_path / name
            dirs[name].mkdir()
            _write_csv(dirs[name] / "pixel_metrics.csv", [{"FOV": "A/0/0", "Timepoint": 0, "FG_frac": frac}])
        table = comparison_table(dirs, metrics=["FG_frac"])
        assert "\\textbf{" not in to_latex(table, bold_best=True)

    def test_fg_columns_are_not_default_pixel_metrics(self):
        """Default tables are unchanged: FG_* are reported only when asked for."""
        from dynacell.reporting.tables import PIXEL_METRICS

        assert not any(m.startswith("FG_") for m in PIXEL_METRICS)


class TestToLatexCaptionLabel:
    """Tests for to_latex caption and label wrapping."""

    def test_with_caption_and_label(self, tmp_path):
        """caption/label wraps output in table environment."""
        dir_a = tmp_path / "model_a"
        dir_a.mkdir()
        _write_csv(dir_a / "pixel_metrics.csv", [{"FOV": "A/0/0", "Timepoint": 0, "PCC": 0.9}])
        table = comparison_table({"ModelA": dir_a}, metrics=["PCC"])
        latex = to_latex(table, caption="My caption", label="tab:test")
        assert "\\begin{table}" in latex
        assert "\\caption{My caption}" in latex
        assert "\\label{tab:test}" in latex

    def test_without_caption_no_table_env(self, tmp_path):
        """Without caption/label, no table environment wrapper."""
        dir_a = tmp_path / "model_a"
        dir_a.mkdir()
        _write_csv(dir_a / "pixel_metrics.csv", [{"FOV": "A/0/0", "Timepoint": 0, "PCC": 0.9}])
        table = comparison_table({"ModelA": dir_a}, metrics=["PCC"])
        latex = to_latex(table)
        assert "\\begin{table}" not in latex


class TestToLatexSingleModel:
    """Tests for single-model table behavior."""

    def test_single_model_no_bolding(self, tmp_path):
        """Single-model table skips bolding."""
        dir_a = tmp_path / "model_a"
        dir_a.mkdir()
        _write_csv(dir_a / "pixel_metrics.csv", [{"FOV": "A/0/0", "Timepoint": 0, "PCC": 0.9}])
        table = comparison_table({"ModelA": dir_a}, metrics=["PCC"])
        latex = to_latex(table, bold_best=True)
        assert "\\textbf{" not in latex


class TestLoadAndAggregate:
    """Tests for load_and_aggregate."""

    def test_empty_dir_returns_empty(self, tmp_path):
        """Empty results dir returns empty DataFrame and empty metrics list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        agg, available = load_and_aggregate(empty_dir, ["PCC", "SSIM"])
        assert agg.empty
        assert available == []

    def test_missing_metric_filtered(self, tmp_path):
        """Requested metrics not in CSV are silently dropped."""
        dir_a = tmp_path / "model_a"
        dir_a.mkdir()
        _write_csv(dir_a / "pixel_metrics.csv", [{"FOV": "A/0/0", "Timepoint": 0, "PCC": 0.9}])
        agg, available = load_and_aggregate(dir_a, ["PCC", "NonexistentMetric"])
        assert "PCC" in available
        assert "NonexistentMetric" not in available
