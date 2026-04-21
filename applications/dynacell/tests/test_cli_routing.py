"""Tests for dynacell CLI subcommand routing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from dynacell.__main__ import (
    _HYDRA_COMMANDS,
    _inject_external_configs,
    main_cli,
)


class TestCliRouting:
    """Tests for the main_cli router."""

    def test_lightning_commands_delegate_to_viscy(self):
        """fit/predict/validate fall through to viscy_utils.cli.main."""
        with (
            patch("sys.argv", ["dynacell", "fit", "--help"]),
            patch("dynacell.__main__.importlib") as mock_importlib,
            patch("viscy_utils.cli.main") as mock_main,
        ):
            mock_main.side_effect = SystemExit(0)
            try:
                main_cli()
            except SystemExit:
                pass
            mock_main.assert_called_once()
            mock_importlib.import_module.assert_not_called()

    def test_evaluate_routes_to_hydra(self):
        """'evaluate' imports and calls the evaluation pipeline entry point."""
        mock_module = MagicMock()
        with (
            patch("sys.argv", ["dynacell", "evaluate", "--help"]),
            patch("importlib.import_module", return_value=mock_module) as mock_import,
        ):
            main_cli()
            mock_import.assert_called_once_with("dynacell.evaluation.pipeline")
            mock_module.evaluate_model.assert_called_once()

    def test_report_routes_to_hydra(self):
        """'report' imports and calls the reporting CLI entry point."""
        mock_module = MagicMock()
        with (
            patch("sys.argv", ["dynacell", "report", "--help"]),
            patch("importlib.import_module", return_value=mock_module) as mock_import,
        ):
            main_cli()
            mock_import.assert_called_once_with("dynacell.reporting.cli")
            mock_module.generate_report.assert_called_once()

    def test_precompute_gt_routes_to_hydra(self):
        """'precompute-gt' imports and calls the precompute CLI entry point."""
        mock_module = MagicMock()
        with (
            patch("sys.argv", ["dynacell", "precompute-gt", "--help"]),
            patch("importlib.import_module", return_value=mock_module) as mock_import,
        ):
            main_cli()
            mock_import.assert_called_once_with("dynacell.evaluation.precompute_cli")
            mock_module.precompute_gt.assert_called_once()

    def test_missing_deps_prints_install_hint(self, capsys):
        """ModuleNotFoundError gives a helpful install message."""
        with (
            patch("sys.argv", ["dynacell", "evaluate"]),
            patch(
                "importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'cubic'"),
            ),
        ):
            try:
                main_cli()
            except SystemExit as e:
                assert e.code == 1
            captured = capsys.readouterr()
            assert "dynacell[eval]" in captured.out

    def test_hydra_commands_dict_is_complete(self):
        """All Hydra commands have module path, function name, and extra."""
        for cmd, (mod, func, extra) in _HYDRA_COMMANDS.items():
            assert isinstance(cmd, str)
            assert "." in mod
            assert isinstance(func, str)
            assert isinstance(extra, str)


class TestInjectExternalConfigs:
    """Tests for the hydra.searchpath injection that exposes HPC-specific config
    instances living outside the Python package."""

    def test_appends_searchpath_when_external_dirs_present(self, tmp_path: Path):
        """When external configs dirs exist, inject a hydra.searchpath override
        encoding all roots as comma-separated file:// URIs in one token."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        with patch("dynacell.__main__._external_configs_dirs", return_value=[a, b]):
            argv = ["dynacell", "leaf=er/ipsc_confocal/celldiff/eval/ipsc_confocal"]
            result = _inject_external_configs(argv)
        assert result[:-1] == argv
        assert result[-1] == f"hydra.searchpath=[file://{a},file://{b}]"

    def test_noop_when_external_dirs_absent(self):
        """Wheel installs without the repo have no external dirs — argv stays unchanged."""
        with patch("dynacell.__main__._external_configs_dirs", return_value=[]):
            argv = ["dynacell", "target_name=er", "io.pred_path=/x", "save.save_dir=/y"]
            result = _inject_external_configs(argv)
        assert result == argv

    def test_appended_not_prepended(self, tmp_path: Path):
        """Injection goes at the end so Hydra's argparse doesn't misread it as a
        positional before diagnostic flags like ``-c job``."""
        with patch("dynacell.__main__._external_configs_dirs", return_value=[tmp_path]):
            result = _inject_external_configs(["dynacell", "-c", "job", "leaf=x"])
        assert result[0] == "dynacell"
        assert result[1:4] == ["-c", "job", "leaf=x"]
        assert result[4].startswith("hydra.searchpath=[file://")
