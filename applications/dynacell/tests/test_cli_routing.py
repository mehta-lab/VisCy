"""Tests for dynacell CLI subcommand routing."""

from unittest.mock import MagicMock, patch

from dynacell.__main__ import _HYDRA_COMMANDS, main_cli


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
