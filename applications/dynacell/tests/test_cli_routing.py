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
    """Tests for ``_inject_external_configs``."""

    def test_injects_searchpath_when_external_dirs_present(self, tmp_path: Path):
        """When external configs dirs exist, inject a hydra.searchpath override
        encoding all roots as comma-separated file:// URIs in one token."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        with patch("dynacell.__main__._external_configs_dirs", return_value=[a, b]):
            argv = ["dynacell", "leaf=er/ipsc_confocal/celldiff/eval/ipsc_confocal"]
            result = _inject_external_configs(argv)
        expected_token = f"hydra.searchpath=[file://{a},file://{b}]"
        assert expected_token in result
        assert len(result) == len(argv) + 1

    def test_noop_when_external_dirs_absent(self):
        """Wheel installs without the repo have no external dirs — argv stays unchanged."""
        with patch("dynacell.__main__._external_configs_dirs", return_value=[]):
            argv = ["dynacell", "target_name=er", "io.pred_path=/x", "save.save_dir=/y"]
            result = _inject_external_configs(argv)
        assert result == argv

    def test_inserts_adjacent_to_positional_when_flag_leads(self, tmp_path: Path):
        """Flag-first layout: token inserts among positionals so argparse sees
        all ``overrides`` contiguous."""
        with patch("dynacell.__main__._external_configs_dirs", return_value=[tmp_path]):
            result = _inject_external_configs(["dynacell", "-c", "job", "leaf=x"])
        # Token must land next to `leaf=x` (the only existing positional),
        # not after `-c job` which would scatter positionals across a flag.
        assert result[0] == "dynacell"
        assert result[1:3] == ["-c", "job"]
        assert result[3].startswith("hydra.searchpath=[file://")
        assert result[4] == "leaf=x"

    def test_inserts_adjacent_to_positional_when_flag_trails(self, tmp_path: Path):
        """Flag-trailing layout: token inserts before the first positional so
        argparse's ``overrides`` nargs="*" collects everything in one run."""
        with patch("dynacell.__main__._external_configs_dirs", return_value=[tmp_path]):
            result = _inject_external_configs(["dynacell", "leaf=x", "-c", "job"])
        assert result[0] == "dynacell"
        assert result[1].startswith("hydra.searchpath=[file://")
        assert result[2:] == ["leaf=x", "-c", "job"]

    def test_appends_when_no_positional_overrides(self, tmp_path: Path):
        """With only flags (e.g. ``--help``), append at the end."""
        with patch("dynacell.__main__._external_configs_dirs", return_value=[tmp_path]):
            result = _inject_external_configs(["dynacell", "--help"])
        assert result[:2] == ["dynacell", "--help"]
        assert result[2].startswith("hydra.searchpath=[file://")
