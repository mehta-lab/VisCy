"""Tests for dynacell CLI subcommand routing."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dynacell.__main__ import (
    _HYDRA_COMMANDS,
    _inject_external_configs,
    _maybe_set_shared_hf_cache,
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
            argv = ["dynacell", "leaf=er/celldiff/ipsc_confocal/eval__ipsc_confocal"]
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


class TestMaybeSetSharedHfCache:
    """Tests for the HF_HUB_CACHE auto-setter that points repo-checkout jobs at
    the team-shared Hugging Face cache so gated models (DINOv3) download
    once per team instead of once per user."""

    def test_user_set_wins(self, tmp_path: Path, monkeypatch):
        """An existing HF_HUB_CACHE in env takes precedence."""
        monkeypatch.setenv("HF_HUB_CACHE", "/user/chose/this")
        with (
            patch("dynacell.__main__._external_configs_dirs", return_value=[tmp_path]),
            patch("dynacell.__main__._SHARED_HF_CACHE", tmp_path),
        ):
            _maybe_set_shared_hf_cache()
        assert os.environ["HF_HUB_CACHE"] == "/user/chose/this"

    def test_noop_in_wheel_install(self, tmp_path: Path, monkeypatch):
        """Wheel installs (no external config dirs) don't set HF_HUB_CACHE."""
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        with (
            patch("dynacell.__main__._external_configs_dirs", return_value=[]),
            patch("dynacell.__main__._SHARED_HF_CACHE", tmp_path),
        ):
            _maybe_set_shared_hf_cache()
        assert "HF_HUB_CACHE" not in os.environ

    def test_noop_when_shared_dir_missing(self, tmp_path: Path, monkeypatch):
        """If the shared cache dir doesn't exist on this machine, skip."""
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        missing = tmp_path / "does_not_exist"
        with (
            patch("dynacell.__main__._external_configs_dirs", return_value=[tmp_path]),
            patch("dynacell.__main__._SHARED_HF_CACHE", missing),
        ):
            _maybe_set_shared_hf_cache()
        assert "HF_HUB_CACHE" not in os.environ

    def test_sets_on_repo_checkout_when_dir_exists(self, tmp_path: Path, monkeypatch):
        """Repo checkout + shared dir present + user hasn't set HF_HUB_CACHE ⇒ set it."""
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        with (
            patch("dynacell.__main__._external_configs_dirs", return_value=[tmp_path]),
            patch("dynacell.__main__._SHARED_HF_CACHE", tmp_path),
        ):
            _maybe_set_shared_hf_cache()
        assert os.environ["HF_HUB_CACHE"] == str(tmp_path)


class TestResolverThreading:
    """main_cli() → viscy_utils.cli.main(resolver=...) → _maybe_compose_config(resolver=...)."""

    def test_resolver_threaded_to_maybe_compose(self, monkeypatch, tmp_path):
        """Full wiring: dynacell main_cli passes the ref resolver to viscy_utils.cli.main."""
        import sys

        import yaml as _yaml

        from dynacell._compose_hook import _dynacell_ref_resolver

        fixture_root = Path(__file__).resolve().parent / "fixtures" / "manifests"
        monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(fixture_root))
        repo_root = Path(__file__).resolve().parents[3]
        leaf = (
            repo_root / "applications/dynacell/configs/benchmarks/virtual_staining/er/celldiff/ipsc_confocal/train.yml"
        )
        monkeypatch.setattr(sys, "argv", ["dynacell", "fit", "-c", str(leaf)])

        captured: dict = {}

        def fake_main(*, resolver=None):
            # Confirm the dynacell resolver was injected, not None.
            captured["resolver"] = resolver
            # Reproduce the viscy_utils composition step so the temp file is written.
            from viscy_utils.cli import _maybe_compose_config

            _maybe_compose_config(resolver=resolver)
            # sys.argv[-1] now points at the rewritten temp YAML.
            captured["temp_path"] = sys.argv[-1]

        monkeypatch.setattr("viscy_utils.cli.main", fake_main)
        main_cli()

        assert captured["resolver"] is _dynacell_ref_resolver
        composed = _yaml.safe_load(Path(captured["temp_path"]).read_text())
        ia = composed["data"]["init_args"]
        assert ia["data_path"].endswith("train/SEC61B.zarr")
        assert ia["source_channel"] == "Phase3D"
        assert ia["target_channel"] == "Structure"


@pytest.fixture
def _clear_hydra():
    """Reset Hydra's GlobalHydra singleton around each test.

    Hydra's @hydra.main decorator registers per-invocation state in
    a process-wide singleton; without this, consecutive tests that
    invoke main_cli() see stale state and throw
    ``ValueError: GlobalHydra is already initialized``.
    """
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


class TestHydraResolverErrorCatch:
    """main_cli() catches dataset-resolver errors raised inside Hydra entry
    points and converts them to a clean ``SystemExit(2)`` with a message on
    stderr (no traceback). These tests drive the Hydra branch end-to-end so
    any future refactor that accidentally breaks the catch (e.g. removes
    ``HYDRA_FULL_ERROR=1`` or the try/except) gets caught.
    """

    _FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "manifests"

    @staticmethod
    def _base_overrides(tmp_path: Path) -> list[str]:
        """Minimal overrides so Hydra can compose ``eval.yaml`` to a leaf where
        the ``apply_dataset_ref`` hook fires."""
        return [
            "target=er_sec61b",
            "predict_set=ipsc_confocal",
            "io.pred_path=/tmp/fake",
            f"save.save_dir={tmp_path}",
            "hydra.run.dir=.",
            "hydra.output_subdir=null",
        ]

    def test_evaluate_no_manifest_roots_exits_cleanly(self, monkeypatch, capsys, tmp_path, _clear_hydra):
        """``dynacell evaluate`` with no roots configured ⇒ exit 2, clean stderr."""
        monkeypatch.delenv("DYNACELL_MANIFEST_ROOTS", raising=False)
        # Stub entry-point-registered roots so this test is independent
        # of whichever providers happen to be installed in the dev env.
        monkeypatch.setattr("dynacell.data.resolver._entry_point_roots", lambda: [])
        monkeypatch.setattr(
            "sys.argv",
            ["dynacell", "evaluate", *self._base_overrides(tmp_path)],
        )
        with pytest.raises(SystemExit) as exc:
            main_cli()
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "No dynacell manifest roots configured" in captured.err
        assert "DYNACELL_MANIFEST_ROOTS" in captured.err
        assert "Traceback" not in captured.err

    def test_evaluate_manifest_not_found_exits_cleanly(self, monkeypatch, capsys, tmp_path, _clear_hydra):
        """Unknown ``dataset_ref.dataset`` slug ⇒ exit 2, stderr lists searched paths."""
        monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(self._FIXTURE_ROOT))
        monkeypatch.setattr(
            "sys.argv",
            [
                "dynacell",
                "evaluate",
                *self._base_overrides(tmp_path),
                "benchmark.dataset_ref.dataset=nonexistent",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            main_cli()
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "dataset 'nonexistent' not found" in captured.err
        assert "Searched:" in captured.err
        assert "nonexistent/manifest.yaml" in captured.err
        assert "Traceback" not in captured.err

    def test_evaluate_target_not_found_exits_cleanly(self, monkeypatch, capsys, tmp_path, _clear_hydra):
        """Unknown ``dataset_ref.target`` slug ⇒ exit 2, stderr lists available targets."""
        monkeypatch.setenv("DYNACELL_MANIFEST_ROOTS", str(self._FIXTURE_ROOT))
        monkeypatch.setattr(
            "sys.argv",
            [
                "dynacell",
                "evaluate",
                *self._base_overrides(tmp_path),
                "benchmark.dataset_ref.target=bogus_target",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            main_cli()
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "target 'bogus_target' not found in dataset 'aics-hipsc'" in captured.err
        assert "Available targets:" in captured.err
        assert "Traceback" not in captured.err

    def test_precompute_gt_no_manifest_roots_exits_cleanly(self, monkeypatch, capsys, tmp_path, _clear_hydra):
        """Symmetry check: same catch fires on ``dynacell precompute-gt``."""
        monkeypatch.delenv("DYNACELL_MANIFEST_ROOTS", raising=False)
        monkeypatch.setattr("dynacell.data.resolver._entry_point_roots", lambda: [])
        # precompute-gt requires gt_cache_dir instead of save_dir; the
        # resolver error fires before either is validated, but Hydra still
        # demands required fields be present for struct-mode composition.
        monkeypatch.setattr(
            "sys.argv",
            [
                "dynacell",
                "precompute-gt",
                "target=er_sec61b",
                "predict_set=ipsc_confocal",
                "io.pred_path=/tmp/fake",
                f"save.save_dir={tmp_path}",
                f"io.gt_cache_dir={tmp_path}/cache",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            main_cli()
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "No dynacell manifest roots configured" in captured.err
        assert "DYNACELL_MANIFEST_ROOTS" in captured.err
        assert "Traceback" not in captured.err
