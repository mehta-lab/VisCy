"""Smoke tests for the viscy CLI entry point."""

import subprocess
import sys
from datetime import datetime

import pytest
from jsonargparse import Namespace

from viscy_utils.cli import _configure_wandb_logger


@pytest.fixture
def run_viscy():
    """Run the viscy CLI as a subprocess."""

    def _run(*args):
        return subprocess.run(
            [sys.executable, "-m", "viscy_utils.cli", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    return _run


def test_cli_help(run_viscy):
    result = run_viscy("--help")
    assert result.returncode == 0
    assert "fit" in result.stdout
    assert "predict" in result.stdout
    assert "validate" in result.stdout
    assert "test" in result.stdout


def test_cli_subcommands_registered(run_viscy):
    expected = [
        "fit",
        "validate",
        "test",
        "predict",
        "preprocess",
        "export",
        "precompute",
        "convert_to_anndata",
    ]
    result = run_viscy("--help")
    for cmd in expected:
        assert cmd in result.stdout, f"Subcommand '{cmd}' not found in CLI help"


def test_cli_fit_help(run_viscy):
    result = run_viscy("fit", "--help")
    assert result.returncode == 0
    assert "model" in result.stdout
    assert "trainer" in result.stdout


def test_cli_predict_help(run_viscy):
    result = run_viscy("predict", "--help")
    assert result.returncode == 0
    assert "model" in result.stdout
    assert "ckpt_path" in result.stdout


def _make_wandb_fit_config(name: str = "FNet3D_iPSC_SEC61B") -> Namespace:
    return Namespace(
        fit=Namespace(
            trainer=Namespace(
                logger=Namespace(
                    class_path="lightning.pytorch.loggers.WandbLogger",
                    init_args=Namespace(project="dynacell", name=name),
                )
            )
        )
    )


def test_configure_wandb_logger_stamps_name_and_defaults_group(monkeypatch):
    monkeypatch.delenv("VISCY_WANDB_GROUP", raising=False)
    monkeypatch.delenv("VISCY_WANDB_LAUNCH", raising=False)
    config = _make_wandb_fit_config()

    _configure_wandb_logger(config, "fit", now=datetime(2026, 4, 1, 14, 30, 45))

    init_args = config["fit"]["trainer"]["logger"]["init_args"]
    assert init_args["name"] == "20260401-143045_FNet3D_iPSC_SEC61B"
    assert init_args["group"] == "FNet3D_iPSC_SEC61B"
    assert init_args["job_type"] == "fit"


def test_configure_wandb_logger_prefers_launch_group(monkeypatch):
    monkeypatch.delenv("VISCY_WANDB_GROUP", raising=False)
    monkeypatch.setenv("VISCY_WANDB_LAUNCH", "20260401-augfix-r1")
    config = _make_wandb_fit_config()

    _configure_wandb_logger(config, "fit", now=datetime(2026, 4, 1, 14, 30, 45))

    init_args = config["fit"]["trainer"]["logger"]["init_args"]
    assert init_args["group"] == "20260401-augfix-r1"


def test_configure_wandb_logger_prefers_explicit_group_override(monkeypatch):
    monkeypatch.setenv("VISCY_WANDB_GROUP", "sec61b-restart-r2")
    monkeypatch.setenv("VISCY_WANDB_LAUNCH", "20260401-augfix-r1")
    config = _make_wandb_fit_config()

    _configure_wandb_logger(config, "fit", now=datetime(2026, 4, 1, 14, 30, 45))

    init_args = config["fit"]["trainer"]["logger"]["init_args"]
    assert init_args["group"] == "sec61b-restart-r2"


def test_configure_wandb_logger_does_not_double_prefix(monkeypatch):
    monkeypatch.delenv("VISCY_WANDB_GROUP", raising=False)
    monkeypatch.delenv("VISCY_WANDB_LAUNCH", raising=False)
    config = _make_wandb_fit_config(name="20260401-143045_FNet3D_iPSC_SEC61B")

    _configure_wandb_logger(config, "fit", now=datetime(2026, 4, 1, 15, 0, 0))

    init_args = config["fit"]["trainer"]["logger"]["init_args"]
    assert init_args["name"] == "20260401-143045_FNet3D_iPSC_SEC61B"
