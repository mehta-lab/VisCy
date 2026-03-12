"""Smoke tests for the viscy CLI entry point."""

import subprocess
import sys

import pytest


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
