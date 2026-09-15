"""Tests for the eval numeric-provenance contract."""

import json
import re
from importlib.metadata import version
from pathlib import Path

import pytest

from dynacell.evaluation.provenance import (
    PROVENANCE_FILENAME,
    REQUIRED_CUBIC_VERSION,
    check_cubic_pin,
    installed_versions,
    metrics_provenance_matches,
    write_metrics_provenance,
)

_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def test_required_version_matches_every_pyproject_pin():
    """The declared constant must equal every ``cubic`` pin in pyproject.

    Two extras pin cubic independently. If either drifts from
    ``REQUIRED_CUBIC_VERSION``, ``check_cubic_pin`` would reject the very
    environment ``uv sync`` builds, so they have to move together.
    """
    pins = re.findall(r"cubic @ git\+[^@]+@v([^\"']+)", _PYPROJECT.read_text())
    assert pins, "no cubic git pin found in pyproject.toml"
    assert set(pins) == {REQUIRED_CUBIC_VERSION}, (
        f"pyproject pins cubic at {sorted(set(pins))} but REQUIRED_CUBIC_VERSION is {REQUIRED_CUBIC_VERSION!r}"
    )


def test_installed_versions_records_cubic():
    assert installed_versions()["cubic"] == version("cubic")


def test_roundtrip_matches_running_environment(tmp_path):
    write_metrics_provenance(tmp_path)
    payload = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())
    assert payload["versions"]["cubic"] == version("cubic")
    assert metrics_provenance_matches(tmp_path)


def test_missing_sidecar_is_not_a_match(tmp_path):
    """An unstamped save_dir must not read as compatible.

    Every cache written before the stamp existed is exactly the ambiguous
    0.8.0a2-or-0.9.0a1 case that has to be recomputed.
    """
    assert not metrics_provenance_matches(tmp_path)


def test_foreign_cubic_version_is_not_a_match(tmp_path):
    write_metrics_provenance(tmp_path)
    path = tmp_path / PROVENANCE_FILENAME
    payload = json.loads(path.read_text())
    payload["versions"]["cubic"] = "0.8.0a2"
    path.write_text(json.dumps(payload))
    assert not metrics_provenance_matches(tmp_path)


def test_check_cubic_pin_accepts_the_declared_version():
    """The environment running the tests must satisfy the declared pin."""
    if version("cubic") != REQUIRED_CUBIC_VERSION:
        pytest.skip(f"environment holds cubic {version('cubic')}, not the declared pin")
    check_cubic_pin()


def test_check_cubic_pin_rejects_a_mismatch(monkeypatch):
    monkeypatch.setattr("dynacell.evaluation.provenance.version", lambda _name: "0.8.0a2")
    with pytest.raises(RuntimeError, match="not comparable across"):
        check_cubic_pin()
