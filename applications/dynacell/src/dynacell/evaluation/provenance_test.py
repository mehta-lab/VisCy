"""Tests for the eval numeric-provenance contract."""

import json
import re
from importlib.metadata import version
from pathlib import Path

import pytest

from dynacell.evaluation.provenance import (
    CUBIC_VERSIONS_EQUIVALENT_TO_REQUIRED,
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
    write_metrics_provenance(tmp_path, cp_reference_sha256="ref-abc", cp_space_sha256="abc")
    payload = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())
    assert payload["versions"]["cubic"] == version("cubic")
    assert payload["cp_reference_sha256"] == "ref-abc"
    assert payload["cp_space_sha256"] == "abc"
    assert metrics_provenance_matches(tmp_path, cp_space_sha256="abc")


def test_feature_less_stamp_matches_only_a_feature_less_run(tmp_path):
    """A run without feature metrics stamps ``None``; a CP-scoring run must not reuse it."""
    write_metrics_provenance(tmp_path, cp_reference_sha256=None, cp_space_sha256=None)
    assert metrics_provenance_matches(tmp_path, cp_space_sha256=None)
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256="abc")


def test_feature_less_run_ignores_the_cp_key(tmp_path):
    """compute_feature_metrics=false reuses no CP value, so a missing or foreign hash is irrelevant."""
    (tmp_path / PROVENANCE_FILENAME).write_text(json.dumps({"versions": {"cubic": version("cubic")}}))
    assert metrics_provenance_matches(tmp_path, cp_space_sha256=None)
    write_metrics_provenance(tmp_path, cp_reference_sha256="ref-abc", cp_space_sha256="abc")
    assert metrics_provenance_matches(tmp_path, cp_space_sha256=None)


def test_other_cp_reference_is_not_a_match(tmp_path):
    """CP values scored in another reference are not reusable after a rebuild."""
    write_metrics_provenance(tmp_path, cp_reference_sha256="ref-old", cp_space_sha256="old")
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256="new")


def test_stamp_predating_cp_reference_is_not_a_match(tmp_path):
    """A sidecar written before the CP reference existed carries no hash; a CP-scoring run refuses it."""
    (tmp_path / PROVENANCE_FILENAME).write_text(json.dumps({"versions": {"cubic": version("cubic")}}))
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256="abc")


def test_missing_sidecar_is_not_a_match(tmp_path):
    """An unstamped save_dir must not read as compatible.

    Every cache written before the stamp existed is exactly the ambiguous
    0.8.0a2-or-0.9.0a1 case that has to be recomputed.
    """
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256=None)


def _stamp_cubic(save_dir, cubic, *, cp_space_sha256):
    """Write a sidecar as if a run under ``cubic`` had produced it."""
    cp_reference_sha256 = None if cp_space_sha256 is None else f"ref-{cp_space_sha256}"
    write_metrics_provenance(save_dir, cp_reference_sha256=cp_reference_sha256, cp_space_sha256=cp_space_sha256)
    path = save_dir / PROVENANCE_FILENAME
    payload = json.loads(path.read_text())
    payload["versions"]["cubic"] = cubic
    path.write_text(json.dumps(payload))


def test_foreign_cubic_version_is_not_a_match(tmp_path):
    _stamp_cubic(tmp_path, "0.8.0a2", cp_space_sha256=None)
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256=None)


@pytest.fixture
def running_the_declared_pin(monkeypatch):
    """Report the declared cubic as installed, whatever the test venv holds."""
    monkeypatch.setattr("dynacell.evaluation.provenance.version", lambda _name: REQUIRED_CUBIC_VERSION)


def test_equivalent_versions_exclude_the_declared_one():
    assert "0.9.0a1" in CUBIC_VERSIONS_EQUIVALENT_TO_REQUIRED
    assert REQUIRED_CUBIC_VERSION not in CUBIC_VERSIONS_EQUIVALENT_TO_REQUIRED


def test_equivalent_cubic_version_is_a_match(tmp_path, running_the_declared_pin):
    """A 0.9.0a1 cache holds the values the declared pin reproduces, so it is reused."""
    _stamp_cubic(tmp_path, "0.9.0a1", cp_space_sha256=None)
    assert metrics_provenance_matches(tmp_path, cp_space_sha256=None)


def test_older_cubic_version_is_still_not_a_match(tmp_path, running_the_declared_pin):
    """0.8.0a2 moved FSC/FRC/Spectral_PCC, so its caches stay refused."""
    _stamp_cubic(tmp_path, "0.8.0a2", cp_space_sha256=None)
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256=None)


def test_equivalent_cubic_version_keeps_the_cp_rules(tmp_path, running_the_declared_pin):
    """Accepting the cubic stamp does not relax the CP-space binding check."""
    _stamp_cubic(tmp_path, "0.9.0a1", cp_space_sha256="abc")
    assert metrics_provenance_matches(tmp_path, cp_space_sha256="abc")
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256="other")
    _stamp_cubic(tmp_path, "0.9.0a1", cp_space_sha256=None)
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256="abc")


def test_equivalence_only_applies_under_the_declared_pin(tmp_path, monkeypatch):
    """An environment off the pin gets no equivalence: only its own version matches."""
    monkeypatch.setattr("dynacell.evaluation.provenance.version", lambda _name: "0.8.0a2")
    _stamp_cubic(tmp_path, "0.9.0a1", cp_space_sha256=None)
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256=None)


def test_check_cubic_pin_accepts_the_declared_version():
    """The environment running the tests must satisfy the declared pin."""
    if version("cubic") != REQUIRED_CUBIC_VERSION:
        pytest.skip(f"environment holds cubic {version('cubic')}, not the declared pin")
    check_cubic_pin()


def test_check_cubic_pin_rejects_a_mismatch(monkeypatch):
    monkeypatch.setattr("dynacell.evaluation.provenance.version", lambda _name: "0.8.0a2")
    with pytest.raises(RuntimeError, match="not comparable across"):
        check_cubic_pin()


def test_stamp_without_a_cp_space_binding_is_not_a_match(tmp_path):
    """A stamp carrying only the whole-reference hash (no dataset binding) is refused by a CP-scoring run."""
    payload = {"versions": {"cubic": version("cubic")}, "cp_reference_sha256": "abc"}
    (tmp_path / PROVENANCE_FILENAME).write_text(json.dumps(payload))
    assert not metrics_provenance_matches(tmp_path, cp_space_sha256="abc")


def test_half_a_cp_stamp_is_refused(tmp_path):
    with pytest.raises(ValueError, match="both be given or both be None"):
        write_metrics_provenance(tmp_path, cp_reference_sha256="abc", cp_space_sha256=None)
