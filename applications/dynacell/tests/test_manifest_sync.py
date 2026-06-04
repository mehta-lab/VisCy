"""Verify ``dynacell._manifests`` stays in sync with dynacell-paper canonical.

VisCy ships its own bundled manifest registry under
``applications/dynacell/src/dynacell/_manifests/`` — the resolver discovers
this via the ``dynacell.manifest_roots`` entry point, so the registry
works on a fresh clone without ``DYNACELL_MANIFEST_ROOTS`` configured.

dynacell-paper is the source of truth for manifest *authoring*; this test
guards against drift between the two by parsing both copies and asserting
semantic equality. YAML-level differences (em-dash escaping etc.) parse to
the same Python object so they're tolerated; **content** changes are
flagged.

Skipped if ``DYNACELL_PAPER_PATH`` env var is not set or the resolved path
doesn't contain ``_configs/datasets/``. CI configures ``DYNACELL_PAPER_PATH``
when both repos are checked out side-by-side.
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


_VISCY_REGISTRY = Path(str(files("dynacell._manifests")))


def _canonical_root() -> Path | None:
    """Locate dynacell-paper's _configs/datasets/ via env var.

    Returns ``None`` (test will skip) if the env var is unset or the
    resolved directory doesn't contain a ``_configs/datasets/`` subtree.
    """
    raw = os.environ.get("DYNACELL_PAPER_PATH")
    if not raw:
        return None
    root = Path(raw) / "dynacell_paper" / "_configs" / "datasets"
    if not root.is_dir():
        # Try without the dynacell_paper/ subdir (different checkout layouts).
        root = Path(raw) / "_configs" / "datasets"
    if not root.is_dir():
        return None
    return root


def _viscy_to_canonical(viscy_name: str, canonical_root: Path) -> Path | None:
    """Map ``a549-mantis-2024_11_07`` → ``a549-mantis/2024_11_07/``.

    Canonical's a549 plates use a nested ``a549-mantis/<plate>/`` layout;
    aics-hipsc is flat. VisCy's bundled registry flattens both to
    dash-joined dataset names so the resolver doesn't need a tree walker.
    """
    if not viscy_name.startswith("a549-mantis-"):
        return canonical_root / viscy_name
    plate = viscy_name.removeprefix("a549-mantis-")
    return canonical_root / "a549-mantis" / plate


def _registered_datasets() -> list[str]:
    return sorted(p.name for p in _VISCY_REGISTRY.iterdir() if p.is_dir() and not p.name.startswith("_"))


@pytest.fixture(scope="module")
def canonical_root() -> Path:
    """dynacell-paper canonical registry root, skipping when unavailable."""
    root = _canonical_root()
    if root is None:
        pytest.skip("DYNACELL_PAPER_PATH not set or _configs/datasets/ missing")
    return root


@pytest.mark.parametrize("name", _registered_datasets())
def test_manifest_matches_canonical(canonical_root: Path, name: str):
    """VisCy registry's manifest YAML parses identically to canonical.

    VisCy may carry additive fields canonical lacks (e.g., ``gt_cache_dir``
    on aics-hipsc). The check is one-directional: every key in canonical
    must appear in VisCy with matching value; VisCy may have extras.
    """
    canonical_dir = _viscy_to_canonical(name, canonical_root)
    if canonical_dir is None or not canonical_dir.is_dir():
        pytest.skip(f"{name}: no canonical mapping (VisCy-only dataset)")

    viscy_yaml = _VISCY_REGISTRY / name / "manifest.yaml"
    canonical_yaml = canonical_dir / "manifest.yaml"
    assert canonical_yaml.is_file(), f"canonical manifest missing: {canonical_yaml}"

    viscy_doc = yaml.safe_load(viscy_yaml.read_text())
    canonical_doc = yaml.safe_load(canonical_yaml.read_text())

    _assert_subset(canonical_doc, viscy_doc, path=name)


@pytest.mark.parametrize("name", _registered_datasets())
def test_splits_match_canonical(canonical_root: Path, name: str):
    """Every splits/<gene>_*.yaml in VisCy parses identically to canonical."""
    canonical_dir = _viscy_to_canonical(name, canonical_root)
    if canonical_dir is None or not canonical_dir.is_dir():
        pytest.skip(f"{name}: no canonical mapping (VisCy-only dataset)")

    viscy_splits_dir = _VISCY_REGISTRY / name / "splits"
    canonical_splits_dir = canonical_dir / "splits"
    if not canonical_splits_dir.is_dir():
        pytest.skip(f"{name}: no canonical splits dir")
    assert viscy_splits_dir.is_dir(), f"VisCy missing splits/ for {name}"

    canonical_splits = {p.name for p in canonical_splits_dir.glob("*.yaml")}
    viscy_splits = {p.name for p in viscy_splits_dir.glob("*.yaml")}
    missing = canonical_splits - viscy_splits
    assert not missing, f"{name}: canonical splits missing in VisCy: {sorted(missing)}"

    for split_name in canonical_splits:
        viscy_doc = yaml.safe_load((viscy_splits_dir / split_name).read_text())
        canonical_doc = yaml.safe_load((canonical_splits_dir / split_name).read_text())
        assert viscy_doc == canonical_doc, f"{name}/{split_name} differs from canonical"


def _assert_subset(expected: object, actual: object, path: str):
    """Assert every key/value in expected appears in actual (recursively).

    Lists must match exactly. Dicts allow actual to have extra keys. Scalars
    must be equal.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: type mismatch (expected dict)"
        for key, exp_value in expected.items():
            assert key in actual, f"{path}: missing key {key!r}"
            _assert_subset(exp_value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert expected == actual, f"{path}: list differs"
    else:
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"
