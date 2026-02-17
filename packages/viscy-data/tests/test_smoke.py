"""Smoke tests for viscy_data package import and public API surface.

Testing strategy:
    1. Verify the base package imports without error.
    2. Verify every name in ``__all__`` is accessible via ``getattr``.
    3. Pin the ``__all__`` count at 45 to detect accidental additions/removals.
    4. Verify optional-dep modules contain ``pip install`` error-message hints
       (checked via ``inspect.getsource`` so tests pass regardless of whether
       the optional deps are installed).
    5. Verify importing ``viscy_data`` does not pull in the old ``viscy.data``
       namespace.
"""

from __future__ import annotations

import importlib
import inspect
import sys

import pytest

import viscy_data

# ---------------------------------------------------------------------------
# Test 1: Basic import
# ---------------------------------------------------------------------------


def test_import_viscy_data():
    """Importing viscy_data succeeds and the module exposes __all__."""
    assert hasattr(viscy_data, "__all__"), "viscy_data should have __all__"


# ---------------------------------------------------------------------------
# Test 2: Every name in __all__ is importable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", viscy_data.__all__)
def test_all_exports_importable(name: str):
    """Each name in viscy_data.__all__ is accessible via getattr."""
    obj = getattr(viscy_data, name, None)
    assert obj is not None, f"viscy_data.__all__ advertises '{name}' but getattr returned None"


# ---------------------------------------------------------------------------
# Test 3: __all__ count pinned at 45
# ---------------------------------------------------------------------------


def test_all_count():
    """__all__ contains exactly 45 names (detect accidental add/remove)."""
    assert len(viscy_data.__all__) == 45, f"Expected 45 names in __all__, got {len(viscy_data.__all__)}"


# ---------------------------------------------------------------------------
# Test 4: Optional-dep error messages contain pip install hints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name,expected_pattern",
    [
        ("viscy_data.triplet", "pip install 'viscy-data[triplet]'"),
        ("viscy_data.mmap_cache", "pip install 'viscy-data[mmap]'"),
        ("viscy_data.livecell", "pip install 'viscy-data[livecell]'"),
        (
            "viscy_data.cell_classification",
            "pip install pandas",
        ),
    ],
    ids=["triplet", "mmap_cache", "livecell", "cell_classification"],
)
def test_optional_dep_error_messages(module_name: str, expected_pattern: str):
    """Optional-dep modules contain pip install instructions in source.

    Since the optional dependencies may already be installed in the test
    environment, we cannot trigger the ImportError guards directly. Instead
    we inspect the module source code to confirm the error-message patterns
    are present.
    """
    mod = importlib.import_module(module_name)
    src = inspect.getsource(mod)
    assert expected_pattern in src, f"Module {module_name} does not contain expected install hint: {expected_pattern!r}"


# ---------------------------------------------------------------------------
# Test 5: viscy_data does not depend on old viscy.data namespace
# ---------------------------------------------------------------------------


def test_no_viscy_dependency():
    """Importing viscy_data must not pull in the old viscy.data namespace.

    If ``viscy`` happens to be installed alongside ``viscy_data``, importing
    ``viscy_data`` should still not trigger ``viscy.data`` imports -- the
    packages are independent.
    """
    # viscy_data is already imported at module level; check sys.modules.
    assert "viscy.data" not in sys.modules, "viscy_data should not import from the legacy viscy.data namespace"
