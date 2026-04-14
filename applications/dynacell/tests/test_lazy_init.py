"""Tests that dynacell subpackages can be imported without loading engine."""

import importlib
import sys


def test_data_import_does_not_load_engine():
    """Importing dynacell.data should not force dynacell.engine into sys.modules."""
    # Remove cached modules so we get a fresh import
    mods_to_clear = [k for k in sys.modules if k.startswith("dynacell")]
    for mod in mods_to_clear:
        sys.modules.pop(mod, None)

    importlib.import_module("dynacell.data")

    assert "dynacell.engine" not in sys.modules

    # Restore dynacell modules for subsequent tests
    mods_to_clear = [k for k in sys.modules if k.startswith("dynacell")]
    for mod in mods_to_clear:
        sys.modules.pop(mod, None)


def test_lazy_export_still_works():
    """from dynacell import DynacellUNet should still work via __getattr__."""
    from dynacell import DynacellFlowMatching, DynacellUNet

    assert DynacellUNet is not None
    assert DynacellFlowMatching is not None
