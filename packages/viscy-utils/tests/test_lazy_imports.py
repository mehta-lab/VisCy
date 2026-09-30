import subprocess
import sys

import pytest

import viscy_utils

# Submodules that evaluation code imports without training anything.
LIGHT_SUBMODULES = [
    "viscy_utils.compose",
    "viscy_utils.mp_utils",
    "viscy_utils.normalize",
    "viscy_utils.prediction_metadata",
]


@pytest.mark.parametrize("module", LIGHT_SUBMODULES)
def test_light_submodule_skips_torch_and_lightning(module):
    """Importing a light submodule must not load torch or lightning via the package."""
    code = f"import sys, {module}; print(sorted(m for m in ('torch', 'lightning') if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("name", viscy_utils.__all__)
def test_public_names_importable(name):
    assert getattr(viscy_utils, name) is not None
    exec(f"from viscy_utils import {name}", {})


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        viscy_utils.not_a_name  # noqa: B018
