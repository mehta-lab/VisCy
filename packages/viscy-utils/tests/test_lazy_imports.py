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


def test_light_submodules_skip_torch_and_lightning():
    """Importing light submodules must not load torch or lightning via the package."""
    code = (
        f"import sys, {', '.join(LIGHT_SUBMODULES)}; "
        "print(sorted(m for m in ('torch', 'lightning') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("name", viscy_utils.__all__)
def test_public_names_importable(name):
    exec(f"from viscy_utils import {name}", {})


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        viscy_utils.not_a_name  # noqa: B018


def test_dir_lists_public_names():
    assert set(viscy_utils.__all__) <= set(dir(viscy_utils))
