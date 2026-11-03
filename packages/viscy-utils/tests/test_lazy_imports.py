import subprocess
import sys

import pytest

import viscy_utils
import viscy_utils.callbacks

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


def test_health_callbacks_import_without_the_eval_extra():
    """The callbacks VisCyCLI injects into every fit must import without scikit-learn.

    ``sys.modules["sklearn"] = None`` makes every sklearn import fail, as on a base
    install; OnlineEvalCallback imports it at module level.
    """
    code = (
        "import sys; sys.modules['sklearn'] = None; "
        "from viscy_utils.callbacks import MonitorHealthCheck, OptimizerHealthCheck"
    )
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)


@pytest.mark.parametrize("name", viscy_utils.__all__)
def test_public_names_importable(name):
    exec(f"from viscy_utils import {name}", {})


@pytest.mark.parametrize("name", viscy_utils.callbacks.__all__)
def test_public_callbacks_importable(name):
    exec(f"from viscy_utils.callbacks import {name}", {})


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        viscy_utils.not_a_name  # noqa: B018


def test_dir_lists_public_names():
    assert set(viscy_utils.__all__) <= set(dir(viscy_utils))
