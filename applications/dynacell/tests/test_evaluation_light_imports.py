"""Light ``dynacell.evaluation`` modules must import without loading torch.

The paper generators and table scripts import these modules only for cache
readers, feature names and focus helpers; torch costs seconds per process there.
"""

import subprocess
import sys

LIGHT_MODULES = [
    "dynacell.evaluation.cp_reference",
    "dynacell.evaluation.cross_condition_probe",
    "dynacell.evaluation.feature_select",
    "dynacell.evaluation.focus",
    "dynacell.evaluation.instance_metrics",
    "dynacell.evaluation.linear_probe",
    "dynacell.evaluation.metrics",
    "dynacell.evaluation.paths",
]

# One interpreter for all modules: shared heavy deps (iohub, cubic, zarr) load once.
# Checking after each import names the first module that pulls torch in.
_PROBE = """
import importlib, sys
for name in sys.argv[1:]:
    importlib.import_module(name)
    if "torch" in sys.modules:
        print(name)
        break
"""


def test_light_evaluation_modules_do_not_import_torch() -> None:
    """Importing the light modules in a fresh interpreter leaves torch unloaded."""
    result = subprocess.run([sys.executable, "-c", _PROBE, *LIGHT_MODULES], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "", f"{result.stdout.strip()} imports torch at module load"
