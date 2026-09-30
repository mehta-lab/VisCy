"""Light ``dynacell.evaluation`` modules must import without loading torch.

The paper generators and table scripts import these modules only for cache
readers, feature names and focus helpers; torch costs seconds per process there.
"""

import subprocess
import sys

import pytest

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


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_light_evaluation_module_does_not_import_torch(module: str) -> None:
    """Importing ``module`` in a fresh interpreter leaves torch unloaded."""
    code = f"import sys, {module}; print('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False", f"{module} imports torch at module load"
