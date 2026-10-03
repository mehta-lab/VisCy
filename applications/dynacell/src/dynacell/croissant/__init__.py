"""Croissant 1.1 RAI metadata generation for DynaCell release trees.

Builds a single JSON-LD descriptor per release dataset by scanning the
packed OZX archives directly. The generated JSON-LD validates against
the ML Commons Croissant 1.1 schema and the RAI extension fields
required by NeurIPS Evaluations & Datasets.
"""

from dynacell.croissant.builder import (
    build_croissant_from_release,
    merge_croissant_docs,
)
from dynacell.croissant.static import StaticFields

# ``validate_croissant`` is deliberately NOT re-exported here: its module imports
# mlcroissant at top level, so an eager re-export makes the optional ``croissant``
# extra a hard requirement for ``builder`` too -- defeating cli.py's lazy imports
# and the --no-validate flag. Import it from dynacell.croissant.validate directly.
__all__ = [
    "StaticFields",
    "build_croissant_from_release",
    "merge_croissant_docs",
]
