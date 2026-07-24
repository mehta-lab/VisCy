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
from dynacell.croissant.validate import validate_croissant

__all__ = [
    "StaticFields",
    "build_croissant_from_release",
    "merge_croissant_docs",
    "validate_croissant",
]
