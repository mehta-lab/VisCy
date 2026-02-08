# Coding Conventions

**Analysis Date:** 2026-02-07

## Naming Patterns

**Files:**
- Module files use snake_case with leading underscore for private modules: `_flip.py`, `_crop.py`, `_normalize.py`
- Public API modules exported in `__init__.py`
- Test files prefixed with `test_`: `test_flip.py`, `test_crop.py`
- Example: `_typing.py` for type definitions, `_monai_wrappers.py` for wrapper implementations

**Functions:**
- Transform classes use PascalCase: `BatchedRandFlip`, `BatchedRandScaleIntensity`, `NormalizeSampled`
- Dictionary transform variants append `d` suffix: `BatchedRandFlipd`, `BatchedRandScaleIntensityd`
- Private methods use leading underscore: `_match_image()`, `_normalize()`
- Static methods for utility functions: `_match_image` as `@staticmethod`

**Variables:**
- Instance variables use snake_case: `self.spatial_axes`, `self.gamma_range`, `self.remove_meta`
- Boolean flags use `is_`, `has_`, or `allow_` prefixes: `allow_missing_keys`, `randomize`
- Tensor variables use descriptive names: `img`, `data`, `sample`, `out`, `batch_data`

**Types:**
- Use `Tensor` (from `torch`) for PyTorch tensor type hints
- Use `TypedDict` from `typing` for dictionary structure definitions (e.g., `Sample`, `NormMeta`)
- Use `Sequence`, `Iterable` from `typing_extensions` for collections
- Use `Literal` from `typing_extensions` for constrained string enums: `Literal["fov_statistics", "dataset_statistics"]`
- Use `OneOrSeq[T]` pattern for flexible input: `OneOrSeq[Tensor]` accepts single tensor or sequence

## Code Style

**Formatting:**
- Ruff (linter + formatter) for all Python code
- Line length: 120 characters
- Indent: 4 spaces
- Quote style: double quotes (`"string"` not `'string'`)
- Skip magic trailing comma: false (multiline collections get trailing commas)
- Format docstrings with code examples

**Linting:**
- Ruff selected rules: `D`, `E`, `F`, `I`, `NPY`, `PD`, `W`
  - `D`: pydocstring (NumPy convention)
  - `E`: pycodestyle errors
  - `F`: Pyflakes
  - `I`: isort imports
  - `NPY`: NumPy-specific issues
  - `PD`: pandas-specific issues
  - `W`: pycodestyle warnings
- Per-file ignores:
  - Notebooks (`**/*.ipynb`): ignore `D` (docstring requirement)
  - Package inits (`**/__init__.py`): ignore `D104` (module docstring), `F401` (unused imports)
  - Docs (`**/docs/**`): ignore `I` (import sorting)
  - Tests (`**/tests/**`): ignore `D` (docstring requirement)
- NumPy docstring convention enforced

## Import Organization

**Order:**
1. Standard library imports (`torch`, `numpy`, built-in modules)
2. Third-party imports (`monai`, `typing_extensions`, `numpy.typing`)
3. Local imports (`from viscy_transforms._flip import ...`)

**Path Aliases:**
- No custom path aliases configured in codebase
- Relative imports within packages: `from viscy_transforms._typing import Sample`
- Workspace imports in monorepo: `from viscy_transforms._monai_wrappers import ...`

**Example from `_flip.py`:**
```python
from collections.abc import Sequence

import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
```

## Error Handling

**Patterns:**
- Use `ValueError` for invalid arguments with descriptive messages
- Validate parameters in `__init__` methods
- Include expected vs. actual values in error messages

**Examples:**
```python
# In _adjust_contrast.py
if isinstance(gamma, (int, float)):
    self.gamma_range = (gamma, gamma)
elif isinstance(gamma, tuple) and len(gamma) == 2:
    self.gamma_range = (min(gamma), max(gamma))
else:
    raise ValueError("Gamma must be a float or a tuple of two floats.")
if self.gamma_range[0] <= 0.0:
    raise ValueError("Gamma must be a positive value.")

# In _crop.py
if random_size:
    raise ValueError("Batched transform does not support random size.")
if len(self._batch_slices[0]) != 3:
    raise ValueError("BatchedRandSpatialCrop only supports 3D data")
```

**No try/except blocks** in transform implementations; errors bubble up for caller handling.

## Logging

**Framework:** No explicit logging framework. Use `warnings.warn()` for non-fatal issues:
```python
from warnings import warn
warn("Divide by zero (a_min == a_max)")
```

**Patterns:**
- Warn on edge cases and numerical issues
- No debug logging in transforms (assume vectorized execution)

## Comments

**When to Comment:**
- Explain non-obvious implementation choices (e.g., why memory access pattern matters)
- Document performance notes: "NOTE: Copying one-by-one is slightly faster than vectorized indexing possibly due to memory access pattern"
- Explain workarounds for PyTorch limitations: "TODO: address pytorch#64947 to improve performance"

**No excessive comments** on obvious code. Code should be self-documenting via:
- Clear function names
- Type hints
- Comprehensive docstrings

## Docstrings

**Style:** NumPy docstring convention

**Structure:**
1. One-line summary (present tense, imperative)
2. Extended description (if needed)
3. Parameters section with types and descriptions
4. Returns section with type and description
5. Optional "See Also" for related functions

**Example from `_crop.py`:**
```python
class BatchedRandSpatialCrop(RandSpatialCrop):
    """
    Batched version of RandSpatialCrop that applies random spatial cropping to a batch of images.

    Each image in the batch gets its own random crop parameters. When random_size=True,
    all crops use the same randomly chosen size to ensure consistent output tensor shapes.

    Parameters
    ----------
    roi_size : Sequence[int] | int
        Expected ROI size to crop. e.g. [224, 224, 128]. If int, same size used for all dimensions.
    max_roi_size : Sequence[int] | int | None, optional
        Maximum ROI size when random_size=True. If None, defaults to input image size.
    random_center : bool, optional
        Whether to crop at random position (True) or image center (False). Default is True.
    random_size : bool, optional
        Not supported in batched mode, must be False.
    """
```

**Method docstrings** include Parameters, Returns, and extended description:
```python
def __call__(self, img: Tensor, randomize: bool = True) -> Tensor:
    """Apply batched random spatial crop to input tensor.

    Parameters
    ----------
    img : torch.Tensor
        Input tensor of shape (B, C, H, W, D) or (B, C, H, W).
    randomize : bool, optional
        Whether to generate new random parameters. Default is True.

    Returns
    -------
    torch.Tensor
        Cropped tensor with same batch size.
    """
```

## Function Design

**Size Guidelines:**
- Methods typically 20-50 lines
- Complex transforms may span 80+ lines (e.g., `_crop.py` implementations with gather-based ops)
- Each class file focuses on 1-2 transform variants (non-dict + dict versions)

**Parameters:**
- Use keyword-only arguments for optional parameters
- Provide sensible defaults (e.g., `prob=0.5`, `prob=0.1`)
- Accept union types for flexibility: `factors: tuple[float, float] | float`

**Return Values:**
- Transform `__call__` methods return `Tensor` or `dict[str, Tensor]`
- Deterministic operations return identical results on identical input
- Side effects stored in instance variables: `self._flip_spatial_dims`, `self._gamma_values`

**Randomization pattern:**
- Implement `randomize()` method to generate random parameters
- Store randomization state in instance variables (prefixed with underscore)
- `__call__` accepts `randomize: bool = True` parameter
- When `randomize=False`, reuses previously stored state (for batch consistency)

## Module Design

**Exports:**
- Each module defines `__all__` listing public classes
- Example from `_flip.py`:
```python
__all__ = ["BatchedRandFlip", "BatchedRandFlipd"]
```

**Barrel Files:**
- Main package `__init__.py` re-exports all public classes
- Example from `viscy_transforms/__init__.py`:
```python
from viscy_transforms._flip import BatchedRandFlip, BatchedRandFlipd
__all__ = ["BatchedRandFlip", "BatchedRandFlipd", ...]
```

**Module docstrings:**
- Each module includes header docstring explaining purpose
- Example from `_flip.py`:
```python
"""Batch-aware flip transforms."""
```

## Class Hierarchy

**Transform classes inherit from MONAI base classes:**
- `RandomizableTransform` for stochastic transforms
- `MapTransform` for dictionary transforms
- Direct subclasses of MONAI transforms when implementing batched versions: `class BatchedRandFlip(RandSpatialCrop)`

**Multiple inheritance pattern for dictionary transforms:**
```python
class BatchedRandFlipd(MapTransform, RandomizableTransform):
    # Both MapTransform (dict iteration) and RandomizableTransform (probability)
    def __init__(self, keys, spatial_axes=[0, 1, 2], prob=0.5, allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
```

---

*Convention analysis: 2026-02-07*
