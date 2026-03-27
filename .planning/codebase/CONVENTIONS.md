# Coding Conventions

**Analysis Date:** 2026-03-27

## Naming Patterns

**Files:**
- Module files use lowercase with underscores: `cell_index.py`, `gpu_aug.py`, `test_sampler.py`
- Private modules prefixed with underscore: `_utils.py`, `_typing.py`
- Test files use `test_{feature}.py` pattern (co-located with source when possible)

**Functions:**
- snake_case for all functions: `validate_cell_index()`, `build_timelapse_cell_index()`, `format_markdown_table()`
- Private functions prefixed with single underscore: `_build_hcs()`, `_make_valid_df()`, `_reconstruct_lineage()`
- Single or double underscore for internal helpers only

**Variables:**
- snake_case for all variables: `perturbation_wells`, `channel_names`, `global_track_id`
- Module-level logger constant: `_logger = logging.getLogger(__name__)` or `_logger = getLogger("lightning.pytorch")`
- Constants use UPPERCASE: `CELL_INDEX_SCHEMA`, `CELL_INDEX_CORE_COLUMNS`

**Types:**
- PascalCase for classes: `FlexibleBatchSampler`, `ExperimentEntry`, `ResNet3dEncoder`
- Use type hints throughout: function parameters and return types always annotated
- Use `from __future__ import annotations` at file top for forward references

**Class Methods:**
- Private methods: single underscore prefix `_find_root()`, `_get_convnext_stage()`
- Public methods: no prefix `forward()`, `validate()`, `__init__()`

## Code Style

**Formatting:**
- Tool: Ruff formatter (centralized in root `pyproject.toml`)
- Quote style: double quotes (`"string"` not `'string'`)
- Line length: 120 characters
- Indent: 4 spaces (not tabs)
- Magic trailing comma: disabled (skip-magic-trailing-comma = false)

**Linting:**
- Tool: Ruff lint
- Rules selected: `D` (docstrings), `E` (errors), `F` (flakes), `I` (imports), `NPY` (numpy), `PD` (pandas), `W` (warnings)
- Rules ignored in tests: `D` (docstrings not required in test files)
- Rules ignored in notebooks: `D`, `E402`, `E501`, `PD`
- Per-file ignores in `__init__.py`: `D104` (undocumented public package), `F401` (unused imports allowed for re-exports)

**Configuration:**
- All ruff config is in root `/hpc/mydata/eduardo.hirata/repos/viscy/pyproject.toml`
- Do NOT add `[tool.ruff.*]` sections to sub-package `pyproject.toml` files — this silently overrides root config
- Target Python: 3.11+
- Docstring convention: numpy style (`convention = "numpy"`)

## Import Organization

**Order:**
1. `from __future__ import annotations` (required for forward references and type hints)
2. Standard library imports (e.g., `from pathlib import Path`, `import logging`)
3. Third-party imports (e.g., `import pandas as pd`, `import torch`, `from monai import ...`)
4. Local application imports (e.g., `from viscy_data.cell_index import ...`, `from viscy_models.components import ...`)
5. TYPE_CHECKING imports (wrapped in `if TYPE_CHECKING:` block for circular reference avoidance)

**Import Style:**
- Absolute imports only: `from viscy_data.cell_index import read_cell_index` (not relative)
- Do NOT modify `sys.path` for imports
- Do NOT use inline imports without strong reason (import at module top)
- Re-exports in `__init__.py` use explicit `from X import Y` (not `from X import *`)

**Path Aliases:**
- No path aliases configured; use absolute imports from packages
- Standard workspace packages: `viscy_data`, `viscy_models`, `viscy_transforms`, `viscy_utils`, `dynaclr`

## Error Handling

**Patterns:**
- Prefer raising errors rather than silently catching
- Use specific error types: `ValueError` for invalid data, `TypeError` for wrong type, `KeyError` for missing dict keys, `FileNotFoundError` for missing files
- Include context in error messages: `raise ValueError(f"Missing required columns: {sorted(missing)}")` not just `raise ValueError("missing columns")`
- Do NOT catch errors to provide fallback values — let libraries raise their own errors. Only catch when there's a good reason (e.g., retrying HTTP requests)

**Examples from codebase:**
```python
# cell_index.py - explicit error with context
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# sampler.py - specific condition validation
if batch_group_by not in df.columns:
    raise ValueError(f"batch_group_by='{batch_group_by}' not found in anchors columns")

# cell_index.py - clear tracking error
if len(csv_files) > 1:
    raise ValueError(f"Expected exactly one tracking CSV in {tracks_dir}, found: {csv_files}")
```

## Logging

**Framework:** Python standard logging module
- Root logger used: `logging.getLogger(__name__)` or `getLogger("lightning.pytorch")` for PyTorch Lightning integration
- Variable name: `_logger = logging.getLogger(__name__)` (module-level)
- Log levels: `debug()` for detailed traces, `info()` for progress, `warning()` for potential issues

**Patterns:**
- Debug logs for detailed iteration state: `_logger.debug(f"Caching for index {idx}")`
- Info logs for milestone events: `_logger.info(f"Number of test samples: {len(self)}")`
- Use f-strings for all log messages (not % formatting)

**Examples from codebase:**
```python
_logger = logging.getLogger("lightning.pytorch")
_logger.debug(f"Filtering tracks to specific cells: {self.include_fov_names}")
_logger.info(f"Number of test samples: {len(self)}")
```

## Comments

**When to Comment:**
- Do NOT add comments for obvious code. The code itself should be self-documenting
- Comment non-obvious algorithmic choices or performance hacks
- Comment workarounds and their reason (e.g., "Context manager needed to avoid zarr file handle leak")
- Comment complex calculations or data transformations that aren't immediately clear

**JSDoc/TSDoc Style:**
- Use numpy-style docstrings for all public functions, classes, and modules
- Required sections: `Parameters`, `Returns`, `Raises` (if applicable)
- Optional sections: `Notes`, `Examples`, `See Also`
- Private functions (prefixed `_`) do NOT require docstrings but may have them for clarity

**Docstring Example from codebase:**
```python
def validate_cell_index(df: pd.DataFrame, *, strict: bool = False) -> list[str]:
    """Validate a cell index DataFrame against the canonical schema.

    Parameters
    ----------
    df : pd.DataFrame
        Cell index to validate.
    strict : bool
        If ``True``, require **all** schema columns (not just core + grouping).

    Returns
    -------
    list[str]
        Warnings (e.g. nullable columns that are entirely null).

    Raises
    ------
    ValueError
        If required columns are missing or ``(cell_id, channel_name)`` is not unique.
    """
```

## Function Design

**Size:** Keep functions under 50 lines when possible. If a function exceeds 100 lines, consider breaking it into smaller helpers.

**Parameters:**
- Use keyword-only parameters for clarity when there are many optional arguments (prefix with `*`)
- Include type hints on all parameters
- Default to required positional args (no defaults) unless there's a good reason
- Use `None` as sentinel for optional parameters (not empty strings or empty lists)

**Return Values:**
- Always annotate return type, including `None`
- Return early to avoid deep nesting: use `if condition: return` patterns
- For multiple returns, use tuple unpacking: `return (embedding, projections)` with clear type hints

**Example from codebase:**
```python
def read_cell_index(path: str | Path) -> pd.DataFrame:
    """Read a cell index parquet into a pandas DataFrame.

    [docstring...]
    """
    table = pq.read_table(str(path), schema=CELL_INDEX_SCHEMA)
    return table.to_pandas()
```

## Module Design

**Exports:**
- Use `__all__` to explicitly list public exports
- Place `__all__` near top of module (after imports, before class/function definitions)
- Private implementation details (prefixed `_`) are NOT in `__all__`

**Example from codebase:**
```python
__all__ = [
    "CELL_INDEX_SCHEMA",
    "build_ops_cell_index",
    "build_timelapse_cell_index",
    "convert_ops_parquet",
    "read_cell_index",
    "validate_cell_index",
    "write_cell_index",
]
```

**Barrel Files:**
- `__init__.py` files re-export key classes and functions for clean public API
- Do NOT import everything with `*` — be explicit about what's re-exported
- Example: `from viscy_data.cell_index import read_cell_index, write_cell_index`

**Organization:**
- Group related functions into logical sections with comment headers: `# -----------`, `# Schema`, `# I/O`, `# Validation`
- Separate concerns: keep schema definitions, I/O, validation, and builders in separate logical sections

## Context Managers

**Requirement:** Always use context managers (`with` statements) when opening external resources.

**Pattern:**
```python
# CORRECT — resource automatically closed
with open_ome_zarr(path, mode="r") as plate:
    for _pos_path, position in plate.positions():
        ...

# WRONG — resource never closed, leaks file handles
plate = open_ome_zarr(path, mode="r")
# ... code here ... plate is never closed
```

**Resources requiring context managers:**
- Zarr stores: `open_ome_zarr()`, `zarr.open()`
- File handles: `open()`
- Database connections: any client with `.close()`
- Large memory contexts: tensorstore, memory-mapped arrays

---

*Convention analysis: 2026-03-27*
