# Testing Patterns

**Analysis Date:** 2026-03-27

## Test Framework

**Runner:**
- pytest 9.0+ (configured in root `pyproject.toml`)
- Config file: `pyproject.toml` (no separate `pytest.ini`)

**Assertion Library:**
- pytest's built-in assertions (no additional library needed)

**Test Discovery:**
- Test paths: `packages/*/tests`, `applications/*/tests`
- Import mode: `importlib` (specified in `addopts = ["--import-mode=importlib"]`)

**Run Commands:**
```bash
uv run pytest                          # Run all tests
uv run pytest packages/viscy-data/    # Single package tests (data)
uv run pytest packages/viscy-models/  # Single package tests (models)
uv run pytest -v                       # Verbose output
uv run pytest -xvs                     # Stop on first failure, verbose, no capture
```

## Test File Organization

**Location:**
- Co-located with source when possible: `src/viscy_data/module.py` paired with `tests/test_module.py`
- Or in a parallel `tests/` directory at package level
- Pattern: `tests/` directory at same level as `src/`

**Naming:**
- Test files: `test_{feature}.py`
- Test functions: `def test_{scenario}()` (not `def test_scenario()`)
- Test classes: `class Test{Feature}:` (e.g., `TestValidation`, `TestExperimentAware`)

**Structure:**
```
packages/viscy-data/
├── src/viscy_data/
│   ├── cell_index.py
│   └── ...
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_cell_index.py    # Tests for cell_index.py
│   └── ...
└── pyproject.toml
```

## Test Structure

**Suite Organization:**
```python
"""Module docstring describing test scope."""

from __future__ import annotations

import pytest

# Imports organized by: stdlib, third-party, local

# Optional: module-level constants/fixtures

class TestFeatureName:
    """Tests for feature_name.

    Tests cover:
    - Scenario A
    - Scenario B
    - Scenario C
    """

    def test_scenario_a(self):
        """Brief description of what this test verifies."""
        # Arrange
        data = setup_test_data()

        # Act
        result = function_under_test(data)

        # Assert
        assert result == expected_value

    def test_scenario_b(self):
        """Another test scenario."""
        # Test code here
        pass
```

**Patterns:**
- Use test classes to group related tests by feature or concern
- One test class per major feature or component
- Each test method tests one specific behavior
- Include docstrings on test classes (describing covered scenarios) and test methods (what is verified)

**Example from codebase:**
```python
class TestValidation:
    """Tests for validate_cell_index."""

    def test_valid_df_passes(self):
        """1. Valid DataFrame passes validate_cell_index()."""
        df = _make_valid_df()
        warnings = validate_cell_index(df)
        assert isinstance(warnings, list)

    def test_missing_core_columns_raises(self):
        """2. Missing core columns raise ValueError."""
        df = _make_valid_df()
        df = df.drop(columns=["cell_id", "experiment"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_cell_index(df)
```

## Test Fixtures

**Fixture Scope:**
- `scope="function"` (default): fresh fixture per test
- `scope="session"`: fixture created once per test session (use for expensive setup)
- `scope="module"`: fixture created once per test file

**Common Fixtures from codebase:**

**HCS Dataset Fixtures:**
```python
@fixture(scope="session", params=[False, True], ids=["zarr_v2", "zarr_v3"])
def preprocessed_hcs_dataset(tmp_path_factory: TempPathFactory, request: FixtureRequest) -> Path:
    """Provides a preprocessed HCS OME-Zarr dataset (v2 and v3)."""
    # Parameterized: tests run twice (once with v2, once with v3)

@fixture(scope="function")
def small_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a small, not preprocessed HCS OME-Zarr dataset."""
    # Fresh dataset per test
```

**Device Fixture:**
```python
@pytest.fixture
def device() -> str:
    """Return the best available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"
```

**Test Data Fixtures:**
```python
@pytest.fixture()
def two_experiment_anchors() -> pd.DataFrame:
    """DataFrame with 2 experiments, 2 conditions each, 200 rows total."""
    # Helper function constructs fixture data
    rows = []
    for exp_name in ["exp_A", "exp_B"]:
        for cond in ["infected", "uninfected"]:
            for i in range(50):
                rows.append({...})
    return pd.DataFrame(rows)
```

**Location:** All fixtures in `conftest.py` at package test root:
- `packages/viscy-data/tests/conftest.py` (shared across all data package tests)
- `packages/viscy-models/tests/conftest.py`
- `applications/dynaclr/tests/conftest.py`

## Mocking

**Framework:** unittest.mock (standard library)

**When to Mock:**
- External services (APIs, databases) — use fixtures instead of real connections
- Expensive operations (training, inference) — create lightweight synthetic alternatives
- State-dependent behaviors — mock for deterministic testing

**When NOT to Mock:**
- Core library functions (numpy, pandas, torch) — these are stable and well-tested
- Functions under test — always test the real function with real data
- Data transformations — test with actual sample data to catch real bugs

**Pattern (from dynaclr tests):**
```python
# Synthetic tensor data instead of mocking
SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W = 1, 1, 4, 4

def test_some_transform(device):
    x = torch.randn(1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W, device=device)
    result = transform(x)
    assert result.shape == (1, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W)
```

**Avoided Pattern:** Do NOT mock the function under test:
```python
# WRONG — defeats the purpose of the test
with patch("viscy_data.cell_index.validate_cell_index"):
    result = validate_cell_index(df)  # This is now a mock, not the real function
```

## Parametrization

**Approach:** Use pytest `@pytest.mark.parametrize` or fixture params to test multiple scenarios.

**Fixture Parametrization:**
```python
@fixture(scope="session", params=[False, True], ids=["zarr_v2", "zarr_v3"])
def preprocessed_hcs_dataset(tmp_path_factory, request):
    # request.param is True or False
    _build_hcs(..., sharded=request.param)
```

**Test Function Parametrization:**
```python
@pytest.mark.parametrize(
    "condition,expected",
    [
        ("infected", "ZIKV"),
        ("uninfected", "mock"),
    ]
)
def test_perturbation_resolution(condition, expected):
    result = resolve_perturbation({condition: [...]}, ...)
    assert result == expected
```

## Fixtures and Factories

**Test Data Builders:**
- Helper functions (not fixtures) for constructing test data
- Prefix with `_make_` or `_build_`
- Accept configurable parameters for flexibility

**Examples from codebase:**
```python
def _make_valid_df(n: int = 5) -> pd.DataFrame:
    """Create a minimal valid cell index DataFrame."""
    return pd.DataFrame({...})

def _build_hcs(path: Path, channel_names: list[str], zyx_shape: tuple[int, int, int], ...):
    """Create a mock HCS OME-Zarr dataset with specified shape and parameters."""
    dataset = open_ome_zarr(path, ...)
    # Build structure
```

**Location:** In same test file or in `conftest.py` if shared across multiple test files.

## Coverage

**Requirements:** Not enforced at commit time; no coverage threshold configured.

**View Coverage:**
```bash
uv run pytest --cov=packages/viscy-data --cov-report=html
# Opens HTML report in browser for line-by-line analysis
```

**Coverage Strategy:**
- Aim for high coverage on critical data pipelines and validation logic
- 100% coverage is not required for research code
- Focus on edge cases and error paths, not trivial getters/setters

## Test Types

**Unit Tests:**
- Scope: Single function or method in isolation
- Approach: Use synthetic data, test specific behavior
- Location: `test_{module}.py` files
- Example: `test_validate_cell_index()` tests the `validate_cell_index()` function with mock DataFrames

**Integration Tests:**
- Scope: Multiple components working together (e.g., builder reads from zarr, writes to parquet)
- Approach: Use realistic sample data (HCS datasets, collection YAMLs)
- Location: Often in `tests/test_{feature}_integration.py` or within main test file with descriptive name
- Example: `test_round_trip_preserves_dtypes()` tests write + read cycle together

**E2E Tests:**
- Framework: Not used in VisCy codebase
- Alternative: Integration tests cover end-to-end pipeline scenarios
- Examples: `test_flat_parquet_e2e.py`, `test_training_integration.py` in dynaclr

## Common Patterns

**Async Testing:**
Not applicable (VisCy is synchronous).

**Error Testing:**
```python
def test_missing_core_columns_raises(self):
    """Missing core columns raise ValueError."""
    df = _make_valid_df()
    df = df.drop(columns=["cell_id", "experiment"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_cell_index(df)
```

**Parametrized Error Testing:**
```python
@pytest.mark.parametrize("bad_input", [None, [], "invalid"])
def test_invalid_input_raises(bad_input):
    with pytest.raises(TypeError):
        process_input(bad_input)
```

**Assertion Patterns:**
```python
# Exact match
assert result == expected_value

# Collection length
assert len(df) == expected_rows

# Membership
assert "A/1" in df["well"].unique()

# Type/dtype
assert pd.api.types.is_string_dtype(result["cell_id"])
assert isinstance(warnings, list)

# Floating point with tolerance
assert y == pytest.approx(20.0)
assert not torch.allclose(tensor1, tensor2)

# All elements satisfy condition
assert not df["cell_id"].duplicated().any()
assert (df["well"] == "A/1").all()
```

**Context Manager Testing:**
```python
def test_round_trip(tmp_path):
    """Write and read preserve data."""
    path = tmp_path / "cell_index.parquet"
    write_cell_index(df, path)        # Act
    result = read_cell_index(path)    # Act
    assert len(result) == len(df)     # Assert
```

## Skip Markers

**GPU/HPC Tests:**
Tests requiring HPC paths or GPU can be skipped conditionally:

```python
# In conftest.py
_GPU_AVAILABLE = torch.cuda.is_available()
_HPC_PATHS_AVAILABLE = all(p.exists() for p in [path1, path2])

# In test file
@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CUDA not available")
def test_gpu_intensive():
    ...

@pytest.mark.skipif(not _HPC_PATHS_AVAILABLE, reason="HPC paths not available")
def test_with_real_data():
    ...
```

## Test Organization Best Practices

**Isolation:** Each test should be independent and not depend on others.
- Use fresh fixtures per test
- No test order dependencies
- Clean up resources (use context managers, fixtures handle this)

**Clarity:** Test names should describe what they test:
- Good: `test_lineage_reconstruction_links_daughters_to_root_ancestor()`
- Bad: `test_lineage()`

**Descriptive Docstrings:** Include what the test verifies in the docstring:
```python
def test_lineage_reconstruction(self, tmp_path):
    """7. Lineage reconstruction links daughters to root ancestor."""
```

**Realistic Data:** Use realistic sample data structures (even if synthetic):
- Use actual channel names, well IDs, FOV patterns
- Match real data dimensions and dtypes
- Avoid overly simplified edge cases that won't occur in practice

---

*Testing analysis: 2026-03-27*
