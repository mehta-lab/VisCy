# Testing Patterns

**Analysis Date:** 2026-02-07

## Test Framework

**Runner:**
- pytest 9.0.2+
- Config: `pyproject.toml` in root (monorepo-aware)
- Test paths: `packages/*/tests` and `tests/`

**Run Commands:**
```bash
uv run pytest                              # Run all tests
uv run pytest --cov=viscy_transforms       # Coverage report
uv run pytest --cov-report=term-missing    # Coverage with missing lines
pytest -xvs tests/test_flip.py             # Single file, verbose, stop on first failure
```

**CI Configuration:**
- GitHub Actions workflow: `.github/workflows/test.yml`
- Matrix testing: Python 3.11, 3.12, 3.13 on Ubuntu, macOS, Windows
- Coverage required with `--cov=viscy_transforms --cov-report=term-missing`

## Test File Organization

**Location:**
- Co-located with source: Tests in `packages/*/tests/` parallel to `packages/*/src/`
- Example: `packages/viscy-transforms/src/viscy_transforms/_flip.py` → `packages/viscy-transforms/tests/test_flip.py`

**Naming:**
- Test modules: `test_<transform_name>.py`
- Test functions: `test_<feature_or_condition>()`
- Parametrized test names expand with parameters: `test_batched_rand_flip[0.0-[0, 1, 2]-cpu]`

**Structure:**
```
packages/viscy-transforms/tests/
├── __init__.py              # Empty, marks as package
├── conftest.py              # Shared pytest fixtures
├── test_flip.py             # Tests for _flip.py
├── test_crop.py             # Tests for _crop.py
├── test_scale_intensity.py  # Tests for _scale_intensity.py
└── test_transforms.py       # Integration tests across transforms
```

## Test Structure

**Suite Organization:**
All tests use standard pytest structure:

```python
import pytest
import torch
from viscy_transforms import BatchedRandFlip, BatchedRandFlipd

@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
def test_batched_rand_flip(device, prob, spatial_axes):
    # SETUP
    img = torch.arange(32 * 2 * 2 * 2 * 2, device=device).reshape(32, 2, 2, 2, 2).float()
    transform = BatchedRandFlip(prob=prob, spatial_axes=spatial_axes)

    # EXECUTE
    out = transform(img)

    # ASSERT
    assert out.shape == img.shape
    changed = (out != img).any(dim=tuple(range(1, img.ndim)))
    if prob == 1.0:
        assert changed.all()
    elif prob == 0.0:
        assert not changed.any()
```

**Patterns:**

1. **Setup-Execute-Assert pattern:** Clear separation of test phases
2. **Descriptive variable names:** `img`, `data`, `batch_data`, `transform`, `out`, `output`
3. **Single assertion per concern:** Test one behavior per assert, use multiple asserts for different aspects
4. **Device-aware testing:** Tests parametrize across CPU/CUDA when available:
   ```python
   @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
   ```

## Fixtures

**Shared fixtures in `conftest.py`:**

```python
@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def seed():
    """Set deterministic random seed for reproducible tests."""
    torch.manual_seed(42)
    return 42
```

**Fixture usage:**
```python
def test_batched_rand_flip(device, prob, spatial_axes):
    img = torch.arange(32 * 2 * 2 * 2 * 2, device=device).reshape(...)
    # device fixture automatically injected
```

**Test data creation patterns:**
- Random tensors: `torch.rand(shape)`, `torch.randint(0, 2, shape)`
- Sequential data for verification: `torch.arange(size).reshape(shape)`
- Clone for dictionary tests: `{"key": tensor.clone()}`

## Parametrization

**Pattern:** `@pytest.mark.parametrize()` with multiple parameters

```python
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("spatial_axes", [[0, 1, 2], [1, 2], [0]])
@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_batched_rand_flip(device, prob, spatial_axes):
    # Test runs 3 × 3 × 2 = 18 times with all combinations
```

**Coverage strategy:**
- Probability boundary testing: `[0.0, 0.5, 1.0]`
- Dimension/size combinations
- Device compatibility (CPU at minimum, GPU when available)
- Edge cases: empty, single-element, large batches

## Test Types

**Unit Tests (majority):**
- Test individual transform behavior: `test_batched_rand_flip()`
- Verify output shapes match input shapes
- Validate probability behavior (0% never changes, 100% always changes)
- Test parameter validation (raises ValueError for invalid args)

**Determinism Tests:**
```python
def test_batched_rand_flip_dict():
    img = torch.arange(16 * 2 * 4 * 8 * 8).reshape(16, 2, 4, 8, 8).float()
    data = {"a": img.clone(), "b": img.clone()}
    transform = BatchedRandFlipd(keys=["a", "b"], prob=1.0, spatial_axes=[0, 1, 2])
    out = transform(data)
    assert torch.equal(out["a"], out["b"])  # Same input, same transform → same output
```

**Randomization Control Tests:**
```python
def test_batched_rand_spatial_crop_randomize_control():
    """Test randomize parameter behavior."""
    transform = BatchedRandSpatialCrop(roi_size=[8, 8, 8], random_center=True)

    # First call: randomize=True generates new params
    output_1 = transform(data, randomize=True)

    # Second call: randomize=False reuses stored params
    output_2 = transform(data, randomize=False)

    # Third call: randomize=True generates new params
    output_3 = transform(data, randomize=True)

    assert torch.equal(output_1, output_2)      # Reused state matches
    assert not torch.equal(output_1, output_3)  # New randomization differs
```

**Edge Case Tests:**
```python
def test_batched_center_spatial_crop_edge_cases():
    """Test edge cases for BatchedCenterSpatialCrop."""
    data = torch.rand(2, 1, 64, 64, 32)

    # ROI larger than image (should not crop)
    transform_large = BatchedCenterSpatialCrop(roi_size=[128, 128, 64])
    output_large = transform_large(data)
    assert output_large.shape == data.shape

    # ROI same as image (should not crop)
    transform_same = BatchedCenterSpatialCrop(roi_size=[64, 64, 32])
    output_same = transform_same(data)
    assert output_same.shape == data.shape
```

**Integration/Comparison Tests:**
Tests verify batched transforms match MONAI single-image behavior:

```python
@pytest.mark.parametrize("factor_value", [-0.3, -0.1, 0.0, 0.2, 0.5])
def test_batched_scale_intensity_vs_monai(factor_value):
    """Test that batched transform produces same results as individual MONAI transforms."""
    batch_size = 4
    img_batch = torch.rand(batch_size, 2, 8, 8) + 0.1

    # Batched version
    batched_transform = BatchedRandScaleIntensity(prob=1.0, factors=(factor_value, factor_value))
    batched_result = batched_transform(img_batch)

    # Compare against MONAI on each element
    monai_results = []
    for i in range(batch_size):
        sample = img_batch[i]
        monai_transform = RandScaleIntensity(factors=(factor_value, factor_value), prob=1.0)
        monai_results.append(monai_transform(sample))

    monai_batch_result = torch.stack(monai_results)
    assert torch.allclose(batched_result, monai_batch_result, atol=1e-6, rtol=1e-5)
```

**Comparison loop tests:**
Comprehensive matrix testing across configurations:

```python
def test_batched_center_spatial_crop_vs_monai_loop():
    """Test against MONAI's CenterSpatialCrop in a loop with multiple configurations."""
    test_configs = [
        (1, 1, 64, 64, 32, [32, 32, 16]),
        (3, 2, 128, 128, 64, [64, 64, 32]),
        (2, 1, 96, 96, 48, [48, 48, 24]),
        # ... more configs
    ]

    for batch_size, channels, H, W, D, roi_size in test_configs:
        torch.manual_seed(hash((batch_size, channels, H, W, D)) % 10000)
        batch_data = torch.rand(batch_size, channels, H, W, D)

        monai_transform = CenterSpatialCrop(roi_size=roi_size)
        batch_transform = BatchedCenterSpatialCrop(roi_size=roi_size)
        batch_result = batch_transform(batch_data)

        for i in range(batch_size):
            single_img = batch_data[i]
            monai_result = monai_transform(single_img)
            batch_single_result = batch_result[i]
            assert torch.allclose(monai_result, batch_single_result, rtol=1e-6, atol=1e-6)
```

## Async Testing

Not applicable. All transforms are synchronous PyTorch operations.

## Error Testing

**Invalid parameter validation:**

```python
def test_batched_rand_spatial_crop_random_size_error():
    """Test that random_size parameter raises ValueError."""
    with pytest.raises(ValueError, match="Batched transform does not support random size"):
        BatchedRandSpatialCrop(roi_size=[8, 8, 8], random_size=True)

def test_batched_rand_spatial_crop_2d_error():
    """Test that 2D data raises appropriate error."""
    data_2d = torch.rand((4, 2, 32, 32))
    transform = BatchedRandSpatialCrop(roi_size=(16, 16), random_center=True)
    with pytest.raises(ValueError, match="only supports 3D data"):
        transform(data_2d)
```

## Mocking

**No mocking framework used** (unittest.mock not present in test files).

**Approach:** Tests use real PyTorch operations and tensors. MONAI transforms are real implementations.

**Rationale:**
- Transforms are pure tensor operations (no side effects)
- Comparing batched vs. unbatched MONAI is the primary verification
- Device compatibility (CPU/GPU) tested with actual hardware availability

## Test Coverage

**Requirements:** No minimum enforced, but `--cov` flag recommended in CI

**View Coverage:**
```bash
uv run pytest --cov=viscy_transforms --cov-report=term-missing
```

**Coverage strategy:**
- Unit tests for all public transform classes
- Both base and dictionary variants tested
- Edge cases explicitly tested (boundary probabilities, size limits)
- Device coverage (CPU always, GPU if available)

## Common Test Patterns

**Shape assertions (most common):**
```python
assert output.shape == input.shape
assert output.shape == (batch_size, channels, roi_size[0], roi_size[1], roi_size[2])
```

**Value range assertions:**
```python
assert torch.all(output >= 0) and torch.all(output <= 1)
assert torch.all((output == 0) | (output == 1))  # For binary masks
```

**Probability assertions:**
```python
if prob == 1.0:
    assert changed.all()  # All samples modified
elif prob == 0.0:
    assert not changed.any()  # No samples modified
elif prob == 0.5:
    assert changed.any() and not changed.all()  # Some modified
```

**Numerical equality (for floating point):**
```python
assert torch.equal(exact_match, expected)           # Exact equality
assert torch.allclose(approx, expected, atol=1e-6, rtol=1e-5)  # Approximate with tolerance
assert torch.allclose(median, torch.zeros_like(median), atol=1e-6)  # Near-zero checks
```

**Dictionary transform assertions:**
```python
assert torch.equal(out["source"], out["target"])  # Consistency across keys
assert out["other"] == "unchanged"                 # Non-tensor fields preserved
assert "nonexistent" not in output                 # Missing keys not added
```

---

*Testing analysis: 2026-02-07*
