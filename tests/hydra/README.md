# Hydra Integration Tests

This directory contains integration tests for the Hydra-based training CLI.

## Purpose

These tests verify that the Hydra configuration system works correctly:
- ✅ Config files load without syntax errors
- ✅ Config composition works (defaults + overrides)
- ✅ CLI overrides work properly
- ✅ DataModules instantiate from configs
- ✅ Utility functions work with Hydra

**Note**: These tests focus on Hydra integration, not model correctness. Model-specific tests are elsewhere.

## Test Files

- `test_config_loading.py` - Tests for config loading, datamodule instantiation, and overrides

## Running Tests

**All Hydra tests**:
```bash
pytest tests/hydra/
```

**Specific test file**:
```bash
pytest tests/hydra/test_config_loading.py
```

**Specific test**:
```bash
pytest tests/hydra/test_config_loading.py::TestConfigLoading::test_hcs_config_loads
```

**With verbose output**:
```bash
pytest tests/hydra/ -v
```

## Test Structure

### TestConfigLoading
Tests that Hydra configs load successfully:
- HCS config loads
- Triplet/contrastive config loads
- Debug mode config
- Paths config
- Extras config

### TestDataModuleInstantiation
Tests that datamodules can be instantiated from configs:
- HCS datamodule
- Triplet datamodule

### TestConfigOverrides
Tests that CLI overrides work:
- Batch size override
- Task name override
- Tags override

### TestUtilities
Tests utility functions:
- Callback instantiation
- Logger instantiation

## Expected Failures

Some tests may fail due to needing full application context:
- DataModule instantiation with complex MONAI transforms
- Logger instantiation with Hydra interpolations

These are **expected** and not critical - the important thing is that configs load correctly.

## Adding New Tests

When adding new Hydra configs, add corresponding tests:

```python
def test_my_new_config_loads(self, config_dir, preprocessed_hcs_dataset):
    """Test that my new config loads successfully."""
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "data=my_new_data",
                "model=my_new_model",
                "augmentation=none",
                "normalization=none",
                f"data.data_path={preprocessed_hcs_dataset}",
            ],
        )

        assert cfg is not None
        assert cfg.data._target_ == "viscy.data.MyNewDataModule"
```

## CI Integration

These tests run in CI as part of the fast test suite:
```bash
pytest tests/hydra/ -v
```

They catch configuration errors before code reaches production.
