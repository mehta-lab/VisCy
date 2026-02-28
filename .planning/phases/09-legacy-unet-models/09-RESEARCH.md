# Phase 9: Legacy UNet Models - Research

**Researched:** 2026-02-13
**Domain:** Unet2d/Unet25d nn.Module migration, unittest-to-pytest conversion
**Confidence:** HIGH

## Summary

Phase 9 migrates the two legacy UNet architectures (Unet2d and Unet25d) from the v0.3.3 monolithic codebase into viscy-models. These are simpler architectures than UNeXt2/FCMAE -- they do NOT use timm backbones or shared `_components/` modules. Instead they compose ConvBlock2D and ConvBlock3D (already migrated to `unet/_layers/` in Phase 6) with standard PyTorch pooling and upsampling layers. The migration is primarily a copy-and-update-imports exercise, with careful attention to the `register_modules`/`add_module` pattern that governs state dict keys.

The original tests are written in unittest style with a combinatorial explosion approach -- each test class runs all 144 parameter configurations via Cartesian product. They also depend on `viscy.utils.cli_utils.show_progress_bar`, a utility not available in viscy-models. The tests have a known shape mismatch: the `squeeze(2)`/`unsqueeze(2)` was added to Unet2D's forward method in commit `0e2b575` after the tests were written, and the test expected shapes were never updated. The test conversion must fix this mismatch, remove the cli_utils dependency, and convert from unittest to idiomatic pytest using `@pytest.mark.parametrize` for configuration coverage.

Both models use `num_filters=[]` (mutable list default) which must be converted to `num_filters=()` per COMPAT-02. The `up_list` in both models is a plain Python list (not registered via `register_modules`) because `nn.Upsample` has no learnable parameters -- this is fine and must be preserved as-is.

**Primary recommendation:** Copy Unet2d and Unet25d verbatim from v0.3.3, update imports to `viscy_models.unet._layers`, fix `num_filters` mutable default, and write new pytest tests that verify correct output shapes (accounting for squeeze/unsqueeze), state dict key patterns, and a representative subset of parameter configurations.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.10 | nn.Module base, Conv2d/3d, AvgPool, Upsample | All model components are pure PyTorch |
| numpy | >=2.4.1 | `np.linspace` in ConvBlock2D/3D filter step calculation | Indirect dependency via ConvBlock layers |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=9.0.2 | Test framework | pytest.mark.parametrize for config combos |

### Not Needed (Unlike Other Phases)
| Library | Not Needed Because |
|---------|-------------------|
| timm | Unet2d/Unet25d do not use pretrained backbones |
| monai | Unet2d/Unet25d do not use MONAI blocks |
| viscy.utils.cli_utils | Test dependency on `show_progress_bar` must be removed |

**Installation:**
```bash
uv sync --package viscy-models  # Already configured from Phase 6
```

## Architecture Patterns

### File Placement
```
packages/viscy-models/
    src/viscy_models/
        unet/
            __init__.py          # Add Unet2d, Unet25d to exports
            _layers/
                conv_block_2d.py # Already exists (Phase 6)
                conv_block_3d.py # Already exists (Phase 6)
            unet2d.py            # NEW: Unet2d class
            unet25d.py           # NEW: Unet25d class
    tests/
        test_unet/
            test_unet2d.py       # NEW: pytest tests for Unet2d
            test_unet25d.py      # NEW: pytest tests for Unet25d
```

### Pattern 1: Import Path Update (Only Change to Model Code)
**What:** Update the import of ConvBlock2D/ConvBlock3D from old monolithic path to new package path.
**When to use:** Every model file migration.

```python
# BEFORE (v0.3.3 source):
from viscy.unet.networks.layers.ConvBlock2D import ConvBlock2D

# AFTER (viscy-models):
from viscy_models.unet._layers.conv_block_2d import ConvBlock2D
```

### Pattern 2: Mutable Default Fix
**What:** Replace `num_filters=[]` with `num_filters=()` in both Unet2d and Unet25d constructors.
**When to use:** Both model classes.

```python
# BEFORE:
def __init__(self, ..., num_filters=[], ...):
    if len(num_filters) != 0:
        ...
    else:
        self.num_filters = [pow(2, i) * 16 for i in range(num_blocks + 1)]

# AFTER:
def __init__(self, ..., num_filters=(), ...):
    if len(num_filters) != 0:
        ...
    else:
        self.num_filters = [pow(2, i) * 16 for i in range(num_blocks + 1)]
```

The internal code continues to work because `len(())` is 0, and `len(some_tuple)` returns the correct length for non-empty tuples too.

### Pattern 3: register_modules / add_module (Preserve Verbatim)
**What:** Both Unet2d and Unet25d define their own `register_modules()` method that calls `self.add_module()`. This creates state dict keys like `down_samp_0`, `down_conv_block_0`, `up_conv_block_0`, etc.
**Why critical:** These exact key names must be preserved for checkpoint compatibility.
**Do NOT refactor to nn.ModuleList.**

```python
# This pattern MUST be preserved exactly:
def register_modules(self, module_list, name):
    for i, module in enumerate(module_list):
        self.add_module(f"{name}_{str(i)}", module)
```

### Pattern 4: unet/__init__.py Export Update
**What:** Add Unet2d and Unet25d to the unet subpackage exports.

```python
# packages/viscy-models/src/viscy_models/unet/__init__.py
"""UNet family architectures."""

from viscy_models.unet.fcmae import FullyConvolutionalMAE
from viscy_models.unet.unet2d import Unet2d
from viscy_models.unet.unet25d import Unet25d
from viscy_models.unet.unext2 import UNeXt2

__all__ = ["UNeXt2", "FullyConvolutionalMAE", "Unet2d", "Unet25d"]
```

### Anti-Patterns to Avoid
- **Refactoring register_modules to nn.ModuleList:** Changes state dict keys from `down_conv_block_0.*` to `down_conv_blocks.0.*`. Breaks checkpoint loading.
- **Removing the up_list as plain Python list:** nn.Upsample has no parameters. Registering it would add useless entries to state dict. Keep as-is.
- **Removing the squeeze(2)/unsqueeze(2) from Unet2d:** This was added intentionally (commit `0e2b575`) to normalize the 2D model to accept 5D input (BCZYX) matching the 2.5D/3D models.
- **Running the full 144-configuration Cartesian product in pytest:** This creates 432+ slow tests. Use representative subsets with `@pytest.mark.parametrize`.
- **Importing from viscy.utils or any monolithic viscy path:** viscy-models is standalone.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 2D conv blocks | Custom conv+norm+act | `ConvBlock2D` from `unet._layers` | Already migrated, tested, state-dict-verified |
| 3D conv blocks | Custom conv+norm+act | `ConvBlock3D` from `unet._layers` | Already migrated, tested, state-dict-verified |
| Module registration | nn.ModuleList | `register_modules` + `add_module` | Must match legacy state dict key format |
| Test progress bars | tqdm or custom | Remove entirely | Tests should use pytest output, not custom progress bars |

**Key insight:** Unlike UNeXt2/FCMAE/ContrastiveEncoder/VAE, the legacy UNets do NOT use `_components/` at all. They are self-contained models that only depend on their respective ConvBlock layers. This makes the migration simpler.

## Common Pitfalls

### Pitfall 1: Unet2d Test Shape Mismatch (Known Bug in Original Tests)
**What goes wrong:** Original test expected shapes assume 4D output `(B, C, H, W)` but after commit `0e2b575`, Unet2d returns 5D `(B, C, 1, H, W)` due to `unsqueeze(2)`.
**Why it happens:** The `squeeze(2)/unsqueeze(2)` normalization was added after the tests were written, and the test expected shapes were never updated.
**How to avoid:** Write new pytest tests with correct 5D expected shapes: `(B, C, 1, H, W)` for 4D input (Z=1 gets squeezed then unsqueezed) and `(B, C, 1, H, W)` for 5D input with Z=1 squeezed dimension.
**Verification:**
```python
# 4D input: (1, 1, 256, 256) -> squeeze(2) does nothing (dim 2 = 256, not 1)
# -> conv processing -> unsqueeze(2) -> (1, 1, 1, 256, 256)
# 5D input: (1, 1, 1, 256, 256) -> squeeze(2) -> (1, 1, 256, 256)
# -> conv processing -> unsqueeze(2) -> (1, 1, 1, 256, 256)
```

### Pitfall 2: Original Test Dependency on viscy.utils.cli_utils
**What goes wrong:** Original tests import `viscy.utils.cli_utils.show_progress_bar` which does not exist in viscy-models.
**Why it happens:** Legacy test utility for progress bars during exhaustive configuration testing.
**How to avoid:** Do not port the cli_utils dependency. Write new pytest-native tests that do not need progress bar utilities.

### Pitfall 3: Cartesian Product Test Explosion
**What goes wrong:** Original tests run 144 configurations x 3 tests = 432 forward passes per model, taking several minutes.
**Why it happens:** Exhaustive testing via `itertools.product` of all parameter combinations.
**How to avoid:** Use `@pytest.mark.parametrize` with a representative subset that covers the important axes of variation:
- `num_blocks`: 1 (minimum), 4 (standard)
- `residual`: True, False
- `task`: "reg", "seg"
- `kernel_size`: (3, 3) (standard)
- `dropout`: 0.0 (off), 0.25 (on)
- Multi-channel I/O (single test)

This gives ~8-12 well-chosen test cases instead of 144, covering all meaningful code paths.

### Pitfall 4: Unet25d skip_conv_layers Dimension Coupling
**What goes wrong:** The `skip_conv_layers` in Unet25d use `kernel_size=(1 + in_stack_depth - out_stack_depth, 1, 1)`, tightly coupling the Z-compression to the depth parameters. Using wrong depth values produces dimension errors.
**Why it happens:** The 2.5D architecture intentionally compresses Z in both the bottom transition and skip connections.
**How to avoid:** Tests must verify the standard case (in=5, out=1) and at least one case where in_depth equals out_depth (kernel_z=1, Z preserved).

### Pitfall 5: State Dict Key Patterns Differ Between Unet2d and Unet25d
**What goes wrong:** Assuming both models have identical state dict structure.
**Why it happens:** Similar architecture but key differences:
- Unet2d: Uses `ConvBlock2D`, `nn.AvgPool2d`, `nn.Upsample(mode='bilinear')`. No skip conv layers.
- Unet25d: Uses `ConvBlock3D`, `nn.AvgPool3d`, `nn.Upsample(mode='trilinear')`, and has additional `skip_conv_layer_N` modules.
**How to avoid:** Test state dict key prefixes independently for each model. Unet25d should have `skip_conv_layer_0`, Unet2d should not.

### Pitfall 6: ConvBlock2D Dropout Registration Asymmetry
**What goes wrong:** Tests pass with dropout=0 but fail with dropout>0 because of a subtle difference.
**Why it happens:** ConvBlock2D does NOT register dropout modules via `register_modules` (just stores them in `self.drop_list`), while ConvBlock3D DOES register them. This means ConvBlock2D dropout won't be in the state dict. This is the original behavior and must be preserved.
**How to avoid:** Don't test for dropout module registration in Unet2d state dict. Only test it for Unet25d (which uses ConvBlock3D that does register dropout).

## Code Examples

### Unet2d Migration (Complete)
```python
# packages/viscy-models/src/viscy_models/unet/unet2d.py
"""2D UNet with variable depth and configurable convolutional blocks."""

import torch
import torch.nn as nn

from viscy_models.unet._layers.conv_block_2d import ConvBlock2D

__all__ = ["Unet2d"]


class Unet2d(nn.Module):
    # ... copy verbatim from v0.3.3 ...
    # ONLY changes:
    # 1. Import path (above)
    # 2. num_filters=[] -> num_filters=()
    # 3. Add __all__ and module docstring
```

### Unet25d Migration (Complete)
```python
# packages/viscy-models/src/viscy_models/unet/unet25d.py
"""2.5D UNet that learns 3D-to-2D compression for virtual staining."""

import torch
import torch.nn as nn

from viscy_models.unet._layers.conv_block_3d import ConvBlock3D

__all__ = ["Unet25d"]


class Unet25d(nn.Module):
    # ... copy verbatim from v0.3.3 ...
    # ONLY changes:
    # 1. Import path (above)
    # 2. num_filters=[] -> num_filters=()
    # 3. Add __all__ and module docstring
```

### Unet2d State Dict Key Pattern
```python
# For Unet2d(in_channels=1, out_channels=1, num_blocks=2):
# Expected state dict key prefixes:
#   down_samp_0          (nn.AvgPool2d - no params, not in state dict)
#   down_samp_1          (nn.AvgPool2d - no params, not in state dict)
#   down_conv_block_0.*  (ConvBlock2D)
#   down_conv_block_1.*  (ConvBlock2D)
#   bottom_transition_block.*  (ConvBlock2D)
#   up_conv_block_0.*    (ConvBlock2D)
#   up_conv_block_1.*    (ConvBlock2D)
#   terminal_block.*     (ConvBlock2D)
```

### Unet25d State Dict Key Pattern
```python
# For Unet25d(in_channels=1, out_channels=1, num_blocks=2):
# Expected state dict key prefixes (includes everything above PLUS):
#   skip_conv_layer_0.*  (nn.Conv3d)
#   skip_conv_layer_1.*  (nn.Conv3d)
#   bottom_transition_block.*  (nn.Conv3d when bottom_block_spatial=False)
```

### Pytest Test Pattern (Unet2d)
```python
# packages/viscy-models/tests/test_unet/test_unet2d.py
"""Forward-pass tests for Unet2d covering representative configurations."""

import pytest
import torch

from viscy_models.unet import Unet2d


def test_unet2d_default_forward():
    """Default Unet2d: 1ch in/out, num_blocks=4, output shape check."""
    model = Unet2d(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 1, 256, 256)  # 5D input (B,C,Z=1,H,W)
    with torch.no_grad():
        out = model(x)
    # squeeze(2) removes Z=1, conv processes 4D, unsqueeze(2) adds it back
    assert out.shape == (1, 1, 1, 256, 256)


@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_unet2d_variable_depth(num_blocks):
    """Test different encoder/decoder depths."""
    model = Unet2d(in_channels=1, out_channels=1, num_blocks=num_blocks)
    x = torch.randn(1, 1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)


def test_unet2d_state_dict_keys():
    """Verify state dict key prefixes match legacy checkpoint format."""
    model = Unet2d(in_channels=1, out_channels=1, num_blocks=2)
    keys = list(model.state_dict().keys())
    # Verify the add_module naming pattern
    assert any("down_conv_block_0." in k for k in keys)
    assert any("down_conv_block_1." in k for k in keys)
    assert any("bottom_transition_block." in k for k in keys)
    assert any("up_conv_block_0." in k for k in keys)
    assert any("terminal_block." in k for k in keys)
    # down_samp uses AvgPool2d (no params) -- should NOT be in state dict
    assert not any("down_samp" in k for k in keys)
```

### Pytest Test Pattern (Unet25d)
```python
# packages/viscy-models/tests/test_unet/test_unet25d.py
"""Forward-pass tests for Unet25d covering representative configurations."""

import pytest
import torch

from viscy_models.unet import Unet25d


def test_unet25d_default_forward():
    """Default Unet25d: 1ch in/out, depth 5->1, output shape check."""
    model = Unet25d(in_channels=1, out_channels=1, in_stack_depth=5, out_stack_depth=1)
    x = torch.randn(1, 1, 5, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 1, 64, 64)


def test_unet25d_preserved_depth():
    """Unet25d with in_depth == out_depth: Z dimension preserved."""
    model = Unet25d(in_channels=1, out_channels=1, in_stack_depth=5, out_stack_depth=5)
    x = torch.randn(1, 1, 5, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 5, 64, 64)


def test_unet25d_state_dict_has_skip_conv():
    """Unet25d has skip_conv_layer modules (unlike Unet2d)."""
    model = Unet25d(in_channels=1, out_channels=1, num_blocks=2)
    keys = list(model.state_dict().keys())
    assert any("skip_conv_layer_0." in k for k in keys)
    assert any("skip_conv_layer_1." in k for k in keys)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| unittest.TestCase + setUp | pytest functions + fixtures | Phase 7 (2026-02-12) | Consistent with all viscy-models tests |
| Cartesian product exhaustive tests | Representative parametrize | Phase 9 (current) | 144 configs -> ~10 focused tests per model |
| PascalCase filenames (Unet2D.py) | snake_case (unet2d.py) | Convention from Phase 6 | Consistent with ruff and Python conventions |
| `num_filters=[]` mutable default | `num_filters=()` tuple default | COMPAT-02 from Phase 6 research | Prevents shared state bugs |
| `from viscy.unet.networks...` | `from viscy_models.unet._layers...` | Phase 6 migration | Clean package boundary |

**Deprecated/outdated:**
- `viscy.utils.cli_utils.show_progress_bar`: Not available in viscy-models; replaced by pytest native output
- Original unittest test structure: Replaced by idiomatic pytest with parametrize

## Open Questions

1. **Should we preserve the `validate_input` parameter in Unet2d.forward?**
   - What we know: Unet2d.forward has `validate_input=False` parameter that enables input shape assertions. Unet25d.forward does not.
   - What's unclear: Whether downstream code relies on the `validate_input` parameter.
   - Recommendation: Preserve it exactly as-is. It's part of the public API signature.

2. **Should we preserve the `__name__` method on both classes?**
   - What we know: Both classes define `def __name__(self): return "Unet2d"` / `"Unet25d"`. This overrides the dunder protocol (normally `__name__` is a class attribute, not an instance method).
   - What's unclear: Whether any code calls `model.__name__()`.
   - Recommendation: Preserve it. Removing it would change the public API. It's harmless.

3. **Should we test the `task="reg"` vs `task="seg"` terminal block difference?**
   - What we know: When `task="reg"`, the terminal block uses `activation="linear"`. When `task="seg"`, it uses `activation="relu"`.
   - Recommendation: Yes, test both via `@pytest.mark.parametrize("task", ["reg", "seg"])`. This is a meaningful behavioral difference.

4. **Should we test residual mode?**
   - What we know: `residual=True/False` changes the ConvBlock behavior (skip connection vs. no skip). The original tests explicitly verified residual mode doesn't add extra parameters.
   - Recommendation: Yes, test both via parametrize. Include a parameter count comparison test if desired.

## Sources

### Primary (HIGH confidence)
- **v0.3.3 tag -- `viscy/unet/networks/Unet2D.py`** -- Complete Unet2d source (169 lines)
- **v0.3.3 tag -- `viscy/unet/networks/Unet25D.py`** -- Complete Unet25d source (214 lines)
- **v0.3.3 tag -- `tests/unet/networks/Unet2D_tests.py`** -- Original unittest tests (175 lines)
- **v0.3.3 tag -- `tests/unet/networks/Unet25D_tests.py`** -- Original unittest tests (195 lines)
- **Commit `0e2b575`** -- "normalize shape for 2D nets" -- Added squeeze(2)/unsqueeze(2) to Unet2d.forward
- **Phase 6 migrated layers** -- `packages/viscy-models/src/viscy_models/unet/_layers/conv_block_2d.py` and `conv_block_3d.py` (already tested, 10 layer tests passing)
- **Existing test suite** -- 46 tests across 10 test files, all passing

### Secondary (MEDIUM confidence)
- **Phase 6 RESEARCH.md** -- Mutable default inventory, state dict key preservation patterns, architecture structure

### Tertiary (LOW confidence)
- None. All findings verified against source code.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Both models use only torch and ConvBlock layers (already migrated)
- Architecture: HIGH -- Complete source analysis of both models; all code paths traced
- Pitfalls: HIGH -- Shape mismatch bug verified by commit history; test dependency identified; state dict patterns verified empirically
- Test conversion: HIGH -- Original test structure fully analyzed; clear path to pytest conversion

**Research date:** 2026-02-13
**Valid until:** 2026-03-13 (stable domain; no external dependency changes expected)
