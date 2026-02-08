# Architecture

**Analysis Date:** 2026-02-07

## Pattern Overview

**Overall:** Modular monorepo with layered separation of concerns. VisCy uses a uv workspace structure with independent packages. The core pattern is **Transform-based pipelines** inspired by MONAI (Medical Open Network for AI), allowing composable image preprocessing and augmentation.

**Key Characteristics:**
- **Workspace monorepo:** Root `pyproject.toml` declares workspace members in `packages/`. Each package is independently installable.
- **Transform composition:** Transforms follow MONAI's dictionary-based pipeline pattern where data flows through a series of operations defined in configuration.
- **Batched GPU optimization:** Custom batched transforms operate on tensor batches (B, C, D, H, W) with per-sample randomization on GPU, avoiding CPU-GPU transfers.
- **Type-safe transforms:** Explicit type hints and Pydantic-compatible wrappers enable config-driven pipelines (jsonargparse/LightningCLI).

## Layers

**Package Layer (Workspace):**
- Purpose: Organize related functionality into independently-versioned packages
- Location: `packages/` root directory
- Contains: Multiple specialized packages (transforms, future: data, models, airtable)
- Depends on: Python 3.11+, hatchling for building
- Used by: External projects via PyPI or local development via `uv sync`

**Transform Package (`viscy-transforms`):**
- Purpose: GPU-accelerated image transforms for microscopy preprocessing and augmentation
- Location: `/packages/viscy-transforms/src/viscy_transforms/`
- Contains: Transform classes, type definitions, MONAI wrappers, utilities
- Depends on: PyTorch, Kornia (GPU kernels), MONAI (transform base classes), NumPy
- Used by: Training pipelines, data loading, virtual staining applications

**Core Umbrella Package (`viscy`):**
- Purpose: Virtual staining umbrella package that coordinates subpackages
- Location: `/src/viscy/`
- Contains: Version metadata, minimal re-exports
- Depends on: `viscy-transforms` as primary dependency
- Used by: End-users installing from PyPI

## Data Flow

**Transform Pipeline Architecture:**

1. **Input:** Raw microscopy image data as dictionary (Sample type) or tensor
2. **Transform Chain:** User-defined sequence of transforms from `viscy_transforms`
   - Example: `[Decollate, StackChannelsd, NormalizeSampled, BatchedRandAffined, ToDeviced]`
3. **Per-Sample Randomization:** Batched transforms randomize once per batch, apply per-sample (see `BatchedRandFlip.randomize()`)
4. **Output:** Preprocessed data ready for model training or inference

**Data Structure (Sample TypedDict):**
```
Sample = {
    "index": HCSStackIndex,        # Plate/well/timepoint identifier
    "source": Tensor | list[Tensor],  # Input image(s)
    "target": Tensor | list[Tensor],  # Ground truth/target image(s)
    "weight": Tensor | list[Tensor],  # Per-sample loss weights
    "labels": Tensor | list[Tensor],  # Instance segmentation masks
    "norm_meta": NormMeta | None       # Normalization statistics
}
```

**Normalization Metadata Flow:**
- Precomputed statistics stored in `norm_meta` dict (FOV-level + dataset-level per channel)
- `NormalizeSampled` reads from `norm_meta`, applies (x - mean) / std normalization
- Statistics can be computed separately using external tools (e.g., preprocessing pipelines)

**State Management:**
- Transforms are **stateless** except for `RandomizableTransform` instances
- Randomization state stored temporarily in transform instance attributes (`self._do_transform`, `self._batch_slices`, etc.)
- For batched transforms, randomization occurs once per batch call, then applied per-sample

## Key Abstractions

**Transform (Base Pattern):**
- Purpose: Define repeatable operations on image data (forward-compatible with MONAI)
- Examples: `BatchedRandFlip`, `NormalizeSampled`, `StackChannelsd`
- Pattern:
  - Non-dictionary transforms inherit from `RandomizableTransform` or `Transform`
  - Dictionary transforms (suffix `d`) inherit from `MapTransform`, operate on keys
  - Randomizable transforms implement `randomize()` method
- Location: Each in `/packages/viscy-transforms/src/viscy_transforms/_<transform_name>.py`

**Dictionary Transform Pattern (MapTransform):**
- Purpose: Apply transforms to specified dictionary keys, enabling multi-stream processing
- Pattern: Initialize with `keys` parameter, iterate via `self.key_iterator(data)`
- Example: `BatchedRandFlipd` applies `BatchedRandFlip` to multiple keys independently but with synchronized randomization
- Location: All transforms have paired versions (e.g., `BatchedRandFlip` → `BatchedRandFlipd`)

**Batched Transform Pattern:**
- Purpose: Optimize GPU performance by batch-processing with per-sample randomization
- Pattern: Randomize once using first element's shape, apply per-sample via vectorized indexing
- Example in `BatchedRandSpatialCrop`:
  ```python
  def randomize(self, img_size):  # Called once per batch
      self._batch_slices = []
      for _ in range(img_size[0]):  # For each sample in batch
          super().randomize(spatial_size)  # Generate random params
          self._batch_slices.append(self._slices)

  def __call__(self, img):  # Apply all-at-once
      windows = img.unfold(...)  # Vectorized indexing
      return windows[batch_indices, :, ...]  # Extract per-sample crops
  ```

**Type Definitions (Typing Module):**
- Purpose: Define data structures for type-safe transforms and config validation
- Examples: `Sample`, `NormMeta`, `HCSStackIndex`, `ChannelMap`, `OneOrSeq`
- Location: `viscy_transforms/_typing.py`
- Used by: MONAI MapTransform implementations, jsonargparse config parsing

**MONAI Wrappers:**
- Purpose: Re-export MONAI transforms with explicit constructor signatures for config introspection
- Pattern: Inherit from MONAI transform, declare all parameters explicitly to expose to jsonargparse
- Examples: `RandAffined`, `RandFlipd`, `Decollated`, `ToDeviced`
- Location: `viscy_transforms/_monai_wrappers.py`
- Why needed: MONAI uses `**kwargs`, preventing tools like LightningCLI from introspecting parameters

## Entry Points

**Package Installation Entry:**
- Location: `packages/viscy-transforms/pyproject.toml`
- Triggers: `pip install viscy-transforms` or `uv sync`
- Responsibilities: Installs transform library with dependencies (torch, monai, kornia)

**Module Import Entry:**
- Location: `viscy_transforms/__init__.py`
- Triggers: `from viscy_transforms import BatchedRandFlip` etc.
- Responsibilities: Export public API (28+ transform classes)

**Type Entry:**
- Location: `viscy_transforms/_typing.py`
- Triggers: Import of data type definitions
- Responsibilities: Provide TypedDict definitions for IDE autocomplete and type checking

## Error Handling

**Strategy:** Fail-fast with explicit validation in transform initialization and execution.

**Patterns:**
- **Parameter validation:** Constructor checks for conflicting parameters (e.g., `BatchedRandSpatialCrop` rejects `random_size=True`)
- **Shape validation:** `__call__` methods verify input tensor shapes (e.g., 5D for batched transforms)
- **Division by zero prevention:** Add epsilon (1e-8) in normalization to avoid NaN (see `NormalizeSampled._normalize`)
- **Device consistency:** Transforms preserve input device (e.g., CPU tensors stay CPU, GPU tensors stay GPU)
- **Missing key handling:** Dictionary transforms respect `allow_missing_keys` parameter to gracefully skip absent keys

## Cross-Cutting Concerns

**Logging:** Not implemented in transforms themselves. Expected to be handled by training framework (Lightning, Hydra).

**Validation:** Two levels:
1. **Shape validation:** Most transforms check spatial dimensions (e.g., 3D vs 5D)
2. **Type validation:** Type hints used for IDE support; runtime validation via MONAI parent classes

**Device Management:** Transforms operate in-place on device:
- `ToDeviced` moves data to specified device (GPU/CPU)
- Other transforms preserve input device and avoid unnecessary transfers
- GPU kernels (Kornia) only invoked for transforms with GPU acceleration (`BatchedZoom`, `BatchedRandAffined`)

**Randomization:**
- Controlled via `prob` parameter (0.0-1.0)
- For batched transforms, per-sample randomization matrix shape (B, num_parameters)
- Seeding via `torch.manual_seed()` (test suite example in `conftest.py`)

---

*Architecture analysis: 2026-02-07*
