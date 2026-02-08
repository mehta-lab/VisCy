# Codebase Concerns

**Analysis Date:** 2026-02-07

## Tech Debt

**PyTorch quantile performance regression (pytorch#64947):**
- Issue: `torch.quantile()` in `BatchedScaleIntensityRangePercentiles._normalize()` has known performance limitations in PyTorch
- Files: `packages/viscy-transforms/src/viscy_transforms/_percentile_scale.py:63`
- Impact: Percentile-based intensity scaling operations are slower than they could be during preprocessing pipelines
- Fix approach: Monitor PyTorch releases for fixes; consider alternative quantile computation methods or vendor-specific CUDA kernels if this becomes a bottleneck

**Missing documentation site placeholder:**
- Issue: Documentation URL commented out in pyproject.toml with TODO note
- Files: `packages/viscy-transforms/pyproject.toml:48`
- Impact: Users cannot access full documentation; unclear where to find API reference and usage guides
- Fix approach: Set up documentation site (Sphinx/MkDocs) and update URL in configuration after monorepo migration completes

**MONAI wrapper constructors require **kwargs:**
- Issue: MONAI transforms use `**kwargs` in constructors, preventing jsonargparse introspection for config-driven pipelines
- Files: `packages/viscy-transforms/src/viscy_transforms/_monai_wrappers.py` (all wrapper classes)
- Impact: Custom wrappers must re-declare all parameters explicitly; adds maintenance burden when MONAI updates parameters
- Fix approach: Continue wrapping pattern; consider auto-generating wrappers from MONAI source if parameter churn becomes problematic

## Known Limitations

**3D-only support for batched spatial cropping:**
- Limitation: `BatchedRandSpatialCrop` explicitly rejects 2D data
- Files: `packages/viscy-transforms/src/viscy_transforms/_crop.py:94-96`
- Scope: Only affects users processing 2D microscopy images
- Workaround: Use MONAI's standard `RandSpatialCrop` for 2D data; VisCy transforms optimized for 3D volumetric data
- Rationale: Batched implementation uses 3D unfold operations; 2D support requires separate code path

**Batched spatial crop does not support random size:**
- Limitation: `BatchedRandSpatialCrop` parameter `random_size` must be False
- Files: `packages/viscy-transforms/src/viscy_transforms/_crop.py:46-48`
- Impact: All crops in batch must use same fixed size to ensure consistent tensor shapes
- Workaround: Pre-process data to consistent dimensions or use MONAI's standard `RandSpatialCrop` for variable sizes
- Rationale: Batched tensor operations require uniform output shapes; supporting variable sizes would complicate GPU memory management

**Limited support for arbitrary tensor dimensions:**
- Limitation: Most transforms assume (B, C, D, H, W) format for 3D or (B, C, H, W) for 2D
- Files: Throughout `packages/viscy-transforms/src/viscy_transforms/`
- Impact: Users with non-standard dimensions (e.g., time-series with T dimension at different position) must reshape data
- Scope: Core microscopy use case; edge cases not tested

## Performance Bottlenecks

**Memory allocation in BatchedRandFlip:**
- Problem: `torch.zeros_like(data)` allocates full output tensor upfront, then selectively copies/flips elements
- Files: `packages/viscy-transforms/src/viscy_transforms/_flip.py:38-50`
- Cause: Implementation pre-allocates all memory rather than modifying in-place
- Observation: Comment notes one-by-one copying is "slightly faster than vectorized indexing"; optimization already applied
- Improvement path: Consider in-place flip operations for unchanging samples; measure if worthwhile relative to readability

**PercentileScale reshape overhead:**
- Problem: `torch.quantile()` reshape pattern creates intermediate tensors with shape (2, batch_size, 1, 1, 1, 1)
- Files: `packages/viscy-transforms/src/viscy_transforms/_percentile_scale.py:64-68`
- Cause: Need to broadcast quantiles per-sample; reshape adds memory overhead
- Impact: Minimal for typical batch sizes; becomes noticeable at very large batches (>512)
- Improvement path: Use einsum or explicit dimension expansion to reduce reshape overhead

**Device mismatch assertions instead of graceful handling:**
- Problem: `BatchedRandAffined` uses `assert d[key].device == data.device` for validation
- Files: `packages/viscy-transforms/src/viscy_transforms/_affine.py:130`
- Cause: Assertions are disabled with `-O` flag; should use explicit error checking
- Impact: Silent failures in optimized Python execution; assert removed with `python -O`
- Fix approach: Replace with explicit `if` check raising `RuntimeError` with helpful message

**Exceptional retry logic without limits:**
- Problem: `BatchedRandAffined.__call__()` retries on `RuntimeError` without retry count limit
- Files: `packages/viscy-transforms/src/viscy_transforms/_affine.py:125-129`
- Cause: Kornia's RandomAffine3D occasionally fails with device memory issues
- Risk: Infinite loop if error persists; no logging of retry events
- Fix approach: Add `max_retries` parameter (default 3), log warnings on retry, raise if all attempts fail

## Fragile Areas

**Kernel caching in BatchedRandSharpend:**
- Files: `packages/viscy-transforms/src/viscy_transforms/_sharpen.py:65-82`
- Why fragile: Device switching invalidates cached kernel; depends on exact device equality check
- Safe modification: Add unit tests for multi-device scenarios; document that kernel recomputed on device change
- Test coverage: No current tests for GPU device switching or mixed-device batches

**Parameter coordinate axis ordering conversions:**
- Files: `packages/viscy-transforms/src/viscy_transforms/_affine.py:91-106` (MONAI ZYX → Kornia XYZ)
- Why fragile: Axis order conversions are implicit; easy to introduce bugs when refactoring coordinate math
- Safe modification: Add comprehensive docstring examples; add round-trip unit tests verifying conversion symmetry
- Test coverage: Basic tests exist; need matrix of rotation + shear combinations

**Tensor contiguity assumptions in cropping:**
- Files: `packages/viscy-transforms/src/viscy_transforms/_crop.py:112,121`
- Why fragile: `unfold()` + indexing requires contiguous memory layout; non-contiguous inputs will fail
- Safe modification: Add explicit contiguity checks with informative errors; document in docstring
- Test coverage: Only tested with fresh `torch.rand()` tensors (always contiguous)

## Test Coverage Gaps

**2D data handling not tested:**
- What's not tested: 2D variants of 3D-only transforms
- Files: `packages/viscy-transforms/src/viscy_transforms/_crop.py`, `_sharpen.py`, `_elastic.py`
- Risk: Potential ValueError exceptions not caught before user encounters them
- Priority: Medium (documented limitation; users should read docs)

**Device switching and multi-GPU scenarios:**
- What's not tested: Kernel caching across GPU changes, mixed-device batches
- Files: `packages/viscy-transforms/src/viscy_transforms/_sharpen.py`
- Risk: Silent correctness issues if cached kernel computed on wrong device
- Priority: Medium (primarily affects advanced multi-GPU users)

**Exceptional error paths in affine transforms:**
- What's not tested: Kornia RandomAffine3D failure modes, retry logic limits
- Files: `packages/viscy-transforms/src/viscy_transforms/_affine.py`
- Risk: Silent infinite loops on persistent GPU memory errors
- Priority: High (affects reproducibility and debugging)

**Large batch memory behavior:**
- What's not tested: OOM scenarios with very large batches (>512); memory growth patterns
- Files: `packages/viscy-transforms/src/viscy_transforms/` (all transforms)
- Risk: Unpredictable OOM kills rather than graceful degradation
- Priority: Low (users should profile; memory management is PyTorch's responsibility)

## Architectural Concerns

**Monorepo initialization incomplete:**
- Issue: Main `src/viscy/__init__.py` is empty placeholder
- Files: `src/viscy/__init__.py`
- Impact: Cannot import from top-level `viscy`; requires importing from subpackages directly
- Status: README indicates "More packages coming soon"; likely intentional
- Fix approach: Add re-exports when first subpackage completes (planned `viscy-data`, `viscy-models`)

**Pre-commit hook configuration exists but not enforced:**
- Issue: `.pre-commit-config.yaml` defines hooks but contributing guide mentions `prek` (faster runner) as optional
- Files: `.pre-commit-config.yaml`, `CONTRIBUTING.md`
- Impact: Code style inconsistency across contributors; inconsistent enforcement on CI vs local
- Recommendation: Document hook installation as required step; add CI check to ensure all commits are formatted

## Dependencies at Risk

**MONAI version pinning (>=1.5.2):**
- Risk: MONAI releases new versions frequently with API changes; pinning to >= allows broad range
- Impact: User environment may have incompatible MONAI if locked dependencies update
- Mitigation: uv.lock file pins exact versions; users with `pip install` may encounter breakage
- Recommendation: Add upper bound constraint (e.g., `>=1.5.2,<2.0`) for safer range

**Kornia experimental APIs:**
- Risk: `kornia.augmentation.RandomAffine3D` may not be stable API
- Impact: Future Kornia versions could break `BatchedRandAffined`
- Mitigation: Explicitly use `@torch.no_grad()` to avoid accidental gradient computation
- Recommendation: Monitor Kornia releases; add deprecation warning if API changes detected

## Security Considerations

**No input validation for shape constraints:**
- Risk: Transform assumptions about tensor shapes (e.g., 5D for 3D + batch + channel) not validated
- Files: All transform `__call__` methods
- Current mitigation: Exceptions raised by downstream operations (e.g., unfold), but error messages unclear
- Recommendation: Add explicit shape validation early in `__call__` with descriptive errors

**Device transfers not protected:**
- Risk: `ToDeviced` transfers data to device without validation of device availability
- Files: `packages/viscy-transforms/src/viscy_transforms/_monai_wrappers.py:112-135`
- Current mitigation: PyTorch raises exception if device invalid; user loses work
- Recommendation: Check device availability before transfer; log device allocation

## Missing Critical Features

**No built-in data parallelization utilities:**
- Problem: Large 3D volumes require custom data loading strategies; transforms assume pre-batched data
- Blocks: Multi-GPU training without external wrapper libraries
- Status: Likely addressed in `viscy-data` package (not yet released)

**Limited channel normalization per-timepoint:**
- Problem: Commit 44b25b9 indicates normalization per-timepoint was just added; may still be incomplete
- Blocks: Proper handling of temporal variations in microscopy time-series
- Fix approach: Test comprehensive normalization modes (global, per-frame, per-channel-per-frame)

---

*Concerns audit: 2026-02-07*
