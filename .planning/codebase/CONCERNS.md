# Codebase Concerns

**Analysis Date:** 2025-03-27

## Tech Debt

### Unfinished UNet Implementations

**Issue:** Multiple upsampling modes in UNet architectures are intentionally stubbed out.

**Files:**
- `packages/viscy-models/src/viscy_models/unet/unet2d.py` (lines 68-71, 97-111)
- `packages/viscy-models/src/viscy_models/unet/unet25d.py` (lines 79-83, 110, 135)

**Impact:** Code paths for "conv" and "tconv" upsampling modes raise `NotImplementedError` at runtime. Users attempting to use these modes will encounter failures. The hardcoded defaults (avgpool, bilinear) work but are not configurable despite being marked as TODO.

**Fix approach:** Either implement the missing modes, remove the enum options, or make the defaults truly static and remove the TODO markers.

### Unimplemented Metric Counting in cytoland

**Issue:** Metrics calculation does not handle variable num_outputs.

**Files:**
- `applications/cytoland/src/cytoland/engine.py` (line 140)

**Impact:** The `VSUNet` module has commented-out code attempting to detect output channels from the terminal block. Currently num_outputs is hardcoded in metric computations, which may break with multi-task heads.

**Fix approach:** Refactor metrics to introspect model outputs and adapt dynamically, or document the single-output constraint explicitly.

### Hardcoded Static Parameters in UNet

**Issue:** UNet models have multiple parameters marked as TODO to become static.

**Files:**
- `packages/viscy-models/src/viscy_models/unet/unet2d.py` (lines 68-71)
- `packages/viscy-models/src/viscy_models/unet/unet25d.py` (lines 79-82)

**Impact:** Code is confusing about what is configurable vs. hardcoded. The "TODO set static" comments suggest these should be configuration options but are currently hardcoded as module-level choices with no mechanism to override them.

**Fix approach:** Decide whether these should be configurable parameters or remove the TODOs and document them as fixed choices.

## Known Issues

### Triplet Dataset Border Clamping Not Validated

**Issue:** Future timepoints are not validated to exist after sampling interval.

**Files:**
- `packages/viscy-data/src/viscy_data/triplet.py` (line 168)

**Impact:** When `time_interval` is specified, the code filters anchors but does not guarantee that the anchor + interval timepoint actually has matching track data. Silent exclusion of valid anchors could lead to unbalanced or smaller-than-expected datasets with no warning.

**Fix approach:** Add validation that checks for the next timepoint's existence, or add logging to report how many anchors were filtered out.

### HCS Module Weight Map Uses Only First Target Channel

**Issue:** Weight computation for segmentation is delegated to the first target channel only.

**Files:**
- `packages/viscy-data/src/viscy_data/hcs.py` (line 169)

**Impact:** Multi-target segmentation tasks only use one channel's intensity for weighting, ignoring other target channels. This may produce biased loss weighting in multi-channel segmentation tasks.

**Fix approach:** Either compute composite weights from all target channels (e.g., max, mean, sum) or make the weight channel configurable.

### cytoland Test Metrics Assume Batch Size 1

**Issue:** Metric logging in test step only works for batch size 1.

**Files:**
- `applications/cytoland/src/cytoland/engine.py` (line 274)

**Impact:** The code explicitly indexes `pred[0]` and `pred_labels[0]`, assuming single-sample batches. Running inference with batch_size > 1 will compute metrics for only the first sample, silently dropping others.

**Fix approach:** Refactor to handle arbitrary batch sizes or raise an error if batch_size > 1 is detected.

## Resource Management Issues

### File Handles Not Closed in Linear Classifier Utils

**Issue:** OME-Zarr plate handles are opened but not consistently closed.

**Files:**
- `applications/dynaclr/src/dynaclr/evaluation/linear_classifiers/utils.py` (lines 279, 333)

**Impact:** Resource leaks. The `plate` object is opened in `get_z_range()` and `_compute_focus()` but only closed on some code paths (error cases). Success paths may leave file handles open, especially at line 333 where the function returns immediately after reading, leaving the handle open.

**Fix approach:** Wrap all `open_ome_zarr()` calls in context managers (`with` statements).

### Zarr Store Cache Not Cleaned Up

**Issue:** Zarr plate handles are cached indefinitely without explicit cleanup.

**Files:**
- `applications/dynaclr/src/dynaclr/data/index.py` (line 356)

**Impact:** The `_store_cache` dictionary in `MultiExperimentIndex` accumulates open file handles for the lifetime of the index object. In long-running processes or when loading many experiments, this can exhaust file handle limits on the system.

**Fix approach:** Implement a destructor (`__del__`) to close all cached plates, or use weak references with cleanup callbacks. Alternatively, add an explicit `close()` method and document that it must be called.

### Broad Exception Handling in Dimensionality Reduction

**Issue:** PHATE and PCA computation failures are caught and swallowed.

**Files:**
- `packages/viscy-utils/src/viscy_utils/callbacks/embedding_writer.py` (lines 190-191, 200-202)

**Impact:** When PHATE or PCA computation fails (e.g., due to numerical issues, insufficient data), the error is logged as a warning but execution continues. Users may not realize dimensionality reductions are missing from their output.

**Fix approach:** Either let these errors propagate and fail fast (preferred for reproducibility), or require explicit opt-in with validation that datasets are suitable for PHATE/PCA before attempting computation.

### Exception Swallowing in Visualization Engine

**Issue:** Multiple broad `except Exception` clauses in the visualization app.

**Files:**
- `packages/viscy-utils/src/viscy_utils/evaluation/visualization.py` (lines 791, 987, 1543, 1595, 1667, 1678, 1684, 2194, 2240)

**Impact:** Errors in interactive visualization callback handlers are caught and logged but execution continues. UI state can become inconsistent, and users see no clear error message about what failed.

**Fix approach:** Catch only specific exceptions (e.g., `ValueError`, `KeyError`) or let errors propagate to the Dash error boundary. Add user-facing error notifications for expected failure modes.

## Performance Concerns

### Large Complex Files

**Issue:** Several files exceed 700 lines, mixing multiple responsibilities.

**Files:**
- `packages/viscy-utils/src/viscy_utils/evaluation/visualization.py` (2244 lines)
- `applications/dynaclr/tests/test_index.py` (1173 lines)
- `applications/dynaclr/examples/demos/infection_analysis/utils.py` (1171 lines)
- `applications/dynaclr/scripts/pseudotime/infection_onset_distribution.py` (1028 lines)
- `applications/dynaclr/scripts/workflow/workflow.py` (944 lines)
- `applications/cytoland/src/cytoland/engine.py` (900 lines)

**Impact:** High cognitive load, difficult to test individual functions, increased merge conflict risk, harder to navigate and understand code flow.

**Fix approach:** Break into smaller modules with clear separation of concerns. Extract test utilities into fixtures. Extract visualization logic into separate classes.

### Percentile Scale Transform Performance TODO

**Issue:** PyTorch issue #64947 affects percentile computation performance.

**Files:**
- `packages/viscy-transforms/src/viscy_transforms/_percentile_scale.py` (line 63)

**Impact:** Performance degradation in percentile-based normalization on certain PyTorch versions. Workaround is noted but not implemented.

**Fix approach:** Monitor PyTorch issue resolution and update implementation when the underlying bug is fixed.

## Fragile Areas

### CLI Config Parsing with SystemExit Catching

**Issue:** SystemExit is caught in argument parsing.

**Files:**
- `packages/viscy-utils/src/viscy_utils/cli.py` (line 49-50)

**Impact:** The code catches `SystemExit` to work around Lightning issue #21255. This masks legitimate error conditions and makes the CLI harder to debug. If the underlying issue is fixed, this workaround will break other error handling.

**Fix approach:** Monitor Lightning repository for fix; add explicit test for the issue being worked around so removal can be validated.

### Chunk-based Sampling with Fixed Indices

**Issue:** Triplet dataset sampling uses fixed time intervals without validation.

**Files:**
- `packages/viscy-data/src/viscy_data/triplet.py` (lines 177-186)

**Impact:** When `time_interval` is specified, the code assumes fixed timepoint spacing. For irregular time series or with missing timepoints, sampling will fail silently or produce incorrect pairs.

**Fix approach:** Add explicit validation that assumes regular time spacing, or dynamically compute interval based on actual frame numbers.

### Batch Composition Assumptions in DynaCLR Engine

**Issue:** Contrastive loss computation assumes specific batch structure.

**Files:**
- `applications/dynaclr/src/dynaclr/engine.py` (lines 76-85, 141-150)

**Impact:** If custom losses or auxiliary heads change batch structure, the assumption about negative mining and metric computation breaks. The try/except at line 78 masks the actual error.

**Fix approach:** Add explicit validation of batch structure at the start of training, or use structured batch types with validation.

## Incomplete Documentation

### Dataset Loading Pattern Not Documented

**Issue:** The `__getitems__` + `collate_fn=lambda x:x` + `on_after_batch_transfer` pattern is powerful but undocumented in code.

**Files:**
- `applications/dynaclr/src/dynaclr/data/dataset.py` (entire module)
- Design documented in `CLAUDE.md` but not in docstrings

**Impact:** Future developers may not understand why the pattern is used and may refactor to standard per-sample iteration, defeating performance optimizations.

**Fix approach:** Add comprehensive docstring to `MultiExperimentTripletDataset.__getitems__()` explaining the batched reads and GPU transfer strategy.

### Collection YAML Schema Not Validated

**Issue:** Collection YAML format is described in CLAUDE.md but not validated at load time.

**Files:**
- `applications/dynaclr/src/dynaclr/data/experiment.py` (not explicitly validated)

**Impact:** Malformed YAML silently produces confusing downstream errors. Users have no clear feedback about schema violations.

**Fix approach:** Add a schema validator (pydantic model or jsonschema) that runs at collection load time with clear error messages.

## Test Coverage Gaps

### Linear Classifier Cross-Validation Not Covered

**Issue:** Complex cross-validation logic in linear classifiers has limited test coverage.

**Files:**
- `applications/dynaclr/src/dynaclr/evaluation/linear_classifiers/cross_validation.py` (861 lines)

**Impact:** Changes to CV strategy or folds can break silently. Edge cases (e.g., single fold, fold with zero samples) may not be handled correctly.

**Fix approach:** Add parametrized tests for edge cases and realistic multi-fold scenarios.

### Visualization App Interactivity Not Tested

**Issue:** Interactive callbacks in visualization engine are not covered by tests.

**Files:**
- `packages/viscy-utils/src/viscy_utils/evaluation/visualization.py` (most callbacks)

**Impact:** Dash app state changes and error paths are not validated. UI regressions only discovered at runtime.

**Fix approach:** Add unit tests for individual callback handlers with mocked Dash context.

## Dependencies at Risk

### PyTorch Lightning Integration Risk

**Issue:** Code depends on Lightning internals and workarounds for specific versions.

**Files:**
- `packages/viscy-utils/src/viscy_utils/cli.py` (SystemExit workaround)
- `applications/dynaclr/src/dynaclr/engine.py` (checkpoint loading workaround)

**Impact:** Lightning version upgrades may break these workarounds. Current workarounds suggest fragile integration points.

**Fix approach:** Consolidate all Lightning version-specific code into a single compatibility layer. Add CI tests with multiple Lightning versions.

### tensorstore Optional but Heavily Used

**Issue:** tensorstore is optional (try/except import) but required at runtime for key functionality.

**Files:**
- `packages/viscy-data/src/viscy_data/triplet.py` (lines 19-22)
- `applications/dynaclr/src/dynaclr/data/dataset.py` (lines 30-33)

**Impact:** Import-time errors are confusing. Better to make tensorstore a required dependency or provide a clear fallback implementation.

**Fix approach:** Either move tensorstore to required dependencies or implement a non-tensorstore code path with clear performance tradeoffs.

---

*Concerns audit: 2025-03-27*
