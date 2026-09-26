# dynacell/evaluation — Claude Code reference

Code in this directory uses **cubic** (CUDA-accelerated 3D bioimage
computing) for any GPU-accelerated numerical work — image preprocessing
before / after model inference, metric calculations, cropping/resizing,
percentile clips, Gaussian filters, etc. Cubic is a hard runtime
dependency of the eval extras (`applications/dynacell/pyproject.toml`
pins `cubic @ git+…@v0.9.0a1`). Do not gate cubic imports behind `try/except`
or fall back to scipy/skimage paths.

**The cubic version is part of the numeric contract, not just a dep.** Metric
values move across pins — measured 0.8.0a2 → 0.9.0a1: `Z_FSC_Resolution`
+30.9%, `XY_FSC_Resolution` +16.7%, `Spectral_PCC` ≤0.57%, `FRC_Resolution`
+0.52%, with `PCC`/`SSIM`/`NRMSE`/`PSNR`/`SI_*`/`PerCell_*`/`AP_*`/`mAP`/
`instance_dice`/`MicroMS3IM`/CP columns all bit-identical. So:

- `dynacell.evaluation.provenance.REQUIRED_CUBIC_VERSION` is the single
  declared version; `provenance_test.py` pins it to every pyproject pin.
- `check_cubic_pin()` runs at both eval entry points and **fails the run**
  when the environment disagrees. Do not downgrade it to a warning.
- `save_metrics` stamps `metrics_provenance.json` beside the CSVs, and
  `_final_metrics_cache_valid` refuses an unstamped or foreign-stamped cache.
  Unstamped means "written before the stamp existed", i.e. an unknown pin —
  it must recompute, unlike the per-extractor `preprocess_version` bootstrap
  which treats an untagged entry as unconstrained.

The GPU-resident Cellpose-SAM entry point is
`cubic.segmentation.segment_cpsam` (single host→device upload, masks
returned to host; GPU-only by contract). The marker-controlled watershed
helper `segment_watershed` is **not** re-exported from
`cubic.segmentation` — import it from
`cubic.segmentation.segment_utils`.

Below is the same guidance the upstream cubic repository ships in its
`AGENTS.md`, condensed and adapted for this module. **Read it before
adding GPU-aware code here.** It is also fine to read `cubic/AGENTS.md`
directly at `../cubic/` for the canonical version.

## ⚠️ Always verify data format + normalization for pretrained models

Every pretrained model used at eval time — segmentors (Cellpose /
CELL-DINO / cpsam) and deep feature extractors (DINOv3, DynaCLR,
CELL-DINO, MorphEm) — has a **specific input contract**: expected value
range, dtype, channel count, and normalization recipe. These are *not*
interchangeable, and getting one wrong fails silently (plausible-looking
but meaningless features / masks), not loudly. Before wiring or editing
any model call, confirm all of the following against the model's own
docs/source — do not assume:

- **Value range and normalization.** Look up what the model was *trained*
  on and match it. Common recipes here: robust percentile clip
  (`_robust_norm`, the shared `build_crops` recipe), per-image spatial
  z-score (DynaCLR's `NormalizeSampled`, MorphEm's `PerImageNormalize` /
  InstanceNorm2d), ImageNet mean/std (DINOv3). A train/test normalization
  mismatch (e.g. feeding bounded [0,1] to a z-score-trained encoder) is a
  real bug even when nothing crashes.
- **No double-normalization.** If crops are already normalized upstream
  in `build_crops`, a processor/transform that *also* rescales (e.g. the
  HF `AutoImageProcessor`'s default `do_rescale=True` → ÷255, or a second
  z-score) silently destroys the signal. Trace the full path from raw
  array to model input and count every rescale exactly once. Pass
  `do_rescale=False` when the input is already float-normalized.
- **GT vs prediction parity.** GT and predicted intensities live in
  **different value ranges**. The normalization must be applied
  identically to both so their features are comparable — a min/max or
  raw-range recipe makes GT and pred asymmetric; a shared robust/affine
  recipe keeps them aligned.
- **dtype.** `np.percentile` upcasts to float64; model weights are
  float32. Cast crops back to float32 before the model (`.astype(
  np.float32, copy=False)`) or conv/linear layers raise `Input type
  (double) and bias type (float) should be the same`.
- **Cache invalidation.** Any change to a recipe must bump that
  extractor's `PREPROCESS_VERSION` (see
  `_auto_invalidate_on_preprocess_version_mismatch` in
  `pipeline_cache.py`), or stale caches from the old recipe silently
  survive.

When adding a new pretrained model, read its published preprocessing
(model card / training repo) and add a regression test that asserts the
tensor reaching the model has the expected range/dtype — see the
`test_dinov3_*` / `test_dynaclr_*` / `test_morphem_*` tests in
`tests/test_evaluation_extractors.py` and the hot-pixel test in
`tests/test_evaluation_metrics.py`.

## Device management (`cubic.cuda`)

Core utilities for device-agnostic computation:

- `CUDAManager` – Singleton managing CuPy/cuCIM resources
- `get_array_module(array)` – Returns `np` or `cp` based on array location (**use sparingly** - prefer `np.` directly)
- `asnumpy(array)` / `ascupy(array)` – Transfer arrays between CPU/GPU (**preferred** over direct CuPy calls)
- `to_device(array, device)` – Move array to specific device (`"CPU"` or `"GPU"`)
- `to_same_device(source, reference)` – Move source array to same device as reference
- `check_same_device(*arrays)` – Verify all arrays are on the same device
- `get_device(array)` – Returns `"CPU"` or `"GPU"`

**Important**:

- `get_array_module()` should only be used when creating new arrays that
  must be on a specific device. For most operations, use `np.` directly
  — NumPy functions work on both NumPy and CuPy arrays through duck
  typing.
- **Always use `cubic.cuda` functions** for device operations (moving
  arrays, checking devices) rather than directly calling CuPy functions.
  This maintains the abstraction layer and ensures consistent behavior.

## Device-agnostic wrappers

- `cubic.scipy` – Proxy module for device-agnostic SciPy / cupyx.scipy access
- `cubic.skimage` – Proxy module for device-agnostic scikit-image / cuCIM access
- `cubic.cucim` – CuCIM integration for GPU-accelerated image I/O

These modules automatically route function calls to CPU (NumPy / SciPy /
scikit-image) or GPU (CuPy / cuCIM) implementations based on the input
array's device.

## ⚠️ CRITICAL: device-agnostic code pattern

**All functions in `cubic` automatically support both CPU and GPU
without any code changes** — they work with NumPy arrays (CPU) or CuPy
arrays (GPU) based solely on the input array's device location. The same
function call works on both devices; just transfer the input array to
the desired device using `cubic.cuda` functions.

**Avoid using `xp` (array module) interface as much as possible.** Prefer
`np.` or array methods (`.func()`) to maximize code portability between
NumPy and CuPy without modifications.

**Preferred approach** (use `np.` directly):

```python
import numpy as np

# NumPy functions work on both NumPy and CuPy arrays
result = np.fft.fftn(image)            # ✅ Works on both CPU/GPU arrays
result = np.abs(array)                 # ✅ Works on both CPU/GPU arrays
result = np.bincount(bin_id, weights)  # ✅ Works on both CPU/GPU arrays
result = np.sqrt(k0 * k0 + k1 * k1)    # ✅ Works on both CPU/GPU arrays
result = array.ravel()                 # ✅ Array methods work on both
result = array.astype(np.float32)      # ✅ Array methods work on both
```

**Avoid when possible** (using `xp` interface):

```python
from cubic.cuda import get_array_module

xp = get_array_module(array)
result = xp.fft.fftn(image)  # ⚠️ Only use when necessary
result = xp.asarray(data)    # ⚠️ Only use when creating new arrays on specific device
```

**When `xp` is OK** (limited cases):

- Creating new arrays that must be on the same device as existing arrays: `xp.asarray()`, `xp.zeros()`, `xp.ones()`
- Device-specific functions not available in NumPy: `xp.fft.fftfreq()` for device placement
- Functions that don't work with NumPy's duck-typing: rare, prefer `np.` when possible

## Device operations (use `cubic.cuda` functions)

When you need to move arrays between devices or check device placement,
**always use functions from `cubic.cuda`** rather than directly calling
CuPy functions:

```python
from cubic.cuda import asnumpy, ascupy, to_device, to_same_device, check_same_device, get_device

# ✅ Preferred: Use cubic.cuda functions
cpu_array = asnumpy(gpu_array)                  # Move to CPU
gpu_array = ascupy(cpu_array)                   # Move to GPU
target_array = to_device(source_array, "GPU")   # Move to specific device
aligned_array = to_same_device(array1, array2)  # Move to same device as reference
check_same_device(array1, array2)               # Verify same device
device = get_device(array)                      # Check current device

# ❌ Avoid: Direct CuPy calls
import cupy as cp
cpu_array = cp.asnumpy(gpu_array)               # Don't do this — breaks abstraction
```

**Rationale**: Using `np.` directly allows code to work seamlessly with
both NumPy and CuPy arrays through duck typing. This maximizes
portability and allows users to port NumPy code in/out with minimal
modifications. The `xp` interface should only be used when absolutely
necessary for device placement or when NumPy functions don't support
CuPy arrays (rare). For device operations, always use `cubic.cuda`
functions to maintain the abstraction layer and ensure consistent
behavior.

## Concrete example in this module

`segmentation.py`'s `_smooth_nucleus_input` is the minimal canonical
shape:

```python
from cubic.cuda import ascupy, asnumpy
from cubic.skimage import filters as _cubic_filters

def _smooth_nucleus_input(img, sigma=NUCLEUS_GAUSSIAN_SIGMA):
    img_dev = ascupy(img.astype(np.float32, copy=False))                    # move to GPU
    smoothed = _cubic_filters.gaussian(img_dev, sigma=sigma, preserve_range=True)  # cubic proxy auto-dispatches
    return asnumpy(smoothed)                                                # caller wants numpy
```

No `try/except` around the cubic imports, no scipy fallback. If CUDA
isn't available the call route falls through cubic's own CPU path
(scikit-image), and that's the right outcome — but in practice the eval
pipeline already requires CUDA for the SuperModel inference downstream,
so the GPU path is what runs.

## Don't

- Don't add a scipy / skimage fallback alongside a cubic call. Pick one
  via cubic — it already handles both backends.
- Don't add `if torch.cuda.is_available():` dispatches around cubic calls
  for the same reason; cubic decides the backend from the input array
  type.
- Don't `import cupy as cp` and call CuPy directly. Use `cubic.cuda.*`
  for device transfers and `cubic.skimage` / `cubic.scipy` for array
  operations.
- Don't gate cubic imports with `try/except ImportError: None`. Cubic is
  a hard dep here. If it's missing, the pipeline is broken — fail loud.

## Eval runtime parallelism — grouped eval is serial-amortized; `--parallel` is a predict lever

`dynacell evaluate-grouped` loads the model stack (SuperModel + DINOv3 +
DynaCLR + CELL-DINO, ~60 s, ~15 GB) **once** and reuses it across every
condition in the bucket. That amortization is the whole point of the grouped
path — **do not parallelize conditions or FOVs inside it.** `--parallel` and
`runtime.executor=process` are *predict* levers, not eval ones: on a grouped
bucket each condition's worker pool independently reloads the full stack
(`evaluate_predictions_grouped` prints a `!!! WARNING !!!` — `~30-90 s ×
N_workers × N_conditions` of redundant cold-start), and the deep-feature
forwards still serialize on the one GPU under the fcntl lock, so there is no
compute win to offset the reloads.

- **Predicts are the opposite** (cheap load, ~10 GB, GPU-light): `--parallel 2`
  is a confirmed win there (2-up on A40). See `applications/dynacell/CLAUDE.md`
  "Predict submission modes".
- `submit_evaluation_batch.py` **cannot** drive cpdino grouped buckets: it emits
  `uv run dynacell evaluate` through the shared `.venv` (broken for cpdino —
  needs the `cpdino-eval` venv) and requires one `(organelle, model, train_set)`
  per call, while a bucket spans many models.
- **The real eval-side parallelism lever is bucket-level:** run independent
  grouped buckets on separate GPUs. When buckets share a GT cache
  (`sec61b`/`tomm20`, keyed by `(gene_cond, halfwidth)`), a warm-first `afterok`
  chain (one bucket warms, the rest read) is a *true* dependency only while the
  cache is cold — once it is warm at the target halfwidth, that afterok is a
  **false sync point** and the buckets can run concurrently (all read-only).
  Drop it only after confirming warm-state, else concurrent buckets race the
  drop-and-recompute write.

## Cache-staleness checks: gate on the mean PCC, and prove staleness before recomputing

`PCC` is the right **signal** for "was this eval cache produced from the arrays that
are on disk now?" — it is affine-invariant, so a metric-definition change (e.g. the
min-max → scale-invariant switch) cannot move it, and only a genuine array change
can. `pixel_scaling_backfill.py` uses it as its write gate for exactly that reason.

**But a per-row relative tolerance is the wrong test.** PCC's numerator is a
difference of nearly-cancelling sums, so its float32 error is set by cancellation,
not by magnitude. On a near-degenerate prediction the value itself is tiny — FNet3D
on HEK mitochondria sits at **PCC 0.003**, no correlation at all — and cancellation
alone moves one row by ~1e-3, i.e. **30% relative**. Any `rtol` check fires.
Measured 2026-08-03: `rtol=1e-3, atol=1e-5` per row flagged **19 of 648** conditions
as stale cache, and every one that fed a table was a false positive.

Gate on the **mean** instead: `PCC_MEAN_ATOL = 5e-4`, tighter than the 3-decimal
rounding the tables publish, so a mean inside it cannot change a published cell
while a real re-predict moves it well outside. A per-row failure that clears the
mean is reported `reproduced_mean` — written, but still visible in the audit. Do not
collapse that into `reproduced`, and do not widen the per-row `rtol` instead.

**A staleness verdict is a hypothesis, not a finding. Run these four before
spending GPU on a re-eval** — all cheap, and together they settled all 19:

1. **Compare the mean against an independent published artifact.** The one flagged
   cell that fed a table reproduced its published `0.540` as `0.5396`. PCC is
   affine-invariant, so identical PCC ⇒ identical arrays. That one check was
   decisive.
2. **Chunk-mtime prediction *and* GT against the eval CSV mtime**, full scan:
   `find <store> -type f -not -name '*.json' -printf '%T@\n' | sort -rn | head -1`.
   All three suspect dirs had both stores older than the eval and untouched since,
   so no data change was possible. Never sample via `find | head | xargs stat` — a
   truncated listing gives a wrong "newest chunk" and sends you down a false trail.
3. **Check for a live writer.** Two flagged dirs were being rewritten at that moment
   by a running eval; the gate was comparing against a cache mid-write. `squeue`
   before concluding anything about a cache.
4. **`git log` the config surface** against the eval date: the leaf YAML, the dataset
   manifest under `_manifests/<name>/manifest.yaml`, and `_configs/eval.yaml`.

Note that `focus.*` / `segmentation.slice_selection` affect **instance segmentation
and the deep-feature slab only — not pixel metrics.** Don't reach for them to
explain a moved pixel number.

A gate that fails **closed** (refuses to write, reports `stale_cache`) is safe to
leave over-sensitive, because its only cost is a line in a report. So prefer false
positives in the gate — but never let a false positive drive a re-run.
