# dynacell/evaluation — Claude Code reference

Code in this directory uses **cubic** (CUDA-accelerated 3D bioimage
computing) for any GPU-accelerated numerical work — image preprocessing
before / after model inference, metric calculations, cropping/resizing,
percentile clips, Gaussian filters, etc. Cubic is a hard runtime
dependency of the eval extras (`applications/dynacell/pyproject.toml`
pins `cubic==0.7`, resolved from PyPI). Do not gate cubic imports behind `try/except`
or fall back to scipy/skimage paths.

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
