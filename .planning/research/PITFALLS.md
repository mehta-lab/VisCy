# Pitfalls Research: viscy-data Extraction

**Domain:** Extracting a complex data module package from a uv workspace monorepo
**Researched:** 2026-02-13
**Confidence:** MEDIUM-HIGH (based on codebase analysis, architecture docs, v1.0 extraction experience, and domain knowledge of PyTorch/Lightning data loading patterns)

**Note:** WebSearch was unavailable during this research. Pitfalls are derived from: (1) analysis of the actual viscy/data module architecture (README.md), (2) v1.0 extraction experience documented in existing research files, (3) the pyproject.toml and CI configurations, and (4) domain expertise with PyTorch multiprocessing, Lightning DataModules, tensorstore, and C-extension packages. Confidence is MEDIUM-HIGH rather than HIGH because some cross-platform build claims could not be verified against current package indices.

---

## Critical Pitfalls

Mistakes that cause rewrites, major blockers, or silent runtime failures that are hard to trace.

---

### Pitfall 1: Lazy Import Guard Ordering Breaks at Runtime, Not Import Time

**What goes wrong:**
Optional dependencies (tensorstore, tensordict, pycocotools) are guarded with lazy imports like `try: import tensorstore` in module-level code. But if the guard is at the module level and the import is used inside a class method, the error only surfaces when that specific code path is exercised -- potentially deep inside a training loop after hours of preprocessing. Worse: if you guard the import at module level with a sentinel (`HAS_TENSORSTORE = False`) but forget to check the sentinel before using the library in a method, you get `NameError: name 'tensorstore' is not defined` instead of a helpful message.

**Why it happens:**
During extraction, developers convert `import tensorstore` to a guarded import but don't systematically audit every usage site. The module has 13 files; it's easy to miss a call site in a nested helper. The triplet.py module in particular uses tensorstore inside a dataset `__getitem__` which is called by DataLoader workers -- the error appears in a worker subprocess stack trace, obscuring the real cause.

**How to avoid:**
1. Use a centralized lazy import pattern in `_imports.py`:
   ```python
   def _require_tensorstore():
       try:
           import tensorstore
           return tensorstore
       except ImportError:
           raise ImportError(
               "tensorstore is required for TripletDataModule. "
               "Install it with: pip install viscy-data[triplet]"
           ) from None
   ```
2. Call `_require_tensorstore()` at the top of every method that uses it, not at module level.
3. Add a test for each optional-dep module that verifies the ImportError message when the dep is missing. Use `pytest.importorskip` in tests, but also test the error path.
4. In CI, run one test job with only base deps installed (no extras) to catch unguarded imports.

**Warning signs:**
- Any `import X` at module top-level for an optional dependency
- Tests that always install `[all]` extras -- they never catch missing-dep errors
- `NameError` in DataLoader worker processes during training

**Phase to address:**
Package scaffolding phase (pyproject.toml + `_imports.py` helper). Verify in CI phase with a "minimal deps" test job.

---

### Pitfall 2: Manager().dict() Shared Cache Is Not Picklable Across spawn Contexts

**What goes wrong:**
`CachedOmeZarrDataset` uses `multiprocessing.Manager().dict()` as a shared cache across DataLoader workers. This works with PyTorch's default `fork` start method on Linux, but fails on macOS (default `spawn`) and Windows (only `spawn`). The `Manager` proxy objects must be passed to workers via pickling, and depending on when the Manager is created relative to DataLoader fork/spawn, you get either: (a) `RuntimeError: cannot pickle 'weakref' object` or (b) a silently separate cache per worker (defeating the purpose).

**Why it happens:**
The original code was developed and tested on Linux HPC clusters where `fork` is the default. The extraction needs cross-platform CI (Ubuntu, macOS, Windows per existing test.yml matrix). The `spawn` start method creates fresh Python processes that must pickle everything passed to them. `Manager().dict()` proxy objects work across `fork` boundaries but behave differently under `spawn`.

**How to avoid:**
1. Create the `Manager` in `setup()` (Lightning hook), not in `__init__`. Lightning's `setup()` runs after the Trainer has configured the multiprocessing context.
2. Store the manager reference on `self` so it persists for the lifetime of the DataModule.
3. Test with `mp_start_method="spawn"` explicitly in at least one CI configuration:
   ```python
   @pytest.fixture
   def spawn_trainer():
       return Trainer(accelerator="cpu", devices=1,
                      strategy=SingleDeviceStrategy(device="cpu"))
   ```
4. Consider replacing `Manager().dict()` with `torch.multiprocessing.Queue` or a file-based cache (memory-mapped array via tensordict's `MemoryMappedTensor`) which is already used in `mmap_cache.py`.
5. Document that `num_workers=0` avoids the issue entirely (useful for debugging).

**Warning signs:**
- Tests pass on Linux but fail on macOS/Windows CI
- `pickle` or `weakref` errors in DataLoader worker stack traces
- Cache hit rate is 0% despite repeated access to same volumes

**Phase to address:**
Code migration phase. Must be addressed before CI matrix runs on macOS/Windows.

---

### Pitfall 3: Base Class Extraction Creates Hidden Import Cycles

**What goes wrong:**
`HCSDataModule` is both a concrete DataModule (used directly for translation fine-tuning) and the base class for `TripletDataModule` and `CellDivisionTripletDataModule`. It contains shared utility methods like `_read_norm_meta()` and references to iohub types. When extracting shared utilities into `_utils.py`, developers often create circular imports: `hcs.py` imports from `_utils.py`, but `_utils.py` needs types defined in `_typing.py` which imports from something that imports from `hcs.py`.

More subtly: if `_utils.py` imports `Sample` from `_typing.py`, and `hcs.py` imports from both `_utils.py` and `_typing.py`, the import order depends on which module is imported first. If `triplet.py` imports `hcs.py` before `_typing.py` is loaded, you get a partially initialized module.

**Why it happens:**
The original `hcs.py` was a single module that served as both a concrete implementation and a utility library. The README.md explicitly notes this dual role: "_hcs.py serves as both a concrete data module AND a utility library_." Extracting utilities without carefully analyzing the import dependency graph creates cycles.

**How to avoid:**
1. Map the complete import graph BEFORE moving any code. The dependency graph in `viscy/data/README.md` is the starting point, but also trace function-level dependencies.
2. Follow a strict layering rule:
   ```
   _typing.py  (types only, no imports from viscy_data)
       |
   _utils.py   (imports only from _typing, stdlib, and external libs)
       |
   hcs.py      (imports from _typing and _utils)
       |
   triplet.py  (imports from _typing, _utils, and hcs)
   ```
3. Test the import order explicitly: `python -c "from viscy_data import TripletDataModule"` should work without importing `hcs` first.
4. Use `ruff` rule `I001` (isort) to keep imports consistent, but also add a custom test that verifies no circular imports exist.

**Warning signs:**
- `ImportError: cannot import name 'X' from partially initialized module`
- Tests pass when run individually but fail when run as a suite (import order dependency)
- `AttributeError` on a module object (partially loaded module)

**Phase to address:**
Code migration phase -- specifically the `_utils.py` extraction plan.

---

### Pitfall 4: Optional Extras Create a 2^N CI Matrix Explosion

**What goes wrong:**
With 4 optional dependency groups (`[triplet]`, `[livecell]`, `[mmap]`, `[all]`), testing all combinations across 3 Python versions and 3 OS targets creates a massive matrix: (base + triplet + livecell + mmap + all) x 3 Python x 3 OS = 45 jobs. But some combinations are invalid (pycocotools doesn't build on Windows; tensorstore has limited arm64 support on macOS). The matrix either explodes in cost or has confusing include/exclude rules.

**Why it happens:**
viscy-transforms had no optional dependencies -- all deps were required. viscy-data is the first package with optional extras. The existing CI (`test.yml`) uses a simple 3x3 matrix. Naively extending it for extras creates an unmaintainable workflow.

**How to avoid:**
1. Do NOT test all combinations. Use this strategy:
   - **Base job** (no extras): 3 Python x 3 OS = 9 jobs. Tests only modules that work with base deps.
   - **Full extras job** (`[all]`): 1 Python (3.12) x 1 OS (ubuntu-latest) = 1 job. Tests everything.
   - **Per-extra smoke test**: 1 Python x 1 OS per extra = 3 jobs. Tests that the extra installs correctly.
2. Use pytest markers to skip tests when optional deps are missing:
   ```python
   tensorstore = pytest.importorskip("tensorstore")
   ```
3. Exclude known-broken combinations using the GitHub Actions matrix `exclude` key:
   ```yaml
   exclude:
     - os: windows-latest
       extras: livecell  # pycocotools build fails on Windows
   ```
4. For the `[all]` extra, document that it may not install on all platforms. This is acceptable -- scientific Python packages routinely have platform limitations.

**Warning signs:**
- CI takes >30 minutes due to matrix size
- Frequent red CI from platform-specific build failures in optional deps
- Maintainers start ignoring CI failures ("oh that's just the Windows pycocotools thing")

**Phase to address:**
CI phase. Design the matrix before writing the workflow file.

---

### Pitfall 5: pycocotools Build Failure Blocks Entire CI on Windows

**What goes wrong:**
`pycocotools` requires a C compiler to build from source. On Ubuntu, `gcc` is available by default. On macOS, `clang` via Xcode Command Line Tools. On Windows, there is no default C compiler -- users need Visual Studio Build Tools installed. If the CI matrix includes Windows + `[livecell]` or `[all]`, the job fails at pip install time, not at test time, with a cryptic compilation error.

**Why it happens:**
pycocotools does not always have pre-built wheels for all platform/Python version combinations. When a wheel is missing, pip falls back to building from source, which requires a C compiler. Windows GitHub Actions runners have MSVC available, but the environment may not be configured correctly for all Python versions.

**How to avoid:**
1. Use the `pycocotools` package from conda-forge or the `pycocotools-windows` fork if targeting Windows.
2. Better: mark `[livecell]` as Linux/macOS only in documentation. LiveCell is a research dataset used primarily on HPC Linux clusters.
3. In CI, exclude the livecell extra from Windows:
   ```yaml
   exclude:
     - os: windows-latest
       python-version: "3.11"  # if no wheel available
   ```
4. Alternatively, use `cython` + build isolation: `pip install pycocotools --no-build-isolation` can sometimes help, but this is fragile.
5. Pin pycocotools to a version that has wheels for your target platforms. Check PyPI for available wheels before pinning.

**Warning signs:**
- `error: Microsoft Visual C++ 14.0 or greater is required` in CI logs
- CI green on Ubuntu/macOS, red on Windows
- Intermittent failures when new Python versions lack pre-built wheels

**Phase to address:**
Package scaffolding phase (dependency specification) + CI phase (matrix exclusions).

---

### Pitfall 6: iohub API Coupling Creates Fragile Type Boundaries

**What goes wrong:**
The data modules use iohub's `Position`, `Plate`, `Well`, and `ImageArray` types extensively -- not just for I/O, but as type annotations on public methods and in constructor signatures. If iohub releases a breaking change (e.g., renaming `Position` to `FOV`, or changing the `ImageArray` interface), every data module breaks simultaneously. More immediately: if the viscy-data pyproject.toml pins `iohub>=0.X` but a user has `iohub==0.X-1`, they get confusing `AttributeError`s instead of a clean version conflict.

**Why it happens:**
iohub is developed by the same team (CZ Biohub), so API stability was implicitly assumed. The tight coupling is intentional -- these types ARE the domain model. But as a separately packaged library, iohub's release cycle is now decoupled from viscy-data's.

**How to avoid:**
1. Pin iohub with a minimum version that has the API you depend on: `iohub>=0.2.0` (or whatever version stabilized the `Position`/`Plate` API).
2. DO NOT pin an upper bound (`iohub<1.0`) -- this causes resolver hell for users. Trust semantic versioning.
3. Add an integration test that imports all iohub types used by viscy-data and verifies they have the expected attributes. This catches API drift early:
   ```python
   def test_iohub_api_surface():
       from iohub.ngff import Position, Plate, ImageArray
       assert hasattr(Position, "__getitem__")  # etc.
   ```
4. Document which iohub version was tested against in the README.

**Warning signs:**
- iohub release notes mention "breaking changes" or "renamed"
- Users report `AttributeError` on iohub objects
- `uv lock` resolves a newer iohub than was tested

**Phase to address:**
Package scaffolding phase (dependency pinning). Integration test in test migration phase.

---

### Pitfall 7: ThreadDataLoader from MONAI Breaks Isolation Testing

**What goes wrong:**
`TripletDataModule` uses `monai.data.ThreadDataLoader` instead of PyTorch's `DataLoader`. When running `uv sync --package viscy-data` for isolated testing, the test environment has MONAI installed (it's a required dep). But `ThreadDataLoader` has threading behavior that interacts badly with pytest's test isolation -- specifically, thread-local state can leak between tests, and `tensorstore`'s async I/O in threads can deadlock if the event loop is not properly managed.

More specifically: `ThreadDataLoader` creates `ThreadPoolExecutor` workers instead of subprocess workers. These threads share the GIL and the process's memory space. If a test creates a `ThreadDataLoader` and doesn't explicitly shut it down, the threads persist into the next test, causing resource leaks and eventual `OSError: too many open files`.

**Why it happens:**
`ThreadDataLoader` was chosen for DynaCLR/triplet because tensorstore performs async I/O that releases the GIL. Thread-based workers are more efficient than subprocess workers for GIL-releasing I/O. But this design choice has testing implications that aren't obvious until you run a full test suite.

**How to avoid:**
1. Use pytest fixtures that explicitly create and tear down DataLoaders:
   ```python
   @pytest.fixture
   def triplet_dataloader(triplet_datamodule):
       triplet_datamodule.setup("fit")
       loader = triplet_datamodule.train_dataloader()
       yield loader
       # Explicit cleanup
       if hasattr(loader, '_executor'):
           loader._executor.shutdown(wait=False)
   ```
2. Run triplet-related tests in a separate pytest session or mark them with `@pytest.mark.forked` (requires `pytest-forked`).
3. Set `num_workers=0` in test fixtures to avoid threading entirely. Reserve `num_workers>0` tests for integration/slow test marks.
4. Add resource leak detection: `pytest --tb=short -W error::ResourceWarning`.

**Warning signs:**
- Tests hang indefinitely when run as a full suite but pass individually
- `OSError: [Errno 24] Too many open files` in CI
- Intermittent test failures ("flaky tests") that depend on test execution order

**Phase to address:**
Test migration phase. Configure fixtures before migrating test files.

---

## Moderate Pitfalls

Mistakes that cause delays, confusion, or technical debt but are recoverable.

---

### Pitfall 8: `__init__.py` Re-exports Create Eager Import Chains

**What goes wrong:**
A natural approach for `viscy_data/__init__.py` is to re-export all public classes:
```python
from viscy_data.hcs import HCSDataModule
from viscy_data.triplet import TripletDataModule
from viscy_data.livecell import LiveCellDataModule
```
But `triplet.py` has `import tensorstore` (even if guarded), and `livecell.py` has `import pycocotools`. If the guard fails or is at module level, `import viscy_data` itself fails for users who only installed the base package without extras.

**Why it happens:**
viscy-transforms got away with eager re-exports because all its dependencies are required. viscy-data has optional deps that make eager re-exports dangerous.

**How to avoid:**
1. Only re-export classes whose modules have no optional dependencies in `__init__.py`:
   ```python
   # viscy_data/__init__.py
   from viscy_data.hcs import HCSDataModule, SlidingWindowDataset
   from viscy_data.gpu_aug import GPUTransformDataModule
   from viscy_data.combined import CombinedDataModule
   from viscy_data.distributed import ShardedDistributedSampler
   # Do NOT import from triplet, livecell, mmap_cache at top level
   ```
2. For optional-dep modules, use `__all__` and document the import path:
   ```python
   # Users must import explicitly:
   # from viscy_data.triplet import TripletDataModule
   ```
3. Alternatively, use `__getattr__` for lazy module-level imports (PEP 562):
   ```python
   def __getattr__(name):
       if name == "TripletDataModule":
           from viscy_data.triplet import TripletDataModule
           return TripletDataModule
       raise AttributeError(f"module 'viscy_data' has no attribute {name}")
   ```

**Warning signs:**
- `import viscy_data` fails on a fresh install without `[all]`
- Users report `ModuleNotFoundError: No module named 'tensorstore'` when they only wanted `HCSDataModule`

**Phase to address:**
Package scaffolding phase (design `__init__.py` strategy before code migration).

---

### Pitfall 9: Zarr/iohub Test Fixtures Are Expensive and Non-Trivial

**What goes wrong:**
Data module tests require actual OME-Zarr stores with HCS plate structure (plates > wells > FOVs > images). Creating these fixtures is not trivial -- it requires iohub to write proper OME-Zarr metadata, multiple resolution levels, and channel metadata. If each test creates its own fixture, the test suite becomes extremely slow. If fixtures are shared (session-scoped), tests can interfere with each other through mutations.

Additionally, tensorstore tests need properly structured Zarr arrays, and livecell tests need COCO-format JSON annotations plus TIFF images. These are three different fixture ecosystems within one package.

**Why it happens:**
viscy-transforms tests only needed tensor fixtures (easy: `torch.randn(1, 1, 32, 32)`). Data module tests need filesystem-backed fixtures with specific directory structures and metadata. This is a qualitative jump in fixture complexity.

**How to avoid:**
1. Create a `conftest.py` with session-scoped, read-only fixtures:
   ```python
   @pytest.fixture(scope="session")
   def ome_zarr_store(tmp_path_factory):
       """Create a minimal HCS OME-Zarr store for testing."""
       store_path = tmp_path_factory.mktemp("data") / "test.zarr"
       # Use iohub to create a proper HCS plate
       from iohub.ngff import open_ome_zarr
       with open_ome_zarr(store_path, layout="hcs", mode="w") as plate:
           position = plate.create_position("A", "1", "0")
           position.create_image("0", data=np.random.rand(1, 2, 5, 64, 64))
       return store_path
   ```
2. Use `tmp_path_factory` (session-scoped) not `tmp_path` (function-scoped) for expensive I/O fixtures.
3. Mark tests that need real I/O fixtures with `@pytest.mark.slow` and skip them in the fast CI job.
4. For livecell: create a minimal COCO JSON fixture as a Python dict, write it to a temp file. Do not download real LiveCell data in CI.
5. NEVER use real data paths in tests. The existing Hydra configs reference `/hpc/projects/...` paths that are HPC-specific.

**Warning signs:**
- Test suite takes >5 minutes due to fixture creation
- Tests fail on CI but pass locally (missing data files)
- Flaky tests due to shared mutable fixtures

**Phase to address:**
Test migration phase. Design fixture strategy BEFORE writing test files.

---

### Pitfall 10: Lightning CLI / Hydra Config Class Paths Break Silently

**What goes wrong:**
Downstream training configs (Hydra YAML, Lightning CLI YAML) reference data modules by fully qualified class path:
```yaml
# Old path (broken after extraction)
data:
  class_path: viscy.data.hcs.HCSDataModule

# New path (correct)
data:
  class_path: viscy_data.hcs.HCSDataModule
```
The breakage is silent -- configs are just YAML files, so no linter or type checker catches the stale path. The error only appears when someone runs training, potentially weeks after the extraction.

Hydra is explicitly out of scope per PROJECT.md, but Lightning CLI is used by the training pipelines. The existing Hydra output configs in `applications/dynacell/outputs/` show the config structure but don't reference class paths (they use nested config, not `_target_`). However, Lightning CLI configs DO use `class_path`.

**Why it happens:**
Config files are stringly-typed. Import path changes are invisible to them. The v1.0 extraction of viscy-transforms did not have this problem because transforms are passed as constructor arguments (MONAI `Compose`), not referenced by class path in configs.

**How to avoid:**
1. Search all YAML/JSON files for `viscy.data` references:
   ```bash
   grep -r "viscy\.data\." applications/ examples/ configs/ --include="*.yaml" --include="*.json"
   ```
2. Create a migration guide document listing old -> new import paths for every public class.
3. Add a deprecation shim module (even though clean break is the policy):
   ```python
   # viscy/data/__init__.py (in the umbrella package, if it exists)
   import warnings
   warnings.warn(
       "viscy.data is deprecated. Use viscy_data instead. "
       "See migration guide: https://...",
       DeprecationWarning, stacklevel=2
   )
   from viscy_data import *
   ```
   This is optional and contradicts the "clean break" decision, but it prevents silent failures for existing users.
4. If clean break is firm (per PROJECT.md): ensure all known configs in the repository are updated in the same PR that extracts the package.

**Warning signs:**
- Training scripts fail with `ModuleNotFoundError` or `ClassNotFoundError` weeks after extraction
- Users open issues about broken configs
- Saved experiment configs become non-reproducible

**Phase to address:**
Code migration phase (update all in-repo configs). Document in a migration guide during docs phase.

---

### Pitfall 11: `SelectWell` Mixin + Multiple Inheritance MRO Fragility

**What goes wrong:**
`CachedOmeZarrDataModule(GPUTransformDataModule, SelectWell)` and `MmappedDataModule(GPUTransformDataModule, SelectWell)` use multiple inheritance with a mixin. During extraction, if the base classes are reorganized or their `__init__` signatures change, Python's Method Resolution Order (MRO) can break. Specifically: if `GPUTransformDataModule.__init__` calls `super().__init__()` but `SelectWell.__init__` expects different arguments, you get `TypeError: __init__() got an unexpected keyword argument`.

**Why it happens:**
Multiple inheritance with `super()` requires cooperative `__init__` chains. If any class in the MRO doesn't properly forward `**kwargs`, the chain breaks. This is fragile even without extraction, but extraction increases the risk because developers may refactor `__init__` signatures to "clean up" the extracted code.

**How to avoid:**
1. Do NOT change `__init__` signatures during extraction. Copy first, refactor later.
2. Add explicit MRO tests:
   ```python
   def test_cached_ome_zarr_mro():
       mro = CachedOmeZarrDataModule.__mro__
       assert mro.index(GPUTransformDataModule) < mro.index(SelectWell)
   ```
3. Ensure all mixins accept `**kwargs` and forward them to `super().__init__(**kwargs)`.
4. Document the inheritance hierarchy in the package README (the existing mermaid diagram in `viscy/data/README.md` is excellent -- preserve it).

**Warning signs:**
- `TypeError: __init__() got an unexpected keyword argument` when constructing DataModules
- Tests pass for base classes but fail for subclasses
- Behavior changes when import order changes (MRO is import-order-independent, but some metaclass tricks are not)

**Phase to address:**
Code migration phase. Copy-first, refactor-never (during extraction).

---

### Pitfall 12: tensordict MemoryMappedTensor Cleanup in Tests

**What goes wrong:**
`MmappedDataModule` creates `MemoryMappedTensor` objects backed by files in a scratch directory. These memory-mapped files hold open file descriptors. In tests, if the fixture doesn't explicitly close/unmap these tensors, you get: (a) `PermissionError` on Windows when pytest tries to clean up `tmp_path`, (b) memory leaks in CI, (c) stale memory-mapped files accumulating in `/tmp` on long-running test sessions.

**Why it happens:**
Memory-mapped files are OS resources, not Python objects managed by the garbage collector. `del tensor` doesn't immediately release the mmap. On Windows, you cannot delete a file that has an open memory map.

**How to avoid:**
1. Use a fixture with explicit cleanup:
   ```python
   @pytest.fixture
   def mmap_module(tmp_path):
       module = MmappedDataModule(scratch_dir=tmp_path, ...)
       module.setup("fit")
       yield module
       # Explicit cleanup
       module.teardown("fit")
       # Force garbage collection to release mmaps
       import gc; gc.collect()
   ```
2. Override `teardown()` in `MmappedDataModule` to explicitly close all `MemoryMappedTensor` handles.
3. On Windows CI, use `shutil.rmtree(tmp_path, ignore_errors=True)` as a fallback.
4. Consider using `pytest-tmp-files` or a custom plugin that handles mmap cleanup.

**Warning signs:**
- `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process` on Windows CI
- `/tmp` filling up on CI runners
- Tests pass locally but leave zombie files

**Phase to address:**
Test migration phase.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `[all]` extra in CI instead of per-extra testing | Simpler CI config | Never catches missing lazy import guards; broken base installs ship | Never in isolation; always combine with a base-deps-only job |
| Copying hcs.py without extracting `_utils.py` | Faster initial extraction | Shared helpers remain in a concrete class; 5 modules import from hcs.py for utility functions, not for `HCSDataModule` | Never -- extract `_utils.py` first, it's the prerequisite |
| Keeping `Manager().dict()` as-is on all platforms | No refactoring needed | macOS/Windows CI failures; users on non-Linux can't use CachedOmeZarrDataModule | Acceptable if you document Linux-only + skip in CI |
| Re-exporting all classes in `__init__.py` | Nice DX, `from viscy_data import X` for everything | Import of `viscy_data` fails without optional deps | Never for a package with optional deps |
| Ignoring the typing.py `DictTransform` duplication | No cross-package dependency | Two identical types that can drift apart if definition changes | Acceptable -- it's a one-line type alias; duplication is cheaper than coupling |

## Integration Gotchas

Common mistakes when connecting viscy-data to external services and libraries.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| **iohub OME-Zarr** | Not pinning minimum iohub version; API changes break at runtime | Pin `iohub>=X.Y` where X.Y is the version that stabilized Position/Plate API |
| **Lightning Trainer** | Testing DataModule.setup() without a Trainer; missing DDP context | Use `Trainer(accelerator="cpu", devices=1)` in test fixtures even for CPU tests |
| **MONAI transforms** | Importing `monai.transforms` triggers MONAI's full init (slow); `set_track_meta(False)` leaks between tests | Set `monai.utils.set_track_meta(False)` in a session-scoped fixture; reset after |
| **tensorstore** | Assuming tensorstore is available on all platforms; arm64 macOS wheels may be missing | Guard with `pytest.importorskip("tensorstore")` and add platform skip markers |
| **tensordict** | Creating `MemoryMappedTensor` without checking `scratch_dir` exists | Always `Path(scratch_dir).mkdir(parents=True, exist_ok=True)` in setup() |
| **pycocotools** | Including in `[all]` extra without platform guards | Add `; sys_platform != "win32"` environment marker or document Windows limitation |
| **uv workspace** | Forgetting `workspace = true` source for iohub/other workspace members | viscy-data should NOT have workspace sources for iohub (it's an external dep, not a workspace member) |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Session-scoped zarr fixtures shared mutably | First test that modifies the fixture corrupts all subsequent tests | Make session fixtures read-only; use function-scoped for mutable tests | >10 tests using same fixture |
| Creating new iohub `open_ome_zarr()` per test function | Test suite takes minutes; OS file handle limit hit | Session-scoped fixture; reuse same store | >50 data module tests |
| `num_workers>0` in ALL test DataLoaders | Each test spawns worker processes; CI runs out of memory | `num_workers=0` default in test fixtures; `>0` only for marked integration tests | >20 DataModule tests on CI runner with 4GB RAM |
| Full CI matrix with all extras on all platforms | 45+ jobs; 30+ minute CI; GitHub Actions quota consumed | Tiered CI: fast (base, 3x3) + full (all extras, 1x1) | When extras count > 2 |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Package installs:** Verify `pip install viscy-data` works AND `import viscy_data` works without extras -- test the base install path
- [ ] **Lazy imports:** Every optional dependency has a guarded import AND a test that verifies the error message when missing
- [ ] **`_utils.py` extraction:** All 5 helper functions moved out of `hcs.py` AND all 5 callers updated to import from `_utils.py`
- [ ] **Cross-platform Manager().dict():** Tested with `multiprocessing.set_start_method("spawn")` not just default `fork`
- [ ] **`__init__.py` re-exports:** Only re-export classes from modules with required deps; optional-dep modules are import-on-demand
- [ ] **CI minimal-deps job:** At least one CI job installs viscy-data WITHOUT any extras and runs the base test suite
- [ ] **iohub version pin:** `iohub>=X.Y` set to a version you have actually tested against, not just `iohub`
- [ ] **Config path grep:** All YAML/JSON files in the repo searched for `viscy.data` references and updated
- [ ] **MRO preserved:** Multiple inheritance order unchanged from original code; explicit MRO tests exist
- [ ] **MemoryMappedTensor cleanup:** Tests that use mmap_cache explicitly close resources in fixture teardown
- [ ] **ThreadDataLoader lifecycle:** Triplet tests explicitly shut down thread pools in fixture teardown
- [ ] **DDP compatibility:** `ShardedDistributedSampler` imported and tested even in single-device test configuration

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Lazy import guard missing (runtime NameError) | LOW | Add guard, release patch version. No API change. |
| Manager().dict() fails on macOS/Windows | MEDIUM | Refactor to file-based cache or document Linux-only; may require API change to setup() |
| Circular import from _utils.py extraction | LOW | Re-order imports or move offending function to correct module. Usually a 1-line fix. |
| CI matrix explosion | LOW | Simplify matrix in workflow YAML. No code change needed. |
| pycocotools Windows build failure | LOW | Add environment marker `; sys_platform != "win32"` to extra. 1-line pyproject.toml change. |
| iohub API break | HIGH | Must update all data modules that use the changed API. May require coordinated release. |
| `__init__.py` eager import breaks base install | LOW | Change to lazy imports or remove re-export. Patch release. |
| Zarr fixtures too slow in CI | MEDIUM | Refactor to session-scoped fixtures. Requires rewriting test structure. |
| Lightning CLI config class_path broken | LOW per config | Find-and-replace `viscy.data.X` -> `viscy_data.X`. Tedious but mechanical. |
| MRO break from refactored __init__ | MEDIUM | Revert to original inheritance order. May require undoing "cleanup" refactoring. |
| MemoryMappedTensor file leak on Windows | MEDIUM | Add explicit cleanup in teardown. May require adding teardown() to DataModule. |
| ThreadDataLoader resource leak in tests | MEDIUM | Add fixture teardown. May require restructuring test isolation. |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1: Lazy import guards | Scaffolding (create `_imports.py` pattern) | CI minimal-deps job passes |
| P2: Manager().dict() spawn | Code migration | macOS CI job passes with `num_workers>0` |
| P3: Import cycles from _utils.py | Code migration (extract _utils.py first) | `python -c "from viscy_data import X"` for all public classes |
| P4: CI matrix explosion | CI phase | CI completes in <15 min with full coverage |
| P5: pycocotools Windows | Scaffolding (env markers) + CI (matrix exclude) | Windows CI job passes for base deps |
| P6: iohub API coupling | Scaffolding (version pin) + Tests (API surface test) | Integration test verifies iohub types |
| P7: ThreadDataLoader test leaks | Test migration (fixture design) | Full test suite passes without resource warnings |
| P8: `__init__.py` eager imports | Scaffolding (design `__init__.py`) | `import viscy_data` works without extras installed |
| P9: Expensive zarr fixtures | Test migration (conftest.py design) | Test suite completes in <3 min on CI |
| P10: Config class_path breakage | Code migration (grep + update) | No `viscy.data.` references in repo YAML files |
| P11: MRO fragility | Code migration (copy-first rule) | MRO test for every multi-inherited class |
| P12: MemoryMappedTensor cleanup | Test migration (fixture teardown) | Windows CI passes without PermissionError |

## Sources

### Project-Specific (HIGH confidence)
- `viscy/data/README.md` -- Module inventory, class hierarchy, dependency graph, GPU transform patterns (primary source for all architecture pitfalls)
- `.planning/PROJECT.md` -- Decisions on clean break imports, no viscy-transforms dependency, optional extras
- `.planning/research/PITFALLS.md` (v1.0) -- Workspace-level pitfalls from first extraction (referenced but not duplicated)
- `.planning/research/ARCHITECTURE.md` -- Workspace structure, anti-patterns
- `packages/viscy-transforms/pyproject.toml` -- Pattern for package structure, existing CI approach
- `.github/workflows/test.yml` -- Current 3x3 CI matrix structure

### Domain Knowledge (MEDIUM confidence)
- PyTorch multiprocessing `fork` vs `spawn` behavior with Manager proxies -- well-documented in PyTorch docs and numerous issue reports
- Python MRO and cooperative `super()` in multiple inheritance -- standard Python language behavior
- Memory-mapped file handle semantics on Windows vs Unix -- OS-level behavior, well-established
- `monai.data.ThreadDataLoader` threading model -- MONAI documentation
- pycocotools build requirements -- package metadata and common CI failure pattern
- tensorstore platform availability -- observed from PyPI wheel listings

### Training Data (LOW confidence, flagged)
- Specific tensorstore arm64 macOS wheel availability -- may have changed; verify against current PyPI
- Exact pycocotools Windows wheel coverage -- verify against current PyPI for target Python versions
- tensordict `MemoryMappedTensor` cleanup API -- verify against current tensordict docs

---
*Pitfalls research for: viscy-data extraction (milestone v1.1)*
*Researched: 2026-02-13*
