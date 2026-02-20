---
phase: 18-training-validation
plan: 01
subsystem: testing
tags: [lightning, contrastive-learning, fast_dev_run, tensorboard, yaml-config]

# Dependency graph
requires:
  - phase: 15-17 (v2.0 DynaCLR manual phases)
    provides: ContrastiveModule engine, configs, package structure
provides:
  - Training integration tests proving ContrastiveModule trains end-to-end
  - Config class_path resolution validation for fit.yml and predict.yml
  - Workspace exclude fix for non-package application directories
affects: [19-inference-validation]

# Tech tracking
tech-stack:
  added: [tensorboard (test dep)]
  patterns: [synthetic TripletSample data for Lightning fast_dev_run, parametrized config validation]

key-files:
  created:
    - applications/dynacrl/tests/test_training_integration.py
  modified:
    - applications/dynacrl/pyproject.toml
    - pyproject.toml
    - uv.lock

key-decisions:
  - "Used TensorBoardLogger with tmp_path instead of logger=False to exercise full on_epoch_end logging code path"
  - "Used (1,1,4,4) tensor shape to produce valid 2D images after detach_sample mid-depth slicing"
  - "Added tensorboard as test dependency rather than mocking _log_samples"
  - "Added workspace exclude for non-package application directories (benchmarking, contrastive_phenotyping, qc)"

patterns-established:
  - "Integration test pattern: SimpleEncoder + SyntheticTripletDataModule + fast_dev_run for Lightning training loop validation"
  - "Config validation pattern: recursive class_path extraction + importlib resolution"

requirements-completed: [TRAIN-01, TRAIN-02]

# Metrics
duration: 5min
completed: 2026-02-20
---

# Phase 18 Plan 01: Training Integration Tests Summary

**ContrastiveModule fast_dev_run training loop validated with TripletMarginLoss and NTXentLoss code paths, plus YAML config class_path resolution for all fit.yml and predict.yml references**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-20T07:23:28Z
- **Completed:** 2026-02-20T07:29:23Z
- **Tasks:** 1
- **Files modified:** 4

## Accomplishments
- ContrastiveModule completes full Lightning training loop (training_step, validation_step, on_train_epoch_end with TensorBoard image logging, on_validation_epoch_end, configure_optimizers) via fast_dev_run
- Both loss function code paths validated: TripletMarginLoss (anchor/positive/negative) and NTXentLoss (anchor/positive only, label-based)
- All class_path strings in fit.yml and predict.yml verified to resolve to importable Python classes (dynacrl.engine, viscy_models, viscy_data, viscy_transforms, viscy_utils)
- Full test suite (6 tests) passes without regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create fast_dev_run training integration test for ContrastiveModule** - `5c34dc47` (feat)

**Plan metadata:** (pending final commit)

## Files Created/Modified
- `applications/dynacrl/tests/test_training_integration.py` - Training integration tests (4 tests: fast_dev_run with TripletMarginLoss, fast_dev_run with NTXentLoss, config class_path resolution for fit.yml and predict.yml)
- `applications/dynacrl/pyproject.toml` - Added tensorboard to test dependencies
- `pyproject.toml` - Added workspace exclude for non-package application directories
- `uv.lock` - Updated lock file with tensorboard dependency tree

## Decisions Made
- Used TensorBoardLogger with tmp_path instead of `logger=False` to exercise the full `on_train_epoch_end` -> `_log_samples` -> `render_images` -> `add_image` code path, proving the logging pipeline works end-to-end
- Used (C=1, D=1, H=4, W=4) tensor shapes instead of (1,1,1,10) so that `detach_sample` produces valid 2D numpy arrays that `render_images` can process (mid-depth slice + squeeze yields 4x4 images)
- Added `tensorboard` as a test dependency rather than mocking `_log_samples`, since production configs use TensorBoardLogger
- Fixed workspace config with `exclude` patterns for non-package application directories that lack pyproject.toml

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed workspace config excluding non-package application directories**
- **Found during:** Task 1 (running tests)
- **Issue:** `applications/*` glob in `[tool.uv.workspace].members` matched `benchmarking`, `contrastive_phenotyping`, `qc` directories that have no pyproject.toml, causing `uv run` to fail
- **Fix:** Added `exclude = ["applications/benchmarking", "applications/contrastive_phenotyping", "applications/qc"]` to workspace config
- **Files modified:** pyproject.toml
- **Verification:** `uv run --package dynacrl pytest` succeeds
- **Committed in:** 5c34dc47 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed tensor shape for render_images compatibility**
- **Found during:** Task 1 (test_contrastive_fast_dev_run failure)
- **Issue:** Plan-specified shape (1,1,1,10) produces 1D arrays after detach_sample mid-depth slicing, which render_images cannot process (expects 2D images)
- **Fix:** Changed to (1,1,4,4) producing proper 4x4 images after slicing, with FLAT_DIM=16 for SimpleEncoder
- **Files modified:** applications/dynacrl/tests/test_training_integration.py
- **Verification:** All 4 tests pass
- **Committed in:** 5c34dc47 (Task 1 commit)

**3. [Rule 3 - Blocking] Added tensorboard test dependency**
- **Found during:** Task 1 (TensorBoardLogger ModuleNotFoundError)
- **Issue:** TensorBoardLogger requires tensorboard or tensorboardX, neither installed in test dependencies
- **Fix:** Added `tensorboard` to `[dependency-groups].test` in applications/dynacrl/pyproject.toml
- **Files modified:** applications/dynacrl/pyproject.toml, uv.lock
- **Verification:** TensorBoardLogger initializes without error
- **Committed in:** 5c34dc47 (Task 1 commit)

**4. [Rule 1 - Bug] Fixed config path resolution in test_config_class_paths_resolve**
- **Found during:** Task 1 (config path assertion failure)
- **Issue:** Plan specified `parents[2]` which resolves to `applications/` instead of `applications/dynacrl/`
- **Fix:** Changed to `parents[1]` to correctly reach `applications/dynacrl/examples/configs/`
- **Files modified:** applications/dynacrl/tests/test_training_integration.py
- **Verification:** Both config tests pass
- **Committed in:** 5c34dc47 (Task 1 commit)

---

**Total deviations:** 4 auto-fixed (2 bugs, 2 blocking)
**Impact on plan:** All fixes necessary for tests to run. No scope creep.

## Issues Encountered
- Stale `__pycache__` in `applications/` directory initially caused workspace resolution failure (removed via Python shutil)
- Stale numpy `__pycache__` in `.venv` caused `uv` package installation failure (removed via Python shutil)

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Training integration validated, ready for Phase 19 (inference/prediction validation)
- All 6 dynacrl tests pass (2 smoke tests + 4 integration tests)
- Checkpoint loading tests (Phase 19) will need real checkpoint paths from user

## Self-Check: PASSED

- FOUND: applications/dynacrl/tests/test_training_integration.py
- FOUND: .planning/phases/18-training-validation/18-01-SUMMARY.md
- FOUND: commit 5c34dc47

---
*Phase: 18-training-validation*
*Completed: 2026-02-20*
