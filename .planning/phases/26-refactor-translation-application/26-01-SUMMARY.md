---
phase: 26-refactor-translation-application
plan: 01
subsystem: infra
tags: [viscy-utils, callbacks, losses, workspace, lightning-cli, ome-zarr]

# Dependency graph
requires:
  - phase: 15-shared-infrastructure
    provides: viscy-utils package with callbacks, evaluation, cli
  - phase: 06-package-scaffolding
    provides: viscy-data package with HCSDataModule, Sample types
provides:
  - HCSPredictionWriter callback in viscy_utils.callbacks
  - MixedLoss reconstruction loss in viscy_utils.losses
  - Translation application scaffold at applications/translation/
  - viscy-translation workspace registration in root pyproject.toml
  - Example YAML configs for fit and predict workflows
affects: [26-02-engine-migration]

# Tech tracking
tech-stack:
  added: [viscy-translation]
  patterns: [TYPE_CHECKING guard for cross-package type imports, losses submodule pattern in viscy-utils]

key-files:
  created:
    - packages/viscy-utils/src/viscy_utils/callbacks/prediction_writer.py
    - packages/viscy-utils/src/viscy_utils/losses/__init__.py
    - packages/viscy-utils/src/viscy_utils/losses/mixed_loss.py
    - applications/translation/pyproject.toml
    - applications/translation/README.md
    - applications/translation/src/viscy_translation/__init__.py
    - applications/translation/src/viscy_translation/__main__.py
    - applications/translation/tests/__init__.py
    - applications/translation/examples/configs/fit.yml
    - applications/translation/examples/configs/predict.yml
  modified:
    - packages/viscy-utils/src/viscy_utils/callbacks/__init__.py
    - packages/viscy-utils/src/viscy_utils/callbacks/embedding_snapshot.py
    - packages/viscy-utils/src/viscy_utils/callbacks/embedding_writer.py
    - pyproject.toml
    - uv.lock

key-decisions:
  - "TYPE_CHECKING guard for viscy_data imports in HCSPredictionWriter to avoid runtime dependency on viscy-data from viscy-utils"
  - "viscy-translation registered in workspace sources following dynaclr pattern exactly"
  - "__main__.py delegates to viscy_utils.cli.main for shared LightningCLI entry point"

patterns-established:
  - "TYPE_CHECKING guard pattern: use from __future__ import annotations + TYPE_CHECKING for cross-package type-only imports"
  - "Losses submodule: viscy_utils.losses as location for shared reconstruction losses"

requirements-completed: []

# Metrics
duration: 43min
completed: 2026-02-27
---

# Phase 26 Plan 01: Shared Infrastructure Extraction + Application Scaffold Summary

**HCSPredictionWriter and MixedLoss extracted to viscy-utils, translation app scaffold created with workspace registration and LightningCLI entry point**

## Performance

- **Duration:** 43 min
- **Started:** 2026-02-27T23:10:18Z
- **Completed:** 2026-02-27T23:53:43Z
- **Tasks:** 3
- **Files modified:** 15

## Accomplishments
- HCSPredictionWriter callback extracted from viscy/translation/predict_writer.py to viscy_utils.callbacks with TYPE_CHECKING guard for viscy_data types
- MixedLoss reconstruction loss extracted from viscy/translation/engine.py to new viscy_utils.losses submodule using ms_ssim_25d from viscy_utils.evaluation.metrics
- Translation application scaffold created at applications/translation/ with proper src layout, README.md, LightningCLI entry point, and example YAML configs
- viscy-translation registered in root pyproject.toml workspace sources, uv sync resolves cleanly

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract HCSPredictionWriter to viscy-utils callbacks** - `995e886` (feat)
2. **Task 2: Extract MixedLoss to viscy-utils losses submodule** - `8ce020b` (feat)
3. **Task 3: Create translation application scaffold with workspace registration** - `59cd777` (feat)

## Files Created/Modified
- `packages/viscy-utils/src/viscy_utils/callbacks/prediction_writer.py` - HCSPredictionWriter callback with _pad_shape, _resize_image, _blend_in helpers
- `packages/viscy-utils/src/viscy_utils/callbacks/__init__.py` - Added HCSPredictionWriter re-export
- `packages/viscy-utils/src/viscy_utils/losses/mixed_loss.py` - MixedLoss (L1 + L2 + MS-DSSIM) reconstruction loss
- `packages/viscy-utils/src/viscy_utils/losses/__init__.py` - MixedLoss re-export
- `applications/translation/pyproject.toml` - Package config with hatchling build, workspace deps
- `applications/translation/README.md` - Minimal README required by hatchling
- `applications/translation/src/viscy_translation/__init__.py` - Package placeholder
- `applications/translation/src/viscy_translation/__main__.py` - LightningCLI entry point
- `applications/translation/tests/__init__.py` - Empty test package
- `applications/translation/examples/configs/fit.yml` - Example training config
- `applications/translation/examples/configs/predict.yml` - Example prediction config with HCSPredictionWriter
- `pyproject.toml` - Added viscy-translation workspace source
- `uv.lock` - Updated lockfile with new workspace member
- `packages/viscy-utils/src/viscy_utils/callbacks/embedding_snapshot.py` - Fixed INDEX_COLUMNS -> ULTRACK_INDEX_COLUMNS
- `packages/viscy-utils/src/viscy_utils/callbacks/embedding_writer.py` - Fixed INDEX_COLUMNS -> ULTRACK_INDEX_COLUMNS

## Decisions Made
- Used TYPE_CHECKING guard for viscy_data imports (HCSDataModule, Sample) in HCSPredictionWriter to avoid adding viscy-data as a runtime dependency of viscy-utils
- Followed dynaclr pyproject.toml pattern exactly for translation application scaffold
- __main__.py delegates to viscy_utils.cli.main for shared LightningCLI entry point (same entry point as the `viscy` console script)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed INDEX_COLUMNS -> ULTRACK_INDEX_COLUMNS in embedding callbacks**
- **Found during:** Task 1 (HCSPredictionWriter extraction)
- **Issue:** embedding_snapshot.py and embedding_writer.py imported `INDEX_COLUMNS` from `viscy_data._typing`, but the actual export is `ULTRACK_INDEX_COLUMNS`. This prevented importing from `viscy_utils.callbacks` entirely.
- **Fix:** Renamed all references from `INDEX_COLUMNS` to `ULTRACK_INDEX_COLUMNS` in both files
- **Files modified:** embedding_snapshot.py, embedding_writer.py
- **Verification:** `from viscy_utils.callbacks import HCSPredictionWriter` succeeds
- **Committed in:** 995e886 (Task 1 commit)

**2. [Rule 1 - Bug] Added missing docstrings to embedding_snapshot.py public methods**
- **Found during:** Task 1 (pre-commit hook failure)
- **Issue:** Three public methods in EmbeddingSnapshotCallback lacked docstrings, triggering D102 ruff violations
- **Fix:** Added one-line docstrings to on_validation_epoch_start, on_validation_batch_end, on_validation_epoch_end
- **Files modified:** embedding_snapshot.py
- **Verification:** ruff check passes
- **Committed in:** 995e886 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for import chain to work. No scope creep.

## Issues Encountered
- HPC filesystem cache issue with `uv sync` install step (failed to remove pycache dirs) -- not related to our changes, `uv sync --dry-run` confirms resolution is clean, and `uv sync --all-packages` installed successfully

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- HCSPredictionWriter and MixedLoss are importable from viscy-utils, ready for Plan 02 to reference
- Translation application scaffold is in place with workspace registration
- Plan 02 can now populate the scaffold with VSUNet, FcmaeUNet engine code and tests

## Self-Check: PASSED

All 10 created files verified present. All 3 task commits (995e886, 8ce020b, 59cd777) verified in git log.

---
*Phase: 26-refactor-translation-application*
*Completed: 2026-02-27*
