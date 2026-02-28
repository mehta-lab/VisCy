# Phase 26: Refactor Translation Application - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Extract the `viscy/translation/` module into a standalone application at `applications/translation/` that composes shared packages (viscy-data, viscy-models, viscy-transforms, viscy-utils). The translation app handles virtual staining (fluorescence prediction from phase images) using UNet-family architectures. Reusable components (prediction writer, losses, metrics, image logging) are extracted to shared packages.

</domain>

<decisions>
## Implementation Decisions

### Package Boundary
- VSUNet LightningModule (engine.py) → `applications/translation/src/viscy_translation/engine.py` (app-specific)
- HCSPredictionWriter (predict_writer.py) → `packages/viscy-utils/src/viscy_utils/callbacks/` (reusable)
- MixedLoss → `packages/viscy-utils/src/viscy_utils/losses/` (reusable reconstruction loss)
- Metric functions (VOI, POD, mAP, MS-SSIM) → `packages/viscy-utils/src/viscy_utils/evaluation/` (reusable)
- SegmentationMetrics2D test runner → `applications/translation/` (app-specific)
- Image logging utilities (log_images) → `packages/viscy-utils/` (reusable)
- Architecture dispatch dict (_UNET_ARCHITECTURE) → stays app-level in the translation app

### Application Structure
- Follow dynaclr layout exactly: `applications/translation/` with `src/viscy_translation/`, `tests/`, `examples/`, `evaluation/`
- Installable package name: `viscy-translation` (pip install viscy-translation)
- Top-level import: `from viscy_translation import VSUNet`
- Exclude from uv workspace (same as dynaclr)
- YAML config files (fit.yml, predict.yml) via Lightning CLI
- Move code as-is with minimal cleanup — no refactoring of VSUNet during migration

### Import Path Design
- `from viscy_translation import VSUNet` — top-level re-export in __init__.py
- `from viscy_utils.callbacks import HCSPredictionWriter` — follows EmbeddingWriter pattern
- `from viscy_utils.losses import MixedLoss` — new losses submodule in viscy-utils
- `from viscy_utils.evaluation import ms_ssim_25d, mean_average_precision` — existing evaluation module
- Lightning CLI entry point: `python -m viscy_translation fit --config fit.yml`
- No umbrella package re-exports (clean break)

### Backward Compatibility
- Clean break on imports: `from viscy.translation import X` stops working
- Update example configs to use new class_path (`viscy_translation.engine.VSUNet`)
- Delete old `viscy/translation/` directory after migration is verified
- Checkpoint compatibility via state_dict loading (instantiate new class, load weights)
- Include state dict regression test to verify key compatibility (COMPAT pattern from viscy-models)

### Dependencies to Verify
- Verify old imports (viscy.data.combined, viscy.data.gpu_aug, viscy.data.typing, viscy.utils.log_images) have equivalents in viscy-data/viscy-utils
- Migrate log_images to viscy-utils if not already there
- Research step should confirm what's available and what needs migrating

### Testing
- Include basic tests: import tests, config validation, smoke test
- State dict regression test for checkpoint compatibility

### Claude's Discretion
- Exact submodule organization within viscy_translation (whether to split engine.py into multiple files)
- YAML config content (what defaults to use)
- Which evaluation metrics to re-export at top level
- Test fixture design and synthetic data approach

</decisions>

<specifics>
## Specific Ideas

- Follow the dynaclr application as the reference pattern — same directory structure, same workspace exclusion, same Lightning CLI approach
- MixedLoss from engine.py should be extracted cleanly since it's a standalone nn.Module with no app-specific dependencies
- The COMPAT pattern from viscy-models Phase 10 (state dict key regression tests) should be replicated for VSUNet

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 26-refactor-translation-application*
*Context gathered: 2026-02-27*
