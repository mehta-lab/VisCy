# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-12)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Phase 7 - Core UNet Models

## Current Position

Phase: 7 of 10 (Core UNet Models) -- PHASE COMPLETE
Plan: 2 of 2 in current phase
Status: Phase 7 Complete
Last activity: 2026-02-13 -- Completed 07-02 FCMAE migration

Progress: [=============-----] 68% (v1.0 complete, v1.1 phase 7: 2/2 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 12 (v1.0: 7, v1.1: 5)
- Average duration: ~19 min
- Total execution time: ~3.7 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 2 | ~60m | ~30m |
| 2. Package | 1 | ~30m | ~30m |
| 3. Migration | 3 | ~90m | ~30m |
| 5. CI/CD | 1 | ~30m | ~30m |
| 6. Package Scaffold | 3 | ~10m | ~3m |
| 7. Core UNet Models | 2 | ~6m | ~3m |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Pure nn.Module in viscy-models: No Lightning/Hydra coupling
- Function-based grouping: unet/, vae/, contrastive/ with shared _components/
- viscy-models independent of viscy-transforms (torch/timm/monai/numpy only)
- 14+ shared components in unext2.py need extraction to _components/
- Mutable defaults must be fixed to tuples during migration
- State dict key compatibility is non-negotiable for checkpoint loading
- Followed viscy-transforms pyproject.toml pattern exactly for consistency
- No optional-dependencies for viscy-models (no notebook extras needed)
- Dev dependency group includes only test (no jupyter for models package)
- Preserved register_modules/add_module pattern verbatim for state dict key compatibility
- Fixed only docstring formatting for ruff D-series compliance, no logic changes to legacy code
- Intra-_components import allowed: heads.py imports icnr_init from blocks.py (no circular risk)
- _get_convnext_stage private but importable; excluded from __all__
- Preserved exact list mutation pattern (decoder_channels = num_channels alias) in UNeXt2 for compatibility
- Marked deconv decoder test as xfail due to pre-existing channel mismatch bug in original code
- Fixed deconv tuple assignment bug in UNeXt2UpStage (trailing comma created tuple instead of module)
- Removed PixelToVoxelShuffleHead duplication from fcmae.py; import from canonical _components.heads location
- Fixed mutable list defaults (encoder_blocks, dims) to tuples in FullyConvolutionalMAE

### Pending Todos

- Fix deconv decoder channel mismatch in UNeXt2UpStage (pre-existing bug, xfailed test documents it)

### Blockers/Concerns

None currently.

## v1.0 Completion Summary

All 5 phases complete (Phase 4 Documentation deferred). See MILESTONES.md.

## Session Continuity

Last session: 2026-02-13
Stopped at: Completed 07-02-PLAN.md (FCMAE migration -- Phase 7 complete)
Resume file: None

---
*State initialized: 2025-01-27*
*Last updated: 2026-02-13 (07-02 summary added, Phase 7 complete)*
