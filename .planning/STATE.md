# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** Harmonization complete — ready for v2.0 milestone definition

## Current Position

Phase: 14 of 14 (all shipped)
Status: v1.0, v1.1, v1.2 milestones complete. v2.0 not yet defined.
Last activity: 2026-02-16 — Harmonized modular-data + modular-models planning docs

Progress: [==================] 100% (all milestones through v1.2 complete)

## Performance Metrics

**Combined velocity (from both branches):**
- Total plans completed: 25 (v1.0: 7, v1.1: 9, v1.2: 9)

**By Milestone:**

| Milestone | Phases | Plans | Branch |
|-----------|--------|-------|--------|
| v1.0 Transforms | 1-5 | 7 | shared |
| v1.1 Data | 6-9 | 9 | modular-data |
| v1.2 Models | 10-14 | 9 | modular-models |

## Accumulated Context

### Decisions

Key decisions carrying forward from all milestones:

**Architecture:**
- Clean break on imports: `from viscy_{pkg} import X` (no backward compatibility)
- hatchling + uv-dynamic-versioning for build system
- src layout, tests inside packages, uv-only tooling
- No cross-package dependencies between transforms, data, and models
- Flat public API pattern (MONAI-style) across all packages

**Data-specific:**
- No viscy-transforms dependency: assert batch shape instead of BatchedCenterSpatialCropd
- Optional dependency groups: tensorstore, tensordict, pycocotools as extras
- Lazy import pattern for optional deps: try/except at module level, guard in __init__
- Extract shared utilities from hcs.py into _utils.py before migration
- combined.py preserved as-is (no split per REF-02 deferral)

**Models-specific:**
- Pure nn.Module in viscy-models: No Lightning/Hydra coupling
- Function-based grouping: unet/, vae/, contrastive/ with shared _components/
- State dict key compatibility non-negotiable for checkpoint loading
- Mutable defaults fixed to tuples during migration
- Deconv decoder channel mismatch in UNeXt2UpStage: pre-existing bug, xfailed test

### Blockers/Concerns

None currently.

## Next Steps

v2.0 milestone needs definition. Candidate features:
- **APP-01**: applications/DynaCLR — ContrastiveModule LightningModule
- **APP-02**: applications/Cytoland — VSUNet/FcmaeUNet LightningModules
- **APP-03**: viscy-airtable — abstract from current Airtable integration
- **HYDRA-***: Hydra infrastructure (BaseModel, ConfigStore, registry)

## Session Continuity

Last session: 2026-02-16
Stopped at: Harmonized .planning/ docs from modular-data + modular-models branches
Resume file: None

---
*State initialized: 2025-01-27*
*Harmonized from modular-data + modular-models branches: 2026-02-16*
