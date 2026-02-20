# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Independent, reusable subpackages with clean import paths
**Current focus:** v2.1 DynaCLR Integration Validation

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-19 — Milestone v2.1 started

## Performance Metrics

**Combined velocity (all branches):**
- Total plans completed: 25 (v1.0: 7, v1.1: 9, v1.2: 9) + v2.0 manual phases

**By Milestone:**

| Milestone | Phases | Plans | Branch |
|-----------|--------|-------|--------|
| v1.0 Transforms | 1-5 | 7 | shared |
| v1.1 Data | 6-9 | 9 | modular-data |
| v1.2 Models | 10-14 | 9 | modular-models |
| v2.0 DynaCLR | 15-17 | manual | app-dynaclr |

## Accumulated Context

### Decisions

Key decisions carrying forward from all milestones:

**Architecture:**
- Clean break on imports: `from viscy_{pkg} import X` (no backward compatibility)
- hatchling + uv-dynamic-versioning for build system
- src layout, tests inside packages, uv-only tooling
- No cross-package dependencies between transforms, data, and models
- Flat public API pattern (MONAI-style) across all packages
- Applications compose packages: `dynacrl` depends on viscy-data, viscy-models, viscy-transforms, viscy-utils

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

**v2.0 DynaCLR-specific:**
- viscy-utils extracts shared training infrastructure (trainer, callbacks, evaluation)
- LazyCommand CLI pattern defers heavy imports; graceful fallback on missing extras
- Evaluation scripts live outside package src/ (standalone); CLI wires them via sys.path
- cli_utils.py provides format_markdown_table() and load_config() (pyyaml dependency)
- dynacrl optional [eval] extras: anndata, natsort, wandb, scikit-learn, phate, umap-learn
- YAML config class_path references: dynacrl.engine, viscy_models.contrastive, viscy_data.triplet, viscy_transforms

### Blockers/Concerns

None currently.

## Next Steps

Milestone v2.1: Define requirements → create roadmap → execute phases.

Future candidates (after v2.1):
- **APP-02**: applications/Cytoland — VSUNet/FcmaeUNet LightningModules
- **APP-03**: viscy-airtable — abstract from current Airtable integration
- **HYDRA-***: Hydra infrastructure (BaseModel, ConfigStore, registry)

## Session Continuity

Last session: 2026-02-19
Stopped at: Starting milestone v2.1 DynaCLR Integration Validation
Resume file: None

---
*State initialized: 2025-01-27*
*Harmonized from modular-data + modular-models branches: 2026-02-16*
*Updated for v2.0 DynaCLR: 2026-02-17*
*Updated for v2.1 DynaCLR Integration Validation: 2026-02-19*
