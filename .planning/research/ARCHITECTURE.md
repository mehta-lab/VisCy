# Architecture Research

**Project:** Airtable AI QC — Microscopy Data Pipeline
**Confidence:** HIGH

## Key Finding: No New Application Needed

The pipeline extends `applications/airtable` and `applications/qc`. A new DAG orchestrator script lives alongside existing CLI entry points. The monorepo constraint "applications must not import from each other" is respected.

**Critical fix first:** `applications/qc/src/qc/config.py` imports `ChannelAnnotationEntry` and `WellExperimentMetadata` from `airtable_utils.schemas`. This cross-application import violation must be fixed by moving shared models to `packages/viscy-data/src/viscy_data/schemas.py` before the pipeline grows further.

## Component Boundaries

### Existing (Extend)

| Component | Location | Role | Changes Needed |
|-----------|----------|------|---------------|
| `airtable-utils` | `applications/airtable/` | Registration CLI, Airtable CRUD | Add post-assembly auto-register, metadata backfill, pipeline state tracking |
| `qc` | `applications/qc/` | Focus QC, preprocessing | Add automated QC trigger, QC parquet writer, cell-level flags |
| `viscy-data` | `packages/viscy-data/` | Schemas, cell_index, samplers | Add QC columns to cell_index schema, shared Pydantic models from airtable |
| `viscy-utils` | `packages/viscy-utils/` | CLI utilities | Add `viscy pipeline` CLI entry point |

### New (Minimal)

| Component | Location | Role |
|-----------|----------|------|
| Pipeline DAG config | `applications/airtable/configs/pipeline/` | YAML step definitions with SLURM dependencies |
| Pipeline runner | `applications/airtable/src/airtable_utils/pipeline.py` | Parse DAG YAML, submit SLURM jobs with `--dependency=afterok` |
| Per-FOV state tracker | `applications/airtable/src/airtable_utils/state.py` | Read/write `.pipeline_state.yaml` per dataset |

## Data Flow

```
[1] Biologist → Airtable platemap (well-level records via Add Platemap view)
         │
[2] biahub reconstruction → OME-Zarr on VAST/NFS (2-assembled/{dataset}.zarr)
         │
[3] Metadata backfill → Write pixel sizes, recon params back to zarr .zattrs
         │
    ┌────┴────┐
    │         │
[4a] airtable-utils register    [4b] qc run (focus.yml)
    │  → FOV records in Airtable      │  → .zattrs updated with QC metrics
    │  → pixel sizes, channels         │  → qc_data.parquet written
    │  → VAST/NFS paths                │
    └────┬────┘
         │
[5] viscy preprocess → Normalized zarr
         │
[6] dynaclr build-cell-index → Flat parquet with QC columns
         │  → is_blank_timepoint, focus_z_index, qc_passed
         │
[7] dynaclr train → MultiExperimentDataModule → ContrastiveModule
         │
[8] Airtable: update pipeline_status per dataset
```

**Key:** Steps 4a and 4b run in parallel (both depend on step 3). Step 5 depends on 4a. Step 6 depends on 4a, 4b, and 5.

## Shared Schema Fix

**Before:**
```
applications/qc/ → imports from → applications/airtable/  (VIOLATION)
```

**After:**
```
applications/qc/ → imports from → packages/viscy-data/
applications/airtable/ → imports from → packages/viscy-data/
```

Models to move to `packages/viscy-data/src/viscy_data/schemas.py`:
- `ChannelAnnotationEntry`
- `WellExperimentMetadata`
- Any other Pydantic models used by both applications

## Build Order

Hard dependency chain — each step requires the previous:

1. **Fix shared schema location** — Move cross-app models to `viscy-data`
2. **Metadata backfill** — Ensure pixel sizes exist in zarr before registration/QC
3. **Extend registration** — Auto-register after assembly, VAST/NFS paths
4. **QC automation** — Auto-run after assembly, write parquet output
5. **Cell index QC integration** — Add QC columns to flat parquet schema
6. **DAG orchestrator** — YAML config + SLURM runner composing steps 2-5
7. **Pipeline checklist** — Status reporting and completeness validation

## Where New Code Lives

| Code | Location | Rationale |
|------|----------|-----------|
| Pipeline YAML configs | `applications/airtable/configs/pipeline/` | Pipeline is Airtable-centric; registration is the pivotal step |
| Pipeline runner (SLURM DAG) | `applications/airtable/src/airtable_utils/pipeline.py` | Orchestration extends the airtable CLI |
| Per-FOV state tracking | `applications/airtable/src/airtable_utils/state.py` | State tracks registration + QC completion |
| Shared Pydantic models | `packages/viscy-data/src/viscy_data/schemas.py` | Consumed by both airtable and qc apps |
| QC parquet writer | `applications/qc/src/qc/` | QC-specific output format |
| Cell index QC columns | `packages/viscy-data/src/viscy_data/cell_index.py` | Part of the shared cell_index schema |

---
*Architecture research: 2026-03-27*
