# Roadmap: Airtable AI QC

## Overview

This project extends the VisCy monorepo with a reliable, end-to-end microscopy data pipeline: from biahub assembly through Airtable registration, QC, and training-ready cell index generation, culminating in a SLURM DAG orchestrator that composes all steps with per-FOV state tracking. The build order is driven by a hard constraint — zarr metadata is lost during biahub assembly, making every downstream step untrustworthy until that is fixed. Foundation correctness comes first, training integration second, and DAG orchestration last (because the orchestrator can only compose steps that already work correctly).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation** - Fix import violation, metadata backfill CLI, and reliable Airtable registration
- [ ] **Phase 2: Training Integration** - QC columns in cell_index parquet and viscy preprocess as a pipeline step
- [ ] **Phase 3: Pipeline Orchestration** - YAML DAG, SLURM runner, per-FOV state tracking, and completeness gate

## Phase Details

### Phase 1: Foundation
**Goal**: A single dataset can be processed correctly end-to-end — pixel sizes survive biahub assembly, Airtable registration is reliable and validates record counts, and shared schema models live in the correct package
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03
**Success Criteria** (what must be TRUE):
  1. After running the metadata backfill CLI on an output zarr, pixel sizes in `.zattrs` match the source zarr — no `(1,1,1)` defaults remain
  2. Running post-assembly registration with 50+ FOVs completes without silent data loss — final Airtable record count matches expected FOV count, even when rate limit 429s occur
  3. `applications/qc` no longer imports from `applications/airtable` — shared Pydantic models (`ChannelAnnotationEntry`, `WellExperimentMetadata`) are importable from `viscy_data.schemas`
  4. A FOV that failed biahub assembly is detected and skipped gracefully during registration without crashing the batch
**Plans**: TBD

### Phase 2: Training Integration
**Goal**: Training-ready cell_index parquets include QC columns, and `viscy preprocess` is runnable as a named pipeline step after Airtable registration
**Depends on**: Phase 1
**Requirements**: TRAIN-01, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. A cell_index parquet built after Phase 1 includes `focus_z_index`, `is_blank_timepoint`, and `qc_passed` columns with sensible defaults when QC data is absent
  2. Running `viscy preprocess` after registration completes without error and produces a parquet compatible with the existing DynaCLR dataloader
  3. Existing cell_index parquets without QC columns are handled without crashing downstream consumers (dataloader, sampler, dataset)
**Plans**: TBD

### Phase 3: Pipeline Orchestration
**Goal**: End-to-end pipeline runs from a single YAML config — SLURM jobs chain with `--dependency=afterok`, per-FOV state is tracked across reruns, and a completeness gate blocks training unless all FOVs passed all steps
**Depends on**: Phase 2
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05
**Success Criteria** (what must be TRUE):
  1. A `pipeline.yaml` DAG config specifies all pipeline steps and their SLURM dependencies; the pipeline runner submits them as a chain without manual job ID tracking
  2. After a partial run, re-invoking the pipeline runner skips FOVs already marked complete in `.pipeline_state.yaml` and only resubmits pending or failed FOVs
  3. Running the completeness check on a dataset with any failed FOVs exits non-zero and reports which FOVs and steps are incomplete — a fully-passed dataset exits zero
  4. A written pipeline checklist document exists that a biologist can follow from platemap entry in Airtable through to a training-ready dataset
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/TBD | Not started | - |
| 2. Training Integration | 0/TBD | Not started | - |
| 3. Pipeline Orchestration | 0/TBD | Not started | - |
