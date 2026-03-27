# Requirements: Airtable AI QC

**Defined:** 2026-03-27
**Core Value:** Every FOV that reaches model training has passed automated QC and is fully registered in Airtable with accurate metadata.

## v1 Requirements

Requirements for initial milestone. Focus: pipeline structure/workflow design + foundation code fixes.

### Foundation

- [ ] **FOUND-01**: Shared Pydantic models (ChannelAnnotationEntry, WellExperimentMetadata) moved from `airtable_utils.schemas` to `viscy_data.schemas`, fixing cross-application import violation
- [ ] **FOUND-02**: Metadata backfill CLI command that reads source zarr pixel sizes and reconstruction parameters, writing them to output zarr `.zattrs` after biahub assembly
- [ ] **FOUND-03**: Post-assembly registration improved with Airtable rate limiting (10 records/batch, 0.2s sleep, 429 retry) and record count validation

### Pipeline Orchestration

- [ ] **PIPE-01**: YAML DAG config defining pipeline steps and their SLURM dependencies (OPS-style `slurm_task_config.yaml`)
- [ ] **PIPE-02**: Per-dataset `.pipeline_state.yaml` tracking step completion per FOV (completed/failed/pending)
- [ ] **PIPE-03**: Pipeline runner CLI that parses DAG YAML and submits SLURM jobs with `--dependency=afterok` chaining
- [ ] **PIPE-04**: Pipeline completeness check that validates all FOVs passed all steps before training
- [ ] **PIPE-05**: Pipeline checklist document from biologist platemap entry through training-ready data

### Training Integration

- [ ] **TRAIN-01**: `viscy preprocess` runnable as a pipeline step after Airtable registration
- [ ] **TRAIN-02**: QC columns added to cell_index parquet schema (is_blank_timepoint, focus_z_index, qc_passed) as optional columns with sensible defaults

## v2 Requirements

Deferred to future milestone.

### Quality Control Implementation

- **QC-01**: Automated focus QC runs after assembly, writes results to zarr .zattrs and qc_data parquet
- **QC-02**: Cell-level QC flags for single-cell filtering in parquet
- **QC-03**: Configurable QC thresholds per experiment (YAML config or Airtable field)
- **QC-04**: Auto `qc_data_parquet_path` update in Airtable Dataset record

### Advanced Integration

- **ADV-01**: VAST/NFS dual-path tracking in Airtable (path_to_bruno + path_to_bruno_vast)
- **ADV-02**: VAST/NFS sync validation (compare zarr metadata across storage systems)
- **ADV-03**: Airtable-driven cell_index build (Airtable as single source of truth for cell index generation)
- **ADV-04**: Harmonized QC columns compatible with OPS single-cell tracking parquet schema

## Out of Scope

| Feature | Reason |
|---------|--------|
| Biahub core rewrite | Owned by biahub team; only fix metadata propagation |
| Web UI for pipeline monitoring | CLI-first; SLURM sacct is sufficient |
| Real-time streaming pipeline | Batch processing is sufficient |
| Snakemake/Nextflow orchestration | Overkill; SLURM dependencies + YAML config is sufficient |
| Replacing Airtable | Airtable stays as central database |
| QC algorithm implementation | v2; v1 lays out structure only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | — | Pending |
| FOUND-02 | — | Pending |
| FOUND-03 | — | Pending |
| PIPE-01 | — | Pending |
| PIPE-02 | — | Pending |
| PIPE-03 | — | Pending |
| PIPE-04 | — | Pending |
| PIPE-05 | — | Pending |
| TRAIN-01 | — | Pending |
| TRAIN-02 | — | Pending |

**Coverage:**
- v1 requirements: 10 total
- Mapped to phases: 0
- Unmapped: 10 (pending roadmap)

---
*Requirements defined: 2026-03-27*
*Last updated: 2026-03-27 after initial definition*
