# Airtable AI QC

## What This Is

An end-to-end data pipeline for fluorescence microscopy that connects raw image acquisition through processing, quality control, Airtable registration, and contrastive model training. Built on top of the VisCy monorepo, it provides CLI tooling, a processing DAG, and per-FOV rerunability so that biologists can go from platemap entry to trained DynaCLR models with confidence in data quality.

## Core Value

Every FOV that reaches model training has passed automated QC and is fully registered in Airtable with accurate metadata.

## Requirements

### Validated

<!-- Existing capabilities inferred from the codebase -->

- ✓ Airtable registration of zarr positions via `airtable-utils register` CLI — existing
- ✓ Cell index parquet schema (one row per cell/timepoint/channel) — existing
- ✓ MultiExperimentDataModule for multi-experiment contrastive training — existing
- ✓ ExperimentRegistry reads collection YAML or cell_index parquet — existing
- ✓ Focus QC via `applications/qc` with YAML configs — existing
- ✓ FlexibleBatchSampler with stratify_by and batch_group_by — existing
- ✓ GPU-first augmentation pipeline via on_after_batch_transfer — existing
- ✓ Airtable MCP integration for reading/writing records — existing
- ✓ viscy-data schemas (CellIndex, ChannelsMetadata, ExperimentMetadata) — existing
- ✓ Pydantic-based data validation for domain objects — existing

### Active

<!-- Current scope. Building toward these. -->

- [ ] Post-assembly CLI that registers all positions and metadata to Airtable automatically
- [ ] Automated QC pipeline that runs in parallel after biahub processing and writes results to `qc_data_parquet_path`
- [ ] Per-FOV rerunability for any pipeline step (biahub reconstruction, QC, registration)
- [ ] Processing DAG (like OPS slurm_task_config.yaml) that defines step dependencies and tracks completion status
- [ ] Pipeline checklist starting from biologist platemap entry through to training-ready data
- [ ] Metadata propagation through biahub (reconstruction parameters, position metadata preserved in output zarr)
- [ ] QC data integration into AnnData/parquet at single-cell level (e.g., blank-timepoint flags)
- [ ] `viscy preprocess` CLI command runnable after registration
- [ ] VAST/NFS dataset sync validation (detect out-of-sync partitions)
- [ ] Harmonized QC columns in cell_index parquet (compatible with OPS single-cell tracking)

### Out of Scope

- Rewriting biahub core processing (reconstruction algorithms) — owned by biahub team
- Building a web UI for pipeline monitoring — CLI-first
- Real-time streaming pipeline — batch processing is sufficient
- Replacing Airtable as the central database — Airtable stays

## Context

- **Biahub** handles raw-to-reconstructed processing but does not copy `ngff.Position()` metadata via `iohub.ngff.utils.create_empty_plate()`, and cannot skip/retry single failed positions.
- **OPS** (`ops_process`) has a configurable SLURM DAG with step dependencies that this project should emulate for the VisCy pipeline.
- **Imaging QC Pipeline** (`imaging-qc-pipeline`) generates tabular QC data that maps to Airtable Dataset → `qc_data_path`.
- The Confluence page "Data Standardization & Flexible Sample/Dataloaders for Contrastive Models" provides the high-level vision; specs live in `viscy-data` schemas and `applications/airtable`.
- Datasets live on both VAST and NFS partitions and frequently fall out of sync.
- The `Add Platemap` Airtable view is the biologist's entry point; everything downstream should be automated from there.

## Constraints

- **Tech stack**: Must build within the VisCy monorepo (packages/ and applications/) using existing patterns (uv workspace, Lightning, iohub, Pydantic)
- **HPC**: Pipeline steps run via SLURM; scripts need `export PYTHONNOUSERSITE=1` and `srun` for DDP
- **Data format**: OME-Zarr (iohub) for images, parquet for cell indices, AnnData for single-cell tabular data
- **Airtable**: Use MCP tools for Airtable operations (not pyairtable SDK) per team convention
- **Dependencies**: `applications/airtable`, `applications/qc`, `packages/viscy-data` are the primary packages to extend; do not create new applications if existing ones can be extended

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Airtable as central database | Already adopted by team; biologists use it daily | — Pending |
| DAG-based pipeline (OPS-style) | Enables per-FOV rerun and dependency tracking | — Pending |
| QC runs in parallel with processing | Reduces wall-clock time for pipeline completion | — Pending |
| CLI-first interface | Composable with SLURM; scriptable by biologists | — Pending |

---
*Last updated: 2026-03-27 after initialization*
