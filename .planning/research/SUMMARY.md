# Project Research Summary

**Project:** Airtable AI QC — Microscopy Data Pipeline
**Domain:** HPC batch data pipeline with Airtable as source of truth
**Researched:** 2026-03-27
**Confidence:** HIGH

## Executive Summary

This project automates the microscopy data pipeline from biahub assembly through training-ready cell index generation, with Airtable as the authoritative record of dataset status and metadata. The core pattern is well-understood: a YAML-driven DAG with SLURM `--dependency=afterok` chaining, modeled directly on the existing OPS pipeline. No new orchestration framework is needed — the team's existing primitives (SLURM dependencies, parquet sentinel files, Airtable records) are sufficient and already in use.

The recommended build order is driven by a hard constraint: zarr metadata is lost during biahub assembly, and every downstream step (registration, QC, cell index) depends on pixel sizes being present in `.zattrs`. Metadata backfill must be the first deliverable. Once that foundation is in place, registration and QC can be extended and automated, followed by cell index QC integration, and finally DAG orchestration composing all steps. The architecture extends two existing applications (`applications/airtable` and `applications/qc`) rather than creating a new one.

The primary risks are (1) silent data corruption from Airtable 429 rate limit drops during bulk registration, (2) QC producing wrong results from default `(1,1,1)` pixel sizes when metadata backfill is skipped, and (3) a pre-existing cross-application import violation (`applications/qc` importing from `applications/airtable`) that must be fixed before adding more shared code. All three are preventable with known mitigations and are addressed in Phase 1.

## Key Findings

### Recommended Stack

The project builds on the existing monorepo stack without new framework dependencies. DAG orchestration uses SLURM `--dependency=afterok` chaining with a YAML config file — the same pattern used in the OPS pipeline — rather than Snakemake or Prefect. QC results are written as parquet files (not touch files) because they feed directly into `qc_data_parquet_path` in Airtable and downstream cell_index filtering. Airtable bulk writes require Python-side batching with 10 records/batch, 0.2s sleep between requests, and 429 retry.

**Core technologies:**
- SLURM `--dependency=afterok`: DAG step chaining — matches existing team patterns, no new dependencies
- Parquet (pyarrow): QC sentinel output — carries content, feeds Airtable and cell_index directly
- YAML state file (`.pipeline_state.yaml`): per-FOV step completion tracking — human-readable, lightweight, no database
- Pydantic 2 (existing): shared schema models — moved to `packages/viscy-data` to fix cross-app import violation
- Airtable direct API (Python batching): bulk registration — MCP tools are single-record; bulk logic must be in Python

### Expected Features

**Must have (table stakes):**
- Metadata backfill — fix pixel sizes in zarr `.zattrs` after biahub assembly, before any downstream step
- Post-assembly Airtable registration — auto-register FOVs with pixel sizes, channels, VAST/NFS paths
- Automated QC — focus slice QC runs after assembly, writes results to `.zattrs` and parquet
- QC columns in cell_index parquet — `focus_z_index`, `is_blank_timepoint`, `qc_passed` materialized for dataloader filtering
- Per-FOV rerunability — any step can be re-run for a single position without reprocessing the entire dataset
- biahub single-position failure handling — detect failed positions, skip gracefully in registration and QC

**Should have (differentiators):**
- OPS-style YAML DAG config with SLURM runner composing all steps
- VAST/NFS dual-path tracking with sync validation
- Airtable-driven cell_index (Airtable as source of truth for dataset membership)
- Harmonized QC columns compatible with OPS single-cell tracking parquet schema
- Pipeline completeness gate before training (all FOVs registered + QC passed)

**Defer (v2+):**
- Full parallel execution (QC and preprocessing simultaneously via SLURM)
- Marker registry integration during registration (channel alias resolution)
- Auto `qc_data_parquet_path` update in Airtable after QC completes

### Architecture Approach

The pipeline extends two existing applications (`applications/airtable` and `applications/qc`) and adds minimal new files: a YAML DAG config, a SLURM pipeline runner (`pipeline.py`), and a per-FOV state tracker (`state.py`), all in `applications/airtable`. A pre-existing cross-application import violation — `applications/qc` importing from `applications/airtable` — must be fixed first by moving shared Pydantic models (`ChannelAnnotationEntry`, `WellExperimentMetadata`) to `packages/viscy-data/src/viscy_data/schemas.py`.

**Major components:**
1. `packages/viscy-data` (schemas.py, cell_index.py) — shared Pydantic models; QC columns added to cell_index schema
2. `applications/airtable` (registration, pipeline.py, state.py) — registration with batching/retry; SLURM DAG runner; per-FOV state tracking
3. `applications/qc` (QC automation, parquet writer) — automated focus QC trigger; writes qc_data.parquet and updates `.zattrs`
4. Pipeline YAML configs (`applications/airtable/configs/pipeline/`) — step definitions, SLURM dependencies, per-experiment QC thresholds

**Data flow:** Biologist platemap (Airtable) → biahub assembly → metadata backfill → [registration || QC, parallel] → viscy preprocess → dynaclr build-cell-index (flat parquet with QC columns) → training.

### Critical Pitfalls

1. **Zarr metadata lost through biahub assembly** — `create_empty_plate()` drops pixel sizes; implement metadata backfill as the first step; downstream steps must not run without it.
2. **Airtable 429 rate limit silently drops batch writes** — chunk to 10 records/batch, sleep 0.2s between requests, retry on 429, validate record count after write.
3. **QC passes with wrong pixel sizes** — if metadata backfill is skipped, QC runs on `(1,1,1)` scale; use a pre-flight check that asserts pixel sizes are non-unit before QC runs.
4. **No per-FOV state — pipeline reruns process everything** — write `.pipeline_state.yaml` per dataset tracking step completion per FOV; check state before submitting SLURM jobs.
5. **Cross-application import violation blocks growth** — fix `applications/qc` → `applications/airtable` import by moving shared models to `packages/viscy-data` before adding any new shared code.

## Implications for Roadmap

Based on research, the dependency chain is strict: metadata backfill must precede registration and QC; the import violation fix must precede adding new shared models; cell index QC columns require QC output to exist; the DAG orchestrator composes everything. This suggests 3 phases with clear deliverables.

### Phase 1: Foundation — Reliable Single-Dataset Pipeline

**Rationale:** The zarr metadata loss and Airtable rate limit pitfalls are blocking issues. Nothing downstream is trustworthy without pixel sizes in `.zattrs` and reliable Airtable record counts. Fix the import violation and these two pitfalls before building anything new.
**Delivers:** A single dataset can be processed correctly end-to-end: metadata backfill → registration (reliable, batched) → automated QC → QC in parquet.
**Addresses:** Table stakes features 1-6 (metadata backfill, registration, automated QC, QC in parquet, per-FOV error handling).
**Avoids:** Pitfalls 1 (zarr metadata loss), 2 (Airtable rate limits), 3 (wrong pixel sizes in QC), 5 (cross-app import violation).
**Research flag:** Standard patterns — metadata backfill and Airtable batching are well-understood.

### Phase 2: Cell Index Integration and QC Columns

**Rationale:** Once QC results are reliably written as parquet, they can be integrated into the flat cell_index schema. This is the prerequisite for training-ready data. QC threshold configurability also belongs here (before the DAG hardcodes them).
**Delivers:** Flat parquet cell_index includes `focus_z_index`, `is_blank_timepoint`, `qc_passed`; dataloader can filter bad samples without manual intervention. QC thresholds are configurable per experiment.
**Addresses:** Table stakes features 5 and 9 (QC columns in parquet, cell-level QC flags); differentiator feature 6 (harmonized OPS columns).
**Avoids:** Pitfall 4 (hardcoded QC thresholds that drift).
**Research flag:** Cell index schema evolution needs care — downstream consumers (sampler, dataset, datamodule) must handle new columns; plan for optional columns with sensible defaults.

### Phase 3: DAG Orchestration and Pipeline Checklist

**Rationale:** With all individual steps working reliably, compose them into a SLURM DAG runner. Per-FOV state tracking enables incremental reruns. VAST/NFS dual-path tracking and completeness gates complete the pipeline.
**Delivers:** End-to-end `viscy pipeline run --config pipeline.yaml` that chains all steps with SLURM dependencies, tracks per-FOV state, and validates completeness before training.
**Addresses:** Differentiator features 1-5 (OPS-style DAG, parallel QC, VAST/NFS tracking, completeness gate); table stakes feature 10 (pipeline checklist).
**Uses:** SLURM `--dependency=afterok`, `.pipeline_state.yaml` per-dataset state file.
**Avoids:** Pitfall 3 (no per-FOV state — reruns re-run everything); pitfall 5 (VAST/NFS out-of-sync paths).
**Research flag:** SLURM DAG pattern is standard but `.pipeline_state.yaml` design needs validation — confirm it handles concurrent SLURM jobs writing to the same state file without race conditions.

### Phase Ordering Rationale

- Phase 1 first because the zarr metadata bug makes all output untrustworthy — no point building on a broken foundation.
- Phase 2 before Phase 3 because the DAG orchestrator must compose steps that already produce correct output.
- The import violation fix is a hard gate at the start of Phase 1 — it blocks adding any new shared Pydantic models.
- VAST/NFS sync validation deferred to Phase 3 because it is a "nice to have" safety check, not a correctness prerequisite.

### Research Flags

Phases needing deeper research during planning:
- **Phase 3:** `.pipeline_state.yaml` concurrent write safety — SLURM array jobs and parallel steps may race on the state file; needs a locking strategy or atomic write pattern.
- **Phase 2:** Cell index schema migration — existing parquet files will not have QC columns; decide whether to regenerate all parquets or handle missing columns in downstream readers.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Airtable batching, zarr metadata I/O, and QC automation are all well-documented in the codebase and existing OPS scripts.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Explicit team conventions and OPS reference pattern; no new dependencies introduced |
| Features | HIGH | Derived from actual pipeline gaps and existing code — not speculative |
| Architecture | HIGH | Based on monorepo structure and existing cross-app import violation discovered in code |
| Pitfalls | HIGH | Zarr metadata loss and Airtable rate limits are confirmed bugs from existing code inspection |

**Overall confidence:** HIGH

### Gaps to Address

- **`.pipeline_state.yaml` concurrent write safety:** Pattern is proposed but not validated for concurrent SLURM jobs. Address during Phase 3 planning — may need file locking or a different sentinel strategy.
- **Airtable MCP vs Python batching tension:** MCP tools are single-record; bulk registration needs Python-side batching that dispatches via direct API calls. Confirm this approach is consistent with team conventions before Phase 1 implementation.
- **QC threshold configurability scope:** Research flags this for Phase 2, but the exact interface (per-experiment YAML vs Airtable field vs CLI flag) was not resolved. Decide during Phase 1 planning to avoid retrofitting.

## Sources

### Primary (HIGH confidence)
- Codebase inspection (`applications/airtable/`, `applications/qc/`, `packages/viscy-data/`) — cross-app import violation, existing registration patterns, Airtable batching logic
- SLURM documentation — `--dependency=afterok` chaining pattern
- Airtable official API docs — 5 req/sec rate limit, 10 records/batch create/update limit
- `qc_and_preprocess_slurm.sh` (existing OPS script) — parallel `&` + `wait` pattern reference

### Secondary (MEDIUM confidence)
- OPS `slurm_task_config.yaml` — YAML DAG config reference architecture (user-specified reference)
- `.pipeline_state.yaml` pattern — derived from OPS conventions; needs validation for concurrent safety

### Tertiary (LOW confidence)
- QC threshold drift behavior — inferred from hardcoded `NA_DET=1.35` in SLURM scripts; actual drift rate not measured

---
*Research completed: 2026-03-27*
*Ready for roadmap: yes*
