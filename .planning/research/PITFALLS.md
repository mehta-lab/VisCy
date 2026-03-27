# Pitfalls Research

**Project:** Airtable AI QC — Microscopy Data Pipeline
**Confidence:** HIGH

## Critical Pitfalls

### 1. Zarr Metadata Lost Through biahub Assembly

**Problem:** `iohub.ngff.utils.create_empty_plate()` does not copy `ngff.Position()` metadata (pixel sizes, reconstruction parameters) to the output zarr. Downstream steps (`register_fovs()`, `FocusSliceMetric`) depend on pixel sizes in `.zattrs`.

**Warning signs:** `(1, 1, 1)` scale values in registered Airtable records; QC focus metrics are wildly wrong.

**Prevention:** Add a metadata backfill step immediately after assembly that reads source zarr and writes missing fields to output zarr `.zattrs`. This must run BEFORE registration or QC.

**Phase:** Phase 1 (prerequisite for everything)

### 2. Airtable 429 Rate Limit Silently Corrupts Batch Writes

**Problem:** Airtable limits to 5 requests/sec and 10 records per batch create/update. Without rate limiting, bulk registration silently drops records when hitting 429 responses.

**Warning signs:** FOV count in Airtable doesn't match zarr position count; intermittent "success" followed by missing records.

**Prevention:** Python-side batching with chunking (10 records/batch), 0.2s sleep between requests, and 429 retry with exponential backoff. Validate record count after batch write.

**Phase:** Phase 1 (registration reliability)

### 3. DAG Step Completion Not Persisted — Reruns Re-run Everything

**Problem:** Without per-FOV status tracking, there's no way to know which FOVs completed which steps. A pipeline retry re-processes all FOVs including those that already succeeded.

**Warning signs:** Pipeline takes the same time on "retry" as on first run; good data gets overwritten.

**Prevention:** Per-dataset `.pipeline_state.yaml` file tracking step completion per FOV. The pipeline runner checks state before submitting jobs.

**Phase:** Phase 2 (DAG orchestration)

### 4. QC Thresholds Hardcoded — Drift Undetected

**Problem:** QC thresholds (e.g., `NA_DET=1.35` in SLURM shell scripts) are hardcoded. As imaging conditions change across experiments, these thresholds become stale and pass bad FOVs or reject good ones.

**Warning signs:** QC pass rate suddenly changes without apparent data quality change; biologists override QC results manually.

**Prevention:** QC thresholds stored in Airtable per-experiment or in experiment-level YAML config. Expose as CLI parameters, not hardcoded values.

**Phase:** Phase 2 (QC automation)

### 5. VAST/NFS Out-of-Sync Paths Break Training

**Problem:** Datasets exist on both VAST and NFS. If `data_path` in Airtable points to one but training reads from the other, the model trains on stale data. The paths can diverge silently.

**Warning signs:** Training loss curves look different for "same" experiment re-runs; file modification times differ across storage systems.

**Prevention:** Track both paths in Airtable (`path_to_bruno` for NFS, `path_to_bruno_vast` for VAST). Add a sync validation step that compares zarr metadata checksums across storage.

**Phase:** Phase 2 or 3

### 6. biahub Cannot Retry Single Failed Positions

**Problem:** biahub processes an entire dataset. If 1 of 100 positions fails, the user must re-run all 100. Re-running can overwrite already-correct positions.

**Warning signs:** Processing time doesn't decrease on retry; positions that were correct get different values after re-processing.

**Prevention:** This is a biahub limitation (out of scope to fix). Mitigate by: (a) detecting failed positions via missing/corrupt zarr data, (b) reporting them clearly, (c) allowing downstream steps (registration, QC) to skip failed positions gracefully.

**Phase:** Phase 1 (error handling in registration/QC)

## Integration Gotchas

### Cross-Application Import Violation

`applications/qc/src/qc/config.py` imports from `airtable_utils.schemas`. This must be fixed before adding more shared code. Move shared models to `packages/viscy-data/`.

### Airtable MCP vs Python SDK

Team convention: use Airtable MCP tools, not pyairtable SDK. But MCP tools are single-record operations. Bulk writes need Python-side batching logic that then dispatches via MCP or direct API calls. This tension needs resolution early.

### Cell Index Schema Evolution

Adding QC columns (`is_blank_timepoint`, `focus_z_index`, `qc_passed`) to the flat parquet changes the cell_index schema. All downstream consumers (sampler, dataset, datamodule) must handle the new columns gracefully. Add them as optional columns with sensible defaults.

## Performance Traps

### VAST/NFS Timestamp Reliability

Filesystem modification times on network storage (VAST, NFS) are not reliable for detecting staleness. Don't use file timestamps for pipeline step completion detection — use explicit state files instead.

### Tensorstore Context Leaks in Pipeline Scripts

Pipeline scripts that open multiple zarr stores in a loop can accumulate tensorstore contexts. Use context managers (`with` statements) for all zarr I/O.

## "Looks Done But Isn't" Checklist

- [ ] Registration "completed" but missing positions (Airtable record count != zarr position count)
- [ ] QC "passed" but using default pixel sizes `(1,1,1)` instead of actual values
- [ ] Pipeline state says "complete" but output parquet has fewer rows than expected
- [ ] VAST and NFS paths both resolve but contain different zarr versions
- [ ] Cell index parquet exists but QC columns are all NaN (QC step was skipped)

## Pitfall-to-Phase Mapping

| Pitfall | Phase | Prevention |
|---------|-------|-----------|
| Zarr metadata loss | Phase 1 | Metadata backfill step |
| Airtable rate limits | Phase 1 | Batching + retry in registration |
| No per-FOV state | Phase 2 | State tracking YAML |
| Hardcoded QC thresholds | Phase 2 | Configurable thresholds |
| VAST/NFS desync | Phase 2-3 | Dual path tracking + validation |
| biahub single-position retry | Phase 1 | Graceful skip in downstream steps |

---
*Pitfalls research: 2026-03-27*
