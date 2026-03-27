# Stack Research

**Project:** Airtable AI QC — Microscopy Data Pipeline
**Confidence:** MEDIUM-HIGH

## Current Stack (Keep)

| Component | Library | Role |
|-----------|---------|------|
| Package manager | uv | Workspace monorepo management |
| ML training | PyTorch Lightning | Distributed training, callbacks |
| Data I/O | iohub, tensorstore | OME-Zarr read/write, batched reads |
| Tabular data | pandas, pyarrow | Cell index parquet, metadata |
| Schemas | Pydantic 2 | Data validation for domain objects |
| CLI | Click | Command-line tools |
| Config | jsonargparse, PyYAML | Structured config parsing |

## New Stack Recommendations

### DAG Orchestration: Lightweight YAML + SLURM Dependencies

**Recommendation:** OPS-style YAML DAG config with SLURM `--dependency=afterok` chaining. Not Snakemake.

**Rationale:**
- The team already uses this pattern in `qc_and_preprocess_slurm.sh` (parallel `&` + `wait`)
- OPS `slurm_task_config.yaml` is the reference architecture the user wants to emulate
- SLURM `--dependency=afterok:$JOB_ID` is the correct primitive for HPC job chaining
- No server process required (unlike Airflow/Prefect)
- Per-FOV rerunability via sentinel files (parquet outputs, not `.done` touch files)

**Confidence:** HIGH — matches existing team patterns and explicit user reference to OPS

**Why NOT Snakemake:**
- Adds a large dependency with its own DSL
- File-timestamp-based staleness detection unreliable on VAST/NFS (modification times can drift)
- Overkill for a pipeline with ~5 linear steps
- Team unfamiliar with Snakemake — learning curve overhead

**Why NOT Prefect/Airflow/Dagster:**
- Require persistent server process — not viable on HPC login nodes
- Heavyweight for a pipeline this size

### QC Sentinel Pattern: Parquet Output Files

**Recommendation:** QC results written as parquet files, not `.done` touch files.

**Rationale:** QC results feed into `qc_data_parquet_path` in Airtable and downstream cell_index filtering. The output file must carry content, not just signal completion.

**Confidence:** HIGH

### Airtable Batch Writes: Chunked with Rate Limiting

**Recommendation:** Python-side batching with chunking + 0.2s sleep + 429 retry.

**Rationale:** Airtable limits: 5 req/sec, 10 records per batch create/update. The existing `register_fovs()` accumulates create/update lists. MCP tools are single-record; bulk logic must live in Python.

**Confidence:** HIGH — official Airtable documentation

### Per-FOV Status Tracking: YAML State File

**Recommendation:** Per-dataset YAML state file tracking step completion per FOV.

```yaml
# .pipeline_state.yaml (next to the zarr store)
dataset: experiment_name.zarr
steps:
  register:
    completed: [A/1/0, A/2/0, A/3/0]
    failed: [A/4/0]
    pending: [A/5/0]
  qc:
    completed: [A/1/0, A/2/0]
    failed: []
    pending: [A/3/0, A/4/0, A/5/0]
```

**Rationale:** Enables per-FOV rerun without re-running the entire dataset. Lightweight, human-readable, no database dependency.

**Confidence:** MEDIUM — pattern derived from OPS; needs validation

## What NOT to Use

| Tool | Reason |
|------|--------|
| Snakemake | Overkill; unreliable timestamps on VAST/NFS; unfamiliar DSL |
| Prefect/Airflow/Dagster | Require persistent server; not HPC-compatible |
| Nextflow | Groovy DSL; poor fit for Python-first team |
| pyairtable SDK | Team convention: use Airtable MCP tools |
| Luigi | Abandoned; no active maintenance |

## Roadmap Implications

1. DAG config YAML + SLURM runner should be introduced early — it is the skeleton everything else hangs on
2. QC rules and registration rules are separate steps with explicit dependencies
3. Airtable write batching needs to be handled in the CLI layer
4. Per-FOV state tracking enables incremental reruns without full-dataset re-processing

---
*Stack research: 2026-03-27*
