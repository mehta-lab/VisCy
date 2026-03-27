# Features Research

**Project:** Airtable AI QC — Microscopy Data Pipeline
**Confidence:** HIGH

## Table Stakes

Features the pipeline must have to be reliable.

| # | Feature | Complexity | Dependencies |
|---|---------|-----------|-------------|
| 1 | **Post-assembly Airtable registration** — `airtable-utils register` runs automatically after `2-assembled/{dataset}.zarr` completes | Low | Existing `register_fovs()` |
| 2 | **Metadata backfill** — Reconstruct pixel sizes, reconstruction parameters in output zarr `.zattrs` when biahub drops them via `create_empty_plate()` | Medium | iohub, source zarr metadata |
| 3 | **Per-FOV rerunability** — Any step can be re-run for a single position without re-processing the entire dataset | Medium | State tracking per FOV |
| 4 | **Automated QC** — Focus slice QC runs after assembly and writes results to zarr `.zattrs` and `qc_data_parquet_path` | Medium | `applications/qc`, pixel sizes in zattrs |
| 5 | **QC columns in cell_index parquet** — `focus_z_index`, `is_blank_timepoint` materialized during parquet generation so dataloader can filter bad samples | Low | QC results in zattrs |
| 6 | **Pipeline completeness check** — Validate all FOVs passed registration + QC before training | Low | State tracking |
| 7 | **VAST/NFS path tracking** — Both `path_to_bruno` (NFS) and `path_to_bruno_vast` (VAST) tracked in Airtable | Low | Schema update |
| 8 | **`viscy preprocess` after registration** — Preprocessing step runnable as CLI after registration completes | Low | Existing `viscy preprocess` |
| 9 | **Cell-level QC flags** — Per-timepoint blank detection, out-of-focus flagging at single-cell level in parquet | Medium | QC results + cell_index schema |
| 10 | **Pipeline checklist** — From biologist platemap entry through to training-ready data, all steps documented and trackable | Low | DAG definition |

## Differentiators

Features that distinguish this from manual workflows.

| # | Feature | Complexity | Dependencies |
|---|---------|-----------|-------------|
| 1 | **OPS-style DAG config** — YAML-driven step dependency graph with SLURM `--dependency=afterok` chaining | Medium | SLURM, YAML config |
| 2 | **Parallel QC execution** — QC and preprocessing run simultaneously, not sequentially | Low | SLURM job submission |
| 3 | **Airtable-driven cell_index** — Cell index parquet built from Airtable records + zarr, ensuring Airtable is source of truth | Medium | Airtable MCP, registration |
| 4 | **Marker registry integration** — Channel aliases resolved via Airtable Marker Registry LUT during registration | Low | Existing marker registry |
| 5 | **Multi-storage tracking** — Airtable tracks both VAST and NFS paths; sync validation detects drift | Medium | Airtable schema, filesystem checks |
| 6 | **Harmonized QC columns** — Compatible with OPS single-cell tracking parquet schema | Medium | OPS schema alignment |
| 7 | **Auto `qc_data_parquet_path` update** — Airtable Dataset record updated with QC parquet path after QC completes | Low | Airtable MCP |

## Anti-Features

Things to deliberately NOT build.

| Feature | Reason |
|---------|--------|
| Web UI for pipeline monitoring | CLI-first; SLURM `sacct` is sufficient |
| Real-time streaming pipeline | Batch processing is sufficient for microscopy workflows |
| Biahub core rewrite | Owned by biahub team; fix metadata propagation only |
| Backward-compatible schema shims | Research codebase; changes are expected |
| Content-hash caching | VAST/NFS filesystem timestamps are unreliable; sentinel files are simpler |
| Snakemake/Nextflow orchestration | Overkill; SLURM dependencies + YAML config is sufficient |

## Feature Dependencies

```
Biologist platemap entry (Airtable)
  └─ biahub reconstruction
       └─ Metadata backfill (#2)
            ├─ Post-assembly registration (#1)
            │    ├─ VAST/NFS path tracking (#7)
            │    ├─ viscy preprocess (#8)
            │    └─ Auto qc_data_parquet_path (D7)
            ├─ Automated QC (#4) [parallel with registration]
            │    ├─ QC columns in parquet (#5)
            │    └─ Cell-level QC flags (#9)
            └─ Per-FOV rerunability (#3)
                 └─ Pipeline completeness check (#6)
                      └─ Pipeline checklist (#10)
                           └─ MultiExperimentDataModule training
```

## MVP Phasing

**v1 (Core Pipeline):**
- Metadata backfill, post-assembly registration, automated QC, QC in parquet, per-FOV rerun

**v1.x (Integration):**
- VAST/NFS tracking, cell-level QC flags, harmonized OPS columns, pipeline checklist

**v2 (Orchestration):**
- Full DAG config, parallel execution, Airtable-driven cell_index, completeness gates

---
*Features research: 2026-03-27*
