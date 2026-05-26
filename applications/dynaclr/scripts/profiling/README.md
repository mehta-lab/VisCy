# DynaCLR I/O profiling scripts

Scripts that validate data-loading performance on VAST/NFS for the DynaCLR
contrastive training pipeline.

## Current scripts

### `benchmark_recheck_cached_data.py`

Measures the effect of `TensorStoreConfig.recheck_cached_data` on NFS read
latency for the DynaCLR contrastive read pattern. Exercises the iohub
tensorstore implementation directly (no training stack involved) so it can
be run **before** the dynaclr datamodule is ported to iohub 0.3.x.

**Prerequisite.** Requires an iohub build with the upstream
`recheck_cached_data` knob on `TensorStoreConfig`. Until that lands, either
install iohub from the feature branch locally, or skip this script.

Run:

```
uv run python applications/dynaclr/scripts/profiling/benchmark_recheck_cached_data.py
```

Output is a markdown table comparing median/p95 batch latency, patches/s,
and MiB/s across three configurations (`none`, `"open"`, `false`). Run
twice back-to-back and compare: if the `none` vs `"open"` gap shrinks on
the second run, the Linux NFS client page cache is masking the
per-chunk revalidation cost on this node.

## Planned follow-ups (after iohub 0.3.x merge into dynadtw)

- **Dataset-level A/B** — same configurations, but driven through
  `MultiExperimentDataModule` + `MultiExperimentTripletDataset` so we
  exercise `_get_position`/`_get_tensorstore`/`_slice_patches` and the
  `ts.stack(...).read().result()` batched read path exactly as training
  does.
- **SLURM DDP A/B** — 200-step fastdev runs with Lightning's
  `SimpleProfiler`, comparing `data_time`/`batch_time` and GETATTR/s
  from `nfsiostat` across ranks.
