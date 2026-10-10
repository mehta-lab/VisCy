# tools

Operational scripts for the DynaCell benchmark: leaf submission, config generation, and one-off data/artifact
builders. Not part of the installable package — run with `uv run python applications/dynacell/tools/<script>`.

## Submission

- `submit_benchmark_job.py` — compose one benchmark leaf → strip `launcher:`/`benchmark:` → render sbatch →
  submit. `--print-script` / `--print-resolved-config` / `--dry-run` for preview. See the top-level
  [README](../README.md#benchmark-submit).
- `submit_benchmark_batch.py` (+ wrapper `predict_batch.sh`) — submit N leaves in one call; modes: serial /
  `--array` / `--parallel P` (see [CLAUDE.md § Predict submission modes](../CLAUDE.md#predict-submission-modes)).
- `predict_local.sh` — foreground predicts on the current host's GPU (`--parallel N` backgrounds concurrent
  processes).
- `run_eval_direct.slurm` — submit one eval leaf (grouped or single) as one sbatch, in the uv project environment.
- `sbatch_template.sbatch`, `sbatch_template_batch.sbatch`, `sbatch_template_array.sbatch` — sbatch templates the
  submit tools render into.

## Config generation

- `generate_grouped_eval_configs.py` — emit the `_internal/leaf/grouped/<bucket>/eval_grouped.yaml` leaves
  (bucketed by organelle × train_set); has a colocated `_test.py`.
- `generate_instance_ap_eval_configs.py` — emit the `_internal/leaf/instance_ap/` 2D instance-AP eval leaves
  (`_test.py` alongside).
- `add_instance_ap_to_ablation_leaves.py` — fold instance-AP config into existing ablation eval leaves.

## Data / artifact builders

- `build_cpdino_seg_cleaned.py` (+ `.slurm`) — build the Cellpose-DINO cleaned cell-segmentation stores used as
  the fixed `io.cell_segmentation_path` for feature crops.
- `fuse_a549_dual_channel_zarr.py` — fuse A549 nucleus+membrane into the dual-channel zarr for two-channel leaves.
- `save_random_init_vscyto3d_ckpts.py` — write untrained (`_randinit`) VSCyto3D checkpoints for ablations.
- `assemble_release_checkpoints.py` — build/update the public S3 checkpoint zoo (`dynacell_v1/models/`) from the
  pinned `predict__*.yml` checkpoints; idempotent (drives the recurring deconv→raw ER/Mito update). See
  [RELEASING_CHECKPOINTS.md](./RELEASING_CHECKPOINTS.md).

## Diagnostics / tests

- `smoke_joint_leaf.py` — smoke a joint (multi-dataset) training leaf.
- `nccl_smoke_test.py` — probe multi-GPU NCCL connectivity before a DDP fit.
- `scan_nan_microssim_caches.py` — read-only: list eval caches stamped with a cubic version (default 0.9.0a1)
  whose MicroMS3IM is entirely NaN, i.e. caches a cubic-equivalence reuse would carry a NaN forward in.
- `*_test.py` — pytest for the generators (`generate_grouped_eval_configs_test.py`,
  `generate_instance_ap_eval_configs_test.py`); further tool tests live in [`../tests/`](../tests/README.md).
- `slurm_logs/` — captured job logs (runtime output, not source).

## Navigation

- Up: [applications/dynacell](../README.md)
