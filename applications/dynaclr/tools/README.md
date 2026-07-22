# DynaCLR model matrix — run many models through train → predict → eval

Launchers for running one or many DynaCLR models through the full pipeline. Models
run **in parallel**; within a model the three stages chain via SLURM
`--dependency=afterok`. Embeddings are written once into the dataset-centric tree
(`<dataset>/2-phenotyping/predictions/{model_family}/{run}/{ckpt_name}/{marker}.zarr`)
and evaluated from there — no re-prediction to iterate on evals.

## The matrix YAML

One file lists the sweep. A `defaults:` block holds shared fields; each model entry
is essentially **one line** — its training `.sh`. `family` / `run` / `train_configs`
are parsed from that script's `export PROJECT= / RUN_NAME= / CONFIGS=` lines (so they
can't drift from what training actually runs). Example:
[`../configs/matrix/example.yml`](../configs/matrix/example.yml).

```yaml
defaults:
  ckpt_name: last                # → checkpoints/last.ckpt (or pin epochN-stepM)
  collection: applications/dynaclr/configs/collections/<...>.yml
  eval_config: applications/dynaclr/configs/evaluation/<...>.yaml
  datasets_root: /hpc/projects/intracellular_dashboard/organelle_dynamics
  predict_flags: {z_range: [15, 45], z_reduction: mip, reference_pixel_size: 0.1494}
models:
  - train_sbatch: applications/dynaclr/configs/training/DynaCLR-2D/<...>.sh
  - train_sbatch: applications/dynaclr/configs/training/DynaCLR-3D/<...>.sh
```

Add a model = add one line. Change something sweep-wide = edit `defaults:`.

## Running

```bash
# see the chained sbatch commands without submitting
uv run dynaclr run-matrix -c <matrix.yml> --dry-run

# submit the full chain (train → predict → eval) for every model, in parallel
uv run dynaclr run-matrix -c <matrix.yml>

# run a subset of stages (e.g. skip training, evaluate existing checkpoints)
uv run dynaclr run-matrix -c <matrix.yml> --stages predict,eval
```

Per model, `dynaclr run-matrix` submits:
`train` → `predict` (`afterok:train`) → `eval` (`afterok:predict`). `train` is the
model's own `.sh`; `predict` and `eval` are the `predict.sbatch` / `eval.sbatch`
wrappers (which call `dynaclr predict-batch` / `dynaclr eval`).

## Preprocessing precondition (AI-ready datasets)

Prediction needs each dataset's FOV zattrs to already carry `normalization` +
`focus_slice` (written by the upstream `prepare` pipeline). `dynaclr run-matrix` runs
an **upfront preflight** across every dataset before submitting anything:

- **normalization** missing → safe to auto-run (`viscy preprocess`, no manual params).
  `dynaclr predict-batch --auto-normalize` will do it; the matrix flags it.
- **focus_slice** missing → **only flagged, never auto-run** — z-focus finding needs
  per-dataset physics params (NA / wavelength / pixel size in `qc_config.yml`).
  Run `qc run -c <dataset>/qc_config.yml` yourself, then retry.

`--skip-preflight` bypasses the check.

## Checkpoint selection

Manual. By default the checkpoint is derived as
`{models_root}/{family}/{run}/checkpoints/{last.ckpt | epoch=N-step=M.ckpt}` from `ckpt_name`
(`last` or `epochN-stepM`). No automatic best-by-metric selection.

**Existing models usually need an explicit `checkpoint:`.** Lightning writes checkpoints under a
`{run}/{PROJECT}/{wandb_run_id}/checkpoints/` subdir (the wandb id isn't derivable from the
identity, and a run may have several). Give the full path in the matrix entry to override the
derived default:

```yaml
models:
  - train_sbatch: .../DynaCLR-2D-MIP-BagOfChannels-single-marker.sh
    ckpt_name: epoch105-step84800
    checkpoint: /hpc/.../{run}/DynaCLR-2D-MIP-BagOfChannels/jbrwhzr3/checkpoints/epoch=105-step=84800.ckpt
```

## The individual launchers

| Tool | Does |
|---|---|
| `dynaclr run-matrix` | the whole matrix — parse `.sh`, preflight, chain train→predict→eval |
| `dynaclr predict-batch` | predict one model over a collection (wraps `predict-triplet`) + AI-ready preflight |
| `dynaclr eval` | evaluate one model's embeddings (launches Nextflow `eval_from_embeddings`) |
| `predict.sbatch` / `eval.sbatch` | the wrapper jobs the matrix chains |

See [`../docs/DAGs/end_to_end.md`](../docs/DAGs/end_to_end.md) for the pipeline overview
and [`../nextflow/README.md`](../nextflow/README.md) for the eval Nextflow entries.
