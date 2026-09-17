# DynaCLR Nextflow Pipelines

Multi-workflow Nextflow layout. `main.nf` is a thin router that dispatches to a
named sub-workflow via `-entry`. Each entry workflow owns its own DAG and lives
under `workflows/`; processes live under `modules/<workflow>/`.

## Layout

```
applications/dynaclr/nextflow/
├── main.nf                            # thin router — -entry <name>
├── nextflow.config                    # shared params + SLURM resource labels
├── workflows/
│   ├── evaluation.nf                  # workflow EVALUATION { take: ... }  (predict→split→DOWNSTREAM)
│   ├── eval_from_embeddings.nf        # workflow EVAL_FROM_EMBEDDINGS { take: ... }  (glob→DOWNSTREAM)
│   ├── _downstream.nf                 # shared DOWNSTREAM sub-workflow (reduce/mmd/lc/append/plot)
│   └── training_preprocessing.nf      # workflow TRAINING_PREPROCESSING { take: ... }
└── modules/
    ├── evaluation/                    # processes used only by evaluation
    │   ├── prepare_configs.nf
    │   ├── predict.nf
    │   ├── split.nf
    │   └── ...
    ├── preprocessing/                 # processes used by training_preprocessing
    │   ├── build_cell_index.nf
    │   └── preprocess_cell_index.nf
    └── shared/                        # processes reused across workflows
```

## Running

```bash
module load nextflow/24.10.5

# Evaluation
nextflow run applications/dynaclr/nextflow/main.nf -entry evaluation \
    --eval_config applications/dynaclr/configs/evaluation/<config>.yaml \
    --workspace_dir /hpc/mydata/eduardo.hirata/repos/viscy \
    -resume

# Eval from pre-computed embeddings (decoupled: no predict/split, reads the frozen tree)
nextflow run applications/dynaclr/nextflow/main.nf -entry eval_from_embeddings \
    --eval_config applications/dynaclr/configs/evaluation/<config>.yaml \
    --embeddings_glob '/hpc/projects/intracellular_dashboard/organelle_dynamics/*/2-phenotyping/predictions/<MODEL>/<RUN>/<CKPT>/*.zarr' \
    --workspace_dir /hpc/mydata/eduardo.hirata/repos/viscy \
    -resume

# Training preprocessing (collection YAML → training-ready parquet)
nextflow run applications/dynaclr/nextflow/main.nf -entry training_preprocessing \
    --collection_yaml applications/dynaclr/configs/collections/<name>.yml \
    --parquet_out /hpc/projects/organelle_phenotyping/models/collections/<name>.parquet \
    --focus_channel Phase3D \
    --workspace_dir /hpc/mydata/eduardo.hirata/repos/viscy \
    -resume

# Local test (no SLURM) — append `-profile local`
```

Running `main.nf` without `-entry` fails loudly with the list of valid entries.

## Inference and evaluation are decoupled

Embeddings are written **once** into the dataset-centric tree
`<dataset>/2-phenotyping/predictions/{model_family}/{run}/{ckpt_name}/{marker}.zarr`
(by `dynaclr predict-triplet`, or by the parquet spine via
`dynaclr split-embeddings --route-by-dataset`). Evaluation reads those **frozen**
embeddings — no GPU, no re-prediction — so you can iterate classifiers and
downstream tasks freely against a fixed set of embeddings.

- **`-entry evaluation`** — the full spine: predict → split → downstream
  (reduce/smoothness/mmd/classifiers/plots). Use for the parquet spine or a
  one-shot run.
- **`-entry eval_from_embeddings`** — skips predict/split; sources per-experiment
  zarrs from `--embeddings_glob` and runs the same shared `DOWNSTREAM` DAG.

Both call the same `DOWNSTREAM` sub-workflow (`workflows/_downstream.nf`), so the
per-experiment-zarr write-order barriers live in one place.

### Selecting which datasets to evaluate

The **glob is the cohort selector** — the `*` in the dataset slot picks which
datasets; the tail pins model/run/ckpt/marker. No manifest to maintain:

```bash
# every dataset for this model/run/ckpt (progressive default)
--embeddings_glob '.../organelle_dynamics/*/2-phenotyping/predictions/M/R/C/*.zarr'
# one dataset
--embeddings_glob '.../organelle_dynamics/<ds>/2-phenotyping/predictions/M/R/C/*.zarr'
# a subset (brace expansion)
--embeddings_glob '.../organelle_dynamics/{ds_a,ds_b}/.../M/R/C/*.zarr'
# one marker across all datasets
--embeddings_glob '.../organelle_dynamics/*/.../M/R/C/SEC61B.zarr'
```

Add a dataset → predict it (writes into the tree) → re-run the same glob; the
cohort grows implicitly. The eval config YAML is the orthogonal lever: the glob
says *which datasets*, the config says *what to compute*.

### One-command launcher

`dynaclr eval` turns a `(model_family, run, ckpt_name)` identity into the glob
and launches the `eval_from_embeddings` entry:

```bash
uv run dynaclr eval \
    --eval-config applications/dynaclr/configs/evaluation/<config>.yaml \
    --model-family DynaCLR-2D-MIP-BagOfChannels \
    --run 2d-mip-fix-shuffler --ckpt-name epoch105-step84800 \
    [--marker SEC61B] [--datasets ds_a ds_b] [--print-cmd]
```

To run **many models** through train → predict → eval in parallel (chained per model
via SLURM `afterok`), use the model matrix — see
[`../tools/README.md`](../tools/README.md) (`dynaclr run-matrix` + a matrix YAML).

### Predict-only runs

To run inference without downstream evals, use the `evaluation` entry with
`steps: [predict, split]`; rerun with more steps later via `-resume` (predict/
split are skipped because the zarrs already exist on disk). See
[docs/DAGs/evaluation.md](../docs/DAGs/evaluation.md#predict-only-runs-inference-without-downstream-evals).

## Adding a new workflow

Follow this four-step recipe. The `training_preprocessing` workflow is the
reference example — copy its structure.

### 1. Create process modules

Each process is a single `.nf` file under `modules/<your_workflow>/`. Prefer
`val` inputs (not `path`) to avoid Nextflow staging zarr/parquet files —
everything is read/written in place on VAST.

```groovy
// modules/<your_workflow>/my_step.nf
process MY_STEP {
    label 'cpu'                         // picks SLURM resources from nextflow.config

    input:
    val input_path
    val workspace_dir

    output:
    val input_path, emit: result

    script:
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr my-command ${input_path}
    """
}
```

Reuse `modules/shared/` for processes used by more than one workflow.

### 2. Create a named sub-workflow

```groovy
// workflows/my_workflow.nf
include { MY_STEP } from '../modules/my_workflow/my_step'

workflow MY_WORKFLOW {
    take:
        input_path
        workspace_dir

    main:
    MY_STEP(input_path, workspace_dir)
}
```

Use `take:` for inputs so the workflow is composable. `main:` holds the DAG.

### 3. Register an entry wrapper in `main.nf`

```groovy
include { MY_WORKFLOW } from './workflows/my_workflow'

workflow my_workflow {          // lowercase name → -entry my_workflow
    if (!params.input_path) {
        error "ERROR: --input_path is required for -entry my_workflow"
    }
    MY_WORKFLOW(params.input_path, params.workspace_dir)
}
```

The wrapper has two jobs: validate required params and bridge CLI flags into
the `take:` arguments of the sub-workflow. Use lowercase entry names so they
don't clash with the imported UPPERCASE workflow symbol.

### 4. Add params to `nextflow.config`

```groovy
params {
    // ... existing params
    input_path = null           // Required for -entry my_workflow
}
```

Resource labels (`cpu`, `cpu_heavy`, `gpu_2d`, `gpu_3d`, `cpu_light`) are
shared across all workflows — don't redefine them per workflow.

## Conventions

- **Entry workflow names are lowercase** (`evaluation`, `training_preprocessing`).
  Sub-workflow symbols are UPPERCASE (`EVALUATION`, `TRAINING_PREPROCESSING`).
- **Process names are UPPERCASE** (`PREDICT`, `BUILD_CELL_INDEX`).
- **Pass paths as `val`, not `path`** — avoids Nextflow staging large zarrs.
- **Always use `-resume`** — every step re-checks existence on disk.
- **Use `PYTHONNOUSERSITE=1`** (already set in `env { }` block) — prevents
  `~/.local/` from shadowing the conda/uv env.
- **Manifest-driven optional steps**: if a workflow generates a JSON manifest
  (like `prepare-eval-configs`), gate steps with `.filter { it.containsKey(...) }`
  and `.ifEmpty('skip')` so the DAG remains connected when a step is disabled.
  See `workflows/evaluation.nf` for the pattern.
