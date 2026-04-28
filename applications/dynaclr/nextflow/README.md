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
│   ├── evaluation.nf                  # workflow EVALUATION { take: ... }
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

## Predict-only runs

Use the `evaluation` entry with `steps: [predict, split]` in your eval config
to run inference on a new dataset without any downstream evals. Rerun with
more steps later using `-resume` — predict/split are skipped because
`embeddings.zarr` and `{exp}.zarr` already exist on disk. See
[docs/DAGs/evaluation.md](../docs/DAGs/evaluation.md#predict-only-runs-inference-without-downstream-evals)
for the full pattern.

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
