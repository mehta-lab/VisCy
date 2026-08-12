# Evaluate embeddings

Use the evaluation orchestrator for normal runs. Individual commands are useful
for debugging or rerunning one generated step.

## Choose the runner

| Starting point | Runner |
| --- | --- |
| Checkpoint and cell-index parquet | Nextflow `evaluation` entry |
| Per-marker zarrs in the standard prediction tree | `dynaclr eval` |
| One already-generated step config | The corresponding direct CLI |

The complete commands, config blocks, output tree, and workflow visual are in
the [evaluation runbook](../DAGs/evaluation.md).

## Evaluation config

Create a model leaf under
[`applications/dynaclr/configs/evaluation/`](../../configs/evaluation/) and
compose shared recipes:

```yaml
base:
  - ../recipes/predict.yml
  - ../recipes/reduce.yml

training_config: /path/to/resolved-training-config.yml
ckpt_path: /path/to/checkpoint.ckpt
cell_index_path: /path/to/cell-index.parquet
output_dir: /path/to/evaluation-output

steps:
  - predict
  - split
  - reduce_dimensionality
  - reduce_combined
  - smoothness
  - plot
  - plot_combined
```

Add `mmd`, `linear_classifiers`, `append_annotations`, or
`append_predictions` only when their config blocks are present.

Validate the resolved step configs before launch:

```sh
uv run dynaclr prepare-eval-configs \
  -c applications/dynaclr/configs/evaluation/<config>.yaml
```

## Launch from existing embeddings

```sh
uv run dynaclr eval \
  --eval-config applications/dynaclr/configs/evaluation/<config>.yaml \
  --model-family <model-family> \
  --run <run-name> \
  --ckpt-name <checkpoint-name>
```

## Direct step commands

Use the YAMLs generated under `<output_dir>/configs/`:

| Task | Command |
| --- | --- |
| Reduction | `uv run dynaclr reduce-dimensionality -c <config>` |
| Combined reduction | `uv run dynaclr combined-dim-reduction -c <config>` |
| Smoothness | `uv run dynaclr evaluate-smoothness -c <config>` |
| Plotting | `uv run dynaclr plot-embeddings -c <config>` |
| MMD | `uv run dynaclr compute-mmd -c <config>` |
| Linear probes | `uv run dynaclr run-linear-classifiers -c <config>` |

Biological `compute-mmd` and `compute-mmd --pooled` runs now default to
time-matched control median/MAD normalization followed by one pooled PCA80
basis per marker. The selected PC counts, scree curves, and fitted contracts
are saved under `<output_dir>/representation/`. Required obs columns are
`experiment`, `marker`, `perturbation`, and `hours_post_perturbation`.
Disable both transforms only for an intentional raw sensitivity analysis:

```yaml
representation:
  normalization: none
  pca_variance: null
```

Cross-experiment `--combined` and correction `--over-time` modes remain in
their supplied coordinates so the preprocessing cannot erase the batch effect
being measured.

Copyable standalone configurations are available at
`applications/dynaclr/configs/evaluation/recipes/mmd.yaml` and
`applications/dynaclr/configs/evaluation/recipes/mmd_pooled.yaml`.

Do not embed PCA, UMAP, or PHATE in prediction configs; reductions are separate,
repeatable evaluation steps. For classifier inputs and outputs, see the
[linear-classifier runbook](../linear_classifiers/README.md).
