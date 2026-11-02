# DynaCLR recipes

Recipes cover input contracts and configuration choices. The
[DAG runbooks](../DAGs/) are the canonical end-to-end workflows and contain the
recommended launch commands.

| Recipe | Use it to |
| --- | --- |
| [Prepare a custom dataset](prepare-custom-dataset.md) | Create the image, tracking, and collection inputs. |
| [Build a cell index](build-cell-index.md) | Build and validate a training-ready parquet. |
| [Train across experiments](train-multi-experiment.md) | Compose and launch a multi-experiment training config. |
| [Choose sampling settings](sampling-strategies.md) | Configure batches, channels, and positive pairs. |
| [Extract embeddings](extract-embeddings.md) | Run collection-driven per-marker inference. |
| [Evaluate embeddings](evaluate-embeddings.md) | Choose the full evaluation runner or an individual step. |
| [Troubleshoot](troubleshooting.md) | Diagnose common configuration and data errors. |

Start with an existing config under
[`applications/dynaclr/configs/`](../../configs/) and change only the dataset,
checkpoint, output, and experiment-specific settings.
