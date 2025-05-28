# DynaCLR Demos

This repository contains examples and demos for using DynaCLR with VisCy.

## Available Demos

- [ImageNet vs DynaCLR embeddings (cell infection)](examples/DynaCLR/DynaCLR-DENV-VS-Ph)
- [Embedding visualization](examples/DynaCLR/embedding-web-visualization)

## Setup

To run the demos, you need to download the data and activate the environment.

> **Note**: The `download_data.sh` script downloads data to `{$HOME}/data/dynaclr/demo` by default. Modify the script to download the data to a different directory if needed.

```bash
# To setup the environment
bash setup.sh

# To download the data
bash download_data.sh
```

## To Generate DynaCLR Embeddings

Alternatively to the pre-computed shared embeddings, you can run the model following these instructions using the `DynaCLR-DENV-VS-Ph` model as an example.

### Modify the Config File

Open the `dynaclr_denv-vs-ph_test_data.yml` and modify the following to point to your download:

Replace the output path where you want to save the xarray `.zarr` file with the embeddings:

```yaml
callbacks:
- class_path: viscy.representation.embedding_writer.EmbeddingWriter
  init_args:
    output_path: '/TODO_REPLACE_TO_OUTPUT_PATH.zarr'  # Select the path to save
```

Point to the downloaded checkpoint for the desired model (e.g., `DynaCLR-DENV-VS+Ph`):

```yaml
ckpt_path: '/downloaded.ckpt'  # Point to ckpt file
```