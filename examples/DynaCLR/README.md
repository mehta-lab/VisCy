# DynaCLR Demos

This repository contains examples and demos to embed cellular dynamics using DynaCLR.

## Available Demos

- [ImageNet vs DynaCLR embeddings (cell infection)](/examples/DynaCLR/DynaCLR-DENV-VS-Ph/README.md)
- [Embedding visualization](/examples/DynaCLR/embedding-web-visualization/README.md)

## Setup

To run the demos, you need to download the data and activate the environment.

> **Note**: The `download_data.sh` script downloads data to `{$HOME}/data/dynaclr/demo` by default. Modify the script to download the data to a different directory if needed.

```bash
# To setup the environment
bash setup.sh

# To download the data
bash download_data.sh
```

## Generate DynaCLR Embeddings

For this demo, we will use the `DynaCLR-DENV-VS-Ph` model as an example.

The datasets and config files for the models can be found:
-  [Test datasets](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/)
-  [Models](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_models/)


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

---
### DynaCLR with classical triplet sampling

To train DynaCLR models using the classical triplet sampling, you need to generate pseudo-tracking data from 2D segmentation masks.

These pseudo-tracks are used to run the same. For more information: [README.md](./DynaCLR-classical-sampling/README.md)

### Exporting DynaCLR models

To export DynaCLR models to ONNX run:

`viscy export -c config.yml`

The `config.yml` is similar to the `fit.yml` which describes the model. An example can be found [here](./examples_cli/dynaclr_microglia_onnx.yml).
