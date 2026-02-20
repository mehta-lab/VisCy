# DynaCLR Examples

## Quick start

- [quickstart/](quickstart/) — Get started with model inference in Python

## Demos

- [demos/infection_analysis/](demos/infection_analysis/) — Compare ImageNet vs DynaCLR-DENV-VS+Ph embeddings for cell infection analysis
- [demos/embedding_explorer/](demos/embedding_explorer/) — Interactive web-based embedding visualization with Plotly Dash

## Data preparation

- [data_preparation/classical_sampling/](data_preparation/classical_sampling/) — Generate pseudo-tracking data from 2D segmentation masks for classical triplet sampling

## Configs

- [configs/](configs/) — Training (`fit.yml`), prediction (`predict.yml`), and ONNX export (`export_onnx.yml`) configuration files, plus SLURM submission scripts

## Generate DynaCLR Embeddings

The datasets and config files for the models can be found:
- [Test datasets](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/)
- [Models](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_models/)

### Modify the Config File

Open the prediction config and modify the following to point to your download:

Replace the output path where you want to save the xarray `.zarr` file with the embeddings:

```yaml
callbacks:
- class_path: viscy_utils.callbacks.embedding_writer.EmbeddingWriter
  init_args:
    output_path: '/TODO_REPLACE_TO_OUTPUT_PATH.zarr'  # Select the path to save
```

Point to the downloaded checkpoint for the desired model (e.g., `DynaCLR-DENV-VS+Ph`):

```yaml
ckpt_path: '/downloaded.ckpt'  # Point to ckpt file
```

### Exporting DynaCLR models

To export DynaCLR models to ONNX run:

```bash
viscy export -c config.yml
```

An example config can be found at [`configs/export_onnx.yml`](configs/export_onnx.yml).
