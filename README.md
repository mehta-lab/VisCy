# DynaCLR

DynaCLR: Contrastive Learning of Cellular Dynamics with Temporal Regularization

# DynaCLR Demos

This repository contains examples and demos for using DynaCLR with viscy.

## Available Demos

- [ImageNet vs DynaCLR embeddings (cell infection)](examples/DynaCLR/DynaCLR-DENV-VS-Ph)
- [Embedding visualization](examples/DynaCLR/embedding-web-visualization)

## Demos' Setup

This will:
- Create a `dynaclr` conda environment with all required dependencies
- Install the VISCY library
- Set up the Python kernel for Jupyter notebooks
- Download the following data (~50GB)
    - Pre-computed features for DynaCLR-DENV-VS+Ph and ImageNet
    - Cell tracks for the dataset
    - Human-annotations of cell state (0-background, 1-uinfected , 2-infected)
    - Test dataset of several FOVs ([A/3/*]-uinfected,[A/4/*]-infected)
    - DynaCLR-DENV-VS+Ph weights

Navigate to the folder you want to download the data and run:
```bash
bash setup.sh
```
> **Note**:
> The full functionality is tested on Linux `x86_64` with NVIDIA Ampere/Hopper GPUs (CUDA 12.4).
> The CTC example configs are also tested on macOS with Apple M1 Pro SoCs (macOS 14.7).
> Apple Silicon users need to make sure that they use
> the `arm64` build of Python to use MPS acceleration.
> Tested to work on Linux on the High Performance cluster, and may not work in other environments.
> The commands below assume a Unix-like system.

## To Generate the DynaCLR embeddings:
Alternatively to the pre-computed shared embeddings, one can run the model following these instructions using `DynaCLR-DENV-VS-Ph` model as an example.

### Modify the config file
Open the `dynaclr_denv-vs-ph_test_data.yml` and modify the following to point to your download:

Replace where you want to save the output xarray `.zarr` with the embeddings.

```yaml
    callbacks:
    - class_path: viscy.representation.embedding_writer.EmbeddingWriter
      init_args:
        output_path: '/TODO_REPLACE_TO_OUTPUT_PATH.zarr'  #Select the path to save
```

Point to the downloaded checkpoint for desired model (i.e `DynaCLR-DENV-VS+Ph`)

 ```yaml
 ckpt_path: '/downloaded.ckpt'  # Point to ckpt file
 ```
### Run inference via CLI

Run the following CLI to run inference:
```bash
viscy predict -c dynaclr_denv-vs-ph_test_data.yml
```

