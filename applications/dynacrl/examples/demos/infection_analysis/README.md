# Cell Infection Analysis Demo: ImageNet vs DynaCLR-DENV-VS+Ph model

This demo compares different feature extraction methods for analyzing infected vs uninfected cells using microscopy images.

As the cells get infected, the red fluorescence protein (RFP) translocates from the cytoplasm into the nucleus.

## Overview

The `demo_infection.py` script demonstrates:

  - PHATE plots from the embeddings generated from  DynaCLR and ImageNet
  - Show the infection progression in cells via Phase and RFP (viral sensor) channels
  - Highlighted trajectories for sample infected and uninfected cells over time

## Key Features

- **Feature Extraction**: Compare ImageNet pre-trained and specialized DynaCLR features
- **Interactive Visualization**: Create plotly-based visualizations with time sliders
- **Side-by-Side Comparison**: Directly compare cell images and PHATE embeddings
- **Trajectory Analysis**: Visualize and track cell trajectories over time
- **Infection State Analysis**: See how different models capture infection dynamics

## Setup

### Download demo data

The `download_data.sh` script downloads the test dataset. By default it saves to `~/data/dynaclr/demo`. You can specify a custom output directory:

```bash
# Default output directory
bash download_data.sh

# Custom output directory
bash download_data.sh /path/to/output
```

For installation instructions, see the [DynaCLR README](../../../README.md).

## Usage

```bash
python demo_infection.py
```

For both of these you will need to ensure to point to the path to the downloaded data:
```python
# Update these paths to your data
input_data_path = "/path/to/registered_test.zarr"
tracks_path = "/path/to/track_test.zarr"
ann_path = "/path/to/extracted_inf_state.csv"

# Update paths to features
dynaclr_features_path = "/path/to/dynaclr_features.zarr"
imagenet_features_path = "/path/to/imagenet_features.zarr"
```

Check out the demo's output visualization:

- [Open Visualization](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/DENV/test/20240204_A549_DENV_ZIKV_timelapse/cell_infection_visualization.html)

Note: You may need to press pause/play for the image to show

## (OPTIONAL) Generating DynaCLR-DENV-VS+PH Features

1. Open the `dynaclr_denv-vs-ph_test_data.yml` and modify the following to point to your download:

- Replace with the output path (`.zarr`) for the embeddings.
```yaml
    callbacks:
    - class_path: viscy_utils.callbacks.embedding_writer.EmbeddingWriter
      init_args:
        output_path: '/TODO_REPLACE_TO_OUTPUT_PATH.zarr'  #Select the path to save
```

- Point to the downloaded checkpoint for DynaCLR-DENV-VS+Ph
```yaml
 ckpt_path: '/downloaded.ckpt'  # Point to ckpt file
 ```

2. Run the following CLI to run inference
```bash
viscy predict -c dynaclr_denv-vs-ph_test_data.yml
```
