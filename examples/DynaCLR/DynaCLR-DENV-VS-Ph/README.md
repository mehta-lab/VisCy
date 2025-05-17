# Cell Infection Analysis Demo: ImageNet vs DynaCLR-DENV-VS+Ph model

This demo compares different feature extraction methods for analyzing infected vs uninfected cells using microscopy images.

## Overview

The `demo_infection.py` script demonstrates:

1. Using ImageNet pre-trained models for feature extraction from cell images
2. Comparing with DynaCLR (Dynamic Contrastive Learning) specialized features
3. Visualizing how each approach distinguishes infected from uninfected cells

## Setup

This will:
- Create a `dynaclr` conda environment with all required dependencies
- Install the VISCY library
- Set up the Python kernel for Jupyter notebooks
- Download the following data (~30GB)
    - Pre-computed features for DynaCLR-DENV-VS+Ph and ImageNet
    - Cell tracks for the dataset
    - Human-annotations of cell state (0-uinfected , 1-infected)
    - Test dataset

```bash
bash setup.sh
```

## Data

The demo uses cellular imaging data with the following components:
- Cell images with Phase and RFP (viral sensor) channels
- Cell tracking data
- Infection state annotations

You can download the data from the provided Google Drive links in the script or use your own data by updating the paths:

```python
# Update these paths to your data
input_data_path = "/path/to/registered_test.zarr"
tracks_path = "/path/to/track_test.zarr"
ann_path = "/path/to/extracted_inf_state.csv"

# Update paths to features 
dynaclr_features_path = "/path/to/dynaclr_features.zarr"
imagenet_features_path = "/path/to/imagenet_features.zarr"
```

## Key Features

- **Feature Extraction**: Compare ImageNet pre-trained and specialized DynaCLR features
- **Interactive Visualization**: Create plotly-based visualizations with time sliders
- **Side-by-Side Comparison**: Directly compare cell images and PHATE embeddings
- **Trajectory Analysis**: Visualize and track cell trajectories over time
- **Infection State Analysis**: See how different models capture infection dynamics

## Usage

After setting up the environment, activate it and run the demo script:

```bash
conda activate dynaclr
python demo_infection.py
```

You can also use the included Jupyter notebook version:

```bash
conda activate dynaclr
jupyter notebook demo_infection.ipynb
```

The script will generate interactive visualizations showing:
- Cell images with Phase and RFP (viral sensor) channels
- PHATE embeddings from both ImageNet and DynaCLR features
- Highlighted trajectories for sample infected and uninfected cells

## Customization

The demo allows for customization:
- Select specific tracks and FOVs for visualization
- Adjust visualization parameters like colors and plot sizes
- Choose different cells to highlight in the embedding space

Example of customizing track selection:
```python
fov_name_mock = "/A/3/9"
track_id_mock = [19]  # Uninfected track
fov_name_inf = "/B/4/9"
track_id_inf = [42]   # Infected track
```

## Utilities

The demo leverages utility functions from `utils.py`:
- `create_plotly_visualization`: Creates interactive embeddings with time slider
- `create_combined_visualization`: Generates side-by-side comparisons of approaches
- `create_image_visualization`: Prepares cell images for display
- Various trajectory visualization and image processing helpers
