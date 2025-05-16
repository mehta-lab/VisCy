# Cell Infection Analysis Demo: ImageNet vs DynaCLR

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
- Download necessary data

**WARNING**: The setup script will download approximately 30GB of data. Make sure you have sufficient disk space and a good internet connection.


```bash
bash setup.sh
```

## Data

The demo uses cellular imaging data with the following components:
- Cell images with Phase and RFP (viral sensor) channels
- Cell tracking data
- Infection state annotations

## Key Features

- **Feature Extraction**: Extract features using both ImageNet pre-trained models and specialized DynaCLR models
- **Dimensionality Reduction**: Apply PHATE to reduce high-dimensional features for visualization
- **Time Series Analysis**: Visualize cell trajectories over time
- **Interactive Visualization**: Create plotly-based interactive visualizations with time sliders
- **Comparative Analysis**: Directly compare general vs specialized feature extraction approaches

## Usage

After setting up the environment, activate it and run the demo script:

```bash
conda activate dynaclr
python demo_infection.py
```

The script will generate visualizations in the `./imagenet_vs_dynaclr/infection` directory.

## Utilities

The demo leverages utility functions from `utils.py`:
- `create_plotly_visualization`: Creates interactive embeddings with time slider
- `create_combined_visualization`: Generates side-by-side comparisons of approaches
- Various helper functions for trajectory visualization and image processing

## Output

The demo produces:
- CSV files containing extracted features and embeddings
- Interactive visualizations comparing the two approaches
- Combined visualization showing cell images alongside embeddings
