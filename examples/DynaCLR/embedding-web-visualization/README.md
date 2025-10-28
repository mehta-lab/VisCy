# Web-based embedding exploration

## Overview

The `interactive_visualize.py` script allows for embedding visualization and exploration.

## Key Features

- **Interactive Visualization**: Plotly-dash visualization of the embeddings
- Lasso selection to display image clusters
- Display Principal Components and PHATE plots
- Single cell selection

## Setup

The demo uses cellular imaging data with the following components:
- Embeds the dynamic cellular response and plots Principal Components or PHATE

You can download the data from the provided Google Drive links in the script or use your own data by updating the paths:

```python
# Update these paths to the downloaded data
download_root = Path.home() / "data/dynaclr/demo"
viz_config = {
    "data_path": download_root / "registered_test.zarr",  # TODO add path to data
    "tracks_path": download_root / "track_test.zarr",  # TODO add path to tracks
    "features_path": download_root
    / "precomputed_embeddings/infection_160patch_94ckpt_rev6_dynaclr.zarr",  # TODO add path to features
    "channels_to_display": ["Phase3D", "RFP"],
    # TODO: Modify for specific FOVs [A/3/*]- Uinfected and [B/4/*]-Infected for 0-9 FOVs. They will be cached in memory.
    "fov_tracks": {
        "/A/3/9": list(range(50)),
        "/B/4/9": list(range(50)),
    },
    "yx_patch_size": (160, 160),
    "num_PC_components": 8,
}
```

## Usage

After setting up the environment, activate it and run the demo script:

```bash
conda activate dynaclr
python interactive_visualizer.py
```

## Demo

### Embeddings per track (click on the track to see the embeddings)
![embeddings_per_track](/examples/DynaCLR/embedding-web-visualization/demo_imgs/demo2_embeddings_visualization_track.png)

### Clustering (use the lasso to select the embeddings)
![embeddings_per_cluster](/examples/DynaCLR/embedding-web-visualization/demo_imgs/demo2_embedding_visualization_cluster.png)
