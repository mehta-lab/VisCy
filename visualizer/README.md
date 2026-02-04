# PHATE Track Viewer

Interactive visualization tool for exploring PHATE embeddings of cell tracking data with multi-dataset support, flexible coloring modes, and track timeline displays.

## Features

- **Multi-Dataset Support**: Load and visualize data from multiple experiments with joint PHATE embeddings
- **Flexible Coloring Modes**: Color points by annotation, time, track ID, or dataset
- **Interactive Track Selection**: Click points to select tracks or use dropdown
- **Track Timelines**: View selected tracks as image timelines across timepoints
- **Trajectory Visualization**: Display temporal trajectories with directional arrows
- **Timepoint Highlighting**: Highlight specific timepoints for selected tracks
- **Colorblind-Friendly**: Uses blue/orange palette by default

## Installation

The visualizer package is part of the viscy repository. Ensure you have the required dependencies:

```bash
pip install dash plotly anndata pandas numpy pillow iohub pyyaml
```

For development and testing:

```bash
pip install -r requirements-test.txt
```

## Quick Start

### Using Configuration Files (Recommended)

Create a `config.yaml` file:

```yaml
datasets:
  - adata_path: /path/to/adata.zarr
    data_path: /path/to/data.zarr
    channels:
      - Phase3D
    z_range: [0, 1]
    fov_filter:
      - A/1
      - A/2
    dataset_id: experiment_1

port: 8050
```

Run from command line:

```bash
python -m visualizer.cli --config config.yaml
```

### Single Dataset (Python API)

```python
from pathlib import Path
from visualizer import create_app, run_app, MultiDatasetConfig, DatasetConfig

config = MultiDatasetConfig(
    datasets=[
        DatasetConfig(
            adata_path=Path("path/to/adata.zarr"),
            data_path=Path("path/to/data.zarr"),
            channels=("Phase3D",),
            z_range=(0, 1),
            fov_filter=["A/1", "A/2"],  # Optional FOV filtering
        )
    ],
    port=8050,
)

app = create_app(config)
run_app(app)
```

### Multiple Datasets with Joint PHATE

```python
config = MultiDatasetConfig(
    datasets=[
        DatasetConfig(
            adata_path=Path("path/to/dataset1_adata.zarr"),
            data_path=Path("path/to/dataset1_data.zarr"),
            channels=("Phase3D", "Nuclei"),
            z_range=(0, 5),
            fov_filter=["B/1", "B/3"],
        ),
        DatasetConfig(
            adata_path=Path("path/to/dataset2_adata.zarr"),
            data_path=Path("path/to/dataset2_data.zarr"),
            channels=("Phase3D",),
            z_range=(0, 1),
            fov_filter=["C/2"],
        ),
    ],
    phate_kwargs={"n_components": 2, "knn": 5, "decay": 40, "scale_embeddings": False},
    port=8050,
)

app = create_app(config)
run_app(app)
```

### With Annotations

```python
config = MultiDatasetConfig(
    datasets=[
        DatasetConfig(
            adata_path=Path("path/to/adata.zarr"),
            data_path=Path("path/to/data.zarr"),
            channels=("Phase3D",),
            z_range=(0, 1),
            annotation_csv=Path("path/to/annotations.csv"),
            annotation_column="infection_status",
            categories={0: "uninfected", 1: "infected"},
        )
    ],
    default_color_mode="annotation",
    port=8050,
)

app = create_app(config)
run_app(app)
```

## Configuration

### DatasetConfig

Configuration for a single dataset source.

**Parameters:**
- `adata_path` (Path): Path to AnnData zarr store with features
- `data_path` (Path): Path to image zarr store
- `channels` (tuple[str, ...]): Channel names to load (e.g., ("Phase3D", "Nuclei"))
- `z_range` (tuple[int, int]): Z slice range (start, end) (e.g., (0, 1))
- `fov_filter` (list[str], optional): FOV patterns to filter (e.g., ["A/1", "B/2"])
- `annotation_csv` (Path, optional): Path to CSV file with annotations
- `annotation_column` (str, optional): Column name in annotation CSV
- `categories` (dict, optional): Dictionary to remap annotation categories
- `dataset_id` (str, optional): Unique identifier (auto-detected if not provided)

### MultiDatasetConfig

Configuration for multi-dataset PHATE viewer.

**Parameters:**
- `datasets` (list[DatasetConfig]): List of dataset configurations. Each dataset must specify its own channels and z_range.
- `phate_kwargs` (dict, optional): PHATE parameters for computing embeddings
  - `None`: Use existing embeddings (single dataset only)
  - `{}`: Recompute with default parameters
  - `{"n_components": 2, "knn": 5, "decay": 40, ...}`: Custom parameters
- `yx_patch_size` (tuple[int, int], default: (160, 160)): Patch size in (Y, X)
- `port` (int, default: 8050): Server port
- `debug` (bool, default: False): Enable debug mode
- `default_color_mode` (str, default: "annotation"): Initial coloring mode

## Module Structure

```
visualizer/
├── __init__.py                # Package exports and version
├── config.py                  # Configuration classes and constants
├── config_loader.py           # YAML/JSON configuration loading
├── data_loading.py            # Data loading and PHATE computation
├── visualization.py           # PHATE plotting functions
├── image_cache.py             # Image loading and caching
├── timeline.py                # Track timeline display
├── app.py                     # Dash application and callbacks
├── cli.py                     # Command-line interface with argparse
├── example_config.py          # Example configuration
├── config.example.yaml        # Example YAML configuration
├── config.example.json        # Example JSON configuration
├── pytest.ini                 # Pytest configuration
├── requirements-test.txt      # Testing dependencies
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── test_config.py
│   ├── test_data_loading.py
│   ├── test_visualization.py
│   └── test_image_cache.py
└── README.md                  # Documentation
```

## Usage

### Running the Application

#### Command-Line Interface

The visualizer provides a flexible CLI with support for configuration files and overrides:

```bash
# Show help message
python -m visualizer.cli --help

# Show version
python -m visualizer.cli --version

# Use example config from code
python -m visualizer.cli

# Load from YAML config
python -m visualizer.cli --config config.yaml

# Load from JSON config
python -m visualizer.cli --config config.json

# Override port and enable debug mode
python -m visualizer.cli --config config.yaml --port 8080 --debug

# Override color mode
python -m visualizer.cli --config config.yaml --color-mode time

# Combined overrides
python -m visualizer.cli --config config.yaml --port 8080 --debug --color-mode dataset
```

#### Python API

```python
from visualizer import create_app, run_app
from visualizer.example_config import CONFIG

app = create_app(CONFIG)
run_app(app, debug=True, port=8050)
```

#### Loading from Configuration Files

```python
from pathlib import Path
from visualizer import load_config_from_yaml, load_config_from_json, create_app, run_app

# Load from YAML
config = load_config_from_yaml(Path("config.yaml"))

# Or load from JSON
config = load_config_from_json(Path("config.json"))

app = create_app(config)
run_app(app)
```

### Coloring Modes

- **Annotation**: Color points by annotation categories (requires annotation data)
- **Time**: Color points by timepoint (continuous colorscale)
- **Track ID**: Color each selected track distinctly
- **Dataset**: Color points by dataset source (multi-dataset mode)

### Interactive Features

- **Click to Select**: Click points on the PHATE plot to select tracks
- **Dropdown Selection**: Use the track dropdown to select multiple tracks
- **Filter by Annotation**: Show/hide annotation categories (annotation mode)
- **Show Trajectories**: Display temporal trajectories with arrows
- **Highlight Timepoint**: Mark specific timepoint with yellow star

### Track Timeline Display

- Shows up to 10 selected tracks
- Displays all channels horizontally for each timepoint
- Color-coded headers by annotation status
- Horizontal scrolling for long tracks
- Fixed channel labels on the left

## Data Requirements

### AnnData Format

The AnnData zarr store must contain:
- `X`: Feature matrix (n_observations × n_features)
- `obsm['X_phate']`: PHATE embeddings (for single dataset with existing embeddings)
- `obs['track_id']`: Track identifiers
- `obs['fov_name']`: Field of view names
- `obs['t']`: Timepoints
- `obs['y']`, `obs['x']`: Centroid coordinates
- `obs['id']`: Observation IDs

### Image Data Format

The image zarr store must be compatible with iohub.open_ome_zarr:
- Multi-position zarr with FOV hierarchy
- Dimensions: (T, C, Z, Y, X)
- Channel names matching `channels` configuration

### Annotation CSV Format

Optional CSV file with columns:
- `id`: Matching observation IDs from AnnData
- `<annotation_column>`: Annotation values (e.g., "infection_status")

## Advanced Features

### Custom PHATE Parameters

```python
config = MultiDatasetConfig(
    datasets=[...],
    phate_kwargs={
        "n_components": 2,
        "knn": 10,           # Number of nearest neighbors
        "decay": 50,         # Decay parameter
        "scale_embeddings": True,
    },
)
```

### Multiple Datasets with Different Channels

Each dataset must specify its own `channels` and `z_range`. This allows different datasets to have different imaging configurations:

```python
config = MultiDatasetConfig(
    datasets=[
        DatasetConfig(
            adata_path=Path("dataset1.zarr"),
            data_path=Path("data1.zarr"),
            channels=("Phase3D", "GFP"),  # Two channels for this dataset
            z_range=(0, 3),                # Larger z-range
        ),
        DatasetConfig(
            adata_path=Path("dataset2.zarr"),
            data_path=Path("data2.zarr"),
            channels=("Phase3D",),         # Single channel
            z_range=(0, 1),                 # Smaller z-range
        ),
    ],
)
```

## Troubleshooting

### FOV Name Mismatches

If images fail to load, check FOV name format:
- AnnData `fov_name` may be shortened (e.g., "001000")
- Zarr store may use full paths (e.g., "A/1/001000")
- The cache automatically attempts to match FOV patterns

Enable debug logging to see matching attempts:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Feature Dimension Mismatch

For multi-dataset mode, all datasets must have the same number of features. The validation will raise an error if dimensions don't match.

### Memory Usage

Large datasets may require significant memory:
- Image caching uses LRU strategy
- Consider filtering FOVs to reduce data size
- Use smaller z_range or yx_patch_size

## Testing

The visualizer package includes a comprehensive test suite using pytest.

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=visualizer --cov-report=term-missing

# Run specific test module
pytest tests/test_config.py -v

# Run tests with specific marker
pytest tests/ -v -m unit
```

### Test Structure

- `test_config.py`: Tests for configuration classes
- `test_data_loading.py`: Tests for data loading (with mocks)
- `test_visualization.py`: Tests for PHATE visualization
- `test_image_cache.py`: Tests for image caching (with mocks)

## Configuration Files

### YAML Format

Example `config.yaml`:

```yaml
datasets:
  - adata_path: /path/to/adata.zarr
    data_path: /path/to/data.zarr
    channels:
      - Phase3D
      - Nuclei
    z_range: [0, 5]
    fov_filter:
      - A/1
      - A/2
    annotation_csv: /path/to/annotations.csv
    annotation_column: infection_status
    categories:
      0: uninfected
      1: infected
    dataset_id: experiment_1

phate_kwargs:
  n_components: 2
  knn: 5
  decay: 40
  scale_embeddings: false

yx_patch_size: [160, 160]
port: 8050
debug: false
default_color_mode: annotation
```

### JSON Format

Example `config.json`:

```json
{
  "datasets": [
    {
      "adata_path": "/path/to/adata.zarr",
      "data_path": "/path/to/data.zarr",
      "channels": ["Phase3D", "Nuclei"],
      "z_range": [0, 5],
      "fov_filter": ["A/1", "A/2"],
      "annotation_csv": "/path/to/annotations.csv",
      "annotation_column": "infection_status",
      "categories": {"0": "uninfected", "1": "infected"},
      "dataset_id": "experiment_1"
    }
  ],
  "phate_kwargs": {
    "n_components": 2,
    "knn": 5,
    "decay": 40,
    "scale_embeddings": false
  },
  "yx_patch_size": [160, 160],
  "port": 8050,
  "debug": false,
  "default_color_mode": "annotation"
}
```

### Saving Configuration Files

```python
from visualizer import save_config_to_yaml, save_config_to_json, MultiDatasetConfig

# Save to YAML
save_config_to_yaml(config, Path("config.yaml"))

# Save to JSON
save_config_to_json(config, Path("config.json"))
```

## Future Enhancements

- Export plots as PNG/SVG
- Save/load viewer state
- Annotation editing interface
- Background PHATE computation

## License

Part of the viscy project. See repository LICENSE for details.
