# Evaluation Suite for Representation Learning Models

This evaluation suite provides command-line tools for evaluating and comparing representation learning models on various metrics including temporal smoothness, clustering quality, and displacement analysis.

## Features

- **Config-based CLI tools** - All tools accept YAML config files for reproducible evaluation
- **Memory-optimized computation** - Efficient algorithms that scale to large datasets (100K+ samples)
- **Markdown output** - Results formatted for easy copy-paste to wikis and confluence
- **Colorblind-friendly visualizations** - Uses blue/orange color scheme
- **Extensible design** - Easy to add new evaluation metrics

## Installation

The evaluation suite is part of the viscy package. All dependencies are already included:
- `click` - CLI framework
- `anndata>=0.12.2` - AnnData format
- `numpy`, `scipy` - Numerical operations
- `scikit-learn` - Preprocessing
- `matplotlib`, `seaborn` - Visualization
- `pandas` - Data manipulation
- `pyyaml` - Config file parsing

## CLI Usage

The evaluation suite is accessible through the `viscy-dynaclr` command-line interface. This provides a unified entry point for all DynaCLR evaluation tools.

**View available commands:**
```bash
viscy-dynaclr --help
```

## Available Tools

### 1. Temporal Smoothness Evaluation

Evaluate how well embeddings capture temporal continuity by comparing distances between adjacent frames vs random frame pairs.

**Command:**
```bash
viscy-dynaclr evaluate-smoothness --config configs/example_smoothness.yaml
```

**Alternative (direct module invocation):**
```bash
python -m applications.DynaCLR.evaluation.evaluate_smoothness \
    --config configs/example_smoothness.yaml
```

**Config file format:**
```yaml
models:
  - path: /path/to/model1.zarr
    label: DynaCLR
  - path: /path/to/model2.zarr
    label: ImageNet

evaluation:
  distance_metric: cosine  # or euclidean
  time_offsets: [1]  # temporal offsets to compute
  output_dir: ./results/smoothness
  save_plots: true
  use_optimized: true  # use memory-efficient computation
  verbose: true
```

**Output:**
- Markdown comparison table (printed to stdout)
- Individual CSV files with statistics for each model
- Distribution plots (PDF + PNG) for each model
- Combined CSV with all results

**Metrics:**
- **Smoothness Score**: Ratio of adjacent frame distance to random frame distance (lower is better)
- **Dynamic Range**: Separation between random and adjacent peaks (higher is better)
- Adjacent/Random frame statistics: mean, std, median, peak

### 2. Tracking Statistics

Compute tracking statistics (lineages, mean length, etc.) from zarr or CSV files at FOV, well, and global levels.

**Command:**
```bash
viscy-dynaclr tracking-stats /path/to/data.csv
```

**Options:**
```bash
viscy-dynaclr tracking-stats INPUT_PATH [OPTIONS]

Arguments:
  INPUT_PATH  Path to tracking data (.zarr or .csv)

Options:
  --min-timepoints INT   Minimum timepoints for lineage inclusion (default: 0)
  --levels [fov|well|global|all]  Statistics levels (can repeat, default: all)
  -o, --output PATH      Output file path for markdown results
```

**Examples:**
```bash
# All statistics (default)
viscy-dynaclr tracking-stats /path/to/annotations.csv

# Only well-level stats with minimum 10 timepoints
viscy-dynaclr tracking-stats /path/to/data.zarr --min-timepoints 10 --levels well

# Save to file
viscy-dynaclr tracking-stats /path/to/data.csv -o results/stats.md
```

**CSV Input Requirements:**
- `fov_name`: Field of view identifier (e.g., 'A/1/000000')
- `track_id`: Track identifier
- `t`: Time point index
- `parent_track_id`: Parent track ID for lineage reconstruction (-1 for root)

**Output:**
Markdown tables for easy copy-paste to wikis:
- FOV-Level: lineages, mean length, std per FOV
- Well-Level: aggregated by well_id (first two path components of fov_name)
- Global: total wells, FOVs, lineages, overall mean/std

### 3. Model Comparison Tool

Compare previously saved results from multiple evaluation runs.

**Command:**
```bash
viscy-dynaclr compare-models --config configs/compare_config.yaml
```

**Alternative (direct module invocation):**
```bash
python -m applications.DynaCLR.evaluation.compare_models \
    --config configs/compare_config.yaml
```

**Config file format:**
```yaml
result_files:
  - path: results/model1_stats.csv
    label: Model1
  - path: results/model2_stats.csv
    label: Model2

comparison:
  metrics:
    - smoothness_score
    - dynamic_range
  output_format: markdown
```

## Usage Examples

### Example 1: Evaluate Single Model

Create `my_model.yaml`:
```yaml
models:
  - path: /data/my_model.zarr
    label: MyModel

evaluation:
  distance_metric: cosine
  output_dir: ./results
  save_plots: true
```

Run:
```bash
viscy-dynaclr evaluate-smoothness --config my_model.yaml
```

### Example 2: Compare Multiple Models

Create `compare_models.yaml`:
```yaml
models:
  - path: /data/dynaclr.zarr
    label: DynaCLR
  - path: /data/imagenet.zarr
    label: ImageNet-ConvNext
  - path: /data/sam2.zarr
    label: SAM2

evaluation:
  distance_metric: cosine
  output_dir: ./results/comparison
  save_plots: true
  verbose: true
```

Run:
```bash
viscy-dynaclr evaluate-smoothness --config compare_models.yaml
```

The output will be a markdown table like:
```markdown
## Temporal Smoothness Evaluation

| Model | smoothness_score | dynamic_range | adjacent_frame_mean | adjacent_frame_peak | random_frame_mean | random_frame_peak |
|-------|------------------|---------------|---------------------|---------------------|-------------------|-------------------|
| DynaCLR | 0.4523 | 0.3201 | 0.2134 | 0.1823 | 0.4720 | 0.5024 |
| ImageNet | 0.6834 | 0.1456 | 0.3421 | 0.3102 | 0.5007 | 0.4558 |
| SAM2 | 0.5234 | 0.2345 | 0.2567 | 0.2134 | 0.4901 | 0.4479 |

**Metrics Interpretation**
- Smoothness Score: Lower is better (adjacent frames are closer)
- Dynamic Range: Higher is better (more separation between adjacent and random)

**Best Models**
- Best smoothness_score: DynaCLR (lowest: 0.4523)
- Best dynamic_range: DynaCLR (highest: 0.3201)
```

Copy this table directly to your wiki or confluence page.

## Memory Optimization

The smoothness evaluation uses a memory-optimized algorithm that avoids creating the full pairwise distance matrix:

- **Traditional approach**: Computes N×N distance matrix → ~104GB for 118K samples
- **Optimized approach**: Computes only temporal neighbors + random samples → ~1GB for 118K samples

This is a **~100x memory reduction** that makes evaluation feasible on large datasets.

The optimization is enabled by default (`use_optimized: true`). You can disable it for small datasets (<50K samples) by setting `use_optimized: false` in the config, though the optimized version is recommended for consistency.

## Input Data Requirements

Embeddings must be stored in AnnData format (`.zarr` files) with the following structure:

**Required `.obs` columns:**
- `fov_name`: Field of view identifier
- `track_id`: Track identifier for temporal sequences
- `t`: Time point index

**Example:**
```python
import anndata as ad
import numpy as np
import pandas as pd

# Create sample embedding
n_samples = 1000
n_features = 512

X = np.random.randn(n_samples, n_features)
obs = pd.DataFrame({
    'fov_name': ['fov_0'] * n_samples,
    'track_id': np.repeat(np.arange(100), 10),
    't': np.tile(np.arange(10), 100),
})

adata = ad.AnnData(X=X, obs=obs)
adata.write_zarr('my_embedding.zarr')
```

## Extending the Suite

The evaluation suite is designed for easy extensibility. To add a new evaluation metric:

1. Implement the core metric function in `viscy/representation/evaluation/`
2. Create a new CLI script in `applications/DynaCLR/evaluation/` following the pattern:
   ```python
   import click
   from .utils import load_config, load_embedding, validate_embedding

   @click.command()
   @click.option("--config", type=click.Path(exists=True), required=True)
   def main(config):
       # Load config
       config_dict = load_config(config)

       # Process each model
       for model in config_dict["models"]:
           features_ad = load_embedding(Path(model["path"]))
           validate_embedding(features_ad)

           # Call your metric function
           results = my_metric_function(features_ad)

           # Format and output results
           print(format_results_table(results, columns))
   ```

3. Add an example config file in `configs/`
4. Update this README with usage instructions

**Planned future CLIs:**
- `evaluate_clustering.py` - Clustering metrics (NMI, ARI, kNN accuracy)
- `evaluate_displacement.py` - Temporal displacement (MSD curves)
- `evaluate_all.py` - Master script to run all metrics

## Colorblind-Friendly Visualizations

All plots use a colorblind-friendly palette:
- **Blue** (`#1f77b4`) for adjacent frames
- **Orange** (`#ff7f0e`) for random samples

Avoid using red/green color schemes as specified in the user's instructions.

## Output Formats

### Markdown Tables
Formatted for easy copy-paste to wikis and confluence pages. This is the default output format.

### CSV Files
Machine-readable format for further analysis:
- Individual model results: `{label}_smoothness_stats.csv`
- Combined results: `combined_smoothness_stats.csv`

### Plots
Both PDF (vector) and PNG (raster) formats:
- PDF for publications and presentations
- PNG for quick viewing and web embedding

## Troubleshooting

### Error: "Embedding missing required metadata columns"
Your embedding file is missing required `.obs` columns (`fov_name`, `track_id`, `t`). Check your embedding generation pipeline.

### Error: "No adjacent frame distances found"
Your dataset may not have tracks with multiple timepoints. Check that:
- Tracks have at least 2 timepoints
- The `track_id` and `t` columns are correctly populated

### Memory Issues
If you encounter memory errors:
1. Ensure `use_optimized: true` in your config (this is the default)
2. Reduce `time_offsets` to `[1]` only
3. Set `save_distributions: false` to avoid saving large arrays

## Citation

If you use this evaluation suite in your research, please cite:

```bibtex
@software{viscy_evaluation_suite,
  title = {Evaluation Suite for Representation Learning Models},
  author = {VisCy Development Team},
  year = {2026},
  url = {https://github.com/mehta-lab/viscy}
}
```

## Contributing

To contribute new evaluation metrics or improvements:
1. Follow the existing code structure and patterns
2. Use numpy-style docstrings
3. Add comprehensive examples to this README
4. Ensure colorblind-friendly visualizations
5. Format output in markdown for easy wiki integration

## Support

For questions or issues:
- Open an issue on GitHub: https://github.com/mehta-lab/viscy/issues
- Check the main VisCy documentation
- Refer to example configs in `configs/`
