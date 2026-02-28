# Temporal Smoothness Evaluation

Evaluate and compare temporal smoothness of cell embedding models. Measures how smoothly embeddings change between adjacent time frames vs random frame pairs.

## Overview

| File | Description |
|------|-------------|
| `evaluate_smoothness.py` | CLI to compute smoothness metrics for one or more models |
| `compare_models.py` | CLI to compare previously saved CSV results |
| `config.py` | Pydantic configuration models |
| `utils.py` | Smoothness-specific utilities |

## Prerequisites

Install DynaCLR with the eval extras:

```bash
pip install -e "applications/dynaclr[eval]"
```

## Workflow

### 1. Evaluate smoothness

Create a config (see `configs/example_smoothness.yaml`):

```yaml
models:
  - path: /path/to/embeddings.zarr
    label: MyModel

evaluation:
  distance_metric: cosine
  output_dir: ./output/smoothness
  save_plots: true
  verbose: true
```

Run the evaluation:

```bash
dynaclr evaluate-smoothness --config configs/example_smoothness.yaml
```

This will:
- Load embeddings from each model's zarr file
- Compute pairwise distances between adjacent and random frame pairs
- Output a markdown comparison table with smoothness metrics
- Save per-model CSV stats and distribution plots to `output_dir`

### 2. Compare results across runs

Once you have CSV results from previous evaluations, create a comparison config (see `configs/example_compare.yaml`):

```yaml
result_files:
  - path: output/smoothness/DynaCLRv3_smoothness_stats.csv
    label: DynaCLRv3
  - path: output/smoothness/ImageNet_smoothness_stats.csv
    label: ImageNet

comparison:
  output_format: markdown
```

Run the comparison:

```bash
dynaclr compare-models --config configs/example_compare.yaml
```

## Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| Smoothness Score | Ratio of adjacent-frame distance to random-frame distance | Lower |
| Dynamic Range | Separation between random and adjacent distribution peaks | Higher |
| Adjacent Frame Mean/Peak | Average/peak distance between consecutive frames | Lower |
| Random Frame Mean/Peak | Average/peak distance between random frame pairs | Higher |
